import numpy as np

import torch
import torch.nn as nn
import gpytorch

from stable_baselines3 import A2C, PPO, DQN, TD3, DDPG
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gp_models import *
from custom_env import ACTION_OFFSET # adjust from action for env to actual trade

ACCEPTED_MODELS = ['a2c', 'ppo', 'dqn', 'td3'] # td3 and ddpg not working - "normal_kernel_cpu" not implemented for 'Long'

class TradingAgent:
    def __init__(self, model='a2c', use_gp=False, gp_params=None, **kwargs):
        # wrapper around stable_baselines RL implemenetations
        assert model in ACCEPTED_MODELS, 'Unknown RL model, must be in {}'.format(ACCEPTED_MODELS)
        if model == 'a2c':
            self.rl = A2C(**kwargs)
        elif model == 'ppo':
            self.rl = PPO(**kwargs)
        elif model == 'dqn':
            self.rl = DQN(**kwargs)
        elif model == 'td3':
            self.rl = TD3(**kwargs)

        self.use_gp = use_gp
        if self.use_gp:
            assert gp_params is not None, 'Must provide parameters such as training data, number of iterations, etc. for GPR'
            self.n_train = gp_params['n_train']
            self.retraining_iter = gp_params['training_iter']
            self.cvar_limit = gp_params['cvar_limit']
            self.gp_limit = gp_params['gp_limit']

            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            if 'data' in gp_params.keys():
                self.X_train = gp_params['data']['X_train']
                self.y_train = gp_params['data']['y_train']
            else:
                self.X_train = torch.zeros(self.n_train, kwargs['env'].num_features) # hard coded to match dimensions of features
                self.y_train = torch.zeros(self.n_train)
            self.gp = ExactGPModel(self.X_train, self.y_train, self.likelihood)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
            self.opt = torch.optim.Adam(self.gp.parameters(), lr=0.1)

            self.shares = 0
            self.cash = 0
            self.obs = [] # holds up to 2 past observations, helps in keeping X, y aligned

            # for plotting
            self.pred_return = 0
            self.pred_lower = 0
            self.pred_upper = 0

            # for debugging
            self.goal_num_shares = 0

    def learn(self, n_steps):
        # when using gp, load pretrained rl agent - no need to train
        if self.use_gp:
            # train GP using fixed number of steps
            self.__train_gp(100)
        else:
            # train RL agent
            self.rl.learn(n_steps)

    def predict(self, obs, deterministic):
        action, state = self.rl.predict(obs, deterministic=deterministic)

        if self.use_gp:
            # slightly retrain
            self.__train_gp(self.retraining_iter, retrain=True)

            # predict next step returns and CI using GP
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.gp(torch.Tensor(obs[2:])[None])
                obs_pred = self.likelihood(output)
                f_mean = output.mean.detach().numpy()[0]
                self.pred_return = f_mean.item()
                f_samples = output.sample(sample_shape=torch.Size((10000,))).detach().numpy()
                lower, upper = obs_pred.confidence_region()
                self.pred_lower = lower.item()
                self.pred_upper = upper.item()

            rl_action = action
            action -= ACTION_OFFSET # adjust from action for env to see actual trade

            # adjust trade size given prediction
            # if self.shares > 0: # long position
            if f_mean > self.gp_limit: # predict positive return over certain threshold
                tail_samples = f_samples[f_samples < lower.item()]
                ps_cvar = np.mean(tail_samples) if len(tail_samples) > 0 else lower.item() # cvar per share
                if ps_cvar < 0:
                    goal_num_shares = self.cvar_limit // ps_cvar
                else:
                    goal_num_shares = self.shares + action # positive return for long - no adjustment needed
                action = min(10, max(0, goal_num_shares - self.shares))
            elif f_mean < -self.gp_limit:
                tail_samples = f_samples[f_samples > upper.item()]
                ps_cvar = np.mean(tail_samples) if len(tail_samples) > 0 else upper.item() # cvar per share
                if ps_cvar < 0:
                    goal_num_shares = self.shares + action # negative return for short - no adjustment needed
                else:
                    goal_num_shares = self.cvar_limit // ps_cvar
                action = max(-10, min(0, goal_num_shares - self.shares))
            else:
                goal_num_shares = self.shares + action
            # print(ps_cvar, lower.item(), upper.item())

            # if not np.isnan(goal_num_shares):
            self.goal_num_shares = goal_num_shares
            # if action > 0: # buy order
            #     action = min(10, max(0, goal_num_shares - self.shares)) # restrict same size trades as original, maintain same direction
            #     # print(goal_num_shares - self.shares, action)
            # elif action < 0: # sell order
            #     action = max(-10, min(0, goal_num_shares - self.shares)) # restrict same size trades as original, maintain same direction

            action += ACTION_OFFSET # adjust for env actions being 1 to N rather than -N/2 to N/2

            # print(f_mean, ps_cvar, self.shares, goal_num_shares, rl_action-ACTION_OFFSET, action-ACTION_OFFSET)

        return action, state

    def update(self, obs, reward=None):
        self.obs.append(obs)
        self.shares, self.cash = obs[:2]
        if reward is not None:
            self.X_train = torch.cat((self.X_train, torch.Tensor(self.obs.pop(0)[2:])[None]))[1:] # self.X_train[1:]
            self.y_train = torch.cat((self.y_train, torch.Tensor([reward])))[1:]

        # print(self.X_train, self.y_train)

        self.gp.set_train_data(self.X_train, self.y_train)

    def save(self, rl_path, gp_path=None):
        self.rl.save(rl_path)
        if gp_path is not None:
            torch.save(self.gp.state_dict(), gp_path)

    def load(self, rl_path, gp_path=None):
        self.rl = A2C.load(rl_path)
        if gp_path is not None:
            state_dict = torch.load(gp_path)
            self.gp.load_state_dict(state_dict)

    def __train_gp(self, n_iter, retrain=False):
        # train GP using fixed number of steps
        self.gp.train()
        self.likelihood.train()

        for i in range(n_iter):
            output = self.gp(self.X_train)
            loss = -self.mll(output, self.y_train)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.gp.eval()
        self.likelihood.eval()
