import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn

from stable_baselines3 import A2C
import gpytorch

from custom_env import *
from model import *
from data_util import *

# TICKERS = ['SPY', 'XLB'] 
TICKERS = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLV']
INITIAL_PORTFOLIO_VALUE = 1000
BORROWING = True
LONG_ONLY = False

def train_rl_agent(ticker):
    # initialize training structures
    train_data_path = '../data/{}_train.csv'.format(ticker.lower())
    train_data = pd.read_csv(train_data_path)
    env = SingleStockTradingEnv(train_data_path,
                                engineer_features, 
                                initial_value=INITIAL_PORTFOLIO_VALUE,
                                borrowing=BORROWING,
                                long_only=LONG_ONLY)

    # create and train agent
    agent = A2C('MlpPolicy', env, gamma=0.1)
    for i in range(10):
        print(ticker, i, env.data.shape[0])
        env.reset()
        agent.learn(env.data.shape[0]) # go through whole history based on each training run
    agent.save('checkpoints/{}_rl_no_restrictions'.format(ticker.lower()))

def compare_models(ticker):
    # initialize structures for evaluation
    train_data_path = '../data/{}_train.csv'.format(ticker.lower())
    val_data_path = '../data/{}_validation.csv'.format(ticker.lower())
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    val_data['Date'] = pd.to_datetime(val_data['Date'])
    env = SingleStockTradingEnv(train_data_path, 
                                engineer_features,
                                initial_value=INITIAL_PORTFOLIO_VALUE, 
                                borrowing=BORROWING,
                                long_only=LONG_ONLY)

    # run evaluation for just RL agent
    rl_checkpoint_path = 'checkpoints/{}_rl_no_restrictions'.format(ticker.lower())
    a2c = A2C.load(rl_checkpoint_path)
    rl_portfolio_values, rl_agent_holdings, rl_agent_actions, rl_goal_num_shares, rl_fig = evaluate(a2c, 
                                                                                                    ticker,
                                                                                                    val_data, 
                                                                                                    INITIAL_PORTFOLIO_VALUE, 
                                                                                                    BORROWING, 
                                                                                                    LONG_ONLY, 
                                                                                                    use_gp=False, 
                                                                                                    plot=True, 
                                                                                                    show_plots=False,
                                                                                                    save_plots=False, 
                                                                                                    env_type='no_restrictions')
    # get features for GP's
    lookback = 5
    train_features = engineer_features(train_data, lookback=lookback)

    # turn data in dataframes into model inputs
    X_train = torch.Tensor(train_features.drop(['Date', 'Volume', 'Returns', 'Close', f'Open -{lookback}'], axis=1).values)
    y_train = torch.Tensor(train_features['Returns'].values)

    gp_params = {'n_train': 20,
                 'training_iter': 10,
                 'cvar_limit': -5, # maximum loss tolerance %
                 'gp_limit': 0.3, # predicted magnitude of GPR such that GP takes over
                 'data': {'X_train': X_train[-20:], 'y_train': y_train[-20:]} # last month's worth of data?
                 }

    # run evaluation for RL w/ GP agent
    a2c_gp = TradingAgent(use_gp=True, gp_params=gp_params, policy='MlpPolicy', env=env)
    a2c_gp.load(rl_path=rl_checkpoint_path)
    a2c_gp.learn(5000)
    a2c_gp.save(rl_path='checkpoints/{}_a2c_gp_no_restrictions_rl'.format(ticker.lower()), 
                gp_path='checkpoints/{}_a2c_gp_no_restrictions_gp'.format(ticker.lower()))
    gp_portfolio_values, gp_agent_holdings, gp_agent_actions, gp_goal_num_shares, gp_fig = evaluate(a2c_gp, 
                                                                                                    ticker,
                                                                                                    val_data, 
                                                                                                    INITIAL_PORTFOLIO_VALUE, 
                                                                                                    BORROWING, 
                                                                                                    LONG_ONLY, 
                                                                                                    use_gp=True, 
                                                                                                    plot=True, 
                                                                                                    show_plots=False,
                                                                                                    save_plots=False, 
                                                                                                    env_type='no_restrictions')
    # plot some stuff that might be interesting to look at
    comp_fig = plt.figure(figsize=(20,5))
    plt.plot(val_data['Date'].iloc[6:], np.exp(val_data['Returns'].iloc[6:].cumsum())*INITIAL_PORTFOLIO_VALUE, label='Buy and Hold')
    plt.plot(val_data['Date'].iloc[6:], rl_portfolio_values, label='A2C')
    plt.plot(val_data['Date'].iloc[6:], gp_portfolio_values, label='A2C + GP')
    plt.title('Performance Comparison - {}'.format(ticker))
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()

    actions_fig = plt.figure(figsize=(20,5))
    plt.plot(val_data['Date'].iloc[6:], rl_agent_actions, label='RL Actions')
    plt.plot(val_data['Date'].iloc[6:], gp_agent_actions, label='GP Actions')
    plt.title('Actions Comparison - {}'.format(ticker))
    plt.legend()

    shares_fig = plt.figure(figsize=(20,5))
    plt.plot(val_data['Date'].iloc[6:], rl_agent_holdings, label='RL Current # Shares')
    plt.plot(val_data['Date'].iloc[6:], gp_goal_num_shares, label='GP Target # Shares')
    plt.plot(val_data['Date'].iloc[6:], gp_agent_holdings, label='GP Current # Shares')
    plt.title('Holdings Comparison - {}'.format(ticker))
    plt.legend()

    # plt.show()

    # save figures
    # rl_fig.savefig('figures/{}_rl_base_no_restrictions.pdf'.format(ticker.lower()), bbox_inches='tight')
    gp_fig.savefig('figures/{}_rl_with_gp_no_restrictions.pdf'.format(ticker.lower()), bbox_inches='tight')
    comp_fig.savefig('figures/{}_rl_gp_comparison_no_restrictions.pdf'.format(ticker.lower()), bbox_inches='tight')
    actions_fig.savefig('figures/{}_actions_comparison_no_restrictions.pdf'.format(ticker.lower()), bbox_inches='tight')
    shares_fig.savefig('figures/{}_num_shares_comparison_no_restrictions.pdf'.format(ticker.lower()), bbox_inches='tight')

    # Calculate and output Sharpe ratios (assume risk-free rate is 0)
    base_log_returns = np.diff(np.log(val_data['Adj Close']))
    base_daily_vol = np.std(base_log_returns)
    base_sharpe = np.sqrt(252) * np.mean(base_log_returns) / base_daily_vol

    rl_log_returns = np.diff(np.log(rl_portfolio_values))
    rl_daily_vol = np.std(rl_log_returns)
    rl_sharpe = np.sqrt(252) * np.mean(rl_log_returns) / rl_daily_vol

    gp_log_returns = np.diff(np.log(gp_portfolio_values))
    gp_daily_vol = np.std(gp_log_returns)
    gp_sharpe = np.sqrt(252) * np.mean(gp_log_returns) / gp_daily_vol

    print('Base: {:.4f}, {:.4f}\tA2C: {:.4f}, {:.4f}\tA2C+GP: {:.4f}, {:.4f}'.format(base_sharpe, base_daily_vol,
                                                                                     rl_sharpe, rl_daily_vol,
                                                                                     gp_sharpe, gp_daily_vol))


def main():
    for tic in TICKERS:
        print('Comparison for {}'.format(tic))
        # train_rl_agent(tic)
        compare_models(tic)

if __name__ == '__main__':
    main()