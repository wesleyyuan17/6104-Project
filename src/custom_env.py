import pandas as pd
import numpy as np

import gym
from gym import spaces

N_DISCRETE_ACTION = 21 # buy/sell/hold - single unit, expand to buying more/less shares?
ACTION_OFFSET = int(N_DISCRETE_ACTION/2)

class SingleStockTradingEnv(gym.Env):
    '''
    Implements trading environment wrapper in OpenAI gym style for easy experimentation
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 stock_data,
                 feature_func, 
                 lookback=5, 
                 initial_value=10000, 
                 borrow_limit=2,
                 short_limit=50, 
                 borrowing=True, 
                 long_only=False):
        # read in price data - assumes trade sizes are small enough to not affect day-to-day price movements
        df = pd.read_csv(stock_data)
        if 'Date' in df.columns:
            df.drop('Date', axis=1, inplace=True)
        self.buy_in_prices = (df['Open'] + df['Adj Close'] - df['Close']).shift(-1)[6:].reset_index(drop=True).ffill() # 6 to match up number of days cut off in feature engineering from lag
        self.close_prices = df['Adj Close'].shift(-1)[6:].reset_index(drop=True).ffill() # 6 to match up number of days cut off in feature engineering from lag
        self.__engineer_features = feature_func
        self.data = self.__engineer_features(df)
        self.features = self.data.drop(['Volume', 'Returns', 'Close', f'Open -{lookback}'], axis=1).values
        self.rewards = 1 + self.data['Returns'].values / 100
        self.num_features = self.features.shape[1] + 2

        # define action and observation spaces
        self.action_space = spaces.Discrete(N_DISCRETE_ACTION)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,))

        # keep track of state of game
        self.terminal = False
        self.curr_state = 0 # index of current data point
        self.borrow_limit = -borrow_limit*initial_value # restrict amount of borrowing allowed?
        self.short_limit = short_limit # restrict amount of shorting allowed
        self.borrowing = borrowing # will we allow cash to go negative
        self.long_only = long_only # allowed to short?

        # keep track of state of agent
        self.agent_shares = 0
        self.agent_cash = initial_value
        self.agent_returns = []
        self.initial_value = initial_value
        self.agent_portfolio_value = initial_value

    def step(self, action):
        action -= ACTION_OFFSET # turn from 0/1/2 into -1/0/1 for sell/hold/buy 

        if self.long_only:
            self.__long_only_step(action)
        else:
            self.__standard_step(action)

        curr_portfolio_value = self.agent_cash + self.agent_shares*self.close_prices[self.curr_state]
        reward = curr_portfolio_value - self.agent_portfolio_value # difference in value is reward
        self.agent_returns.append(np.log(curr_portfolio_value / self.agent_portfolio_value))
        self.agent_portfolio_value = curr_portfolio_value
        self.curr_state += 1
        if self.curr_state >= len(self.close_prices):
            self.terminal = True

        state = np.array([0]*self.num_features) if self.terminal else np.append([self.agent_shares, self.agent_cash], self.features[self.curr_state])

        return state, reward, self.terminal, {}

    def __long_only_step(self, action):
        if action < 0:
            # sell only as many shares as currently hold
            if self.agent_shares < -action:
                shares_sold = self.agent_shares
                self.agent_shares = 0
                self.agent_cash += shares_sold * self.close_prices[self.curr_state]
            else:
                self.agent_shares += action
                self.agent_cash -= action * self.buy_in_prices[self.curr_state]
        else:
            self.__standard_step(action)

    def __standard_step(self, action):
        # trade_dir = np.sign(action)
        if self.borrowing:
            if action < 0:
                max_shares = max(action, -(self.agent_shares + self.short_limit)) # limit on number of shares to be sold
                self.agent_shares += max_shares # self.shares_per_transaction 
                self.agent_cash -= max_shares * self.buy_in_prices[self.curr_state] # self.shares_per_transaction * self.buy_in_prices[self.curr_state]
            elif action > 0:
                max_shares = max(0, (self.agent_cash - self.borrow_limit) // self.buy_in_prices[self.curr_state])
                self.agent_shares += max_shares
                self.agent_cash -= max_shares * self.buy_in_prices[self.curr_state]
        else:
            if action < 0:
                self.agent_shares += action
                self.agent_cash -= action * self.buy_in_prices[self.curr_state]
            elif action > 0:
                max_shares = max(0, self.agent_cash // self.buy_in_prices[self.curr_state])
                self.agent_shares += max_shares
                self.agent_cash -= max_shares * self.buy_in_prices[self.curr_state]

    def reset(self):
        self.curr_state = 0
        self.agent_portfolio_value = self.initial_value
        self.agent_shares = 0
        self.agent_cash = self.initial_value
        self.agent_return = 0
        self.terminal = False

        return np.append([self.agent_shares, self.agent_cash], self.features[self.curr_state])

    def render(self, mode='human'):
        pass

    def get_env(self):
        pass