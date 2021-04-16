import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 18})

from custom_env import *

def engineer_features(df, columns_to_lag=['Open', 'High', 'Low', 'Adj Close'], lookback=5):
    # adjust for dividends
    for col in columns_to_lag:
        if col != 'Adj Close':
            df[col] = df[col] + (df['Adj Close'] - df['Close'])
            
    # calculate exponential moving averages (short and long?)
    normalize_columns = []
    for m in [5, 20, 50]:
        num = 0
        denom = 0
        exp_ma = []
        for p in df['Adj Close']:
            num = np.exp(-1/m)*num + p
            denom = np.exp(-1/m)*denom + 1
            exp_ma.append(num/denom)
        col_name = 'Exp MA {}'.format(m)
        df[col_name] = exp_ma
        normalize_columns.append(col_name)
            
    # create lags in data points
    dfs = [df.copy()]
    normalize_columns += columns_to_lag.copy()
    for i in range(1, lookback+1):
        lag = f' -{i}'
        new_col_names = [cn + lag for cn in columns_to_lag]
        normalize_columns += new_col_names
        new_col_map = dict(zip(columns_to_lag, new_col_names))
        temp_df = df[columns_to_lag].shift(i).rename(mapper=new_col_map, axis=1)
        dfs.append(temp_df)
    output_df = pd.concat(dfs, axis=1)
    
    # normalize relative to Open on first day
    base_col = f'Open -{lookback}'
    normalize_columns.remove(base_col)
    for col in normalize_columns:
        output_df[col] = 100*(output_df[col] - output_df[base_col]) / output_df[base_col]
        
    # align target (returns) with regressors 
    output_df['Returns'] = 100*output_df['Returns'].shift(-1)
    
    return output_df.dropna()

def evaluate(agent, 
             ticker, 
             val_data, 
             initial_port_value, 
             borrowing, 
             long_only, 
             n_steps=None, 
             use_gp=False, 
             plot=False, 
             show_plots=True, 
             save_plots=False, 
             env_type='no_restrictions'):
    # Test the trained agent
    val_env = SingleStockTradingEnv('../data/{}_validation.csv'.format(ticker.lower()),
                                    engineer_features, 
                                    initial_value=initial_port_value, 
                                    borrowing=borrowing, 
                                    long_only=long_only)
    val_data = val_data.iloc[6:] # 6 to match with feature engineered lag start in env
    obs = val_env.reset()
    if use_gp:
        agent.update(obs)
        agent_predictions = []
    if n_steps is None:
        n_steps = val_env.data.shape[0]
    portfolio_values = []
    agent_holdings = []
    agent_actions = []
    goal_num_shares = []
    total_reward = 0
    for step in range(n_steps):
        action, _ = agent.predict(obs, deterministic=True)
        agent_actions.append(action - ACTION_OFFSET)
        # print("Step {}".format(step + 1))
        # print("Action: ", action)
        obs, reward, done, info = val_env.step(action)
        if use_gp:
            agent_predictions.append([agent.pred_return, agent.pred_lower, agent.pred_upper])
            goal_num_shares.append(agent.goal_num_shares)
            # agent.update(obs, reward)
            agent.update(obs, 100*val_data['Returns'].iloc[step]) # to match scale of other features
        else:
            goal_num_shares.append(val_env.agent_shares)

        if not np.isnan(reward):
            portfolio_values.append(val_env.agent_portfolio_value)
            total_reward += reward
            agent_holdings.append(val_env.agent_shares)
        # print('reward=', reward, 'done=', done)
        # env.render(mode='console')
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break

    print(total_reward)

    # print returns over time period
    print(portfolio_values[-1] / initial_port_value, \
          # agent_portfolio_values[1][-1] / initial_port_value, \
          np.exp(val_data['Returns'][:n_steps].cumsum()).iloc[-1])
    
    if plot:
        # visualize PnL
        n_plots = 3 if use_gp else 2
        agent_label = 'A2C + GP' if use_gp else 'A2C'
        fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(20,5*n_plots))
        axes[0].set_title(agent_label + ' - ' + ticker)
        axes[0].plot(val_data['Date'][:n_steps], np.exp(val_data['Returns'][:n_steps].cumsum())*initial_port_value, label='Buy and Hold')
        axes[0].plot(val_data['Date'][:n_steps], portfolio_values, label=agent_label)
        axes[0].set_ylabel('Portfolio Value')
        axes[0].legend()
        
        # axes[1].plot(val_data['Date'][:n_steps], agent_holdings, marker='o', label='Agent Shares')
        axes[1].fill_between(val_data['Date'][:n_steps], agent_holdings, label='Agent Shares', alpha=0.75)
        axes[1].set_ylabel('Number of Shares')

        if use_gp:
            agent_predictions = np.array(agent_predictions).T
            axes[2].plot(val_data['Date'][:n_steps], agent_predictions[0], 'b', label='GP Predicted Return')
            axes[2].plot(val_data['Date'][:n_steps], 100*val_data['Returns'][:n_steps], 'r', alpha=0.5, label='True Return')
            axes[2].fill_between(val_data['Date'][:n_steps], agent_predictions[1], agent_predictions[2], alpha=0.5, label='GP Prediction CI')
            axes[2].set_ylabel('Return (%)')
            axes[2].legend(loc='lower left')

        # fig.suptitle(agent_label + ' - ' + ticker)

        if save_plots:
            fig.savefig('figures/rl_performance_comparison_{}.jpg'.format(env_type), bbox_inches='tight')
        if show_plots:
            plt.show()

    return portfolio_values, agent_holdings, agent_actions, goal_num_shares, fig
