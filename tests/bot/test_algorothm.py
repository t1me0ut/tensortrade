import ssl
import pandas as pd
import os
import numpy as np

from tensortrade.utils import CryptoDataDownload
from tensortrade.data import Node, Module, DataFeed, Stream, Select
from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.data import Stream, DataFeed, Module
from tensortrade.instruments import USD, BTC, ETH
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.environments import TradingEnvironment
# from tensortrade.agents import DQNAgent
from tensortrade.algorithms.algorithms import simple_trend, mooving_avg
import matplotlib.pyplot as plt


ssl._create_default_https_context = ssl._create_unverified_context # Only used if pandas gives a SSLError

cdd = CryptoDataDownload()

data = pd.concat([
    cdd.fetch("Coinbase", "USD", "BTC", "1h").add_prefix("BTC:"),
], axis=1)
data = data.rename({"BTC:date": "date"}, axis=1)


idx = list(range(15000, 20000))

data = data.iloc[idx, :]
print(data.head())

date = data['date'].values
bitcoin_price_close = data['BTC:close'].values
bitcoin_price_open = data['BTC:open'].values
bitcoin_price_low = data['BTC:low'].values
bitcoin_price_high = data['BTC:high'].values
N_AVG = 18
bitcoin_price_close_avg = np.convolve(bitcoin_price_close, np.ones((N_AVG,))/N_AVG, mode='same')


features = []
for c in data.columns[1:]:
    s = Stream(list(data[c])).rename(data[c].name)
    features += [s]

feed = DataFeed(features)
feed.compile()

print(feed.next())

coinbase = Exchange("coinbase", service=execute_order)(
    Stream(list(data["BTC:close"])).rename("USD-BTC"),
)

portfolio = Portfolio(USD, [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 0 * BTC),
])

env = TradingEnvironment(
    feed=feed,
    portfolio=portfolio,
    use_internal=False,
    action_scheme="action-scheme-simple",
    reward_scheme="risk-adjusted",
    window_size=20
)

env.max_episodes = 1
env.max_steps = 100

done = None
iter = 0
action = list()
data_bts_close = np.empty(0)
data_bts_low = np.empty(0)
data_bts_high = np.empty(0)
while not done:
    # action = np.random.randint(-1, 2)
    # action = simple_trend(data_bts_close)
    action.append(mooving_avg(data_bts_close))
    next_state, reward, done, info = env.step(action[-1])
    data_bts_close = np.concatenate([data_bts_close, np.array(next_state[-1, 0], ndmin=1)])
    data_bts_low = np.concatenate([data_bts_low, np.array(next_state[-1, 1], ndmin=1)])
    data_bts_high = np.concatenate([data_bts_high, np.array(next_state[-1, 2], ndmin=1)])
    if iter % 200 == 0:
        print(f'Finish iteration {iter}')

    iter += 1

# agent = DQNAgent(env)
# root_path = '/home/a.antipin/work/trade/tensortrade'
# agent.train(n_steps=200, n_episodes=1, save_path=os.path.join(root_path, 'examples', 'agents'))

orders = env.broker.executed

# print(portfolio.performance.head())

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax1 = fig.add_subplot(121)
ax1.plot(data['date'].values, bitcoin_price_close)
ax1.plot(data['date'].values, bitcoin_price_close_avg)
for ii in range(len(bitcoin_price_low)):
    ax1.vlines(x=data['date'].values[ii], ymin=bitcoin_price_low[ii], ymax=bitcoin_price_high[ii])

account = portfolio.performance['net_worth'].values

ax2 = fig.add_subplot(122)
ax2.plot(data['date'].values, account[0:-2:2])
plt.show()

print('Simulation done!')
