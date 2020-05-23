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
from tensortrade.agents import DQNAgent
from tensortrade.algorithms.algorithms import simple_trend


ssl._create_default_https_context = ssl._create_unverified_context # Only used if pandas gives a SSLError

cdd = CryptoDataDownload()

data = pd.concat([
    cdd.fetch("Coinbase", "USD", "BTC", "1h").add_prefix("BTC:"),
], axis=1)
data = data.rename({"BTC:date": "date"}, axis=1)
print(data.head())


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
data_bts_close = np.empty(0)
while not done:
    # action = np.random.randint(-1, 2)
    action = simple_trend(data_bts_close)
    next_state, reward, done, info = env.step(action)
    tmp = np.array(next_state[-1, 0], ndmin=1)
    data_bts_close = np.concatenate([data_bts_close, tmp])
    if iter % 1000 == 0:
        print(f'Finish iteration {iter}')

    iter += 1

# agent = DQNAgent(env)
# root_path = '/home/a.antipin/work/trade/tensortrade'
# agent.train(n_steps=200, n_episodes=1, save_path=os.path.join(root_path, 'examples', 'agents'))

portfolio.performance.plot()
portfolio.performance.net_worth.plot()

print('Simulation done!')
