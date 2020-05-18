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


ssl._create_default_https_context = ssl._create_unverified_context # Only used if pandas gives a SSLError

cdd = CryptoDataDownload()

data = pd.concat([
    cdd.fetch("Coinbase", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Coinbase", "USD", "ETH", "1h").add_prefix("ETH:")
], axis=1)
data = data.drop(["ETH:date"], axis=1)
data = data.rename({"BTC:date": "date"}, axis=1)

print(data.head())


def rsi(price: Node, period: float):
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)


def macd(price: Node, fast: float, slow: float, signal: float) -> Node:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


features = []
for c in data.columns[1:]:
    s = Stream(list(data[c])).rename(data[c].name)
    features += [s]

btc_close = Select("BTC:close")(*features)
eth_close = Select("ETH:close")(*features)

features += [
    rsi(btc_close, period=20).rename("BTC:rsi"),
    macd(btc_close, fast=10, slow=50, signal=5).rename("BTC:macd"),
    rsi(eth_close, period=20).rename("ETH:rsi"),
    macd(eth_close, fast=10, slow=50, signal=5).rename("ETH:macd")
]

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
while not done:
    action = np.random.randint(-1, 2)
    # action = 1
    next_state, reward, done, _ = env.step(action)
    if iter % 1000 == 0:
        print(f'Finish iteration {iter}')

    iter += 1

# agent = DQNAgent(env)
# root_path = '/home/a.antipin/work/trade/tensortrade'
# agent.train(n_steps=200, n_episodes=1, save_path=os.path.join(root_path, 'examples', 'agents'))

portfolio.performance.plot()
portfolio.performance.net_worth.plot()

print('Simulation done!')
