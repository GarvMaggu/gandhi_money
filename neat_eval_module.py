import neat
import numpy as np
import gym as gym

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import importlib
import trading_env
importlib.reload(trading_env)

# df = pd.read_csv("./binanceData/combined_csv.csv")
# tickers = ["ETHUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "BTCUSDT", "BNBUSDT"]
tickers = ["ETHUSDT"]
column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore', 'ticker']
column_names2 = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
timeframe = '1h'
df = pd.DataFrame(columns=column_names)

start_date = "2019-03"
end_date = "2023-03"

start_date_dt = datetime.strptime(start_date, "%Y-%m")
end_date_dt = datetime.strptime(end_date, "%Y-%m")

current_date = start_date_dt
all_filenames = []
while current_date <= end_date_dt:
    all_filenames.append(current_date.strftime("%Y-%m") + ".csv")
    current_date += relativedelta(months=1)

dfs = []

for ticker in tickers:
    ticker_df = pd.DataFrame(columns=column_names)
    for file_name in all_filenames:
        temp_df = pd.read_csv('./binanceData/spot/monthly/klines/'+ticker+'/'+timeframe +
                              '/'+ticker+'-'+timeframe+'-' + file_name, header=None, names=column_names2)
        temp_df['ticker'] = ticker
        # replace all 0's with 1 in volume column
        temp_df['volume'].replace(to_replace=0, value=1, inplace=True)
        # combined_data = combined_data.c(df, ignore_index=True)
        df = pd.concat([df, temp_df], ignore_index=True)
        ticker_df = pd.concat([ticker_df, temp_df], ignore_index=True)
    dfs.append(ticker_df)

print(df)

runs_per_net = 2

# Use the NN network phenotype and the discrete actuator force function.


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        trading_env.doRegister()
        env = gym.make('stocks-v0', df=df, dfs=dfs, frame_bound=(50, 30000), window_size=50,
                       prediction_size=20, target_percentage=1.0, stoploss_percentage=1.0, isTraining=True)

        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            # Extract values from observation dictionary
            position = observation['position']
            current_trend = observation['current_trend']
            pnl = observation['pnl'][0]  # assuming pnl is a 1-element array
            prices = observation['prices']

            # Combine all values into a single 1D list
            observation_list = [position, current_trend, pnl] + prices.tolist()

            # Now use this list with the activate function
            action = np.argmax(net.activate(observation_list))
            observation, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)
