import gym as gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import random
import pandas as pd

INITIAL_TRADING_BALANCE = 10000


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class Calls(Enum):
    LongEntry = 0
    LongExit = 1
    ShortEntry = 2
    ShortExit = 3
    Hold = 4


class Positions(Enum):
    Short = 0
    Long = 1
    Nothing = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


def getTrend(prices):
    # Calculate the 10-period simple moving average
    short_moving_average = np.convolve(prices, np.ones(10)/10, mode='valid')
    # Calculate the 20-period simple moving average
    long_moving_average = np.convolve(prices, np.ones(20)/20, mode='valid')

    # Compare the moving averages
    if short_moving_average[-1] > long_moving_average[-1]:
        return 1  # Uptrend
    elif short_moving_average[-1] <= long_moving_average[-1]:
        return 0  # Downtrend
    else:
        return -1  # No clear trend


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, dfs, window_size=50, prediction_size=20, target_percentage=1.0, stoploss_percentage=1.0, isTraining=True):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.dfs = dfs
        self.isTraining = isTraining
        self.window_size = window_size
        self.prediction_size = prediction_size
        self.target_percentage = target_percentage
        self.stoploss_percentage = stoploss_percentage
        self.prices, self.signal_features, self.volumes = self._process_data()
        # self.shape = (window_size, self.prices.shape[1])
        self.shape = (window_size, )

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
        # self.observation_space = spaces.Dict({"net_worth": spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64), "prices": spaces.Box(
        #     low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)})
        self.observation_space = spaces.Dict({
            "position": spaces.Discrete(3),
            "current_trend": spaces.Discrete(2),
            "pnl": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "prices": spaces.Box(
                low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64),
            # "volumes": spaces.Box(
            #     low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64),
            # "support_distances": spaces.Box(
            #     low=-np.inf, high=np.inf, shape=(20, ), dtype=np.float64),
            # "support_strengths": spaces.Box(
            #     low=-np.inf, high=np.inf, shape=(20, ), dtype=np.float64),
            # "resistance_distances": spaces.Box(
            #     low=-np.inf, high=np.inf, shape=(20, ), dtype=np.float64),
            # "resistance_strengths": spaces.Box(
            #     low=-np.inf, high=np.inf, shape=(20, ), dtype=np.float64),
        })

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        # self._end_tick = len(self.prices) - 2
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        self.account_balance = INITIAL_TRADING_BALANCE
        self.previous_net_worth = INITIAL_TRADING_BALANCE
        self.net_worth = INITIAL_TRADING_BALANCE
        self.max_account_balance = INITIAL_TRADING_BALANCE
        self.num_tokens_current = 0
        self.num_tokens_bought = 0
        self.num_tokens_sold = 0
        self.calls_history = None
        self.profitable_trades_profit = 0
        self.number_of_profitable_trades = 0
        self.profitable_trades_list = []
        self.unprofitable_trades_loss = 0
        self.number_of_unprofitable_trades = 0
        self.unprofitable_trades_list = []
        self.total_hold_duration = 0
        self.highest_net_worth = INITIAL_TRADING_BALANCE
        self.maximum_drawdown = 0
        self.overtime_net_worth = []
        self.long_trades_count = 0
        self.short_trades_count = 0
        self.reward_summary = {
            "total_reward_entry_count": 0,
            "total_positive_reward_entry_count": 0,
            "total_negative_reward_entry_count": 0,
            "total_reward_exit_count": 0,
            "total_positive_reward_exit_count": 0,
            "total_negative_reward_exit_count": 0,
            "total_reward_hold_count": 0,
            "total_positive_reward_hold_count": 0,
            "total_negative_reward_hold_count": 0,
        }
        self.reward_breakdown = {
            "double_buy_punishment": 0,
            "double_sell_punishment": 0,
            "short_punishment": 0,
            "total_positive_reward_entry": 0,
            "total_negative_reward_entry": 0,
            "total_positive_reward_exit": 0,
            "total_negative_reward_exit": 0,
            "total_positive_reward_hold": 0,
            "total_negative_reward_hold": 0,
        }
        self.candle_movements = []
        self.min_percentage_differences = []
        self.max_percentage_differences = []
        self.actions_taken = {
            "buy": 0,
            "sell": 0,
            "hold": 0
        }
        self.position_holding_since = -1
        self.current_ticker = None
        self.continuous_loss_trades = 0
        self.target_tick = -1
        self.stoploss_tick = -1
        self.discount_factor = 0.9
        self.win_rate = 1.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Nothing
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        self.account_balance = INITIAL_TRADING_BALANCE
        self.previous_net_worth = INITIAL_TRADING_BALANCE
        self.net_worth = INITIAL_TRADING_BALANCE
        self.max_account_balance = INITIAL_TRADING_BALANCE
        self.num_tokens_current = 0
        self.num_tokens_bought = 0
        self.num_tokens_sold = 0
        self.calls_history = (self.window_size * [None]) + [self._position]
        self.profitable_trades_profit = 0
        self.number_of_profitable_trades = 0
        self.profitable_trades_list = []
        self.unprofitable_trades_loss = 0
        self.number_of_unprofitable_trades = 0
        self.total_hold_duration = 0
        self.highest_net_worth = INITIAL_TRADING_BALANCE
        self.maximum_drawdown = 0
        self.overtime_net_worth = []
        self.long_trades_count = 0
        self.short_trades_count = 0
        self.reward_summary = {
            "total_reward_entry_count": 0,
            "total_positive_reward_entry_count": 0,
            "total_negative_reward_entry_count": 0,
            "total_reward_exit_count": 0,
            "total_positive_reward_exit_count": 0,
            "total_negative_reward_exit_count": 0,
            "total_reward_hold_count": 0,
            "total_positive_reward_hold_count": 0,
            "total_negative_reward_hold_count": 0,
        }
        self.reward_breakdown = {
            "double_buy_punishment": 0,
            "double_sell_punishment": 0,
            "short_punishment": 0,
            "total_positive_reward_entry": 0,
            "total_negative_reward_entry": 0,
            "total_positive_reward_exit": 0,
            "total_negative_reward_exit": 0,
            "total_positive_reward_hold": 0,
            "total_negative_reward_hold": 0,
        }
        self.candle_movements = []
        self.min_percentage_differences = []
        self.max_percentage_differences = []
        self.actions_taken = {
            "buy": 0,
            "sell": 0,
            "hold": 0
        }
        self.position_holding_since = -1
        self.current_ticker = None
        self.continuous_loss_trades = 0
        self.target_tick = -1
        self.stoploss_tick = -1
        self.discount_factor = 0.9
        self.win_rate = 1.0

        return self._get_observation()

    def step(self, action):

        is_chart_end_exit = False

        self.current_ticker = self.tickers[self._current_tick]
        self._done = False
        self._current_tick += 1

        if self._current_tick != self._end_tick and self.tickers[self._current_tick] != self.tickers[self._current_tick + 1]:
            print('Ticker Change',
                  self.tickers[self._current_tick], self.tickers[self._current_tick + 1])
            print('Ticker Change',
                  self.prices[self._current_tick], self.prices[self._current_tick + 1])
            if self._position == Positions.Long:
                action = Actions.Sell.value
            elif self._position == Positions.Short:
                action = Actions.Buy.value
            else:
                action = Actions.Hold.value

            is_chart_end_exit = True
            print('Action', action)
            print('Position', self._position)

        step_reward = self._calculate_reward(action, is_chart_end_exit)
        self._total_reward += step_reward

        self._update_profit(action)

        if self._current_tick == self._end_tick \
                or (False and self._get_current_balance() <= INITIAL_TRADING_BALANCE/10):
            self._done = True

        # trade = False
        # if ((action == Actions.Buy.value and self._position == Positions.Short) or
        #         (action == Actions.Sell.value and self._position == Positions.Long)):
        #     trade = True

        # if trade:
        #     # self._position = self._position.opposite()
        #     self._position = Positions.Nothing
        # elif self._position == Positions.Nothing and action != Actions.Hold.value:
        #     self._last_trade_tick = self._current_tick
        #     self._position = Positions.Long if action == Actions.Buy.value else Positions.Short

        re = self._update_portfolio(action)

        self.previous_net_worth = self.net_worth
        self.net_worth = self._get_current_balance()
        self.overtime_net_worth.append(self.net_worth)

        drawdown = 0
        if self.net_worth > self.highest_net_worth:
            self.highest_net_worth = self.net_worth
        else:
            drawdown = self.highest_net_worth - self.net_worth
        if drawdown > self.maximum_drawdown:
            self.maximum_drawdown = drawdown
        # step_reward = (self.net_worth - self.previous_net_worth)
        # step_reward = (self.net_worth - self.previous_net_worth)*(2**(-16))
        # self._total_reward = self.net_worth - INITIAL_TRADING_BALANCE

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        # if action == Actions.Buy.value:
        #     self.signal_features[self._current_tick, -1] = 1
        # elif action == Actions.Sell.value:
        #     self.signal_features[self._current_tick, -1] = -1
        # else:
        #     self.signal_features[self._current_tick, -1] = 0

        total_trades =  (self.number_of_profitable_trades +
                 self.number_of_unprofitable_trades)
        if total_trades != 0:
            self.win_rate = self.number_of_profitable_trades / total_trades
            if total_trades < 10:
                self.win_rate = 1.0
            elif self.win_rate == 0:
                self.win_rate = 0.01

        if self._done and (self.isTraining == False or random.random() < 0.25):
            # if self._done:
            print('= '*50)
            pnl = self._get_current_balance() - INITIAL_TRADING_BALANCE
            print("P&L", pnl)
            print("ROI", (pnl/INITIAL_TRADING_BALANCE) * 100, '%')
            # print('Return  percentage ', (self._get_current_balance() /
            #       INITIAL_TRADING_BALANCE) * 100)
            print('Number of Profitable Trades ',
                  self.number_of_profitable_trades)
            print('Number of Unprofitable Trades ',
                  self.number_of_unprofitable_trades)
            print('Number of Long Trades ',
                  self.long_trades_count)
            print('Number of Short Trades ',
                  self.short_trades_count)
            print('Rewards Summary: ',
                  )
            for key, value in self.reward_summary.items():
                print(key, value)
            print('----------------------------------------')
            for key, value in self.reward_breakdown.items():
                print(key, value)
            print('----------------------------------------')

            if self.reward_summary["total_reward_entry_count"] > 0:
                print('Entry Negative Rewards Percentage ', self.reward_summary["total_negative_reward_entry_count"] /
                      self.reward_summary["total_reward_entry_count"] * 100, '%')
            if self.reward_summary["total_reward_exit_count"] > 0:
                print('Exit Negative Rewards Percentage ', self.reward_summary["total_negative_reward_exit_count"] /
                      self.reward_summary["total_reward_exit_count"] * 100, '%')
            if self.reward_summary["total_reward_hold_count"] > 0:
                print('Hold Negative Rewards Percentage ', self.reward_summary["total_negative_reward_hold_count"] /
                      self.reward_summary["total_reward_hold_count"] * 100, '%')

            if self.number_of_profitable_trades + self.number_of_unprofitable_trades != 0:
                print('Average Holding Duration', self.total_hold_duration /
                      (self.number_of_profitable_trades + self.number_of_unprofitable_trades))
                print('Win Rate (%)', self.number_of_profitable_trades /
                      (self.number_of_profitable_trades + self.number_of_unprofitable_trades) * 100, '%')
                print("Number of Trades", self.number_of_profitable_trades +
                      self.number_of_unprofitable_trades)
                print("Average Trade Return (in $)", (pnl /
                      (self.number_of_profitable_trades + self.number_of_unprofitable_trades)))
                print("Average Profit (in $)", (self.profitable_trades_profit /
                      (self.number_of_profitable_trades + 1e-9)))
                print("Average Loss (in $)", (self.unprofitable_trades_loss /
                      (self.number_of_unprofitable_trades + 1e-9)))

                # print all loss trades sorted by loss amount (descending)
                print("All Loss Trades Sorted by Loss Amount (Descending)")
                unp_count = 0
                for i in sorted(self.unprofitable_trades_list, key=lambda x: x["pnl"]):
                    if unp_count > 10:
                        break
                    unp_count += 1
                    print(i)

                print("All Profit Trades Sorted by Profit Amount (Descending)")
                p_count = 0
                for i in sorted(self.profitable_trades_list, key=lambda x: x["pnl"], reverse=True):
                    if p_count > 10:
                        break
                    p_count += 1
                    print(i)
            print("Maximum Drawdown", self.maximum_drawdown)
            df = pd.DataFrame({"max_percentage_differences": self.max_percentage_differences, "min_percentage_differences": self.min_percentage_differences}, columns=[
                              'max_percentage_differences', 'min_percentage_differences'])
            df.to_csv('perc_differences.csv')
            print("Actions Taken", self.actions_taken)
            print('Total Reward: ', self._total_reward)
            print('= '*50)

        if action == Actions.Sell.value:
            self.actions_taken["sell"] += 1
        elif action == Actions.Buy.value:
            self.actions_taken["buy"] += 1
        elif action == Actions.Hold.value:
            self.actions_taken["hold"] += 1

        return observation, step_reward, self._done, info

    def get_current_position_pnl_perc(self):
        if self._position == Positions.Long:
            return ((self.prices[self._current_tick] - self.prices[self._last_trade_tick]) / self.prices[self._last_trade_tick]) * 100
        elif self._position == Positions.Short:
            return ((self.prices[self._last_trade_tick] - self.prices[self._current_tick]) / self.prices[self._last_trade_tick]) * 100
        else:
            return 0

    def _get_observation(self):
        # return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        # return {"net_worth": [self.net_worth], "prices": self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]}
        # return self.prices_normalized[(self._current_tick-self.window_size+1):self._current_tick+1]
        # support_distances = []
        # support_strengths = []
        # resistance_distances = []
        # resistance_strengths = []

        # next_price = self.prices[self._current_tick + 1 if self._current_tick <
        #                          self._last_trade_tick else self._current_tick]
        # next_price_rounded = int(round(next_price/5.0)*5.0)

        # for i in range(next_price_rounded-50, next_price_rounded+50, 5):
        #     if i in self.srDict:
        #         if self.srDict[i][2] == 1:
        #             support_distances.append(
        #                 ((self.srDict[i][1] - next_price)/next_price) * 100)
        #             support_strengths.append(self.srDict[i][3])
        #             resistance_distances.append(0)
        #             resistance_strengths.append(0)
        #         else:
        #             resistance_distances.append(
        #                 ((self.srDict[i][1] - next_price)/next_price) * 100)
        #             resistance_strengths.append(self.srDict[i][3])
        #             support_distances.append(0)
        #             support_strengths.append(0)
        #     else:
        #         support_distances.append(0)
        #         support_strengths.append(0)
        #         resistance_distances.append(0)
        #         resistance_strengths.append(0)

        return {"position": self._position.value,
                "current_trend": getTrend(self.prices[self._current_tick - self.prediction_size:self._current_tick]),
                "pnl": np.array([self.get_current_position_pnl_perc()/10]),
                "prices": self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1],
                # "volumes": self.volumes[(self._current_tick-self.window_size+1):self._current_tick+1],
                # "support_distances": np.array(support_distances),
                # "support_strengths": np.array(support_strengths),
                # "resistance_distances": np.array(resistance_distances),
                # "resistance_strengths": np.array(resistance_strengths),
                }

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            # elif position == Positions.Nothing:
            #     color = 'black'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def _get_current_balance(self):
        current_stock_balance = self.account_balance
        if self._position == Positions.Long:
            current_stock_balance += (self.num_tokens_current * self.prices[self._last_trade_tick]) + (
                self.num_tokens_current * (self.prices[self._current_tick] - self.prices[self._last_trade_tick]))
        elif self._position == Positions.Short:
            current_stock_balance += (self.num_tokens_current * self.prices[self._last_trade_tick]) + (
                self.num_tokens_current * (self.prices[self._last_trade_tick] - self.prices[self._current_tick]))

        return current_stock_balance

    def render_all(self, mode='human'):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))

        window_ticks = np.arange(len(self.calls_history))
        ax1.plot(self.prices)

        long_entry_ticks = []
        long_exit_ticks = []
        short_entry_ticks = []
        short_exit_ticks = []
        hold_ticks = []

        for i, tick in enumerate(window_ticks):
            if self.calls_history[i] == Calls.LongEntry.value:
                long_entry_ticks.append(tick)
            elif self.calls_history[i] == Calls.LongExit.value:
                long_exit_ticks.append(tick)
            elif self.calls_history[i] == Calls.ShortEntry.value:
                short_entry_ticks.append(tick)
            elif self.calls_history[i] == Calls.ShortExit.value:
                short_exit_ticks.append(tick)
            elif self.calls_history[i] == Calls.Hold.value:
                hold_ticks.append(tick)

        # add radius to the points
        ax1.plot(long_entry_ticks, self.prices[long_entry_ticks], 'g^')
        ax1.plot(long_exit_ticks, self.prices[long_exit_ticks], 'rx')
        ax1.plot(short_entry_ticks, self.prices[short_entry_ticks], 'rv')
        ax1.plot(short_exit_ticks, self.prices[short_exit_ticks], 'gx')
        # ax1.plot(hold_ticks, self.prices[hold_ticks], 'ko')

        current_stock_balance = self._get_current_balance()

        print('self.account_balance: ', self.account_balance)
        print('self._position: ', self._position)
        print('self._last_trade_tick: ', self._last_trade_tick)
        print('self._last_trade_price: ', self.prices[self._last_trade_tick])
        print('self._current_tick: ', self._current_tick)
        print('self._current_price: ', self.prices[self._current_tick])
        print('self.prices[self._last_trade_tick]: ',
              self.prices[self._last_trade_tick])
        print('self.prices[self._current_tick]: ',
              self.prices[self._current_tick])
        print('self.num_tokens_current: ', self.num_tokens_current)

        ax1.set_title(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            # "Total Profit: %.6f" % self._total_profit + ' ~ \n' +
            "Initial: %.6f" % INITIAL_TRADING_BALANCE + ' ~ ' +
            "Current: %.6f" % current_stock_balance
        )
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')

        ax2.set_title("Overtime Net Worth")
        ax2.plot(self.overtime_net_worth)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')

        # plt.tight_layout()

    def _render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        nothing_ticks = []

        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            elif self._position_history[i] == Positions.Nothing:
                nothing_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')
        # plt.plot(nothing_ticks, self.prices[nothing_ticks], 'ko')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def _update_portfolio(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
