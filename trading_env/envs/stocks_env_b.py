import numpy as np
import torch
from torch.utils.data import DataLoader

from .trading_env import TradingEnv, Actions, Calls, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee_bid_percent = 0.01  # unit
        # self.trade_fee_bid_percent = 0  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        # self.trade_fee_ask_percent = 0  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        # position_status = np.zeros_like(prices)
        # signal_features = np.column_stack((prices, diff, position_status))
        signal_features = np.column_stack((prices, diff))
        # convert this to a torch tensor and save in cuda
        # signal_features = torch.tensor(signal_features).cuda()
        # get shape of signal_features
        # signal_features_shape = signal_features.shape

        return prices, signal_features

    def _do_action(self, action):
        step_reward = 0

        trade = False

        if ((action == Actions.Buy.value and self._position == Positions.Long) or
                (action == Actions.Sell.value and self._position == Positions.Short)):
            return -10

        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            # if self._position == Positions.Long:
            step_reward += price_diff

        return step_reward

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False

        if ((action == Actions.Buy.value and self._position == Positions.Long) or
                (action == Actions.Sell.value and self._position == Positions.Short)):
            step_reward = -10

        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                step_reward = current_price - last_trade_price
            else:
                step_reward = last_trade_price - current_price

            # price_diff = current_price - last_trade_price

            # if self._position == Positions.Long:
            # step_reward += price_diff

        return step_reward

    def __calculate_reward(self, action):
        step_reward = 0

        trade = False

        if ((action == Actions.Buy.value and self._position == Positions.Long) or
                (action == Actions.Sell.value and self._position == Positions.Short)):
            return -99999999999

        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            # if self._position == Positions.Long:
            step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit *
                          (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (
                    shares * (1 - self.trade_fee_bid_percent)) * current_price
            elif self._position == Positions.Short:
                shares = (self._total_profit *
                          (1 - self.trade_fee_bid_percent)) / current_price
                self._total_profit = (
                    shares * (1 - self.trade_fee_ask_percent)) * last_trade_price

    def _update_portfolio(self, action):

        reward = 0

        if self._position == Positions.Nothing:
            if action == Actions.Buy.value:
                max_tokens = self.account_balance / \
                    self.prices[self._current_tick]
                # buy all tokens
                self.num_tokens_current = max_tokens
                self.num_tokens_bought += max_tokens
                self.account_balance = 0
                self._position = Positions.Long
                self._last_trade_tick = self._current_tick
                self.calls_history.append(Calls.LongEntry.value)
            elif action == Actions.Sell.value:
                max_tokens = self.account_balance / \
                    self.prices[self._current_tick]
                # short all tokens
                self.num_tokens_current = max_tokens
                self.num_tokens_sold += max_tokens
                self.account_balance = 0
                self._position = Positions.Short
                self._last_trade_tick = self._current_tick
                self.calls_history.append(Calls.ShortEntry.value)
            elif action == Actions.Hold.value:
                # do nothing
                self.calls_history.append(Calls.Hold.value)
                pass

        elif self._position == Positions.Long:
            if action == Actions.Sell.value:
                reward = (self.prices[self._current_tick] -
                          self.prices[self._last_trade_tick]) * self.num_tokens_current
                # sell all tokens
                self.account_balance = self.num_tokens_current * \
                    self.prices[self._current_tick]
                self.num_tokens_current = 0
                self._position = Positions.Nothing
                self.max_account_balance = max(
                    self.max_account_balance, self.account_balance)
                self.calls_history.append(Calls.LongExit.value)
            elif action == Actions.Buy.value:
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass
            elif action == Actions.Hold.value:
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass

        elif self._position == Positions.Short:
            if action == Actions.Buy.value:
                reward = (self.prices[self._last_trade_tick] -
                          self.prices[self._current_tick]) * self.num_tokens_current
                # buy all tokens
                self.account_balance = (self.num_tokens_current * self.prices[self._last_trade_tick]) + (
                    self.num_tokens_current * (self.prices[self._last_trade_tick] - self.prices[self._current_tick]))

                self.num_tokens_current = 0
                self._position = Positions.Nothing
                self.max_account_balance = max(
                    self.max_account_balance, self.account_balance)
                self.calls_history.append(Calls.ShortExit.value)
            elif action == Actions.Sell.value:
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass
            elif action == Actions.Hold.value:
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            # if self._position == Positions.Long:
            shares = (self._total_profit *
                      (1 - self.trade_fee_ask_percent)) / last_trade_price
            self._total_profit = (
                shares * (1 - self.trade_fee_bid_percent)) * current_price

        if reward > 0:
            self.number_of_profitable_trades += 1
        elif reward < 0:
            self.number_of_unprofitable_trades += 1
            
        return reward

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
