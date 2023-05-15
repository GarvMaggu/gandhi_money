import numpy as np

from .trading_env import TradingEnv, Actions, Calls, Positions

# from statsmodels.tsa.stattools import adfuller
# def test_stationary(diff):
#     result = adfuller(diff)
#     print('ADF Statistic: %f' % result[0])
#     print('p-value: %f' % result[1])

#     if result[1] > 0.05:
#         print("not stationary")
#     else:
#         print("stationary")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def support(df1, l, n1, n2):  # n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if (df1.low[i] > df1.low[i-1]):
            return 0
    for i in range(l+1, l+n2+1):
        if (df1.low[i] < df1.low[i-1]):
            return 0
    return 1

# support(df,46,3,2)


def resistance(df1, l, n1, n2):  # n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if (df1.high[i] < df1.high[i-1]):
            return 0
    for i in range(l+1, l+n2+1):
        if (df1.high[i] > df1.high[i-1]):
            return 0
    return 1
# resistance(df, 30, 3, 5)


class StocksEnv(TradingEnv):

    def __init__(self, df, dfs, frame_bound, window_size=50, prediction_size=20, target_percentage=1.0, stoploss_percentage=1.0, isTraining=True):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, dfs, window_size, prediction_size,
                         target_percentage, stoploss_percentage, isTraining)

        self.trading_fees = 0.001
        self.trade_fee_bid_percent = 0.01  # unit
        # self.trade_fee_bid_percent = 0  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        # self.trade_fee_ask_percent = 0  # unit

    def _process_data(self):

        # prices = self.df.loc[:, 'close'].to_numpy()
        # self.tickers = self.df.loc[:, 'ticker'].to_numpy()
        # prices_standardized = (prices - prices.mean()) / prices.std()
        # self.prices_normalized = sigmoid(prices_standardized)

        # validate index (TODO: Improve validation)

        prices = np.array([])
        volumes = np.array([])
        self.tickers = np.array([])
        diff = np.array([])
        diff_vol = np.array([])
        for d in self.dfs:
            _prices = d.loc[:, 'close'].to_numpy()
            _prices = np.diff(_prices) / _prices[:-1] * 100
            _volumes = d.loc[:, 'volume'].to_numpy()
            _volumes = np.diff(_volumes) / _volumes[:-1] * 100
            diff = np.concatenate((diff, _prices))
            diff_vol = np.concatenate((diff_vol, _volumes))
            # concat all prices except the first one
            prices = np.concatenate((prices, d.loc[1:, 'close'].to_numpy()))
            volumes = np.concatenate((volumes, d.loc[1:, 'volume'].to_numpy()))
            self.tickers = np.concatenate(
                (self.tickers, d.loc[1:, 'ticker'].to_numpy()))

        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]
        volumes = volumes[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]
        self.tickers = self.tickers[self.frame_bound[0] -
                                    self.window_size: self.frame_bound[1]]
        diff = diff[self.frame_bound[0] -
                    self.window_size: self.frame_bound[1]]

        print("diff", len(diff))
        print("tickers", len(self.tickers))
        print("prices", len(prices))
        print("volumes", len(volumes))

        # diff = np.diff(prices)
        # convert to percentage change
        # diff = diff / prices[:-1] * 100
        # percentage_changes = np.diff(prices) / prices[:-1] * 100
        # position_status = np.zeros_like(prices)
        # signal_features = np.column_stack((prices, diff, position_status))
        # signal_features = np.column_stack((prices, diff))

        sr = []
        self.srDict = {}
        sr_volumes = []
        n1 = 3
        n2 = 2
        for row in range(n1, len(self.df)-n2):  # len(df)-n2
            if support(self.df, row, n1, n2):
                support_vol = self.df.volume[row-1] + \
                    self.df.volume[row]+self.df.volume[row+1]
                sr_volumes.append(support_vol)
                sr.append([row, self.df.low[row], 1, support_vol])
                # srDict[int(round(self.df.low[row]/5.0)*5.0)] = [row,self.df.low[row],1, support_vol]
            if resistance(self.df, row, n1, n2):
                resistance_vol = self.df.volume[row-1] + \
                    self.df.volume[row]+self.df.volume[row+1]
                sr_volumes.append(resistance_vol)
                sr.append([row, self.df.high[row], 2, resistance_vol])
                # srDict[int(round(self.df.high[row]/5.0)*5.0)] = [row,self.df.high[row],2, resistance_vol]

        # normal sr_volumes to 0-1
        sr_volumes = np.array(sr_volumes)
        sr_volumes = (sr_volumes - sr_volumes.min()) / \
            (sr_volumes.max() - sr_volumes.min())
        # print("sr_volumes", sr_volumes)

        for i in range(len(sr)):
            sr[i][3] = sr_volumes[i]
            self.srDict[int(round(sr[i][1]/5.0)*5.0)] = sr[i]

        # print("sr", sr)
        # print("srDict", srDict)

        # sr.sort(key=lambda x: x[3], reverse=True)
        # signal_features = np.column_stack((percentage_changes))

        return prices, diff, diff_vol

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

    def _calculate_reward(self, action, is_chart_end_exit=False):
        step_reward = 0

        if ((action == Actions.Buy.value and self._position == Positions.Long) or
                (action == Actions.Sell.value and self._position == Positions.Short)):
            if action == Actions.Buy.value:
                self.reward_breakdown["double_buy_punishment"] += -2
            if action == Actions.Sell.value:
                self.reward_breakdown["double_sell_punishment"] += -2
            step_reward += -2

        # Handling Entries
        if self._position == Positions.Nothing and (action == Actions.Buy.value or action == Actions.Sell.value):
            entry_price = self.prices[self._current_tick]
            entry_price = entry_price * \
                (1 + self.trading_fees) if action == Actions.Buy.value else entry_price * \
                (1 - self.trading_fees)
            range_end = self._current_tick + self.prediction_size if self._current_tick + \
                self.prediction_size <= self._end_tick else self._end_tick

            self.target_tick = -1
            self.stoploss_tick = -1

            for i in range(self._current_tick, range_end):

                if i != range_end - 1 and self.tickers[i] != self.tickers[i+1]:
                    break

                current_tick_price = self.prices[i]
                current_tick_price = current_tick_price * \
                    (1 - self.trading_fees) if action == Actions.Buy.value else current_tick_price * \
                    (1 + self.trading_fees)
                if action == Actions.Buy.value:
                    if current_tick_price > entry_price * (1 + self.target_percentage / 100):
                        # buy at $100, sold at $200 (100% profit), ($200 - $100) / $100 = 1
                        step_reward += (current_tick_price -
                                        entry_price) / entry_price
                        self.target_tick = i
                        break
                    elif current_tick_price < entry_price * (1 - self.stoploss_percentage / 100):
                        # buy at $100, sold at $50 (-50% loss), ($50 - $100) / $100 = -0.5
                        step_reward += (current_tick_price -
                                        entry_price) / entry_price
                        # step_reward = step_reward / self.win_rate
                        self.stoploss_tick = i
                        break
                elif action == Actions.Sell.value:
                    if current_tick_price > entry_price * (1 - self.target_percentage / 100):
                        # sell at $100, bought at $50 (50% profit), ($50 - $100) / $100 = -0.5
                        step_reward += (entry_price -
                                        current_tick_price) / entry_price
                        self.target_tick = i
                        break
                    elif current_tick_price < entry_price * (1 + self.stoploss_percentage / 100):
                        # sell at $100, bought at $200 (-100% loss), ($100 - $200) / $100 = -1
                        step_reward += (entry_price -
                                        current_tick_price) / entry_price
                        # step_reward = step_reward / self.win_rate
                        self.stoploss_tick = i
                        break

            if self.target_tick == -1 and self.stoploss_tick == -1:
                step_reward += -1

            if step_reward > 0:
                self.reward_summary["total_reward_entry_count"] += 1
                self.reward_summary["total_positive_reward_entry_count"] += 1
                self.reward_breakdown["total_positive_reward_entry"] += step_reward
            elif step_reward < 0:
                self.reward_summary["total_reward_entry_count"] += 1
                self.reward_summary["total_negative_reward_entry_count"] += 1
                self.reward_breakdown["total_negative_reward_entry"] += step_reward

        # Handling Exits
        if (action == Actions.Sell.value and self._position == Positions.Long) or (action == Actions.Buy.value and self._position == Positions.Short):
            entry_price = self.prices[self._last_trade_tick]
            entry_price = entry_price * \
                (1 + self.trading_fees) if self._position == Positions.Long else entry_price * \
                (1 - self.trading_fees)

            exit_price = self.prices[self._current_tick]
            exit_price = exit_price * \
                (1 - self.trading_fees) if self._position == Positions.Long else exit_price * \
                (1 + self.trading_fees)

            # Early exit
            step_reward += (exit_price - entry_price) / entry_price if self._position == Positions.Long else (
                entry_price - exit_price) / entry_price
            
            if step_reward > 0:
                self.reward_summary["total_reward_exit_count"] += 1
                self.reward_summary["total_positive_reward_exit_count"] += 1
                self.reward_breakdown["total_positive_reward_exit"] += step_reward
                self.continuous_loss_trades = 0
            elif step_reward < 0:
                # step_reward = step_reward / self.win_rate
                self.reward_summary["total_reward_exit_count"] += 1
                self.reward_summary["total_negative_reward_exit_count"] += 1
                self.reward_breakdown["total_negative_reward_exit"] += step_reward
                self.continuous_loss_trades += 1

            self.position_holding_since = -1

        # Handling Holds
        if action == Actions.Hold.value:
            if self._position == Positions.Nothing:
                step_reward += 0
            else:
                self.position_holding_since += 1
                entry_price = self.prices[self._last_trade_tick]
                entry_price = entry_price * \
                    (1 + self.trading_fees) if self._position == Positions.Long else entry_price * \
                    (1 - self.trading_fees)

                if self.target_tick != -1:
                    # it means it is holding towards the target price, which is good, we want to reward it positive
                    target_price = self.prices[self.target_tick] * \
                        (1 - self.trading_fees) if self._position == Positions.Long else self.prices[self.target_tick] * \
                        (1 + self.trading_fees)

                    current_tick_price = self.prices[self._current_tick] * \
                        (1 - self.trading_fees) if self._position == Positions.Long else self.prices[self._current_tick] * \
                        (1 + self.trading_fees)

                    if self._current_tick < self.target_tick:
                        step_reward += ((self.discount_factor) ** (self.target_tick - self._current_tick)) * (
                            ((target_price - entry_price) / entry_price) if self._position == Positions.Long else ((entry_price - target_price) / entry_price))
                        # it means it has achieved the target price, but it is still holding, we want to reward it positive if price is still going up or negative if price is going down
                    else:
                        step_reward += ((self.discount_factor) ** (self._current_tick - self.target_tick)) * (((current_tick_price - target_price) / target_price) if self._position == Positions.Long else (
                            (target_price - current_tick_price) / target_price))

                if self.stoploss_tick != -1:
                    stoploss_price = self.prices[self.stoploss_tick] * \
                        (1 - self.trading_fees) if self._position == Positions.Long else self.prices[self.stoploss_tick] * \
                        (1 + self.trading_fees)

                    current_tick_price = self.prices[self._current_tick] * \
                        (1 - self.trading_fees) if self._position == Positions.Long else current_tick_price * \
                        (1 + self.trading_fees)

                    if self._current_tick < self.stoploss_tick:
                        step_reward += ((self.discount_factor) ** (self.stoploss_tick - self._current_tick)) * (
                            ((stoploss_price - entry_price) / entry_price) if self._position == Positions.Long else ((entry_price - stoploss_price) / entry_price))
                    else:
                        step_reward += ((current_tick_price - entry_price) / entry_price) if self._position == Positions.Long else (
                            (entry_price - current_tick_price) / entry_price)

            if step_reward > 0:
                self.reward_summary["total_reward_hold_count"] += 1
                self.reward_summary["total_positive_reward_hold_count"] += 1
                self.reward_breakdown["total_positive_reward_hold"] += step_reward
            elif step_reward < 0:
                self.reward_summary["total_reward_hold_count"] += 1
                self.reward_summary["total_negative_reward_hold_count"] += 1
                self.reward_breakdown["total_negative_reward_hold"] += step_reward

        return step_reward

    def __calculate_reward(self, action, is_chart_end_exit=False):
        step_reward = 0

        trade = False

        if False and action == Actions.Hold.value:
            if self._position != Positions.Nothing:
                self.position_holding_since += 1
                # step_reward -= (self.position_holding_since * 0.1) / 100
                current_entry_price = self.prices[self._current_tick]
                current_entry_price = current_entry_price * \
                    (1 + self.trading_fees) if self._position == Positions.Long else current_entry_price * \
                    (1 - self.trading_fees)
                range_end = self._current_tick + (self.prediction_size - (self._current_tick - self._last_trade_tick)) if self._current_tick + \
                    (self.prediction_size - (self._current_tick - self._last_trade_tick)) <= self._end_tick else self._end_tick
                average_pnl_perc = 0
                count = 0
                for i in range(self._current_tick, range_end):
                    if i != range_end - 1 and self.tickers[i] != self.tickers[i+1]:
                        break
                    count += 1
                    current_tick_price = self.prices[i]
                    current_tick_price = current_tick_price * \
                        (1 - self.trading_fees) if self._position == Positions.Long else current_tick_price * \
                        (1 + self.trading_fees)
                    average_pnl_perc += current_tick_price - \
                        current_entry_price if self._position == Positions.Long else current_entry_price - \
                        current_tick_price

                if count > 0:
                    # average_pnl_perc = average_pnl_perc - \
                    #     (self.trading_fees * abs(average_pnl_perc))
                    average_pnl_perc = average_pnl_perc / count
                    # convert to percentage
                    average_pnl_perc = (
                        average_pnl_perc / current_entry_price) * 100
                    step_reward += average_pnl_perc / 100

                if step_reward > 0:
                    self.reward_summary["total_reward_hold_count"] += 1
                    self.reward_summary["total_positive_reward_hold_count"] += 1
                    self.reward_breakdown["total_positive_reward_hold"] += step_reward
                elif step_reward < 0:
                    self.reward_summary["total_reward_hold_count"] += 1
                    self.reward_summary["total_negative_reward_hold_count"] += 1
                    self.reward_breakdown["total_negative_reward_hold"] += step_reward

        if ((action == Actions.Buy.value and self._position == Positions.Long) or
                (action == Actions.Sell.value and self._position == Positions.Short)):
            if action == Actions.Buy.value:
                self.reward_breakdown["double_buy_punishment"] += -2
            elif action == Actions.Sell.value and self._position == Positions.Nothing:
                self.reward_breakdown["short_punishment"] += -2
            step_reward += -2

        if self._position == Positions.Nothing and (action == Actions.Buy.value or action == Actions.Sell.value):
            self.position_holding_since += 1
        # if self._position == Positions.Nothing and (action == Actions.Buy.value):
            entry_price = self.prices[self._current_tick]
            entry_price = entry_price * \
                (1 + self.trading_fees) if self._position == Positions.Long else entry_price * \
                (1 - self.trading_fees)
            range_end = self._current_tick + self.prediction_size if self._current_tick + \
                self.prediction_size <= self._end_tick else self._end_tick

            achieved_target_price = False

            average_pnl_perc = 0
            count = 0

            for i in range(self._current_tick, range_end):
                count += 1

                current_tick_price = self.prices[i]
                current_tick_price = current_tick_price * \
                    (1 - self.trading_fees) if self._position == Positions.Long else current_tick_price * \
                    (1 + self.trading_fees)

                if action == Actions.Buy.value:
                    average_pnl_perc += current_tick_price - entry_price
                    if self.prices[i] > entry_price * (1 + self.target_percentage / 100):
                        temp_step_reward = self.prices[i] - entry_price
                        if temp_step_reward > step_reward:
                            step_reward = temp_step_reward
                        achieved_target_price = True

                    elif self.prices[i] < entry_price * (1 - self.stoploss_percentage / 100):
                        if achieved_target_price == False:
                            step_reward = self.prices[i] - entry_price
                        break

                elif action == Actions.Sell.value:
                    average_pnl_perc += entry_price - current_tick_price
                    if self.prices[i] > entry_price * (1 - self.target_percentage / 100):
                        temp_step_reward = entry_price - self.prices[i]
                        if temp_step_reward > step_reward:
                            step_reward = temp_step_reward
                        achieved_target_price = True

                    elif self.prices[i] < entry_price * (1 + self.stoploss_percentage / 100):
                        if achieved_target_price == False:
                            step_reward = entry_price - self.prices[i]
                        break

            if count > 0:
                # average_pnl_perc = average_pnl_perc - \
                #     (self.trading_fees * abs(average_pnl_perc))
                average_pnl_perc = average_pnl_perc / count
                # convert to percentage
                average_pnl_perc = (average_pnl_perc /
                                    entry_price) * 100
                # step_reward += average_pnl_perc

            if step_reward > 0:
                self.reward_summary["total_reward_entry_count"] += 1
                self.reward_summary["total_positive_reward_entry_count"] += 1
                self.reward_breakdown["total_positive_reward_entry"] += step_reward
            elif step_reward < 0:
                self.reward_summary["total_reward_entry_count"] += 1
                self.reward_summary["total_negative_reward_entry_count"] += 1
                self.reward_breakdown["total_negative_reward_entry"] += step_reward

        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            # if position long, exit at current price - 1% trading fees, if position short, exit at current price + 1% trading fees
            current_price = self.prices[self._current_tick]
            current_price = current_price * \
                (1 - self.trading_fees) if self._position == Positions.Long else current_price * \
                (1 + self.trading_fees)
            entry_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                step_reward = current_price - entry_price
            else:
                step_reward = entry_price - current_price

            range_end = self._last_trade_tick + self.prediction_size if self._last_trade_tick + \
                self.prediction_size <= self._end_tick else self._end_tick

            average_exit_pnl_perc = 0
            count = 0

            for i in range(self._last_trade_tick, self._current_tick if is_chart_end_exit else range_end):
                current_tick_price = self.prices[i]
                current_tick_price = current_tick_price * \
                    (1 - self.trading_fees) if self._position == Positions.Long else current_tick_price * \
                    (1 + self.trading_fees)

                if self._position == Positions.Long:
                    average_exit_pnl_perc += current_tick_price - current_price
                else:
                    average_exit_pnl_perc += current_price - current_tick_price
                count += 1

            if count > 0:
                # average_exit_pnl_perc = average_exit_pnl_perc - \
                #     (self.trading_fees * abs(average_exit_pnl_perc))
                average_exit_pnl_perc = average_exit_pnl_perc / count
                # convert to percentage
                average_exit_pnl_perc = (
                    average_exit_pnl_perc / current_price) * 100
                # step_reward += -1 * average_exit_pnl_perc * (1 if self.continuous_loss_trades < 4 else self.continuous_loss_trades)

            # if self._position == Positions.Long:
            #     # get highest element in an array
            #     # if actual_exit_price > highest_price * (1 + self.target_percentage / 100):
            #     #     step_reward = actual_exit_price - entry_price
            #     # elif actual_exit_price < highest_price * (1 - self.stoploss_percentage / 100):
            #     #     step_reward = actual_exit_price - entry_price
            #     # else:
            #     #     step_reward = actual_exit_price - entry_price
            #     step_reward = current_price - entry_price
            # elif self._position == Positions.Short:
            #     # get lowest element in an array
            #     # if actual_exit_price > lowest_price * (1 - self.target_percentage / 100):
            #     #     step_reward = entry_price - actual_exit_price
            #     # elif actual_exit_price < lowest_price * (1 + self.stoploss_percentage / 100):
            #     #     step_reward = entry_price - actual_exit_price
            #     # else:
            #     #     step_reward = entry_price - actual_exit_price
            #     step_reward = entry_price - current_price

            # price_diff = current_price - entry_price

            # if self._position == Positions.Long:
            # step_reward += price_diff

            if step_reward > 0:
                self.reward_summary["total_reward_exit_count"] += 1
                self.reward_summary["total_positive_reward_exit_count"] += 1
                self.reward_breakdown["total_positive_reward_exit"] += step_reward
                self.continuous_loss_trades = 0
            elif step_reward < 0:
                self.reward_summary["total_reward_exit_count"] += 1
                self.reward_summary["total_negative_reward_exit_count"] += 1
                self.reward_breakdown["total_negative_reward_exit"] += step_reward
                self.continuous_loss_trades += 1

            self.position_holding_since = -1

        return step_reward

    def _calculate_reward_profit(self, action):
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
        l_trade_tick = self._last_trade_tick
        num_tokens = self.num_tokens_current
        current_price = self.prices[self._current_tick]

        if self._position == Positions.Nothing:
            if action == Actions.Buy.value:
                max_tokens = self.account_balance / \
                    (current_price * (1 + self.trading_fees))
                # buy all tokens
                self.num_tokens_current = max_tokens
                self.num_tokens_bought += max_tokens
                self.account_balance = 0
                self._position = Positions.Long
                self._last_trade_tick = self._current_tick
                self.calls_history.append(Calls.LongEntry.value)
            elif action == Actions.Sell.value:
                max_tokens = self.account_balance / \
                    (current_price * (1 - self.trading_fees))
                # short all tokens
                self.num_tokens_current = max_tokens
                self.num_tokens_sold += max_tokens
                self.account_balance = 0
                self._position = Positions.Short
                self._last_trade_tick = self._current_tick
                self.calls_history.append(Calls.ShortEntry.value)
            # elif action == Actions.Sell.value:
            #     # do nothing
            #     self.calls_history.append(Calls.Hold.value)
            #     pass
            elif action == Actions.Hold.value:
                # do nothing
                self.calls_history.append(Calls.Hold.value)
                pass

        elif self._position == Positions.Long:
            if action == Actions.Sell.value:
                reward = ((current_price * (1 - self.trading_fees)) -
                          (self.prices[self._last_trade_tick] * (1 + self.trading_fees)))
                # sell all tokens
                self.account_balance = self.num_tokens_current * \
                    (current_price * (1 - self.trading_fees))
                self.account_balance = self.account_balance
                self.num_tokens_current = 0
                self._position = Positions.Nothing
                self._last_trade_tick = -1
                self.max_account_balance = max(
                    self.max_account_balance, self.account_balance)
                self.calls_history.append(Calls.LongExit.value)
            elif action == Actions.Buy.value:
                self.total_hold_duration += 1
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass
            elif action == Actions.Hold.value:
                self.total_hold_duration += 1
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass

        elif self._position == Positions.Short:
            if action == Actions.Buy.value:
                _current_price = current_price * (1 + self.trading_fees)
                reward = ((self.prices[self._last_trade_tick] * (1 - self.trading_fees)) -
                          (_current_price))
                # buy all tokens
                self.account_balance = (self.num_tokens_current * (self.prices[self._last_trade_tick] * (1 - self.trading_fees))) + (
                    self.num_tokens_current * ((self.prices[self._last_trade_tick] * (1 - self.trading_fees)) - _current_price))
                self.account_balance = self.account_balance
                self.num_tokens_current = 0
                self._position = Positions.Nothing
                self._last_trade_tick = -1
                self.max_account_balance = max(
                    self.max_account_balance, self.account_balance)
                self.calls_history.append(Calls.ShortExit.value)
            elif action == Actions.Sell.value:
                self.total_hold_duration += 1
                self.calls_history.append(Calls.Hold.value)
                # do nothing
                pass
            elif action == Actions.Hold.value:
                self.total_hold_duration += 1
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

        if reward != 0:
            if action == Actions.Sell.value:
                self.long_trades_count += 1
            if action == Actions.Buy.value:
                self.short_trades_count += 1

        if reward > 0:
            self.number_of_profitable_trades += 1
            self.profitable_trades_profit += reward * num_tokens
            stat = {
                "pnl": reward * num_tokens,
                "pnl_perc": (reward / (self.prices[l_trade_tick])) * 100,
                "quantity": num_tokens,
                "price_difference": reward,
                "duration in mins": (self._current_tick - l_trade_tick),
                "start": l_trade_tick,
                "end": self._current_tick,
                "entry_price": self.prices[l_trade_tick],
                "exit_price": self.prices[self._current_tick],
                "position": "long" if action == Actions.Sell.value else "short",
                "net_worth": self._get_current_balance(),
            }
            self.profitable_trades_list.append(stat)

            if self.isTraining == False:
                print(stat)

        elif reward < 0:
            self.number_of_unprofitable_trades += 1
            self.unprofitable_trades_loss += reward * num_tokens
            stat = {
                "pnl": reward * num_tokens,
                "pnl_perc": (reward / (self.prices[l_trade_tick])) * 100,
                "quantity": num_tokens,
                "price_difference": reward,
                "duration in mins": (self._current_tick - l_trade_tick),
                "start": l_trade_tick,
                "end": self._current_tick,
                "entry_price": self.prices[l_trade_tick],
                "exit_price": self.prices[self._current_tick],
                "position": "long" if action == Actions.Sell.value else "short",
                "net_worth": self._get_current_balance(),
            }
            self.unprofitable_trades_list.append(stat)

            if self.isTraining == False:
                print(stat)

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
