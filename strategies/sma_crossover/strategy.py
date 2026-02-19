"""SMA Crossover Strategy - Long on fast SMA cross above slow SMA, short on cross below."""

import logging

import pandas as pd

from src.trading_system.core.indicators import calculate_sma
from src.trading_system.core.execution_requests import EntryRequest, ExitRequest, ModifyRequest
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SMACrossover(BaseStrategy):
    def __init__(self, params: dict):
        super().__init__(params)
        self.symbol = params['symbol']
        self.strategy_name = params['strategy_name']
        self.fast_period = params['fast_period']
        self.slow_period = params['slow_period']
        self.sl_points = params['sl_points']

    def generate_entry_signal(self, data: pd.DataFrame) -> Optional[EntryRequest]:
        if data is None or len(data) < self.slow_period + 1:
            return None

        fast = calculate_sma(data, self.fast_period)
        slow = calculate_sma(data, self.slow_period)

        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
            return None

        prev_above = fast.iloc[-2] > slow.iloc[-2]
        curr_above = fast.iloc[-1] > slow.iloc[-1]

        if not prev_above and curr_above:
            signal = 1  # BUY
        elif prev_above and not curr_above:
            signal = -1  # SELL
        else:
            return None

        logger.info(
            f"SMAXo strat={self.strategy_name} | signal={'BUY' if signal == 1 else 'SELL'} | "
            f"fast={fast.iloc[-1]:.5f} | slow={slow.iloc[-1]:.5f}"
        )

        return EntryRequest(
            order_type='market',
            symbol=self.symbol,
            volume=1.0,
            signal=signal,
            sl_points=self.sl_points,
            strategy_name=self.strategy_name,
            comment="SMA_XO",
        )

    def generate_exit_signal(self, pos_dict: dict, data: pd.DataFrame) -> ExitRequest | None:
        if data is None or len(data) < self.slow_period + 1:
            return None

        fast = calculate_sma(data, self.fast_period)
        slow = calculate_sma(data, self.slow_period)

        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
            return None

        is_long = pos_dict['type'] == 0
        fast_above = fast.iloc[-1] > slow.iloc[-1]

        # Exit when SMA cross reverses against the position
        if (is_long and not fast_above) or (not is_long and fast_above):
            return ExitRequest(
                ticket=pos_dict['ticket'],
                portion=1.0,
                exit_reason="SMA cross reversal",
                strategy_name=self.strategy_name,
                comment="SMA_XO_Exit",
            )

        return None

    def generate_modify_signal(self, pos_dict: dict, data: pd.DataFrame) -> ModifyRequest | ExitRequest | None:
        return None

    def calculate_sl(self, data: pd.DataFrame, signal: int) -> float | None:
        return None

    def calculate_tp(self, data: pd.DataFrame, signal: int) -> float | None:
        return None