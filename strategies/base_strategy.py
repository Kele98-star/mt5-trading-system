from abc import ABC, abstractmethod
from datetime import datetime, time as dt_time
from functools import lru_cache
from typing import Optional, Union

import pandas as pd

from src.trading_system.core.execution_requests import EntryRequest, ModifyRequest, ExitRequest


class BaseStrategy(ABC):
    """
    Base strategy interface with entry/exit/modify hooks.

    Strategies must implement:
    - generate_entry_signal(): Return EntryRequest or None
    - generate_exit_signal(): Return ExitRequest or None
    - generate_modify_signal(): Return ModifyRequest or None
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_entry_signal(self, data: pd.DataFrame) -> Optional[EntryRequest]:
        """
        Generate entry signal based on current market data.

        Args:
            data: DataFrame with OHLCV columns, timezone-aware index

        Returns:
            EntryRequest if signal detected, None otherwise
        """
        pass

    @abstractmethod
    def generate_exit_signal(self, pos_dict: dict, data: pd.DataFrame) -> Optional[ExitRequest]:
        """
        Generate exit signal for open position.

        Args:
            pos_dict: Position dict with keys: ticket, symbol, type, volume,
                     price_open, sl, tp, profit
            data: DataFrame with OHLCV columns, timezone-aware index

        Returns:
            ExitRequest if exit condition met, None otherwise.
        """
        pass

    @abstractmethod
    def generate_modify_signal(self, pos_dict: dict, data: pd.DataFrame) -> Union[ModifyRequest, ExitRequest, None]:
        """
        Generate SL/TP modification signal for open position.

        Called BEFORE generate_exit_signal() in monitoring loop.
        Use for trailing stops, breakeven adjustments, or partial profit taking.

        Args:
            pos_dict: Position dict with keys: ticket, symbol, type, volume,
                     price_open, sl, tp, profit
            data: DataFrame with OHLCV columns, timezone-aware index

        Returns:
            ModifyRequest if modification needed, None otherwise.

        Example:
            # Move SL to breakeven when position reaches 0.2% profit
            current_price = data['Close'].iloc[-1]
            is_long = pos_dict['type'] == 0  # 0=BUY, 1=SELL
            direction = 1 if is_long else -1
            returnpct = (current_price - pos_dict['price_open']) / pos_dict['price_open'] * direction

            if returnpct >= 0.002 and pos_dict['sl'] != pos_dict['price_open']:
                return ModifyRequest(
                    ticket=pos_dict['ticket'],
                    newsl=pos_dict['price_open'],  # Set SL to entry price
                    newtp=pos_dict['tp'],          # Keep existing TP
                    comment="Breakeven adjustment"
                )
            return None
        """
        pass

    @staticmethod
    @lru_cache(maxsize=32)
    def _parse_time(time_str: str) -> pd.Timestamp:
        """Parse HH:MM once and reuse cached timestamp object."""
        return pd.to_datetime(time_str, format='%H:%M')

    @staticmethod
    def _parse_time_with_offset(time_str: str, offset: pd.Timedelta) -> dt_time:
        """Parse time string and apply offset."""
        return (BaseStrategy._parse_time(time_str) + offset).time()

    @staticmethod
    def _parse_time_to_str(time_str: str, offset: pd.Timedelta) -> str:
        """Parse time string, apply offset, return formatted string."""
        return (BaseStrategy._parse_time(time_str) + offset).strftime('%H:%M')
