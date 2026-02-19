"""Schedule and timeframe alignment mixin for StrategyRunner.

This mixin provides timeframe-aligned scheduling for entry and exit signal checks.
All datetime operations assume timezone-aware timestamps matching the broker timezone.
"""

from datetime import datetime, timedelta
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class RunnerScheduleMixin:
    """Scheduling and bar-deduplication helpers for timeframe-aligned trading."""

    def _calculate_next_entry_time(self, from_time: datetime) -> datetime:
        """Calculate next entry time aligned to timeframe boundary from given time."""
        offset_td = timedelta(seconds=self.strategy_offset_seconds)

        # Calculate next timeframe boundary
        next_boundary_minute = ((from_time.minute // self.timeframe_minutes) + 1) * self.timeframe_minutes
        hours_to_add = next_boundary_minute // 60
        adjusted_minute = next_boundary_minute % 60

        next_entry = from_time.replace(minute=adjusted_minute, second=0, microsecond=0)
        if hours_to_add:
            next_entry += timedelta(hours=hours_to_add)
        next_entry += offset_td

        # Handle edge case where offset pushes us backwards
        if next_entry <= from_time:
            next_entry += timedelta(minutes=self.timeframe_minutes)

        return next_entry

    def _calculate_next_exit_time(self, from_time: datetime) -> datetime:
        """Calculate next exit time at next minute boundary from given time."""
        offset_td = timedelta(seconds=self.strategy_offset_seconds)

        # Round up to next minute boundary
        next_exit = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        next_exit += offset_td

        # Handle edge case where offset pushes us backwards
        if next_exit <= from_time:
            next_exit += timedelta(minutes=1)

        return next_exit

    def _initialize_schedule(self) -> tuple[datetime, datetime]:
        """Calculate initial next_entry_time and next_exit_time based on timeframe."""
        now = datetime.now()
        next_entry = self._calculate_next_entry_time(now)
        next_exit = self._calculate_next_exit_time(now)
        return next_entry, next_exit

    def _should_check_entry_signal(self, data: pd.DataFrame) -> bool:
        """Check if we should process entry signal (timeframe-aligned).

        For 1M timeframe: always process (every bar is new).
        For multi-minute timeframes: deduplicate by comparing bar timestamps.

        Validates timezone-aware timestamps to prevent comparison errors.
        """
        if data is None or len(data) == 0:
            return False

        # 1M timeframe: process every bar
        if self.timeframe_minutes == 1:
            return True

        # Multi-minute timeframe: deduplicate bars
        current_bar_time = data.index[-1]

        # Validate timezone-aware comparison (DataHandler returns TZ-aware timestamps)
        if self.last_processed_bar_time is not None:
            if current_bar_time.tzinfo is None or self.last_processed_bar_time.tzinfo is None:
                logger.warning(f"BarTZWarn strat={self.strategy_name} | cur_tz={current_bar_time.tzinfo} | last_tz={self.last_processed_bar_time.tzinfo}")
                # Fail-safe: treat as new bar to avoid blocking trades
                self.last_processed_bar_time = current_bar_time
                return True

            if current_bar_time <= self.last_processed_bar_time:
                logger.debug(f"BarSkip strat={self.strategy_name} | reason=already_processed")
                return False

        self.last_processed_bar_time = current_bar_time
        return True
