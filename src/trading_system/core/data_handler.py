from collections import deque
from typing import Any
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timezone
import importlib
import pytz
from src.trading_system.config.broker_config import string_to_timeframe, TIMEFRAME_TO_MINUTES


class DataHandler:
    def __init__(self, broker_tz: pytz.tzinfo.BaseTzInfo):
        self._strategy_configs_cache = {}
        self._strategy_tz_cache = {}
        self.broker_tz = broker_tz
        self._latest_bar_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._bar_rolling_windows: dict[tuple[str, int], Any] = {}

    def _get_strategy_tz(self, strategy_name: str) -> pytz.timezone:
        if strategy_name not in self._strategy_tz_cache:
            config = self._load_strategy_config(strategy_name)
            self._strategy_tz_cache[strategy_name] = pytz.timezone(config['strategy_timezone'])
        return self._strategy_tz_cache[strategy_name]

    def _materialize_from_deque(self, cache_key: tuple[str, int]) -> pd.DataFrame | None:
        """Return cached rolling frame, materializing legacy deque only once."""
        cached_window = self._bar_rolling_windows.get(cache_key)
        if cached_window is None:
            return None
        if isinstance(cached_window, pd.DataFrame):
            return cached_window
        if not cached_window:
            return None
        materialized = pd.concat(list(cached_window), ignore_index=False).sort_index()
        self._bar_rolling_windows[cache_key] = materialized
        return materialized

    def _get_cache_capacity(self, strategy_config: dict) -> int:
        window_size = strategy_config["number_of_bars"]
        if strategy_config.get("filter_enabled", False):
            return max(window_size, int(window_size * 1.3))
        return window_size

    def _return_cached_window(self, cache_key: tuple[str, int], strategy_config: dict) -> pd.DataFrame | None:
        """Materialize cached window, apply session filter, and trim to size."""
        df = self._materialize_from_deque(cache_key)
        if df is None or len(df) == 0:
            return df
        if strategy_config['filter_enabled']:
            df = self._filter_session_hours(df, strategy_config)
        return df.tail(strategy_config['number_of_bars'])

    def _load_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        if strategy_name in self._strategy_configs_cache:
            return self._strategy_configs_cache[strategy_name]
        
        config_module = importlib.import_module(f'strategies.{strategy_name}.config')
        config = config_module.get_config()
        
        params = config.get('params', {})
        trading_hours = config.get('trading_hours', {})
        execution = config.get('execution', {})
        news_filter = config.get('filters', {}).get('news_filter', {})
        
        parsed = {
            'symbol': params.get('symbol'),
            'timeframe': params.get('timeframe'),
            'strategy_timezone': trading_hours.get('timezone'),
            'filter_enabled': trading_hours.get('enabled'),
            'sessions': trading_hours.get('sessions'),
            'number_of_bars': params.get('backcandles', 0) + 1,
            'magic_number': execution.get('magic_number'),
            'deviation': execution.get('deviation'),
            'expiration_time': params.get('expiration_time'),
            'comment_prefix': execution.get('comment_prefix'),
            'min_market_threshold_points': execution.get('min_market_threshold_points'),
            'timezone': trading_hours.get('timezone'),
            'news_filter_enabled': news_filter.get('enabled'),
            'currencies': news_filter.get('currencies'),
            'buffer_minutes': news_filter.get('buffer_minutes'),
            'timeframe_minutes': TIMEFRAME_TO_MINUTES.get(
                string_to_timeframe(params.get('timeframe'))
            ),
        }

        self._strategy_configs_cache[strategy_name] = parsed
        return parsed

    def _update_cache_metadata(self, cache_key: tuple[str, int], bar_time_broker: pd.Timestamp, timeframe_minutes: int) -> None:
        """Centralized cache metadata update to eliminate duplication."""
        tolerance_seconds = 2
        next_bar_complete_at = bar_time_broker + pd.Timedelta(
            minutes=timeframe_minutes,
            seconds=tolerance_seconds
        )
        
        self._latest_bar_cache[cache_key] = {
            'bar_time': bar_time_broker,
            'timeframe_minutes': timeframe_minutes,
            'next_bar_complete_at': next_bar_complete_at,
            'cache_seeded': True
        }

    def _create_tz_aware_dataframe(self, rates: np.ndarray, strategy_tz: pytz.timezone) -> pd.DataFrame:
        """Create timezone-aware DataFrame from MT5 rates array."""
        timestamps = pd.to_datetime(rates['time'], unit='s')
        tz_aware_index = timestamps.tz_localize(self.broker_tz).tz_convert(strategy_tz)
        tz_aware_index.name = 'Time'
        
        df = pd.DataFrame({
            'Open': rates['open'],
            'High': rates['high'],
            'Low': rates['low'],
            'Close': rates['close'],
            'Volume': rates['tick_volume'],
            'spread': rates['spread']
        }, index=tz_aware_index)
        # Ensure chronological order in case MT5 returns newest-first.
        return df.sort_index()

    def _should_skip_mt5_call(self, cached_metadata: dict | None, now_broker: datetime) -> bool:
        """Check if predictive skip applies (current bar still forming)."""
        if not cached_metadata or not cached_metadata.get('cache_seeded', False):
            return False
        
        next_complete_at = cached_metadata.get('next_bar_complete_at')
        return next_complete_at is not None and now_broker < next_complete_at

    def _calculate_bars_elapsed(self, cached_metadata: dict, now_broker: datetime) -> float:
        """Calculate how many bars have elapsed since last cached bar."""
        last_cached_time = cached_metadata['bar_time']
        timeframe_minutes = cached_metadata['timeframe_minutes']
        
        bar_close_time = last_cached_time + pd.Timedelta(minutes=timeframe_minutes)
        time_since_close_minutes = (now_broker - bar_close_time).total_seconds() / 60.0
        return time_since_close_minutes / timeframe_minutes

    def get_latest_bars(self, strategy_name: str) -> pd.DataFrame | None:
        """
        Fetch latest bars with predictive caching to eliminate unnecessary MT5 API calls.
        
        Cache Strategy:
        1. Predictive skip: If current bar incomplete, return cached window without MT5 call
        2. Same bar (cache hit): Bar complete but no new bar yet
        3. New bar (incremental): Fetch 1 bar from MT5, append to window
        4. Gap detected: Full refresh all bars
        """
        strategy_config = self._load_strategy_config(strategy_name)
        symbol = strategy_config['symbol']
        timeframe_mt5 = string_to_timeframe(strategy_config['timeframe'])
        cache_key = (symbol, timeframe_mt5)
        
        strategy_tz = self._get_strategy_tz(strategy_name)
        now_broker = datetime.now(self.broker_tz)
        cached_metadata = self._latest_bar_cache.get(cache_key)

        # Predictive skip: bar still forming
        if self._should_skip_mt5_call(cached_metadata, now_broker):
            cached = self._return_cached_window(cache_key, strategy_config)
            if cached is not None and len(cached) > 0:
                return cached

        # Determine cache strategy: same bar, incremental, or full refresh
        if cached_metadata and cached_metadata.get('cache_seeded', False):
            bars_elapsed = self._calculate_bars_elapsed(cached_metadata, now_broker)
            
            MIN_BARS_FOR_NEW = 1.0
            MAX_BARS_FOR_INCREMENTAL = 1.1
            
            if bars_elapsed < MIN_BARS_FOR_NEW:
                cached = self._return_cached_window(cache_key, strategy_config)
                if cached is not None and len(cached) > 0:
                    return cached
                return self._full_refresh_bars(
                    symbol, strategy_name, strategy_config, cache_key, strategy_tz
                )
            
            elif bars_elapsed <= MAX_BARS_FOR_INCREMENTAL:
                # Incremental update (single new bar)
                return self._fetch_and_append_new_bar(symbol, strategy_name, strategy_config, cache_key, strategy_tz)
        
        # Full refresh (cache miss or gap)
        return self._full_refresh_bars(symbol, strategy_name, strategy_config, cache_key, strategy_tz)

    def _fetch_and_append_new_bar(self,symbol: str,strategy_name: str,strategy_config: dict,cache_key: tuple[str, int],strategy_tz: pytz.timezone) -> pd.DataFrame | None:
        """Fetch single new bar and append to rolling window (incremental update)."""
        timeframe_mt5 = cache_key[1]

        cached_df = self._materialize_from_deque(cache_key)
        if cached_df is None or len(cached_df) == 0:
            return self._full_refresh_bars(
                symbol, strategy_name, strategy_config, cache_key, strategy_tz
            )

        # Fetch 2 bars from position 0 to handle bar transition race condition.
        # At bar boundaries, MT5 may not have created the new forming bar yet, so
        # position 0 could still be the just-completed bar rather than a forming bar.
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, 2)
        if rates is None or len(rates) == 0:
            return self._full_refresh_bars(
                symbol, strategy_name, strategy_config, cache_key, strategy_tz
            )

        new_bar_df = self._create_tz_aware_dataframe(rates, strategy_tz)
        new_bar_df = self._filter_complete_bars(new_bar_df, strategy_config)

        # No complete bars available
        if len(new_bar_df) == 0:
            if cached_metadata := self._latest_bar_cache.get(cache_key):
                cached_metadata['next_bar_complete_at'] = cached_metadata['bar_time'] + pd.Timedelta(minutes=cached_metadata['timeframe_minutes'], seconds=2)
            return self._return_cached_window(cache_key, strategy_config)

        # Take the latest complete bar
        latest_complete_bar = new_bar_df.iloc[[-1]]
        new_bar_time = latest_complete_bar.index[0]
        old_last_bar_time = cached_df.index[-1]
        is_new_bar = old_last_bar_time is None or new_bar_time > old_last_bar_time

        if not is_new_bar:
            return self._return_cached_window(cache_key, strategy_config)

        cache_capacity = self._get_cache_capacity(strategy_config)
        updated_df = pd.concat([cached_df, latest_complete_bar], ignore_index=False)
        self._bar_rolling_windows[cache_key] = updated_df.tail(cache_capacity).copy()

        new_bar_time_broker = latest_complete_bar.index[0].tz_convert(self.broker_tz)
        timeframe_minutes = strategy_config.get('timeframe_minutes', TIMEFRAME_TO_MINUTES[timeframe_mt5])

        self._update_cache_metadata(cache_key, new_bar_time_broker, timeframe_minutes)

        return self._return_cached_window(cache_key, strategy_config)

    def _full_refresh_bars(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: dict,
        cache_key: tuple[str, int],
        strategy_tz: pytz.timezone
    ) -> pd.DataFrame | None:
        """Full refresh of bar data from MT5 (cache miss or gap scenario)."""
        timeframe_mt5 = cache_key[1]
        fetch_count = self._get_cache_capacity(strategy_config)
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, fetch_count)
        
        if rates is None or len(rates) == 0:
            return None
        
        df_raw = self._create_tz_aware_dataframe(rates, strategy_tz)
        df_complete = self._filter_complete_bars(df_raw, strategy_config)

        # Cache latest fetched bar first to avoid repeated MT5 polling while a bar forms.
        last_fetched_bar_time = df_raw.index[-1].tz_convert(self.broker_tz)
        timeframe_minutes = strategy_config.get('timeframe_minutes', TIMEFRAME_TO_MINUTES[timeframe_mt5])
        self._update_cache_metadata(cache_key, last_fetched_bar_time, timeframe_minutes)

        if len(df_complete) == 0:
            self._bar_rolling_windows[cache_key] = pd.DataFrame(columns=df_raw.columns)
            return pd.DataFrame(columns=df_raw.columns)

        self._bar_rolling_windows[cache_key] = df_complete.tail(fetch_count).copy()

        # Use latest complete bar for elapsed-bar calculations.
        latest_complete_bar_time = df_complete.index[-1].tz_convert(self.broker_tz)
        self._update_cache_metadata(cache_key, latest_complete_bar_time, timeframe_minutes)

        return self._return_cached_window(cache_key, strategy_config)

    def _filter_complete_bars(self, df: pd.DataFrame, strategy_config: dict) -> pd.DataFrame:
        """Filter out incomplete bars (bars still forming)."""
        timeframe_minutes = strategy_config.get("timeframe_minutes")
        if timeframe_minutes is None:
            timeframe_minutes = self._get_timeframe_minutes(strategy_config["timeframe"])
        now_strategy = datetime.now(tz=df.index.tz)
        
        bar_close_times = df.index + pd.Timedelta(minutes=timeframe_minutes)
        tolerance = pd.Timedelta(seconds=2)
        mask_complete = bar_close_times <= (now_strategy + tolerance)
        
        return df[mask_complete]

    def get_current_tick(self, symbol: str, strategy_name: str | None = None) -> dict | None:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        tick_time_broker = datetime.fromtimestamp(tick.time, tz=self.broker_tz)
        tick_time_utc = tick_time_broker.astimezone(timezone.utc)

        result = {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'time': tick.time,
            'time_broker': tick_time_broker,
            'time_utc': tick_time_utc
        }

        if strategy_name:
            result['time_strategy'] = tick_time_broker.astimezone(self._get_strategy_tz(strategy_name))

        return result

    def _get_timeframe_minutes(self, timeframe_str: str) -> int:
        """Get timeframe period in minutes."""
        mt5_tf = string_to_timeframe(timeframe_str)
        timeframe_minutes = TIMEFRAME_TO_MINUTES.get(mt5_tf)
        return timeframe_minutes

    def _filter_session_hours(self, df: pd.DataFrame, strategy_config: dict) -> pd.DataFrame:
        """
        Filter bars to trading session hours using vectorized operations.
        Handles midnight-spanning sessions (e.g., 23:00-01:00).
        """
        if not strategy_config.get('filter_enabled', False) or not (sessions := strategy_config.get('sessions')):
            return df

        bar_minutes = df.index.hour * 60 + df.index.minute
        combined_mask = np.zeros(len(df), dtype=bool)
        
        for session in sessions:
            start_h, start_m = map(int, session['start'].split(':'))
            end_h, end_m = map(int, session['end'].split(':'))
            
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m
            
            if end_min >= start_min:
                session_mask = (bar_minutes >= start_min) & (bar_minutes <= end_min)
            else:
                # Midnight-spanning session
                session_mask = (bar_minutes >= start_min) | (bar_minutes <= end_min)
            
            combined_mask |= session_mask
        
        return df.loc[combined_mask]
