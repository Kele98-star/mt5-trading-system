"""DataHandler unit tests."""

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

from tests.util import ensure_mt5_stub, patch_data_handler_dependencies

ensure_mt5_stub()

from trading_system.config.broker_config import string_to_timeframe
from trading_system.core.data_handler import DataHandler


@contextmanager
def mock_data_handler_now(target_time: datetime):
    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            if tz is None:
                return target_time.replace(tzinfo=None)
            return target_time.astimezone(tz)

        @staticmethod
        def fromtimestamp(ts, tz=None):
            return datetime.fromtimestamp(ts, tz=tz)

        @staticmethod
        def fromisoformat(value):
            return datetime.fromisoformat(value)

    with patch("trading_system.core.data_handler.datetime", FrozenDateTime()):
        yield


@pytest.fixture
def broker_tz():
    return pytz.timezone("Europe/Bucharest")


@pytest.fixture
def strategy_tz():
    return pytz.timezone("Europe/Budapest")


@pytest.fixture
def handler(broker_tz):
    data_handler = DataHandler(broker_tz=broker_tz)
    data_handler._get_timeframe_minutes = lambda timeframe: 15 if timeframe == "M15" else 1
    return data_handler


@pytest.fixture
def strategy_config_factory():
    def _factory(
        symbol: str = "EURUSD",
        timeframe: str = "M15",
        strategy_timezone: str = "Europe/Budapest",
        filter_enabled: bool = False,
        sessions=None,
        backcandles: int = 100,
    ) -> dict[str, Any]:
        return {
            "params": {
                "symbol": symbol,
                "timeframe": timeframe,
                "backcandles": backcandles,
                "expiration_time": 60,
            },
            "trading_hours": {
                "timezone": strategy_timezone,
                "enabled": filter_enabled,
                "sessions": sessions or [{"start": "09:00", "end": "17:00"}],
            },
            "execution": {
                "magic_number": 12345,
                "deviation": 10,
                "comment_prefix": "TEST",
                "min_market_threshold_points": 5,
            },
            "filters": {
                "news_filter": {
                    "enabled": False,
                    "currencies": ["USD", "EUR"],
                    "buffer_minutes": 30,
                }
            },
        }

    return _factory


@pytest.fixture
def make_strategy_module():
    def _factory(config):
        module = MagicMock()
        module.get_config = MagicMock(return_value=config)
        module.__spec__ = MagicMock()
        module.__name__ = "mock_strategy_module"
        return module

    return _factory


@pytest.fixture
def make_rates():
    def _factory(
        count: int,
        start_time: datetime,
        timeframe_minutes: int,
        base_price: float = 1.1,
        volatility: float = 0.001,
    ):
        if start_time.tzinfo is None:
            anchor = start_time.replace(tzinfo=timezone.utc)
        else:
            anchor = start_time.astimezone(timezone.utc)

        timestamps = [
            int((anchor + timedelta(minutes=i * timeframe_minutes)).timestamp())
            for i in range(count)
        ]

        np.random.seed(42)
        changes = np.random.normal(0, volatility, count)
        closes = base_price + np.cumsum(changes)
        opens = np.roll(closes, 1)
        opens[0] = base_price
        highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, volatility / 2, count))
        lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, volatility / 2, count))

        dtype = [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
        ]
        rates = np.zeros(count, dtype=dtype)
        rates["time"] = timestamps
        rates["open"] = opens
        rates["high"] = highs
        rates["low"] = lows
        rates["close"] = closes
        rates["tick_volume"] = np.random.randint(50, 500, count)
        rates["spread"] = np.random.randint(1, 5, count)
        return rates

    return _factory


@pytest.fixture
def make_tick():
    def _factory(bid: float = 1.1, ask: float = 1.1002, ts: datetime | None = None):
        timestamp = ts or datetime.now(pytz.UTC)
        tick = MagicMock()
        tick.bid = bid
        tick.ask = ask
        tick.last = (bid + ask) / 2
        tick.time = int(timestamp.timestamp())
        return tick

    return _factory


def test_cache_miss_triggers_full_refresh(
    handler,
    broker_tz,
    make_rates,
    strategy_config_factory,
    make_strategy_module,
):
    config = strategy_config_factory(backcandles=100, filter_enabled=False)
    frozen = broker_tz.localize(datetime(2026, 1, 20, 10, 0, 0))
    start = frozen - timedelta(minutes=99 * 15)
    rates = make_rates(count=100, start_time=start.replace(tzinfo=None), timeframe_minutes=15)

    with mock_data_handler_now(frozen):
        with patch_data_handler_dependencies(
            copy_rates=MagicMock(return_value=rates),
            import_module=MagicMock(return_value=make_strategy_module(config)),
        ) as deps:
            result = handler.get_latest_bars(strategy_name="test_strategy")

    assert result is not None
    assert len(result) == 99
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    deps["copy_rates"].assert_called_once()
    cache_key = ("EURUSD", string_to_timeframe("M15"))
    assert cache_key in handler._latest_bar_cache


def test_cache_hit_returns_from_window(
    handler,
    broker_tz,
    strategy_tz,
    make_rates,
    strategy_config_factory,
    make_strategy_module,
):
    frozen = broker_tz.localize(datetime(2026, 1, 20, 10, 5, 0))
    last_bar = broker_tz.localize(datetime(2026, 1, 20, 10, 0, 0))

    start = last_bar - timedelta(minutes=599 * 15)
    rates = make_rates(count=600, start_time=start.replace(tzinfo=None), timeframe_minutes=15)

    cache_key = ("EURUSD", string_to_timeframe("M15"))
    window = deque(maxlen=600)
    for rate in rates:
        bar_time_broker = datetime.fromtimestamp(rate["time"], tz=broker_tz)
        bar_time_strategy = bar_time_broker.astimezone(strategy_tz)
        window.append(
            pd.DataFrame(
                {
                    "Open": [rate["open"]],
                    "High": [rate["high"]],
                    "Low": [rate["low"]],
                    "Close": [rate["close"]],
                    "Volume": [rate["tick_volume"]],
                    "spread": [rate["spread"]],
                },
                index=pd.DatetimeIndex([bar_time_strategy]),
            )
        )

    handler._bar_rolling_windows[cache_key] = window
    handler._latest_bar_cache[cache_key] = {
        "bar_time": pd.Timestamp(last_bar),
        "timeframe_minutes": 15,
        "cache_seeded": True,
        "next_bar_complete_at": pd.Timestamp(last_bar) + pd.Timedelta(minutes=15, seconds=2),
    }

    config = strategy_config_factory(backcandles=99, filter_enabled=False)
    with mock_data_handler_now(frozen):
        with patch_data_handler_dependencies(
            copy_rates=MagicMock(return_value=[]),
            import_module=MagicMock(return_value=make_strategy_module(config)),
        ) as deps:
            result = handler.get_latest_bars(strategy_name="test_strategy")

    assert result is not None
    assert len(result) == 100
    deps["copy_rates"].assert_not_called()


def test_session_filter_normal_hours(handler, strategy_tz):
    config = {
        "filter_enabled": True,
        "sessions": [{"start": "09:00", "end": "17:00"}],
        "strategy_timezone": "Europe/Budapest",
        "timezone": "Europe/Budapest",
    }
    index = pd.DatetimeIndex(
        [
            "2026-01-20 08:00",
            "2026-01-20 09:00",
            "2026-01-20 12:00",
            "2026-01-20 17:00",
            "2026-01-20 18:00",
        ],
        tz=strategy_tz,
    )
    df = pd.DataFrame({"Open": [1.1] * 5, "Close": [1.1] * 5}, index=index)

    filtered = handler._filter_session_hours(df, config)

    assert [bar.hour for bar in filtered.index] == [9, 12, 17]


def test_session_filter_midnight_spanning(handler, strategy_tz):
    config = {
        "filter_enabled": True,
        "sessions": [{"start": "23:00", "end": "01:00"}],
        "strategy_timezone": "Europe/Budapest",
        "timezone": "Europe/Budapest",
    }
    index = pd.DatetimeIndex(
        [
            "2026-01-20 22:00",
            "2026-01-20 23:00",
            "2026-01-21 00:00",
            "2026-01-21 01:00",
            "2026-01-21 02:00",
        ],
        tz=strategy_tz,
    )
    df = pd.DataFrame({"Open": [1.1] * 5, "Close": [1.1] * 5}, index=index)

    filtered = handler._filter_session_hours(df, config)

    assert [bar.hour for bar in filtered.index] == [23, 0, 1]


def test_filter_incomplete_bars_rejects_forming_bar(handler, strategy_tz):
    config = {"timeframe": "M15", "timeframe_minutes": 15}
    df = pd.DataFrame(
        {"Open": [1.1], "Close": [1.1]},
        index=pd.DatetimeIndex(["2026-01-20 09:00"], tz=strategy_tz),
    )
    current = strategy_tz.localize(datetime(2026, 1, 20, 9, 10, 0))

    with mock_data_handler_now(current):
        filtered = handler._filter_complete_bars(df, config)

    assert len(filtered) == 0


def test_filter_complete_bars_accepts_closed_bar(handler, strategy_tz):
    config = {"timeframe": "M15", "timeframe_minutes": 15}
    df = pd.DataFrame(
        {"Open": [1.1], "Close": [1.1]},
        index=pd.DatetimeIndex(["2026-01-20 09:00"], tz=strategy_tz),
    )
    frozen = strategy_tz.localize(datetime(2026, 1, 20, 9, 17, 0))

    with mock_data_handler_now(frozen):
        filtered = handler._filter_complete_bars(df, config)

    assert len(filtered) == 1


def test_mt5_failure_returns_none(handler, strategy_config_factory, make_strategy_module):
    config = strategy_config_factory()

    with patch_data_handler_dependencies(
        copy_rates=MagicMock(return_value=None),
        import_module=MagicMock(return_value=make_strategy_module(config)),
    ):
        result = handler.get_latest_bars(strategy_name="test_strategy")

    assert result is None


def test_timezone_conversion_broker_to_strategy(
    handler,
    broker_tz,
    make_rates,
    strategy_config_factory,
    make_strategy_module,
):
    config = strategy_config_factory(strategy_timezone="Europe/Budapest", backcandles=10)
    frozen = broker_tz.localize(datetime(2026, 1, 20, 10, 0, 0))
    start = frozen - timedelta(minutes=9 * 15)
    rates = make_rates(count=10, start_time=start.replace(tzinfo=None), timeframe_minutes=15)

    with mock_data_handler_now(frozen):
        with patch_data_handler_dependencies(
            copy_rates=MagicMock(return_value=rates),
            import_module=MagicMock(return_value=make_strategy_module(config)),
        ):
            result = handler.get_latest_bars(strategy_name="test_strategy")

    assert result is not None
    assert result.index.tz.zone == "Europe/Budapest"


def test_get_current_tick_success(handler, make_tick):
    tick_time = datetime(2026, 1, 20, 10, 0, 0, tzinfo=pytz.UTC)

    with patch_data_handler_dependencies(symbol_tick=MagicMock(return_value=make_tick(ts=tick_time))):
        result = handler.get_current_tick(symbol="EURUSD")

    assert result is not None
    assert result["bid"] == 1.1
    assert result["ask"] == 1.1002


def test_get_current_tick_with_strategy_timezone(
    handler,
    make_tick,
    strategy_config_factory,
    make_strategy_module,
):
    tick_time = datetime(2026, 1, 20, 10, 0, 0, tzinfo=pytz.UTC)
    config = strategy_config_factory(strategy_timezone="Europe/Budapest")

    with patch_data_handler_dependencies(
        symbol_tick=MagicMock(return_value=make_tick(ts=tick_time)),
        import_module=MagicMock(return_value=make_strategy_module(config)),
    ):
        handler._load_strategy_config("test_strategy")
        result = handler.get_current_tick(symbol="EURUSD", strategy_name="test_strategy")

    assert "time_strategy" in result
    assert result["time_strategy"].tzinfo.zone == "Europe/Budapest"


def test_deque_window_maintains_600_bars(
    handler,
    broker_tz,
    make_rates,
    strategy_config_factory,
    make_strategy_module,
):
    frozen = broker_tz.localize(datetime(2026, 1, 15, 9, 0, 0))
    start = frozen - timedelta(minutes=599 * 15)
    rates = make_rates(count=600, start_time=start.replace(tzinfo=None), timeframe_minutes=15)

    config = strategy_config_factory(symbol="EURUSD", timeframe="M15", backcandles=600, filter_enabled=False)

    with mock_data_handler_now(frozen):
        with patch_data_handler_dependencies(
            copy_rates=MagicMock(return_value=rates),
            import_module=MagicMock(return_value=make_strategy_module(config)),
        ):
            handler.get_latest_bars(strategy_name="test_strategy")

    cache_key = ("EURUSD", string_to_timeframe("M15"))
    assert len(handler._bar_rolling_windows[cache_key]) == 599
