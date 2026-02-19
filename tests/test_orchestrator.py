"""Orchestrator unit tests."""

from __future__ import annotations

import ctypes
from datetime import datetime
from multiprocessing import Lock, Value
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytz

from tests.util import ensure_mt5_stub

ensure_mt5_stub()

from trading_system.orchestrator import (
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_LOG_INTERVAL_SECONDS,
    Orchestrator,
)


class DummyManager:
    """Simple Manager stand-in for unit tests."""

    def dict(self, initial=None):
        if initial is None:
            return {}
        return dict(initial)


@pytest.fixture
def orchestrator(tmp_path):
    """Build orchestrator with patched Manager and trade ID manager."""
    broker_config = SimpleNamespace(broker_tz=pytz.UTC)
    with patch("trading_system.orchestrator.Manager", return_value=DummyManager()):
        with patch(
            "trading_system.orchestrator.TradeIDSequenceManager",
            return_value=MagicMock(),
        ):
            instance = Orchestrator(
                broker_config=broker_config,
                account_type="BROKER1",
                log_root=str(tmp_path),
            )

    instance.global_position_count = Value(ctypes.c_int, 0)
    instance.global_trade_count = Value(ctypes.c_int, 0)
    instance.position_cache_lock = Lock()
    return instance


def test_init_uses_normalized_log_root_path(tmp_path):
    """trade_id_db_path must derive from normalized Path(log_root)."""
    broker_config = SimpleNamespace(broker_tz=pytz.UTC)
    with patch("trading_system.orchestrator.Manager", return_value=DummyManager()):
        with patch(
            "trading_system.orchestrator.TradeIDSequenceManager",
            return_value=MagicMock(),
        ):
            instance = Orchestrator(
                broker_config=broker_config,
                account_type="BROKER1",
                log_root=str(tmp_path),
            )

    assert instance.trade_id_db_path == Path(tmp_path) / "trade_id_sequence.db"


def test_update_shared_cache_in_place_preserves_proxy_reference(orchestrator):
    """Position cache update should mutate existing shared dict rather than replace proxy."""
    original_cache = orchestrator.manager.dict(
        {
            101: {"ticket": 101, "volume": 0.10},
            102: {"ticket": 102, "volume": 0.20},
        }
    )
    orchestrator.shared_state["position_cache"] = original_cache

    orchestrator._update_shared_cache(
        {
            102: {"ticket": 102, "volume": 0.25},
            103: {"ticket": 103, "volume": 0.30},
        }
    )

    assert orchestrator.shared_state["position_cache"] is original_cache
    assert set(original_cache.keys()) == {102, 103}
    assert original_cache[102]["volume"] == 0.25
    assert original_cache[103]["volume"] == 0.30
    assert orchestrator.shared_state["position_cache_timestamp"] > 0.0


def test_get_system_metrics_uses_global_position_counter(orchestrator):
    """Metrics cache_positions should be sourced from reconciled global counter."""
    with orchestrator.global_position_count.get_lock():
        orchestrator.global_position_count.value = 5
    with orchestrator.global_trade_count.get_lock():
        orchestrator.global_trade_count.value = 9

    orchestrator.shared_state["position_cache"] = orchestrator.manager.dict({1: {}, 2: {}})
    process_mock = Mock()
    process_mock.is_alive.return_value = True
    orchestrator.strategy_processes = {"s1": process_mock}

    metrics = orchestrator._get_system_metrics()

    assert metrics.active_processes == 1
    assert metrics.cache_positions == 5
    assert metrics.total_trades == 9


def test_should_log_heartbeat_uses_configured_interval(orchestrator):
    """Heartbeat log interval should follow HEARTBEAT_LOG_INTERVAL_SECONDS."""
    boundary_time = datetime(2026, 1, 1, 10, 15, 2)
    non_boundary_time = datetime(2026, 1, 1, 10, 7, 0)
    now_ts = boundary_time.timestamp()
    full_interval_ago = now_ts - HEARTBEAT_LOG_INTERVAL_SECONDS
    slight_jitter_ago = now_ts - (HEARTBEAT_LOG_INTERVAL_SECONDS - 1)
    too_recent_ago = now_ts - (
        HEARTBEAT_LOG_INTERVAL_SECONDS - HEARTBEAT_INTERVAL_SECONDS - 1
    )

    with patch("trading_system.orchestrator.time.time", return_value=now_ts):
        assert orchestrator._should_log_heartbeat(boundary_time, full_interval_ago) is True
        assert orchestrator._should_log_heartbeat(boundary_time, slight_jitter_ago) is True
        assert orchestrator._should_log_heartbeat(boundary_time, too_recent_ago) is False
        assert orchestrator._should_log_heartbeat(non_boundary_time, full_interval_ago) is False


def test_monitor_heartbeats_runs_first_cycle_before_sync_sleep(orchestrator):
    """Startup heartbeat cycle should refresh before waiting for next boundary."""
    call_order: list[str] = []
    orchestrator.shared_state["shutdown_flag"] = False

    def refresh_side_effect() -> None:
        call_order.append("refresh")
        orchestrator.shared_state["shutdown_flag"] = True

    def sync_side_effect() -> None:
        call_order.append("sync")

    with patch.object(orchestrator, "refresh_position_cache", side_effect=refresh_side_effect):
        with patch.object(orchestrator, "sync_to_next_heartbeat", side_effect=sync_side_effect):
            with patch.object(orchestrator, "_should_log_heartbeat", return_value=False):
                orchestrator.monitor_heartbeats()

    assert call_order == ["refresh", "sync"]


def test_discover_strategies_is_deterministic_by_name(orchestrator):
    """Strategy discovery order should be deterministic regardless of filesystem order."""
    orchestrator.account_enabled_strategies = {"alpha": True, "zeta": True}
    orchestrator.global_risk_policy["strategy_risk"] = {"alpha": 0.01, "zeta": 0.01}

    alpha_config = {
        "name": "alpha",
        "strategy_module": "strategies.alpha.strategy",
        "strategy_class": "AlphaStrategy",
        "params": {"symbol": "EURUSD", "timeframe": "M15"},
        "execution": {"magic_number": 100001},
    }
    zeta_config = {
        "name": "zeta",
        "strategy_module": "strategies.zeta.strategy",
        "strategy_class": "ZetaStrategy",
        "params": {"symbol": "GBPUSD", "timeframe": "M5"},
        "execution": {"magic_number": 100002},
    }

    def import_side_effect(module_path):
        if module_path.endswith(".alpha.config"):
            return SimpleNamespace(get_config=lambda: alpha_config)
        if module_path.endswith(".zeta.config"):
            return SimpleNamespace(get_config=lambda: zeta_config)
        raise ImportError(module_path)

    with patch("trading_system.orchestrator.importlib.import_module", side_effect=import_side_effect):
        orchestrator.discover_strategies()

    assert list(orchestrator.strategy_configs.keys()) == ["alpha", "zeta"]


def test_discover_strategies_raises_on_name_mismatch(orchestrator):
    orchestrator.account_enabled_strategies = {"alpha": True}
    bad_config = {
        "name": "wrong_name",
        "strategy_module": "strategies.alpha.strategy",
        "strategy_class": "AlphaStrategy",
        "params": {"symbol": "EURUSD", "timeframe": "M15"},
        "execution": {"magic_number": 100001},
    }

    with patch(
        "trading_system.orchestrator.importlib.import_module",
        return_value=SimpleNamespace(get_config=lambda: bad_config),
    ):
        orchestrator.discover_strategies()

    assert orchestrator.strategy_configs["alpha"]["name"] == "wrong_name"


def test_discover_strategies_raises_on_strategy_module_mismatch(orchestrator):
    orchestrator.account_enabled_strategies = {"alpha": True}
    bad_config = {
        "name": "alpha",
        "strategy_module": "strategies.alpha.other",
        "strategy_class": "AlphaStrategy",
        "params": {"symbol": "EURUSD", "timeframe": "M15"},
        "execution": {"magic_number": 100001},
    }

    with patch(
        "trading_system.orchestrator.importlib.import_module",
        return_value=SimpleNamespace(get_config=lambda: bad_config),
    ):
        orchestrator.discover_strategies()

    assert orchestrator.strategy_configs["alpha"]["strategy_module"].endswith(".other")


def test_discover_strategies_raises_on_invalid_strategy_class(orchestrator):
    orchestrator.account_enabled_strategies = {"alpha": True}
    bad_config = {
        "name": "alpha",
        "strategy_module": "strategies.alpha.strategy",
        "strategy_class": "not-valid-class",
        "params": {"symbol": "EURUSD", "timeframe": "M15"},
        "execution": {"magic_number": 100001},
    }

    with patch(
        "trading_system.orchestrator.importlib.import_module",
        return_value=SimpleNamespace(get_config=lambda: bad_config),
    ):
        orchestrator.discover_strategies()

    assert orchestrator.strategy_configs["alpha"]["strategy_class"] == "not-valid-class"
