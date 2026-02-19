"""RiskManager unit tests."""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import pytz

from tests.util import FakeAtomicInt, ensure_mt5_stub, patch_risk_dependencies

ensure_mt5_stub()

from trading_system.core.risk import RiskManager


@dataclass
class MarketCondition:
    is_valid: bool
    reason: str
    spread_points: float = 0.0
    spread_price: float = 0.0
    slippage_points: float = 0.0
    slippage_price: float = 0.0


@pytest.fixture
def risk_env():
    mt = MagicMock()
    account = Mock(balance=10000.0, equity=10000.0)
    symbol = Mock(
        trade_tick_size=0.25,
        trade_tick_value=5.0,
        trade_contract_size=20.0,
        volume_min=1.0,
        volume_max=100.0,
        volume_step=1.0,
        bid=18500.0,
        ask=18500.25,
    )
    mt.account_info.return_value = account
    mt.symbol_info.return_value = symbol
    mt.positions_get.return_value = []
    mt.history_deals_get.return_value = []
    mt.last_error.return_value = (0, "Success")

    calculator_class = MagicMock()
    calculator = Mock()
    calculator.validate_market_conditions.return_value = MarketCondition(
        is_valid=True,
        reason="Market conditions acceptable",
        spread_points=0.5,
        spread_price=0.125,
        slippage_points=0.25,
        slippage_price=0.0625,
    )
    calculator.max_spread_points = 2.0
    calculator.max_slippage_points = 1.0
    calculator_class.return_value = calculator

    with patch_risk_dependencies(
        mt=mt,
        market_cost_calculator=calculator_class,
        news_filter=MagicMock(),
    ):
        yield {
            "mt": mt,
            "account": account,
            "symbol": symbol,
            "calculator": calculator,
        }


@pytest.fixture
def strategy_config():
    return {
        "name": "NQRSI",
        "risk": {
            "max_positions": 3,
            "max_trades": 10,
            "max_spread_points": 2.0,
            "max_slippage_points": 1.0,
            "max_slippage_price": 0.5,
            "max_slippage_pct": 0.05,
        },
        "execution": {"magic_number": 100001},
        "filters": {"news_filter": {"enabled": False}},
    }


@pytest.fixture
def global_policy():
    return {
        "max_total_positions": 5,
        "max_daily_trades": 20,
        "max_daily_drawdown_pct": 0.03,
        "max_drawdown_pct": 0.10,
        "initial_balance": 10000.0,
        "strategy_risk": {"NQRSI": 0.01, "ESGC": 0.015},
        "adaptive_sizing": {
            "enabled": False,
            "scope": "portfolio",
            "drawdown_thresholds": [
                {"drawdown_pct": 0.05, "risk_multiplier": 0.5},
                {"drawdown_pct": 0.03, "risk_multiplier": 0.75},
                {"drawdown_pct": 0.0, "risk_multiplier": 1.0},
            ],
        },
    }


@pytest.fixture
def shared_state():
    return {"daily_trade_counts": {"NQRSI": 0, "ESGC": 0}}


@pytest.fixture
def global_counters():
    return {"positions": FakeAtomicInt(0), "trades": FakeAtomicInt(0)}


@pytest.fixture
def mock_data_handler():
    data_handler = Mock()
    data_handler.get_news_events.return_value = []
    return data_handler


@pytest.fixture
def risk_manager(
    risk_env,
    strategy_config,
    global_policy,
    shared_state,
    global_counters,
    mock_data_handler,
):
    return RiskManager(
        strategy_config=strategy_config,
        global_policy=global_policy,
        shared_state=shared_state,
        global_position_count=global_counters["positions"],
        global_trade_count=global_counters["trades"],
        data_handler=mock_data_handler,
        broker_tz=pytz.timezone("Europe/Bucharest"),
    )


@pytest.fixture
def set_account(risk_env):
    def _set(balance: float, equity: float | None = None):
        risk_env["account"].balance = balance
        risk_env["account"].equity = equity if equity is not None else balance

    return _set


@pytest.fixture
def set_positions(risk_env):
    def _set(magic: int, count: int):
        positions = []
        for idx in range(count):
            pos = Mock()
            pos.magic = magic
            pos.profit = 100.0 * (idx + 1)
            pos.symbol = "NQ"
            positions.append(pos)
        risk_env["mt"].positions_get.return_value = positions

    return _set


@pytest.fixture
def set_market_condition(risk_env):
    def _set(is_valid: bool, reason: str = "OK", spread_points: float = 0.5):
        risk_env["calculator"].validate_market_conditions.return_value = MarketCondition(
            is_valid=is_valid,
            reason=reason,
            spread_points=spread_points,
            spread_price=spread_points * 0.25,
            slippage_points=0.25,
            slippage_price=0.0625,
        )

    return _set


class TestRiskManagerInitialization:
    def test_initialization_success(self, risk_manager, strategy_config):
        assert risk_manager.strategy_name == "NQRSI"
        assert risk_manager.risk_config == strategy_config["risk"]

    def test_drawdown_cache_not_initialized_on_startup(self, risk_manager):
        assert risk_manager._max_dd_cache.initialized is False
        assert risk_manager._daily_dd_cache.initialized is False


class TestGlobalRiskLimits:
    def test_global_position_limit_not_reached(self, risk_manager, global_counters):
        global_counters["positions"].value = 3
        assert risk_manager.check_global_risk().can_trade is True

    def test_global_position_limit_reached(self, risk_manager, global_counters):
        global_counters["positions"].value = 5
        assert risk_manager.check_global_risk().can_trade is False

    def test_daily_trade_limit_not_reached(self, risk_manager, global_counters):
        global_counters["trades"].value = 10
        assert risk_manager.check_global_risk().can_trade is True

    def test_daily_trade_limit_reached(self, risk_manager, global_counters):
        global_counters["trades"].value = 20
        assert risk_manager.check_global_risk().can_trade is False

    def test_max_drawdown_limit_exceeded(self, risk_manager, risk_env):
        risk_manager._max_dd_cache.initialized = True
        risk_manager._max_dd_cache.peak_equity = 10000.0
        risk_manager._max_dd_cache.current_equity = 8500.0
        risk_manager._max_dd_cache.last_deal_time = datetime.now()
        risk_env["account"].equity = 8500.0
        risk_env["mt"].history_deals_get.return_value = []

        result = risk_manager.check_global_risk()

        assert result.can_trade is False
        assert "Drawdown" in result.reason

    def test_daily_drawdown_limit_exceeded(self, risk_manager, risk_env):
        risk_manager._daily_dd_cache.initialized = True
        risk_manager._daily_dd_cache.peak_equity = 10000.0
        risk_manager._daily_dd_cache.current_equity = 9650.0
        risk_manager._daily_dd_cache.last_deal_time = datetime.now()
        risk_manager._daily_dd_cache.cache_date = datetime.now().date()
        risk_env["account"].equity = 9650.0
        risk_env["mt"].history_deals_get.return_value = []

        result = risk_manager.check_global_risk()

        assert result.can_trade is False
        assert "Daily Drawdown" in result.reason

    def test_max_drawdown_limit_not_exceeded(self, risk_manager, risk_env):
        risk_manager._max_dd_cache.initialized = True
        risk_manager._max_dd_cache.peak_equity = 10000.0
        risk_manager._max_dd_cache.current_equity = 9800.0
        risk_manager._max_dd_cache.last_deal_time = datetime.now()
        risk_env["account"].equity = 9800.0
        risk_env["mt"].history_deals_get.return_value = []

        assert risk_manager.check_global_risk().can_trade is True

    def test_daily_drawdown_limit_not_exceeded(self, risk_manager, risk_env):
        risk_manager._daily_dd_cache.initialized = True
        risk_manager._daily_dd_cache.peak_equity = 10000.0
        risk_manager._daily_dd_cache.current_equity = 9800.0
        risk_manager._daily_dd_cache.last_deal_time = datetime.now()
        risk_manager._daily_dd_cache.cache_date = datetime.now().date()
        risk_env["account"].equity = 9800.0
        risk_env["mt"].history_deals_get.return_value = []

        assert risk_manager.check_global_risk().can_trade is True


class TestStrategyLimits:
    def test_strategy_position_limit_not_reached(self, risk_manager, set_positions):
        set_positions(magic=100001, count=2)
        assert risk_manager.check_strategy_limits("NQRSI").can_trade is True

    def test_strategy_position_limit_reached(self, risk_manager, set_positions):
        set_positions(magic=100001, count=3)
        assert risk_manager.check_strategy_limits("NQRSI").can_trade is False

    def test_skip_position_limit_check(self, risk_manager, set_positions):
        set_positions(magic=100001, count=3)
        result = risk_manager.check_strategy_limits("NQRSI", skip_position_limit_check=True)
        assert result.can_trade is True


class TestMarketConditionValidation:
    def test_validate_trade_with_acceptable_spread(self, risk_manager, set_market_condition):
        set_market_condition(is_valid=True, reason="OK", spread_points=0.5)

        result = risk_manager.validate_trade(
            strategy_name="NQRSI",
            symbol="NQ",
            expected_price=18500.0,
            sl_price=18490.0,
            signal=1,
        )

        assert result.can_trade is True

    def test_validate_trade_with_excessive_spread(self, risk_manager, set_market_condition):
        set_market_condition(is_valid=False, reason="Spread exceeds limit", spread_points=3.0)

        result = risk_manager.validate_trade(
            strategy_name="NQRSI",
            symbol="NQ",
            expected_price=18500.0,
            sl_price=18490.0,
            signal=1,
        )

        assert result.can_trade is False
        assert "spread" in result.reason.lower()


class TestPositionSizing:
    def test_basic_position_size_calculation(self, risk_manager):
        volume = risk_manager.calculate_position_size(
            symbol="NQ",
            entry=18500.0,
            sl=18490.0,
            strategy_name="NQRSI",
        )
        assert volume == 1.0

    def test_position_size_zero_for_zero_sl_distance(self, risk_manager):
        volume = risk_manager.calculate_position_size(
            symbol="NQ",
            entry=18500.0,
            sl=18500.0,
            strategy_name="NQRSI",
        )
        assert volume == 0.0


class TestDrawdownCacheInitialization:
    def test_max_drawdown_cache_initializes_with_no_deals(self, risk_manager, risk_env):
        risk_env["mt"].history_deals_get.return_value = []
        drawdown = risk_manager.get_drawdown(scope="portfolio")
        assert drawdown == 0.0
        assert risk_manager._max_dd_cache.initialized is True

    def test_daily_drawdown_uses_start_of_day_equity_baseline(self, risk_manager, risk_env):
        risk_env["account"].balance = 9094.0
        risk_env["account"].equity = 9094.0
        risk_env["mt"].history_deals_get.return_value = []

        drawdown = risk_manager.get_daily_drawdown(scope="portfolio")
        global_check = risk_manager.check_global_risk()

        assert drawdown == 0.0
        assert global_check.can_trade is True

    def test_daily_drawdown_includes_start_balance_when_first_trade_is_loss(self, risk_manager, risk_env):
        risk_env["account"].balance = 9900.0
        risk_env["account"].equity = 9900.0
        risk_env["mt"].history_deals_get.return_value = [
            Mock(
                type=0,
                magic=100001,
                profit=-100.0,
                commission=0.0,
                swap=0.0,
                fee=0.0,
                time_msc=1_700_000_000_000,
                ticket=101,
            )
        ]

        drawdown = risk_manager.get_daily_drawdown(scope="portfolio")
        assert drawdown == pytest.approx(0.01)

    def test_max_drawdown_init_includes_start_balance_when_first_trade_is_loss(self, risk_manager, risk_env):
        risk_env["account"].balance = 9900.0
        risk_env["account"].equity = 9900.0
        risk_env["mt"].history_deals_get.return_value = [
            Mock(
                type=0,
                magic=100001,
                profit=-100.0,
                commission=0.0,
                swap=0.0,
                fee=0.0,
                time_msc=1_700_000_000_000,
                ticket=202,
            )
        ]

        drawdown = risk_manager.get_drawdown(scope="portfolio")
        assert drawdown == pytest.approx(0.01)

    def test_filter_trading_deals_respects_cursor_on_same_millisecond(self, risk_manager):
        first = Mock(
            type=0,
            magic=100001,
            profit=1.0,
            commission=-0.1,
            swap=0.0,
            fee=0.0,
            time_msc=1_700_000_000_000,
            ticket=10,
        )
        second = Mock(
            type=0,
            magic=100001,
            profit=2.0,
            commission=-0.1,
            swap=0.0,
            fee=0.0,
            time_msc=1_700_000_000_000,
            ticket=11,
        )

        filtered = risk_manager._filter_trading_deals(
            [first, second],
            scope="portfolio",
            magic=0,
            min_cursor=(1_700_000_000_000, 10),
        )

        assert filtered is not None
        net_pnl, trade_count, last_trade_cursor, _ = filtered
        assert trade_count == 1
        assert np.allclose(net_pnl, np.array([1.9]), rtol=1e-5)
        assert last_trade_cursor == (1_700_000_000_000, 11)

    def test_calc_dd_pct_handles_non_positive_peak(self, risk_manager):
        risk_manager._max_dd_cache.peak_equity = 0.0
        risk_manager._max_dd_cache.current_equity = -100.0
        assert risk_manager._calc_dd_pct(risk_manager._max_dd_cache) == 0.0


class TestValidateTradeIntegration:
    def test_validate_trade_passes_all_layers(self, risk_manager, set_market_condition):
        set_market_condition(is_valid=True)

        result = risk_manager.validate_trade(
            strategy_name="NQRSI",
            symbol="NQ",
            expected_price=18500.0,
            sl_price=18490.0,
            signal=1,
        )

        assert result.can_trade is True
        assert result.volume > 0.0
