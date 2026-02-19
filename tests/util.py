"""
Shared mock objects and patch bundles for trading system unit tests.
"""

from __future__ import annotations

import ctypes
import sys
import tempfile
import threading
import types
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Lock, Value
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch


IMPORTANT_PATCH_TARGETS: dict[str, str] = {
    "data_handler_copy_rates": "trading_system.core.data_handler.mt5.copy_rates_from_pos",
    "data_handler_symbol_tick": "trading_system.core.data_handler.mt5.symbol_info_tick",
    "runner_execution_symbol_tick": "trading_system.runner_execution_mixin.mt.symbol_info_tick",
    "data_handler_import_module": "trading_system.core.data_handler.importlib.import_module",
    "data_handler_datetime": "trading_system.core.data_handler.datetime",
    "risk_mt": "trading_system.core.risk.mt",
    "risk_market_cost_calculator": "trading_system.core.risk.MarketCostCalculator",
    "risk_news_filter": "trading_system.core.risk.NewsFilter",
    "trade_logger_mt5": "trading_system.core.trade_logger.mt5",
    "strategy_runner_trade_logger": "trading_system.strategy_runner.TradeLogger",
    "strategy_runner_async_trade_logger": "trading_system.strategy_runner.AsyncTradeLogger",
    "time_time": "time.time",
}


def create_mt5_stub_module() -> types.ModuleType:
    """
    Build a MetaTrader5 stub module suitable for importing system modules in tests.
    """
    mt5_stub = types.ModuleType("MetaTrader5")

    constants = {
        "TIMEFRAME_M1": 1,
        "TIMEFRAME_M5": 5,
        "TIMEFRAME_M15": 15,
        "TIMEFRAME_M30": 30,
        "TIMEFRAME_H1": 60,
        "TIMEFRAME_H4": 240,
        "TIMEFRAME_D1": 1440,
        "TIMEFRAME_W1": 10080,
        "TIMEFRAME_MN1": 43200,
        "ORDER_TYPE_BUY": 0,
        "ORDER_TYPE_SELL": 1,
        "ORDER_TYPE_BUY_LIMIT": 2,
        "ORDER_TYPE_SELL_LIMIT": 3,
        "ORDER_TYPE_BUY_STOP": 4,
        "ORDER_TYPE_SELL_STOP": 5,
        "ORDER_TYPE_BUY_STOP_LIMIT": 6,
        "ORDER_TYPE_SELL_STOP_LIMIT": 7,
        "ORDER_FILLING_FOK": 0,
        "ORDER_FILLING_IOC": 1,
        "ORDER_FILLING_RETURN": 2,
        "ORDER_FILLING_BOC": 3,
        "TRADE_ACTION_DEAL": 0,
        "TRADE_ACTION_PENDING": 1,
        "TRADE_ACTION_SLTP": 2,
        "TRADE_ACTION_MODIFY": 3,
        "TRADE_ACTION_REMOVE": 4,
        "TRADE_ACTION_CLOSE_BY": 5,
        "ORDER_TIME_GTC": 0,
        "ORDER_TIME_DAY": 1,
        "ORDER_TIME_SPECIFIED": 2,
        "ORDER_TIME_SPECIFIED_DAY": 3,
        "POSITION_TYPE_BUY": 0,
        "POSITION_TYPE_SELL": 1,
        "TRADE_RETCODE_DONE": 10009,
    }
    for name, value in constants.items():
        setattr(mt5_stub, name, value)

    mt5_stub.initialize = MagicMock(return_value=True)
    mt5_stub.shutdown = MagicMock(return_value=True)
    mt5_stub.login = MagicMock(return_value=True)
    mt5_stub.last_error = MagicMock(return_value=(0, "Success"))
    mt5_stub.account_info = MagicMock(return_value=Mock(balance=10000.0, equity=10000.0))
    mt5_stub.symbol_info = MagicMock(return_value=Mock())
    mt5_stub.symbol_info_tick = MagicMock(return_value=Mock(bid=1.1, ask=1.1002))
    mt5_stub.positions_get = MagicMock(return_value=[])
    mt5_stub.orders_get = MagicMock(return_value=[])
    mt5_stub.history_deals_get = MagicMock(return_value=[])
    mt5_stub.copy_rates_from_pos = MagicMock(return_value=[])
    mt5_stub.order_send = MagicMock(return_value=Mock(retcode=10009))

    def _fallback_constant(_name: str) -> int:
        # MT5 exposes many integer constants. Returning 0 keeps imports/test code stable.
        return 0

    mt5_stub.__getattr__ = _fallback_constant
    return mt5_stub


def ensure_mt5_stub(force: bool = False) -> Any:
    """
    Ensure `MetaTrader5` exists in `sys.modules`.
    """
    if not force and "MetaTrader5" in sys.modules:
        return sys.modules["MetaTrader5"]
    module = create_mt5_stub_module()
    sys.modules["MetaTrader5"] = module
    return module


@dataclass
class MockPatchBundle:
    """
    Handles returned from `patch_important_dependencies`.
    """

    data_handler_copy_rates: Any
    data_handler_symbol_tick: Any
    data_handler_import_module: Any
    risk_mt: Any
    risk_market_cost_calculator: Any
    risk_news_filter: Any
    trade_logger_mt5: Any
    strategy_runner_trade_logger: Any
    strategy_runner_async_trade_logger: Any
    time_time: Any


@contextmanager
def patch_important_dependencies(
    *,
    data_handler_copy_rates: Any | None = None,
    data_handler_symbol_tick: Any | None = None,
    data_handler_import_module: Any | None = None,
    risk_mt: Any | None = None,
    risk_market_cost_calculator: Any | None = None,
    risk_news_filter: Any | None = None,
    trade_logger_mt5: Any | None = None,
    strategy_runner_trade_logger: Any | None = None,
    strategy_runner_async_trade_logger: Any | None = None,
    time_time: Any | None = None,
    data_handler_datetime: Any | None = None,
) -> Generator[MockPatchBundle, None, None]:
    """
    Patch every high-value external dependency used across the test suite.
    """
    ensure_mt5_stub()

    with ExitStack() as stack:
        patched_bundle = MockPatchBundle(
            data_handler_copy_rates=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["data_handler_copy_rates"],
                    data_handler_copy_rates if data_handler_copy_rates is not None else MagicMock(return_value=[]),
                )
            ),
            data_handler_symbol_tick=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["data_handler_symbol_tick"],
                    data_handler_symbol_tick
                    if data_handler_symbol_tick is not None
                    else MagicMock(return_value=Mock(bid=1.1, ask=1.1002)),
                )
            ),
            data_handler_import_module=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["data_handler_import_module"],
                    data_handler_import_module
                    if data_handler_import_module is not None
                    else MagicMock(return_value=Mock(get_config=Mock(return_value={}))),
                )
            ),
            risk_mt=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["risk_mt"],
                    risk_mt if risk_mt is not None else MagicMock(),
                )
            ),
            risk_market_cost_calculator=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["risk_market_cost_calculator"],
                    risk_market_cost_calculator
                    if risk_market_cost_calculator is not None
                    else MagicMock(),
                )
            ),
            risk_news_filter=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["risk_news_filter"],
                    risk_news_filter if risk_news_filter is not None else MagicMock(),
                )
            ),
            trade_logger_mt5=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["trade_logger_mt5"],
                    trade_logger_mt5 if trade_logger_mt5 is not None else MagicMock(),
                )
            ),
            strategy_runner_trade_logger=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["strategy_runner_trade_logger"],
                    strategy_runner_trade_logger
                    if strategy_runner_trade_logger is not None
                    else MagicMock(),
                )
            ),
            strategy_runner_async_trade_logger=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["strategy_runner_async_trade_logger"],
                    strategy_runner_async_trade_logger
                    if strategy_runner_async_trade_logger is not None
                    else MagicMock(side_effect=lambda trade_logger, max_queue_size=100: trade_logger),
                )
            ),
            time_time=stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["time_time"],
                    time_time if time_time is not None else MagicMock(return_value=0.0),
                )
            ),
        )

        if data_handler_datetime is not None:
            stack.enter_context(patch(IMPORTANT_PATCH_TARGETS["data_handler_datetime"], data_handler_datetime))

        yield patched_bundle


@contextmanager
def patch_data_handler_dependencies(
    *,
    copy_rates: Any | None = None,
    symbol_tick: Any | None = None,
    import_module: Any | None = None,
    datetime_mock: Any | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Patch only DataHandler external calls."""
    ensure_mt5_stub()
    with ExitStack() as stack:
        patched = {
            "copy_rates": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["data_handler_copy_rates"],
                    copy_rates if copy_rates is not None else MagicMock(return_value=[]),
                )
            ),
            "symbol_tick": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["data_handler_symbol_tick"],
                    symbol_tick if symbol_tick is not None else MagicMock(return_value=None),
                )
            ),
            "import_module": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["data_handler_import_module"],
                    import_module if import_module is not None else MagicMock(),
                )
            ),
        }
        if datetime_mock is not None:
            stack.enter_context(patch(IMPORTANT_PATCH_TARGETS["data_handler_datetime"], datetime_mock))
        yield patched


@contextmanager
def patch_risk_dependencies(
    *,
    mt: Any | None = None,
    market_cost_calculator: Any | None = None,
    news_filter: Any | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Patch only RiskManager dependencies."""
    ensure_mt5_stub()
    with ExitStack() as stack:
        patched = {
            "mt": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["risk_mt"],
                    mt if mt is not None else MagicMock(),
                )
            ),
            "market_cost_calculator": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["risk_market_cost_calculator"],
                    market_cost_calculator if market_cost_calculator is not None else MagicMock(),
                )
            ),
            "news_filter": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["risk_news_filter"],
                    news_filter if news_filter is not None else MagicMock(),
                )
            ),
        }
        yield patched


@contextmanager
def patch_strategy_runner_loggers(
    *,
    trade_logger: Any | None = None,
    async_trade_logger: Any | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Patch StrategyRunner logger dependencies."""
    with ExitStack() as stack:
        patched = {
            "trade_logger": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["strategy_runner_trade_logger"],
                    trade_logger if trade_logger is not None else MagicMock(),
                )
            ),
            "async_trade_logger": stack.enter_context(
                patch(
                    IMPORTANT_PATCH_TARGETS["strategy_runner_async_trade_logger"],
                    async_trade_logger
                    if async_trade_logger is not None
                    else MagicMock(side_effect=lambda logger, max_queue_size=100: logger),
                )
            ),
        }
        yield patched


@dataclass
class MockPosition:
    """
    Mock MT5 position object for testing.
    
    Mimics MetaTrader5.TradePosition structure with minimal fields.
    Use in tests to avoid MT5 terminal dependency.
    
    Example:
        >>> pos = MockPosition(ticket=12345, symbol="EURUSD", type=0, volume=1.0)
        >>> assert pos.ticket == 12345
    """
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    magic: int
    profit: float = 0.0
    swap: float = 0.0
    time: int = None
    
    def __post_init__(self):
        """set default timestamp if not provided."""
        if self.time is None:
            self.time = int(datetime.now().timestamp())

class FakeAtomicInt:
    """Test double for Atomicint Protocol."""
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value
    @value.setter
    def value(self, val: int):
        self._value = val
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value
    def get_lock(self):
        return self._lock

@dataclass
class MockMT5Config:
    """
    Mock MT5 configuration for testing.
    
    Provides minimal credentials without actual broker validation.
    """
    login: int = 12345678
    password: str = "test_password"
    server: str = "TestBroker-Demo"
    path: str = ""


class MockDataHandler:
    """
    Mock DataHandler for testing strategies without market data.
    
    Returns empty DataFrames or predefined test data.
    """
    
    def __init__(self):
        self.strategy_configs_cache = {}
    
    def _load_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """Return mock strategy configuration."""
        return {
            'symbol': 'EURUSD',
            'timeframe': 'M15',
            'strategy_timezone': 'America/New_York',
            'filter_enabled': False,
            'sessions': [],
            'number_of_bars': 100,
            'magic_number': 100001,
            'deviation': 10,
            'expiration_time': None,
            'comment_prefix': 'TEST_',
            'min_market_threshold_points': 5.0,
            'timezone': 'America/New_York',
            'news_filter_enabled': False,
            'currencies': [],
            'buffer_minutes': 15,
        }
    
    def get_latest_bars(self, symbol: str, strategy_name: str = None):
        """Return empty DataFrame (override in specific tests)."""
        import pandas as pd
        return pd.DataFrame()


class MockTradeLogger:
    """
    Mock TradeLogger that stores trades in memory (no disk I/O).
    
    Validates method calls and parameters without SQLite dependency.
    """
    
    def __init__(self):
        self.fills_logged = []
        self.closes_logged = []
        self.partials_logged = []
        self.open_trades_by_ticket_last_two: dict[int, dict[str, Any]] = {}
    
    def log_fill(
        self,
        trade_id: int | None = None,
        position: Any | None = None,
        expected_entry_price: float | None = None,
        opening_sl: float | None = None,
        strategy_name: str | None = None,
        fill_time_ms: float | None = None,
        volume_multiplier: float | None = None,
        **kwargs: Any,
    ):
        """Record fill for assertion."""
        if position is None and hasattr(trade_id, "trade_id") and hasattr(trade_id, "position"):
            data = trade_id
            trade_id = getattr(data, "trade_id", None)
            position = getattr(data, "position", None)
            expected_entry_price = getattr(data, "expected_entry_price", expected_entry_price)
            opening_sl = getattr(data, "opening_sl", opening_sl)

        if trade_id is None and kwargs:
            trade_id = kwargs.get("trade_id")
            position = kwargs.get("position")
            expected_entry_price = kwargs.get("expected_entry_price")
            opening_sl = kwargs.get("opening_sl")

        self.fills_logged.append({
            'trade_id': trade_id,
            'ticket': getattr(position, "ticket", None),
            'symbol': getattr(position, "symbol", None),
            'volume': getattr(position, "volume", None),
            'entry_price': getattr(position, "price_open", None),
            'expected_entry_price': expected_entry_price,
            'opening_sl': opening_sl,
        })
    
    def log_close(
        self,
        trade_id: int | None = None,
        position: Any | None = None,
        expected_exit_price: float | None = None,
        opening_sl: float | None = None,
        exit_trigger: str | None = None,
        entry_price: float | None = None,
        expected_entry_price: float | None = None,
        strategy_name: str | None = None,
        exit_price: float | None = None,
        **kwargs: Any,
    ):
        """Record close for assertion."""
        if position is None and hasattr(trade_id, "trade_id") and hasattr(trade_id, "position"):
            data = trade_id
            trade_id = getattr(data, "trade_id", None)
            position = getattr(data, "position", None)
            expected_exit_price = getattr(data, "expected_exit_price", expected_exit_price)
            exit_trigger = getattr(data, "exit_trigger", exit_trigger)

        if trade_id is None and kwargs:
            trade_id = kwargs.get("trade_id")
            position = kwargs.get("position")
            expected_exit_price = kwargs.get("expected_exit_price", exit_price)
            exit_trigger = kwargs.get("exit_trigger")

        self.closes_logged.append({
            'trade_id': trade_id,
            'ticket': getattr(position, "ticket", None),
            'expected_exit_price': expected_exit_price,
            'exit_trigger': exit_trigger,
        })
    
    def log_partial_close(
        self,
        trade_id: int | None = None,
        position: Any | None = None,
        closed_volume: float | None = None,
        remaining_volume: float | None = None,
        exit_price: float | None = None,
        expected_exit_price: float | None = None,
        opening_sl: float | None = None,
        strategy_name: str | None = None,
        exit_trigger: str | None = None,
        entry_price: float | None = None,
        expected_entry_price: float | None = None,
        **kwargs: Any,
    ):
        """Record partial close for assertion."""
        if position is None and hasattr(trade_id, "trade_id") and hasattr(trade_id, "position"):
            data = trade_id
            trade_id = getattr(data, "trade_id", None)
            position = getattr(data, "position", None)
            closed_volume = getattr(data, "closed_volume", closed_volume)
            remaining_volume = getattr(data, "remaining_volume", remaining_volume)

        if trade_id is None and kwargs:
            trade_id = kwargs.get("trade_id")
            position = kwargs.get("position")
            closed_volume = kwargs.get("closed_volume")
            remaining_volume = kwargs.get("remaining_volume")

        self.partials_logged.append({
            'trade_id': trade_id,
            'ticket': getattr(position, "ticket", None),
            'closed_volume': closed_volume,
            'remaining_volume': remaining_volume,
        })

    def get_open_trades_by_ticket_last_three(self, tickets):
        """Return configured reconciliation rows for provided tickets."""
        return {
            ticket: dict(self.open_trades_by_ticket_last_two[ticket])
            for ticket in tickets
            if ticket in self.open_trades_by_ticket_last_two
        }


class MockStrategyRunner:
    """
    Lightweight StrategyRunner mock for unit testing.
    
    Includes only essential attributes/methods needed for testing
    bidirectional mapping and trade ID system.
    
    Usage:
        >>> runner = MockStrategyRunner()
        >>> trade_id = runner._generate_trade_id()
        >>> runner.entry_metadata[trade_id] = {...}
    """
    
    def __init__(self, 
                 strategy_name: str = "test_strategy",
                 temp_dir: Path | None = None):
        """
        Initialize mock runner with minimal dependencies.
        
        Args:
            strategy_name: Strategy identifier for logging
            temp_dir: Temporary directory for test files (auto-created if None)
        """
        self.strategy_name = strategy_name
        
        # Create temporary directory for test databases
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp())
        else:
            self.temp_dir = temp_dir
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Core tracking structures
        self.entry_metadata: dict[int, dict[str, Any]] = {}
        self.ticket_to_trade_id: dict[int, int] = {}
        self.known_positions: set[int] = set()
        
        # Atomic counters (process-safe)
        self.trade_id_counter = Value(ctypes.c_int, 0)
        self.global_position_count = Value(ctypes.c_int, 0)
        self.position_state_lock = Lock()
        
        # Mock dependencies
        self.trade_logger = MockTradeLogger()
        self.data_handler = MockDataHandler()
        
        # Global risk policy (minimal)
        self.global_risk_policy = {
            'max_total_positions': 10,
            'max_daily_trades': 50,
            'log_root': str(self.temp_dir),
        }
        
        # Initialize trade ID sequence from database
        self._init_trade_id_sequence()
    
    def _init_trade_id_sequence(self):
        """Initialize trade ID counter from database (or start at 0)."""
        from trading_system.core.trade_id_manager import TradeIDSequenceManager
        
        db_path = self.temp_dir / "trade_id_sequence.db"
        manager = TradeIDSequenceManager(db_path)
        
        with self.trade_id_counter.get_lock():
            self.trade_id_counter.value = manager.get_current_id()
    
    def _generate_trade_id(self) -> int:
        with self.trade_id_counter.get_lock():
            self.trade_id_counter.value += 1
            return self.trade_id_counter.value

    def _persist_trade_id_sequence(self):
        """
        Persist current trade ID sequence to database.
        
        Called explicitly when persistence is needed (e.g., before shutdown).
        """
        from trading_system.core.trade_id_manager import TradeIDSequenceManager
        
        db_path = self.temp_dir / "trade_id_sequence.db"
        manager = TradeIDSequenceManager(db_path)
        
        # Access the database connection directly to update the sequence
        current_id = self.trade_id_counter.value
        
        # Write to database using SQL directly (bypasses unknown API)
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Assume table structure: CREATE TABLE sequence (id INTEGER PRIMARY KEY, current_value INTEGER)
        cursor.execute("UPDATE sequence SET current_value = ? WHERE id = 1", (current_id,))
        if cursor.rowcount == 0:
            # Table doesn't exist or no rows - create/insert
            cursor.execute("CREATE TABLE IF NOT EXISTS sequence (id INTEGER PRIMARY KEY, current_value INTEGER)")
            cursor.execute("INSERT OR REPLACE INTO sequence (id, current_value) VALUES (1, ?)", (current_id,))
        
        conn.commit()
        conn.close()

    
    def _get_trade_id_by_ticket(self, ticket: int) -> int | None:
        """
        O(1) reverse lookup: ticket -> trade_id.
        
        Returns:
            trade_id if found, None if orphaned position
        """
        return self.ticket_to_trade_id.get(ticket)
    
    def _handle_new_fill(self, pos_dict: dict[str, Any]):
        """
        Simplified fill handler for testing.
        
        Tests the bidirectional mapping creation logic.
        """
        ticket = pos_dict['ticket']
        
        # ADDED: Duplicate detection (MUST be first check)
        if ticket in self.known_positions:
            # Position already tracked - skip to avoid double-counting
            return
        
        # Reverse lookup: ticket -> trade_id
        trade_id = self._get_trade_id_by_ticket(ticket)
        
        if trade_id is None:
            # Orphaned position - generate fallback trade_id
            trade_id = self._generate_trade_id()
            
            # Create minimal metadata
            self.entry_metadata[trade_id] = {
                'expected_entry_price': pos_dict['price_open'],
                'opening_sl': pos_dict['sl'],
                'submission_time': datetime.now().timestamp(),
                'volume_multiplier': None,
                'ticket': ticket,
            }
            
            # Store bidirectional mapping
            self.ticket_to_trade_id[ticket] = trade_id
        
        # Increment counters (only reached if not duplicate)
        new_positions = self.atomic_increment_position()
        new_trades = self.atomic_increment_trade()
        
        # Add to known positions
        self.known_positions.add(ticket)
        
        # Log fill (rest of method unchanged)
        position_obj = MockPosition(
            ticket=pos_dict['ticket'],
            symbol=pos_dict['symbol'],
            type=pos_dict['type'],
            volume=pos_dict['volume'],
            price_open=pos_dict['price_open'],
            sl=pos_dict['sl'],
            tp=pos_dict['tp'],
            magic=pos_dict['magic'],
        )
        
        metadata = self.entry_metadata[trade_id]
        self.trade_logger.log_fill(
            trade_id=trade_id,
            position=position_obj,
            expected_entry_price=metadata['expected_entry_price'],
            opening_sl=metadata['opening_sl'],
            strategy_name=self.strategy_name,
        )

    def _log_full_close_execution(self, pos_dict: dict[str, Any]):
        """
        Simplified close handler for testing cleanup logic.
        
        Handles missing metadata gracefully (corrupted state recovery).
        """
        ticket = pos_dict['ticket']
        trade_id = self._get_trade_id_by_ticket(ticket)
        
        # CRITICAL: Always remove from known_positions (even if no metadata)
        self.known_positions.discard(ticket)
        
        # Early return if no metadata (can't log or cleanup further)
        if trade_id is None:
            return
        
        # Cleanup tracking structures
        self.entry_metadata.pop(trade_id, None)
        self.ticket_to_trade_id.pop(ticket, None)
        
        # Log close
        position_obj = MockPosition(
            ticket=pos_dict['ticket'],
            symbol=pos_dict['symbol'],
            type=pos_dict['type'],
            volume=pos_dict['volume'],
            price_open=pos_dict['price_open'],
            sl=pos_dict['sl'],
            tp=pos_dict['tp'],
            magic=pos_dict['magic'],
        )
        
        self.trade_logger.log_close(
            trade_id=trade_id,
            position=position_obj,
            exit_price=pos_dict['price_open'] + 0.0010,  # Mock profit
            expected_exit_price=pos_dict['price_open'] + 0.0010,
            opening_sl=pos_dict['sl'],
            strategy_name=self.strategy_name,
            exit_trigger='TEST_CLOSE',
            entry_price=pos_dict['price_open'],
            expected_entry_price=pos_dict['price_open'],
        )

    def cleanup(self):
        """Clean up temporary test files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def atomic_increment_position(self) -> int:
        """
        Atomically increment global position counter.
        
        Returns:
            New position count after increment
        """
        with self.global_position_count.get_lock():
            self.global_position_count.value += 1
            return self.global_position_count.value
    
    def atomic_decrement_positions(self, count: int) -> int:
        """
        Atomically decrement global position counter.
        
        Args:
            count: Number of positions to decrement
        
        Returns:
            New position count after decrement
        """
        with self.global_position_count.get_lock():
            self.global_position_count.value -= count
            return self.global_position_count.value
    
    def atomic_increment_trade(self) -> int:
        """
        Atomically increment global trade counter.
        
        Returns:
            New trade count after increment
        """
        # Note: MockStrategyRunner doesn't have global_trade_count in current implementation
        # For testing, we'll add a local counter
        if not hasattr(self, 'local_trade_count'):
            self.local_trade_count = Value(ctypes.c_int, 0)
        
        with self.local_trade_count.get_lock():
            self.local_trade_count.value += 1
            return self.local_trade_count.value
