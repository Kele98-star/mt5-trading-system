"""StrategyRunner entrypoint with mixin-based internals."""

import importlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import MetaTrader5 as mt
import pandas as pd

from src.trading_system.config.meta_configs import META_LABELING_CONFIG
from src.trading_system.config.broker_config import TIMEFRAME_STRING_MAP, TIMEFRAME_TO_MINUTES
from src.trading_system.core.async_trade_logger import AsyncTradeLogger
from src.trading_system.core.connection import MT5Connection
from src.trading_system.core.data_handler import DataHandler
from src.trading_system.core.execution import OrderExecutor
from src.trading_system.core.risk import RiskManager
from src.trading_system.core.trade_id_manager import TradeIDSequenceManager
from src.trading_system.core.trade_logger import TradeLogger
from src.trading_system.core.types import AtomicInt, ProcessLock
from src.trading_system.runner_execution_mixin import RunnerExecutionMixin
from src.trading_system.runner_position_mixin import RunnerPositionMixin
from src.trading_system.runner_schedule_mixin import RunnerScheduleMixin
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class ExitLogData:
    """Parameters for exit logging operations."""

    ticket: int
    expected_exit_price: float
    exit_trigger: str
    expected_entry_price: float
    opening_sl: float
    entry_price: float
    closed_volume: float | None = None
    deal_id: int | None = None


@dataclass
class PendingTicket:
    """Pending order tracking data."""

    order_type: str
    symbol: str
    magic: int
    submission_time: float
    ticket: int | None = None
    buy_order_ticket: int | None = None
    sell_order_ticket: int | None = None
    expected_volume: float | None = None


@dataclass
class RunnerConfig:
    """Consolidated configuration for StrategyRunner initialization."""

    strategy_name: str
    strategy_config: dict[str, Any]
    broker_config: Any
    global_risk_policy: dict[str, Any]
    shared_state: dict
    global_trade_count: AtomicInt
    global_position_count: AtomicInt
    position_cache_lock: ProcessLock
    trade_id_db_path: Path
    strategy_offset_seconds: float = 0.0


class StrategyRunner(RunnerPositionMixin, RunnerExecutionMixin, RunnerScheduleMixin):
    """Hook-based strategy runner with cached position reads."""

    pending_ticket_cls = PendingTicket
    exit_log_data_cls = ExitLogData
    _SUPPORTED_EXIT_ORDER_TYPES = ("market", "limit", "stop", "bracket")

    def __init__(self, config: RunnerConfig):
        self.strategy_name = config.strategy_name
        self.config = config.strategy_config
        self.broker_config = config.broker_config
        self.global_risk_policy = config.global_risk_policy
        self.shared_state = config.shared_state
        self.global_trade_count = config.global_trade_count
        self.global_position_count = config.global_position_count
        self.position_cache_lock = config.position_cache_lock
        self.trade_id_db_path = config.trade_id_db_path
        self.strategy_offset_seconds = config.strategy_offset_seconds

        self.connection: MT5Connection | None = None
        self.data_handler: DataHandler | None = None
        self.executor: OrderExecutor | None = None
        self.risk_manager: RiskManager | None = None
        self.strategy: BaseStrategy | None = None
        self.trade_id_manager: TradeIDSequenceManager | None = None

        params = self.config["params"]
        self.symbol = params["symbol"]
        self.timeframe = params["timeframe"]
        self.timeframe_mt5 = TIMEFRAME_STRING_MAP[self.timeframe]
        self.magic_number = self.config["execution"]["magic_number"]
        self.order_type = self.config["order_type"]
        self.strategy_tz = params["timezone"]

        sync_logger = TradeLogger(
            log_root=Path(self.global_risk_policy['log_root']) / "trades",
            strategy_name=self.strategy_name,
            strategy_tz=self.strategy_tz,
        )
        self.trade_logger = AsyncTradeLogger(trade_logger=sync_logger, max_queue_size=100)

        logger.debug(f"AsyncLog strat={self.strategy_name} | q=100")

        self.entry_metadata: dict[int, dict[str, Any]] = {}
        self.ticket_to_trade_id: dict[int, int] = {}
        self.pending_tickets: dict[int, PendingTicket] = {}
        self.pending_by_key: dict[tuple[str, int], list[int]] = {}
        self.pending_by_ticket: dict[int, int] = {}
        self.known_positions: set[int] = set()
        self.position_state_lock = threading.Lock()

        self.meta_model = None
        self.calibration_model = None
        self.feature_extractor = None
        self.meta_min_confidence = 0.0

        self.timeframe_minutes = TIMEFRAME_TO_MINUTES[self.timeframe_mt5]
        self.last_processed_bar_time: pd.Timestamp | None = None
        self.local_position_count: int = 0
        self.next_entry_time: datetime | None = None
        self.next_exit_time: datetime | None = None

        self._symbol_tick_cache: dict[str, Any] = {}
        self._cleanup_done = False

        logger.info(
            f"Init strat={self.strategy_name:<9} | tf={self.timeframe:<3} | "
            f"sym={self.symbol:<7} | m={self.magic_number:>3}"
        )

    def setup(self) -> bool:
        """Initialize MT5 connection, data handler, executor, risk manager, and strategy."""
        logger.debug(f"SetupStart strat={self.strategy_name} | pid={os.getpid()}")

        self.trade_id_manager = TradeIDSequenceManager(self.trade_id_db_path)

        self.connection = MT5Connection(self.broker_config)
        if not self.connection.connect():
            logger.error(f"SetupFail strat={self.strategy_name} | step=mt5_connect")
            return False

        strategy_class = self._load_strategy_class()
        self.strategy = strategy_class(params=self.config["params"])

        self.data_handler = DataHandler(self.broker_config.broker_tz)
        self.executor = OrderExecutor(self.data_handler, self.broker_config.broker_tz)
        self.risk_manager = RiskManager(
            strategy_config=self.config,
            global_policy=self.global_risk_policy,
            shared_state=self.shared_state,
            global_trade_count=self.global_trade_count,
            global_position_count=self.global_position_count,
            data_handler=self.data_handler,
            broker_tz=self.broker_config.broker_tz,
            strategy_runner=self,
        )

        self._load_meta_models()

        if "heartbeats" not in self.shared_state:
            self.shared_state["heartbeats"] = {}
        self._update_heartbeat()

        return True

    def _load_strategy_class(self) -> type:
        """Load strategy class."""
        configured_module = self.config["strategy_module"]
        strategy_class_name = self.config["strategy_class"]

        module = importlib.import_module(configured_module)
        return getattr(module, strategy_class_name)

    def _load_meta_models(self) -> None:
        """Load meta-labeling models and configuration."""
        from src.trading_system.filters.meta_labeling.meta_loader import (
            get_min_confidence,
            load_calibration_model,
            load_features_extractor,
            load_meta_model,
        )

        config = META_LABELING_CONFIG.get(self.strategy_name, {})
        if not config or not config.get("enabled", False):
            self.meta_model = None
            self.calibration_model = None
            self.feature_extractor = None
            self.meta_min_confidence = 0.0
            logger.debug(f"MetaCfg strat={self.strategy_name} | enabled=0")
            return

        self.meta_model = load_meta_model(self.strategy_name)
        self.calibration_model = load_calibration_model(self.strategy_name)
        self.feature_extractor = load_features_extractor(self.strategy_name)
        self.meta_min_confidence = get_min_confidence(self.strategy_name)

        if self.meta_model is None:
            logger.warning(f"MetaCfgWarn strat={self.strategy_name} | enabled=1 | model=missing")

    def _update_heartbeat(self) -> None:
        """Update heartbeat timestamp in shared_state."""
        self.shared_state["heartbeats"][self.strategy_name] = datetime.now().timestamp()

    def _fetch_data(self) -> pd.DataFrame | None:
        """Fetch latest bars for this strategy's symbol/timeframe."""
        return self.data_handler.get_latest_bars(strategy_name=self.strategy_name)

    def _generate_trade_id(self) -> int:
        """Generate next trade ID atomically with SQLite-backed persistence."""
        return self.trade_id_manager.generate_id()

    def run(self) -> None:
        """Main event loop with event-driven timing for low-latency entry processing."""
        try:
            if not self.setup():
                return

            self._init_known_positions()

            logger.debug(f"LoopStart strat={self.strategy_name} | entry={self.timeframe_minutes}m | exit=1m")

            self.next_entry_time, self.next_exit_time = self._initialize_schedule()

            while not self.shared_state.get("shutdown_flag", False):
                sleep_sec = self._seconds_until_next_event()
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

                data = self._fetch_data()
                if data is None:
                    logger.error(f"DataFetchFail strat={self.strategy_name} | retry=5s")
                    time.sleep(5.0)
                    continue

                self._symbol_tick_cache.clear()

                now = datetime.now()
                tolerance = timedelta(seconds=5)

                process_entry = self.next_entry_time and now >= self.next_entry_time - tolerance
                process_exit = now >= self.next_exit_time - tolerance

                self._update_heartbeat()

                if process_entry:
                    if self._should_check_entry_signal(data):
                        self._process_entry_signal(data)
                        self._process_modify_signals(data)
                    self.next_entry_time = self._calculate_next_entry_time(now)

                if process_exit:
                    if self.order_type in self._SUPPORTED_EXIT_ORDER_TYPES:
                        self._monitor_exits(data)
                    if self.order_type == "bracket":
                        self._cancel_opposite_bracket_orders()
                    self.next_exit_time = self._calculate_next_exit_time(now)

        except KeyboardInterrupt:
            logger.debug(f"RunStop strat={self.strategy_name} | reason=keyboard_interrupt")
        except Exception:
            logger.exception(f"RunCrash strat={self.strategy_name}")
            raise
        finally:
            self.cleanup()

    def _seconds_until_next_event(self) -> float:
        """Return seconds until the nearest scheduled event."""
        candidates = [timestamp for timestamp in (self.next_entry_time, self.next_exit_time) if timestamp]
        if not candidates:
            self.next_entry_time, self.next_exit_time = self._initialize_schedule()
            return 1.0
        return (min(candidates) - datetime.now()).total_seconds()

    def cleanup(self) -> None:
        """Close positions for this strategy, cancel its orders, and disconnect MT5."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        logger.debug(f"CleanupStart strat={self.strategy_name}")

        strategy_positions = self._filter_strategy_positions(mt.positions_get() or [])
        if strategy_positions:
            tickets = [pos["ticket"] for pos in strategy_positions]
            results = self.executor.close_positions(tickets=tickets)
            successful = sum(1 for success, _ in results.values() if success)

            if successful > 0:
                self.known_positions.difference_update(tickets)
                logger.info(f"CleanupPos strat={self.strategy_name} | closed={successful}/{len(tickets)}")

        strategy_orders = [
            order
            for order in (mt.orders_get() or [])
            if order.magic == self.magic_number and order.symbol == self.symbol
        ]
        for order in strategy_orders:
            self.executor._cancel_order(order.ticket)
        if strategy_orders:
            logger.info(f"CleanupOrd strat={self.strategy_name} | cancelled={len(strategy_orders)}")

        if self.trade_logger and hasattr(self.trade_logger, "shutdown"):
            logger.debug(f"CleanupTradeLog strat={self.strategy_name} | action=drain")
            self.trade_logger.shutdown(timeout=10.0)

        if self.trade_id_manager:
            self.trade_id_manager.close()

        if self.connection:
            self.connection.disconnect()

        logger.debug(f"CleanupDone strat={self.strategy_name}")


def run_strategy_process(
    strategy_name: str,
    strategy_config: dict[str, Any],
    broker_config: Any,
    global_risk_policy: dict[str, Any],
    shared_state: dict,
    global_trade_count: AtomicInt,
    global_position_count: AtomicInt,
    position_cache_lock: ProcessLock,
    trade_id_db_path: Path,
    strategy_offset_seconds: float = 0.0,
) -> None:
    """Entry point for multiprocessing.Process."""
    log_root = Path(global_risk_policy['log_root'])
    log_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_root / f"{strategy_name}.log"),
            logging.StreamHandler(),
        ],
    )

    runner_config = RunnerConfig(
        strategy_name=strategy_name,
        strategy_config=strategy_config,
        broker_config=broker_config,
        global_risk_policy=global_risk_policy,
        shared_state=shared_state,
        global_trade_count=global_trade_count,
        global_position_count=global_position_count,
        position_cache_lock=position_cache_lock,
        trade_id_db_path=trade_id_db_path,
        strategy_offset_seconds=strategy_offset_seconds,
    )

    runner = StrategyRunner(config=runner_config)

    try:
        runner.run()
    except Exception as error:
        logger.exception(f"ProcCrash strat={strategy_name} | err={error}")
