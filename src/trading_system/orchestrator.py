"""
Orchestrator - Multi-process strategy coordination with shared position cache.
"""
from multiprocessing import Process, Manager, Value, Lock
import ctypes
import os
from typing import Any
from dataclasses import dataclass
import logging
from pathlib import Path
import importlib
import time
from datetime import datetime, date
import MetaTrader5 as mt

from src.trading_system.strategy_runner import run_strategy_process
from src.trading_system.config.broker_config import MT5Config
from src.trading_system.config.risk_policies import (
    ACCOUNT_GLOBAL_POLICIES, ACCOUNT_RISK_POLICIES,
    ACCOUNT_STRATEGY_CONFIGS, SYSTEM_TIMINGS
)
from src.trading_system.core.execution import OrderExecutor
from src.trading_system.core.data_handler import DataHandler
from src.trading_system.core.types import AtomicInt, ProcessLock, normalize_position
from src.trading_system.core.trade_id_manager import TradeIDSequenceManager
from src.trading_system.core.connection import MT5Connection
from src.trading_system.filters.news import preprocess_calendar_file
from src.trading_system.utils.logging_utils import log_section_header, format_price_display

logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SECONDS = SYSTEM_TIMINGS['heartbeat_interval']
HEARTBEAT_LOG_INTERVAL_SECONDS = SYSTEM_TIMINGS['heartbeat_log_interval']


@dataclass
class HeartbeatState:
    """Tracks heartbeat monitoring state."""
    last_reset_date: date
    last_log_time: float


@dataclass
class SystemMetrics:
    """Current system state metrics."""
    active_processes: int
    cache_positions: int
    total_trades: int
    max_positions: int
    max_trades: int


class Orchestrator:
    """Multi-strategy orchestrator with shared position cache."""

    def __init__(self, broker_config: MT5Config, account_type: str, log_root: Path):
        """Initialize orchestrator with account-specific configuration."""
        self.log_root = Path(log_root)
        self.broker_config = broker_config
        self.account_type = account_type

        self._validate_account_type(account_type)
        
        self.account_enabled_strategies = ACCOUNT_STRATEGY_CONFIGS[account_type]
        
        self.global_risk_policy = self._build_risk_policy(account_type)
        
        self.manager: Any = Manager()
        
        self.global_position_count: AtomicInt = Value(ctypes.c_int, 0)
        self.global_trade_count: AtomicInt = Value(ctypes.c_int, 0)
        self.position_cache_lock: ProcessLock = Lock()
        
        self.shared_state: Any = self._initialize_shared_state()
        
        self.strategy_processes: dict[str, Process] = {}
        self.strategy_configs: dict[str, dict[str, Any]] = {}
        self.process_crash_count: dict[str, int] = {}

        self.trade_id_db_path = self.log_root / "trade_id_sequence.db"
        self.trade_id_manager = TradeIDSequenceManager(self.trade_id_db_path)

        self.mt5_connection: MT5Connection | None = None
        self._shutdown_initiated = False

        logger.info(f"OrchInit acct={account_type} | pos_max={self.global_risk_policy['max_total_positions']} | tr_max={self.global_risk_policy['max_daily_trades']}")

    def _validate_account_type(self, account_type: str) -> None:
        """Validate account type exists in configuration."""
        if account_type not in ACCOUNT_STRATEGY_CONFIGS:
            raise ValueError(f"Unknown account type '{account_type}'. Available: {list(ACCOUNT_STRATEGY_CONFIGS.keys())}")

    def _build_risk_policy(self, account_type: str) -> dict[str, Any]:
        """Build complete risk policy with strategy allocations."""
        policy = ACCOUNT_GLOBAL_POLICIES[account_type].copy()
        policy['strategy_risk'] = ACCOUNT_RISK_POLICIES[account_type]
        policy['log_root'] = str(self.log_root)
        return policy

    def _initialize_shared_state(self) -> Any:
        """Initialize shared state dictionary with all required fields."""
        state = self.manager.dict()
        state['shutdown_flag'] = False
        state['position_cache'] = self.manager.dict()
        state['position_cache_timestamp'] = 0.0
        state['heartbeats'] = self.manager.dict()
        state['calendar_cache'] = None
        state['calendar_cache_timestamp'] = 0.0
        state['daily_trade_counts'] = self.manager.dict()
        state['daily_equity_high'] = 0.0
        state['daily_drawdown'] = 0.0
        state['last_equity_update'] = 0.0
        return state

    def _get_magic_numbers(self) -> set[int]:
        """Extract magic numbers from all configured strategies."""
        return {config.get('execution', {}).get('magic_number') for config in self.strategy_configs.values() if config.get('execution', {}).get('magic_number') is not None}

    def _get_managed_positions(self) -> list:
        """Query MT5 for positions managed by this system."""
        all_positions = mt.positions_get()
        if all_positions is None:
            logger.error(f"MT5Err op=positions_get | err={mt.last_error()}")
            return []

        magic_numbers = self._get_magic_numbers()
        if not magic_numbers:
            return []

        return [pos for pos in all_positions if pos.magic in magic_numbers]

    def _get_managed_orders(self) -> list:
        """Query MT5 for pending orders managed by this system."""
        all_orders = mt.orders_get()
        if all_orders is None:
            logger.error(f"MT5Err op=orders_get | err={mt.last_error()}")
            return []

        magic_numbers = self._get_magic_numbers()
        if not magic_numbers:
            return []

        return [order for order in all_orders if order.magic in magic_numbers]

    def _build_position_cache_dict(self, positions: list) -> dict[int, dict[str, Any]]:
        """Build normalized position cache dictionary."""
        return {pos.ticket: normalize_position(pos) for pos in positions}

    def _update_shared_cache(self, cache_dict: dict[int, dict[str, Any]]) -> None:
        """Atomically update shared position cache."""
        with self.position_cache_lock:
            shared_cache = self.shared_state.get('position_cache')
            if shared_cache is None:
                shared_cache = self.manager.dict()
                self.shared_state['position_cache'] = shared_cache

            stale_keys = [ticket for ticket in list(shared_cache.keys()) if ticket not in cache_dict]
            for ticket in stale_keys:
                shared_cache.pop(ticket, None)

            shared_cache.update(cache_dict)
            self.shared_state['position_cache_timestamp'] = time.time()

    def _log_position_details(self, positions: list) -> None:
        """Log details of existing positions for audit trail."""
        for pos in positions:
            position_type = 'BUY' if pos.type == mt.POSITION_TYPE_BUY else 'SELL'
            price_display = format_price_display(pos.price_open)
            logger.info(f"sym={pos.symbol:<7} | side={'B' if position_type == 'BUY' else 'S'} | vol={pos.volume:>4.2f} | px={price_display:>10} | m={pos.magic:>3}")
            logger.debug(f"t={pos.ticket:>10}")

    def sync_existing_positions(self):
        """Sync global_position_count and cache with actual MT5 positions on startup."""
        managed_positions = self._get_managed_positions()

        count = len(managed_positions)
        if count == 0:
            logger.debug("PosSync n=0")
            return

        cache_dict = self._build_position_cache_dict(managed_positions)
        self._update_shared_cache(cache_dict)

        with self.global_position_count.get_lock():
            self.global_position_count.value = count

        logger.info(f"PosSync n={count}")
        self._log_position_details(managed_positions)

    def preload_calendar_cache(self):
        """Preload and parse economic calendar CSV before spawning strategies."""
        calendar_path = Path(os.getenv('MT5_CALENDAR_PATH'))
        
        if not calendar_path.exists():
            logger.warning(f"CalPreloadSkip path={calendar_path} | reason=not_found")
            return

        df, holidays_frozen = preprocess_calendar_file(calendar_path, self.broker_config.broker_tz)
        
        self.shared_state['calendar_cache'] = df.to_dict('records')
        self.shared_state['calendar_holidays'] = list(holidays_frozen)
        self.shared_state['calendar_cache_timestamp'] = time.time()

        event_count = len(df)
        high_impact_count = df['priority'].eq('High').sum()
        holiday_count = len(holidays_frozen)
        
        logger.info(f"CalPreload evt={event_count} | hi={high_impact_count} | hol={holiday_count}")
        
    def _handle_strategy_import_error(
        self, 
        potential_name: str, 
        error: Exception,
        config_module_path: str
    ) -> None:
        """Handle strategy import errors based on enable status."""
        is_enabled = self.account_enabled_strategies.get(potential_name, False)
        
        if is_enabled:
            logger.critical(f"StratLoadFail name={potential_name} | enabled=1 | mod={config_module_path}")
            logger.critical(f"StratLoadErr name={potential_name} | err={error}", exc_info=True)
            raise RuntimeError(f"Cannot start with broken enabled strategy: {potential_name}")
        else:
            logger.warning(f"StratLoadSkip name={potential_name} | enabled=0 | err={error}")

    def discover_strategies(self):
        """Load enabled strategies from account allowlist using strict invariants."""
        enabled_strategy_names = sorted(name for name, is_enabled in self.account_enabled_strategies.items() if is_enabled)

        for strategy_name in enabled_strategy_names:
            config_module_path = f"strategies.{strategy_name}.config"
            config_module = importlib.import_module(config_module_path)
            config = config_module.get_config()

            self.strategy_configs[strategy_name] = config
            params = config.get("params", {})
            logger.debug(f"StratDisc name={strategy_name} | sym={params.get('symbol')} | tf={params.get('timeframe')}")
            
    def reset_daily_counters(self):
        """Reset trade counters at midnight."""
        with self.global_trade_count.get_lock():
            old_count = self.global_trade_count.value
            self.global_trade_count.value = 0
        logger.info(f"DailyReset tr={old_count}->0")

        self.shared_state['daily_trade_counts'] = self.manager.dict()
        self.shared_state['daily_equity_high'] = 0.0
        self.shared_state['daily_drawdown'] = 0.0
        self.shared_state['last_equity_update'] = 0.0
        logger.info("DailyReset eq_hi=0 | dd=0 | counts=0")

    def spawn_strategy_process(
        self,
        strategy_name: str,
        config: dict[str, Any],
        strategy_index: int = 0
    ):
        """Spawn isolated process with connection verification."""
        strategy_offset_seconds = (strategy_index % 6) / 40
        
        process = Process(
            target=run_strategy_process,
            args=(
                strategy_name,
                config,
                self.broker_config,
                self.global_risk_policy,
                self.shared_state,
                self.global_trade_count,
                self.global_position_count,
                self.position_cache_lock,
                self.trade_id_db_path,
                strategy_offset_seconds,
            )
        )
        process.start()
        
        self.strategy_processes[strategy_name] = process
        logger.debug(f"ProcSpawn name={strategy_name} | pid={process.pid}")
        return True

    def sync_to_next_heartbeat(self):
        """Sleep until next heartbeat boundary with precise alignment."""
        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = (now - midnight).total_seconds()
        
        current_multiple = int(seconds_since_midnight // HEARTBEAT_INTERVAL_SECONDS)
        target_seconds = current_multiple * HEARTBEAT_INTERVAL_SECONDS
        
        if seconds_since_midnight > target_seconds:
            target_seconds += HEARTBEAT_INTERVAL_SECONDS
        
        sleep_duration = target_seconds - seconds_since_midnight
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    def _reconcile_position_counter(self, actual_count: int) -> None:
        """Log drift between global position counter and MT5 reality (strategies own the counter)."""
        with self.global_position_count.get_lock():
            current_count = self.global_position_count.value
        if current_count != actual_count:
            drift = current_count - actual_count
            logger.warning(
                f"PosDrift cnt={current_count} | mt5={actual_count} | d={drift:+d} | action=log_only"
            )

    def refresh_position_cache(self):
        """Refresh shared position cache from MT5."""
        try:
            managed_positions = self._get_managed_positions()
            new_cache = self._build_position_cache_dict(managed_positions)
            self._update_shared_cache(new_cache)

            actual_count = len(managed_positions)
            self._reconcile_position_counter(actual_count)

            logger.debug(f"PosCacheRefresh n={len(new_cache)} | cnt={actual_count} | age=0s")
        except Exception as e:
            logger.error(f"PosCacheRefreshFail err={e}", exc_info=True)

    def _should_log_heartbeat(self, current_time: datetime, last_log_time: float) -> bool:
        """Check if heartbeat should be logged according to configured interval."""
        interval_seconds = max(1, int(HEARTBEAT_LOG_INTERVAL_SECONDS))
        boundary_tolerance = max(1, int(HEARTBEAT_INTERVAL_SECONDS))
        time_since_last_log = time.time() - last_log_time
        if time_since_last_log + boundary_tolerance < interval_seconds:
            return False

        midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = int((current_time - midnight).total_seconds())
        return seconds_since_midnight % interval_seconds < boundary_tolerance

    def _get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        active_processes = sum(1 for p in self.strategy_processes.values() if p.is_alive())
        cache_positions = self.global_position_count.value
        total_trades = self.global_trade_count.value
        
        return SystemMetrics(
            active_processes=active_processes,
            cache_positions=cache_positions,
            total_trades=total_trades,
            max_positions=self.global_risk_policy['max_total_positions'],
            max_trades=self.global_risk_policy['max_daily_trades']
        )

    def _log_heartbeat_status(self, current_time: datetime, metrics: SystemMetrics) -> None:
        """Log system status at heartbeat intervals."""
        logger.info(
            f"HB t={current_time.strftime('%H:%M:%S')} | "
            f"proc={metrics.active_processes}/{len(self.strategy_processes)} | "
            f"pos={metrics.cache_positions}/{metrics.max_positions} | "
            f"tr={metrics.total_trades}/{metrics.max_trades}"
        )

    def _check_global_limits(self, metrics: SystemMetrics) -> None:
        """Check and warn if global limits are reached."""
        if metrics.cache_positions >= metrics.max_positions:
            logger.warning(f"LimitHit typ=pos | cur={metrics.cache_positions} | max={metrics.max_positions}")
        if metrics.total_trades >= metrics.max_trades:
            logger.warning(f"LimitHit typ=tr | cur={metrics.total_trades} | max={metrics.max_trades}")

    def _get_dead_processes(self) -> list[tuple[str, Process]]:
        """Identify crashed strategy processes."""
        return [(name, proc) for name, proc in self.strategy_processes.items() if not proc.is_alive()]

    def _handle_crashed_strategies(self, dead_processes: list[tuple[str, Process]]) -> None:
        """Log crashed strategies and update crash counters."""
        for name, proc in dead_processes:
            self.process_crash_count[name] = self.process_crash_count.get(name, 0) + 1
            logger.critical(f"Crash strat={name} | pid={proc.pid if proc.pid else 'N/A'} | code={proc.exitcode}")
        
        logger.critical(f"CrashShutdown n={len(dead_processes)}")

    def monitor_heartbeats(self):
        """Monitor strategy heartbeats and system limits with daily counter reset."""
        heartbeat_state = HeartbeatState(
            last_reset_date=datetime.now().date(),
            last_log_time=time.time()
        )

        while not self.shared_state['shutdown_flag']:
            self.refresh_position_cache()

            current_time = datetime.now()
            metrics = self._get_system_metrics()

            if self._should_log_heartbeat(current_time, heartbeat_state.last_log_time):
                self._log_heartbeat_status(current_time, metrics)
                heartbeat_state.last_log_time = time.time()

            self._check_global_limits(metrics)

            dead_processes = self._get_dead_processes()
            if dead_processes:
                self._handle_crashed_strategies(dead_processes)
                self.shutdown()
                break

            current_date = current_time.date()
            if current_date > heartbeat_state.last_reset_date:
                self.reset_daily_counters()
                heartbeat_state.last_reset_date = current_date

            self.sync_to_next_heartbeat()

    def _log_enabled_strategies(self) -> None:
        """Log configuration of all enabled strategies."""
        log_section_header(logger, f"ENABLED STRATEGIES (Account: {self.account_type})", level=logging.DEBUG)
        for name, config in self.strategy_configs.items():
            risk_pct = self.global_risk_policy['strategy_risk'][name] * 100
            params = config.get('params', {})
            logger.debug(f"StratCfg name={name} | sym={params['symbol']} | tf={params['timeframe']} | risk={risk_pct:.2f}%")

    def _connect_to_mt5(self) -> None:
        """Establish persistent MT5 connection."""
        # log_section_header(logger, "CONNECTING TO MT5")
        
        self.mt5_connection = MT5Connection(self.broker_config)
        if not self.mt5_connection.connect():
            raise RuntimeError("Orchestrator startup failed: Unable to connect to MT5.")
        logger.debug("OrchMT5 conn=ok")

    def _sync_positions(self) -> None:
        """Sync existing positions on startup."""
        log_section_header(logger, "SYNCING EXISTING POSITIONS", level=logging.DEBUG)
        self.sync_existing_positions()
        time.sleep(3)

    def _spawn_all_strategies(self) -> None:
        """Spawn all configured strategy processes."""
        # log_section_header(logger, "SPAWNING STRATEGY PROCESSES")
        
        for i, (name, config) in enumerate(self.strategy_configs.items()):
            self.spawn_strategy_process(name, config, strategy_index=i)
            if i < len(self.strategy_configs) - 1:
                time.sleep(0.1)

    def start(self):
        """Start orchestrator with sequential initialization phases."""
        self.discover_strategies()
        self._log_enabled_strategies()
        self._connect_to_mt5()
        self._sync_positions()
        self.preload_calendar_cache()
        self._spawn_all_strategies()

        log_section_header(logger,f"HEARTBEAT MONITOR STARTING (refresh: {HEARTBEAT_INTERVAL_SECONDS}s)", level=logging.DEBUG)
        self.monitor_heartbeats()

    def _wait_for_graceful_exits(self) -> None:
        """Wait for strategy processes to exit gracefully."""
        for name, process in self.strategy_processes.items():
            if not process.is_alive():
                logger.debug(f"ProcState name={name} | state=already_exited")
                continue
            
            process.join(timeout=3)
            if process.is_alive():
                logger.debug(f"ProcState name={name} | state=join_timeout | action=terminate")

    def _verify_and_close_remaining(self) -> None:
        """Verify managed positions/orders closed and force-close if needed."""
        try:
            managed_positions = self._get_managed_positions()
            managed_orders = self._get_managed_orders()
            managed_pos_count = len(managed_positions)
            managed_ord_count = len(managed_orders)

            if managed_pos_count > 0 or managed_ord_count > 0:
                logger.warning(f"ShutdownVerifyOpen pos={managed_pos_count} | ord={managed_ord_count}")
                self._force_close_all_immediate(managed_positions, managed_orders)
            else:
                logger.debug("ShutdownVerify ok=1")
        except Exception as e:
            logger.error(f"ShutdownVerifyFail err={e}", exc_info=True)

    def _terminate_unresponsive_processes(self) -> None:
        """Terminate or kill processes that didn't exit gracefully."""
        for name, process in self.strategy_processes.items():
            if not process.is_alive():
                logger.debug(f"ProcState name={name} | state=already_exited | code={process.exitcode}")
                continue
            
            process.terminate()
            process.join(timeout=5)
            
            if process.is_alive():
                logger.error(f"ProcKill name={name}")
                process.kill()
            
            logger.debug(f"ProcState name={name} | state=exited | code={process.exitcode}")

    def shutdown(self):
        """Graceful shutdown with guaranteed cleanup of managed positions/orders."""
        if self._shutdown_initiated:
            logger.debug("ShutdownSkip reason=already_in_progress")
            return
        
        self._shutdown_initiated = True

        log_section_header(logger, "ORCHESTRATOR SHUTTING DOWN", level=logging.DEBUG)

        logger.debug("ShutdownPhase n=1 | step=signal_strategies")
        self.shared_state['shutdown_flag'] = True
        time.sleep(2)

        logger.debug("ShutdownPhase n=2 | step=wait_process_exit")
        self._wait_for_graceful_exits()

        logger.debug("ShutdownPhase n=3 | step=verify_managed_items")
        self._verify_and_close_remaining()

        logger.debug("ShutdownPhase n=4 | step=terminate_unresponsive")
        self._terminate_unresponsive_processes()

        logger.debug("ShutdownPhase n=5 | step=close_mt5")
        if self.mt5_connection:
            self.mt5_connection.disconnect()
        logger.debug("ShutdownMT5 conn=closed")

        logger.debug("ShutdownPhase n=6 | step=reset_trade_counter")
        with self.global_trade_count.get_lock():
            self.global_trade_count.value = 0

        log_section_header(logger, "SHUTDOWN COMPLETE", level=logging.DEBUG)

    def _force_close_all_immediate(
        self,
        managed_positions: list | None = None,
        managed_orders: list | None = None,
    ) -> None:
        """Force-close managed positions and cancel managed pending orders."""
        if managed_positions is None:
            managed_positions = self._get_managed_positions()
        if managed_orders is None:
            managed_orders = self._get_managed_orders()

        if not managed_positions and not managed_orders:
            logger.debug("ForceCloseSkip reason=no_managed_items")
            return

        executor = OrderExecutor(
            DataHandler(self.broker_config.broker_tz),
            self.broker_config.broker_tz
        )

        self._force_close_positions(executor, managed_positions)
        self._force_cancel_orders(executor, managed_orders)

    def _force_close_positions(
        self,
        executor: OrderExecutor,
        managed_positions: list
    ) -> None:
        """Force-close managed positions."""
        if not managed_positions:
            return
        tickets = [p.ticket for p in managed_positions]
        results = executor.close_positions(tickets=tickets)
        closed_count = sum(success for success, _ in results.values())
        logger.info(f"ForceClosePos ok={closed_count}/{len(tickets)}")

    def _force_cancel_orders(
        self,
        executor: OrderExecutor,
        managed_orders: list
    ) -> None:
        """Cancel managed pending orders."""
        if not managed_orders:
            return
        cancelled_count = sum(1 for o in managed_orders if executor._cancel_order(o.ticket))
        logger.info(f"ForceCancelOrd ok={cancelled_count}/{len(managed_orders)}")
