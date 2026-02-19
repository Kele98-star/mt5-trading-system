"""
TradeLogger - SQLite-based trade execution logging with atomic updates.

Schema:
    - Single row per trade (updated on exit)
    - Partial closes create new rows with incremented partial_id
    - ACID transactions prevent data corruption
    - Indexed on ticket for O(log n) lookups
"""

import sqlite3
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass
import threading
import pytz
import MetaTrader5 as mt5

from src.trading_system.core.types import (
    PartialClosePositionSnapshot,
    snapshot_partial_close_position,
)
from src.trading_system.utils.logging_utils import format_price_display

logger = logging.getLogger(__name__)


@dataclass
class FillData:
    """Trade fill parameters."""
    trade_id: int
    position: Any
    expected_entry_price: float
    opening_sl: float
    strategy_name: str
    fill_time_ms: float | None = None
    volume_multiplier: float | None = None


@dataclass(slots=True)
class CloseData:
    """Trade close parameters."""
    trade_id: int
    position: Any
    expected_exit_price: float | None
    opening_sl: float
    exit_trigger: str
    entry_price: float
    expected_entry_price: float


@dataclass(slots=True)
class PartialCloseData:
    """Partial close parameters."""
    trade_id: int
    position: PartialClosePositionSnapshot
    closed_volume: float
    remaining_volume: float
    expected_exit_price: float | None
    opening_sl: float
    strategy_name: str
    exit_trigger: str
    entry_price: float
    expected_entry_price: float
    deal_id: int | None = None


class TradeLogger:
    """SQLite-based trade logger with single-row-per-trade design."""
    
    def __init__(self, log_root: Path, strategy_name: str, strategy_tz: str):
        """Initialize SQLite logger for a specific strategy."""
        self._symbol_info_cache_ttl_seconds = 30.0
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.log_root / f"trades_{strategy_name}.db"
        self.strategy_name = strategy_name
        self.strategy_tz = pytz.timezone(strategy_tz)
        self._local = threading.local()
        self._symbol_info_cache: dict[str, tuple[float, Any]] = {}
        self._symbol_info_lock = threading.Lock()
        
        self._init_database()
        logger.debug(f"TradeLogInit db={self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0
            )
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn
    
    def _init_database(self) -> None:
        """Create trades table with composite primary key."""
        conn = self._get_connection()
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER NOT NULL,
                partial_sequence INTEGER NOT NULL DEFAULT 0,
                ticket INTEGER,
                entry_date TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_date TEXT,
                exit_time TEXT,
                magic_number INTEGER NOT NULL,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                expected_entry_price REAL NOT NULL,
                entry_spread REAL NOT NULL,
                exit_price REAL,
                expected_exit_price REAL,
                exit_spread REAL,
                opening_sl REAL,
                tp REAL,
                rrr REAL,
                commission REAL,
                swap REAL,
                gross_pnl REAL,
                net_pnl REAL,
                slippage_cost REAL,
                fill_time_mseconds REAL,
                position_type TEXT NOT NULL,
                exit_trigger TEXT,
                volume_multiplier REAL,
                PRIMARY KEY (trade_id, partial_sequence)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticket ON trades(ticket)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_date ON trades(strategy_name, entry_date)")
        conn.commit()
        
        logger.debug(f"TradeLogSchema db={self.db_path}")

    def get_open_trades_by_ticket_last_three(self, tickets: list[int]) -> dict[int, dict[str, Any]]:
        """
        Return open trade metadata for tickets using strict latest-3 row reconciliation.

        Open row rule: exit_time IS NULL.
        Scope rule: only the three most recent rows per ticket (trade_id DESC, partial_sequence DESC).
        """
        if not tickets:
            return {}

        conn = self._get_connection()
        placeholders = ",".join("?" for _ in tickets)
        query = f"""
            SELECT
                ticket,
                trade_id,
                expected_entry_price,
                opening_sl,
                volume_multiplier,
                exit_time
            FROM (
                SELECT
                    ticket,
                    trade_id,
                    expected_entry_price,
                    opening_sl,
                    volume_multiplier,
                    exit_time,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticket
                        ORDER BY trade_id DESC, partial_sequence DESC
                    ) AS recency_rank
                FROM trades
                WHERE ticket IN ({placeholders})
            )
            WHERE recency_rank <= 3
            ORDER BY ticket ASC, trade_id DESC, recency_rank ASC
        """

        try:
            rows = conn.execute(query, tickets).fetchall()
        except Exception as e:
            logger.error(f"OpenReconFail err={e}", exc_info=True)
            raise

        candidates: dict[int, list[Any]] = {}
        for row in rows:
            ticket = row[0]
            candidates.setdefault(ticket, []).append(row)

        reconciled: dict[int, dict[str, Any]] = {}
        for ticket, ticket_rows in candidates.items():
            open_rows = [row for row in ticket_rows if row[5] is None]
            if not open_rows:
                continue
            if len(open_rows) > 1:
                logger.warning(
                    f"OpenReconWarn t={ticket} | open={len(open_rows)} | "
                    f"reason=open rows in latest 3 | use_id={open_rows[0][1]}"
                )

            selected = open_rows[0]
            reconciled[ticket] = {
                'trade_id': selected[1],
                'expected_entry_price': selected[2],
                'opening_sl': selected[3],
                'volume_multiplier': selected[4],
            }

        return reconciled

    def log_fill(self, data: FillData) -> None:
        """Log position fill with unique trade_id."""
        entry_date, entry_time = self._format_datetime(datetime.now(tz=self.strategy_tz))
        
        size = data.position.volume if data.position.type == 0 else -data.position.volume
        entry_spread = data.position.price_open - data.expected_entry_price
        slippage_cost = self._calculate_slippage_cost(
            data.position.symbol, entry_spread, data.position.volume, data.position.type
        )
        fill_time_mseconds = data.fill_time_ms / 1000.0 if data.fill_time_ms is not None else None
        
        conn = self._get_connection()
        
        try:
            conn.execute("""
                INSERT INTO trades (
                    trade_id, partial_sequence, ticket,
                    entry_date, entry_time,
                    magic_number, strategy_name, symbol, size,
                    entry_price, expected_entry_price, entry_spread,
                    opening_sl, tp,
                    slippage_cost, fill_time_mseconds,
                    position_type, volume_multiplier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.trade_id, 0, data.position.ticket,
                entry_date, entry_time,
                data.position.magic, data.strategy_name, data.position.symbol, size,
                data.position.price_open, data.expected_entry_price, entry_spread,
                data.opening_sl, data.position.tp if data.position.tp != 0.0 else None,
                slippage_cost, fill_time_mseconds,
                "BUY" if data.position.type == 0 else "SELL", data.volume_multiplier
            ))
            conn.commit()
            entry_price_display = format_price_display(data.position.price_open)
            
            logger.info(
                f"FillLog id={data.trade_id} | sym={data.position.symbol} | "
                f"sz={size:+.2f} | px={entry_price_display}"
            )
        
        except sqlite3.IntegrityError as e:
            logger.error(f"FillLogDup id={data.trade_id} | err={e}")
            conn.rollback()
        except Exception as e:
            logger.error(f"FillLogFail id={data.trade_id} | err={e}", exc_info=True)
            conn.rollback()
            
    def log_close(self, data: CloseData) -> None:
        """Log position close (updates existing trade_id row)."""
        position_deals = self._get_position_deals(data.position.ticket, "Exit deal")
        if position_deals is None:
            return

        deal_data = self._get_exit_deal_data(data.position.ticket, deals=position_deals)
        if deal_data is None:
            return

        actual_exit_price = deal_data['exit_price']
        gross_pnl = deal_data['gross_pnl']

        if data.expected_exit_price is not None:
            expected_exit_price = data.expected_exit_price
        else:
            expected_exit_price = self._infer_expected_exit_price(data.position, actual_exit_price)

        raw_exit_spread = self._calculate_exit_spread(actual_exit_price, expected_exit_price)
        exit_spread = self._calculate_exit_spread(
            actual_exit_price, expected_exit_price, data.position.type
        )
        commission = self._get_commission(data.position.ticket, deals=position_deals)
        net_pnl = gross_pnl + commission
        rrr = self._calculate_rrr(data.entry_price, actual_exit_price, data.opening_sl, data.position.type)
        
        slippage_cost = self._calculate_total_slippage_cost(
            data.position.symbol,
            data.entry_price - data.expected_entry_price,
            raw_exit_spread if raw_exit_spread is not None else 0.0,
            data.position.volume,
            data.position.type
        )

        exit_date, exit_time = self._format_datetime(datetime.now(tz=self.strategy_tz))
        
        conn = self._get_connection()
        
        try:
            before_changes = conn.total_changes
            conn.execute("""
                UPDATE trades
                SET exit_date = ?, exit_time = ?, exit_price = ?, expected_exit_price = ?,
                    exit_spread = ?, rrr = ?, commission = ?, swap = ?,
                    gross_pnl = ?, net_pnl = ?, slippage_cost = ?, exit_trigger = ?
                WHERE trade_id = ? AND partial_sequence = 0 AND exit_time IS NULL
            """, (
                exit_date, exit_time, actual_exit_price, expected_exit_price, exit_spread,
                rrr, commission, data.position.swap, gross_pnl, net_pnl, slippage_cost,
                data.exit_trigger, data.trade_id
            ))

            if conn.total_changes == before_changes:
                conn.rollback()
                logger.warning(
                    f"CloseLogSkip id={data.trade_id} | t={data.position.ticket} | "
                    "reason=no_open_entry_row"
                )
                return

            conn.commit()
            rrr_display = f"{rrr:.2f}" if rrr is not None else "NA"
            slippage_display = f"{slippage_cost:.2f}" if slippage_cost is not None else "NA"
            px_display = format_price_display(actual_exit_price)
            
            logger.info(
                f"CloseLog id={data.trade_id} | px={px_display} | "
                f"pnl={net_pnl:.2f} | rrr={rrr_display} | slip={slippage_display}"
            )
        
        except Exception as e:
            logger.error(f"CloseLogFail id={data.trade_id} | err={e}", exc_info=True)
            conn.rollback()

    def log_partial_close(self, data: PartialCloseData) -> None:
        """Log partial position close (creates new row with incremented partial_sequence)."""
        exit_date, exit_time = self._format_datetime(datetime.now(tz=self.strategy_tz))
        position = snapshot_partial_close_position(data.position)
        
        conn = self._get_connection()
        next_partial = self._get_next_partial_sequence(conn, data.trade_id)
        
        deal_data = self._get_latest_partial_exit_deal(position.ticket, data.deal_id)
        if deal_data is None:
            logger.error(f"PartCloseFail id={data.trade_id} | reason=partial_exit_deal_missing")
            return
                
        actual_exit_price = deal_data['exit_price']
        partial_gross_pnl = deal_data['profit']
        
        total_volume = data.closed_volume + data.remaining_volume
        partial_ratio = data.closed_volume / total_volume
        
        commission = self._get_commission(position.ticket)
        partial_commission = commission * partial_ratio if commission else None
        net_pnl = partial_gross_pnl - (partial_commission if partial_commission else 0)
        
        rrr = self._calculate_rrr(data.entry_price, actual_exit_price, data.opening_sl, position.type)
        raw_exit_spread = self._calculate_exit_spread(actual_exit_price, data.expected_exit_price)
        exit_spread = self._calculate_exit_spread(
            actual_exit_price, data.expected_exit_price, position.type
        )
        
        slippage_cost = self._calculate_total_slippage_cost(
            position.symbol,
            data.entry_price - data.expected_entry_price,
            raw_exit_spread if raw_exit_spread is not None else 0.0,
            data.closed_volume,
            position.type
        )
        
        original = self._get_original_entry_data(conn, data.trade_id)
        if original is None:
            logger.error(
                f"PartCloseFail id={data.trade_id} | t={position.ticket} | reason=original_entry_missing"
            )
            return
        
        size = data.closed_volume if position.type == 0 else -data.closed_volume
        
        try:
            conn.execute("""
                INSERT INTO trades (
                    trade_id, partial_sequence, ticket, entry_date, entry_time, exit_date, exit_time,
                    magic_number, strategy_name, symbol, size, entry_price, expected_entry_price, entry_spread,
                    exit_price, expected_exit_price, exit_spread, opening_sl, tp, rrr,
                    commission, swap, gross_pnl, net_pnl, slippage_cost, fill_time_mseconds,
                    position_type, exit_trigger, volume_multiplier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.trade_id, next_partial, position.ticket,
                original['entry_date'], original['entry_time'], exit_date, exit_time,
                original['magic_number'], data.strategy_name, position.symbol, size,
                data.entry_price, original['expected_entry_price'], original['entry_spread'],
                actual_exit_price, data.expected_exit_price, exit_spread,
                original['opening_sl'], original['tp'], rrr,
                partial_commission, position.swap * partial_ratio,
                partial_gross_pnl, net_pnl, slippage_cost, original['fill_time_mseconds'],
                "BUY" if position.type == 0 else "SELL",
                data.exit_trigger, original['volume_multiplier']
            ))
            conn.commit()
            slippage_display = f"{slippage_cost:.2f}" if slippage_cost is not None else "NA"
            exit_price_display = format_price_display(actual_exit_price)
            
            logger.info(
                f"PartClose id={data.trade_id} | p={next_partial} | "
                f"vol={data.closed_volume:.2f} | px={exit_price_display} | "
                f"pnl={net_pnl:.2f} | slip={slippage_display}"
            )
        
        except Exception as e:
            logger.error(f"PartCloseFail id={data.trade_id} | err={e}", exc_info=True)
            conn.rollback()

    def _format_datetime(self, dt: datetime) -> tuple[str, str]:
        """Format datetime into date and time strings."""
        second_key = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        cached_key = getattr(self._local, "last_dt_second", None)
        if cached_key == second_key:
            return self._local.last_dt_formatted

        formatted = (dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"))
        self._local.last_dt_second = second_key
        self._local.last_dt_formatted = formatted
        return formatted

    def _validate_deals(self, deals, ticket: int, context: str) -> bool:
        """Validate deals exist and are non-empty."""
        if deals is None or len(deals) == 0:
            logger.error(f"DealsFail ctx={context} | t={ticket} | reason=no_deals")
            return False
        return True

    def _get_next_partial_sequence(self, conn: sqlite3.Connection, trade_id: int) -> int:
        """Get next partial_sequence number for a trade."""
        cursor = conn.execute(
            "SELECT MAX(partial_sequence) FROM trades WHERE trade_id = ?",
            (trade_id,)
        )
        max_partial = cursor.fetchone()[0]
        return (max_partial if max_partial is not None else 0) + 1

    def _get_original_entry_data(self, conn: sqlite3.Connection, trade_id: int) -> dict[str, Any] | None:
        """Fetch original entry data for a trade."""
        cursor = conn.execute("""
            SELECT entry_date, entry_time, magic_number, ticket,
                expected_entry_price, entry_spread, opening_sl, tp,
                fill_time_mseconds, volume_multiplier
            FROM trades
            WHERE trade_id = ? AND partial_sequence = 0
        """, (trade_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            'entry_date': row[0],
            'entry_time': row[1],
            'magic_number': row[2],
            'ticket': row[3],
            'expected_entry_price': row[4],
            'entry_spread': row[5],
            'opening_sl': row[6],
            'tp': row[7],
            'fill_time_mseconds': row[8],
            'volume_multiplier': row[9]
        }

    @staticmethod
    def _calculate_exit_spread(
        actual_exit_price: float,
        expected_exit_price: float | None,
        position_type: int | None = None,
    ) -> float | None:
        """Calculate exit spread as expected minus actual."""
        if expected_exit_price is None:
            return None
        raw_spread = expected_exit_price - actual_exit_price
        if position_type is None:
            return raw_spread
        direction_multiplier = 1 if position_type == 0 else -1
        return direction_multiplier * raw_spread

    @staticmethod
    def _normalize_protective_level(level: float | None) -> float | None:
        """Normalize protective level by treating sentinel zeros as missing."""
        if level is None:
            return None
        return None if abs(level) <= 1e-12 else level

    def _infer_expected_exit_price(self, position: Any, actual_exit_price: float) -> float | None:
        """Infer expected exit price from position SL/TP if not provided."""
        normalized_sl = self._normalize_protective_level(getattr(position, 'sl', None))
        normalized_tp = self._normalize_protective_level(getattr(position, 'tp', None))

        if normalized_sl is not None:
            sl_distance = abs(position.price_open - normalized_sl)
            tolerance = max(sl_distance / 10, 1e-5)
        else:
            tolerance = 1e-5

        if position.type == 0:  # BUY
            tp_hit = normalized_tp is not None and actual_exit_price >= (normalized_tp - tolerance)
            sl_hit = normalized_sl is not None and actual_exit_price <= (normalized_sl + tolerance)

            if tp_hit:
                return normalized_tp
            if sl_hit:
                return normalized_sl
        else:  # SELL
            tp_hit = normalized_tp is not None and actual_exit_price <= (normalized_tp + tolerance)
            sl_hit = normalized_sl is not None and actual_exit_price >= (normalized_sl - tolerance)

            if tp_hit:
                return normalized_tp
            if sl_hit:
                return normalized_sl

        return None

    def _get_position_deals(self, ticket: int, context: str, max_retries: int = 3) -> list[Any] | None:
        """Retrieve all history deals for a ticket with bounded retry backoff."""
        for attempt in range(max_retries):
            deals = mt5.history_deals_get(position=ticket)
            if deals and len(deals) > 0:
                return list(deals)
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
        logger.error(f"DealsFetchFail ctx={context} | t={ticket} | reason=no_deals")
        return None

    def _get_exit_deals(self, ticket: int, context: str, max_retries: int = 3) -> list[Any] | None:
        """Retrieve exit deals (entry==1) for a position ticket."""
        deals = self._get_position_deals(ticket, context, max_retries=max_retries)
        if deals is None:
            return None
        exit_deals = [deal for deal in deals if getattr(deal, 'entry', None) == 1]
        if not exit_deals:
            logger.error(f"ExitDealsFail ctx={context} | t={ticket} | reason=no_exit_deals")
            return None
        return exit_deals

    def _get_latest_partial_exit_deal(self, ticket: int, deal_id: int | None = None) -> dict[str, Any] | None:
        """Retrieve partial exit deal by deal_id (required)."""
        if deal_id is None:
            logger.error(f"PartExitFail t={ticket} | reason=Missing deal_id")
            return None
        deal = mt5.history_deals_get(ticket=deal_id)
        if deal and len(deal) > 0:
            return {'exit_price': deal[0].price, 'profit': deal[0].profit}
        logger.error(f"PartExitFail deal={deal_id} | t={ticket} | reason=deal_not_found")
        return None

    def _get_symbol_info_cached(self, symbol: str) -> Any | None:
        """Fetch MT5 symbol metadata with short TTL cache."""
        now = time.monotonic()
        with self._symbol_info_lock:
            cached = self._symbol_info_cache.get(symbol)
            if cached is not None:
                cached_ts, cached_info = cached
                if now - cached_ts <= self._symbol_info_cache_ttl_seconds:
                    return cached_info

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"SlipCalcFail sym={symbol} | reason=symbol_info_unavailable")
            return None

        with self._symbol_info_lock:
            self._symbol_info_cache[symbol] = (now, symbol_info)
        return symbol_info

    def _resolve_slippage_params(
        self,
        symbol: str,
        symbol_info: Any,
        position_type: int,
    ) -> tuple[float, float, int] | None:
        """Resolve tick parameters for directional slippage calculation."""
        tick_size = getattr(symbol_info, 'trade_tick_size', None)
        if tick_size is None or tick_size <= 0:
            logger.error(f"SlipCalcFail sym={symbol} | reason=invalid_tick_size | tick_size={tick_size}")
            return None

        tick_value_attr = 'trade_tick_value_loss' if position_type == 0 else 'trade_tick_value_profit'
        tick_value = getattr(symbol_info, tick_value_attr, None)
        if tick_value is None:
            tick_value = getattr(symbol_info, 'trade_tick_value', None)
        if tick_value is None:
            logger.error(f"SlipCalcFail sym={symbol} | reason=tick_value_missing")
            return None

        direction_multiplier = 1 if position_type == 0 else -1
        return tick_size, tick_value, direction_multiplier

    def _calculate_slippage_cost(
        self,
        symbol: str,
        spread: float,
        volume: float,
        position_type: int,
    ) -> float | None:
        """
        Calculate directional slippage cost in account currency.
        
        Formula: spread × volume × (tick_value / tick_size) × direction
        """
        symbol_info = self._get_symbol_info_cached(symbol)
        if symbol_info is None:
            return None
        params = self._resolve_slippage_params(symbol, symbol_info, position_type)
        if params is None:
            return None

        tick_size, tick_value, direction_multiplier = params
        return direction_multiplier * spread * volume * (tick_value / tick_size)

    def _calculate_total_slippage_cost(
        self,
        symbol: str,
        entry_spread: float,
        exit_spread: float,
        volume: float,
        position_type: int,
    ) -> float | None:
        """Calculate total slippage (entry + exit) with asymmetric tick values."""
        symbol_info = self._get_symbol_info_cached(symbol)
        if symbol_info is None:
            return None
        params = self._resolve_slippage_params(symbol, symbol_info, position_type)
        if params is None:
            return None

        tick_size, tick_value, direction_multiplier = params
        total_spread = entry_spread + exit_spread
        return direction_multiplier * total_spread * volume * (tick_value / tick_size)

    def _get_commission(self, ticket: int, deals: list[Any] | None = None) -> float:
        """Query total commission from history deals."""
        if deals is None:
            deals = self._get_position_deals(ticket, "Commission query")
        
        if not self._validate_deals(deals, ticket, "Commission query"):
            return 0.0
        
        return sum(getattr(deal, 'commission', 0.0) for deal in deals)
        
    def _get_exit_deal_data(
        self,
        ticket: int,
        deals: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve exit price and gross P&L from history deals."""
        position_deals = deals if deals is not None else self._get_position_deals(ticket, "Exit deal")
        if not self._validate_deals(position_deals, ticket, "Exit deal"):
            return None

        exit_deals = [deal for deal in position_deals if getattr(deal, 'entry', None) == 1]
        if not exit_deals:
            logger.error(f"ExitDealFail t={ticket} | reason=no_exit_deals")
            return None

        gross_pnl = sum(getattr(deal, 'profit', 0.0) for deal in position_deals)
        return {'exit_price': exit_deals[-1].price, 'gross_pnl': gross_pnl}

    def _calculate_rrr(
        self,
        entry_price: float,
        exit_price: float,
        opening_sl: float,
        position_type: int,
    ) -> float | None:
        """Calculate realized risk-reward ratio."""
        try:
            if position_type == 0:  # BUY
                risk = entry_price - opening_sl
                reward = exit_price - entry_price
            else:  # SELL
                risk = opening_sl - entry_price
                reward = entry_price - exit_price
            
            return 0.0 if risk == 0 else reward / risk
        
        except Exception as e:
            logger.error(f"RRRFail err={e}", exc_info=True)
            return None
    
    def export_to_csv(self, output_path: Path | None = None) -> Path:
        """Export database to CSV format for backward compatibility."""
        if output_path is None:
            output_path = self.log_root / f"trades_{self.strategy_name}.csv"
        
        conn = self._get_connection()
        
        try:
            import pandas as pd
            
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_date, entry_time", conn)
            df.to_csv(output_path, index=False)
            
            logger.info(f"CsvExport rows={len(df)} | path={output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"CsvExportFail err={e}", exc_info=True)
            raise
    
    def close(self):
        """Close database connection (call on shutdown)."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            logger.info(f"TradeLogClose db={self.db_path}")
