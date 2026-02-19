from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
import logging
import time
import numpy as np
import pytz
import MetaTrader5 as mt
from src.trading_system.core.types import AtomicInt
from src.trading_system.filters.news import NewsFilter
from src.trading_system.filters.market_costs import MarketCostCalculator


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    can_trade: bool
    reason: str
    volume: float = 0.0


@dataclass
class DrawdownCache:
    """Holds incremental drawdown calculation state."""
    peak_equity: float
    current_equity: float
    last_deal_time: datetime
    initialized: bool = True
    cache_date: date | None = None
    last_cursor: tuple[int, int] = field(default_factory=lambda: (0, 0))


@dataclass
class TTLCache:
    """Generic TTL-based cache entry."""
    value: Any = None
    timestamp: float = 0.0
    
    def is_valid(self, ttl: float) -> bool:
        return self.value is not None and (time.time() - self.timestamp) < ttl


class RiskManager:
    """
    Centralized risk management with NewsFilter integration.
    
    Risk Layers (ordered by cost):
    0. Global limits (atomic counters - <0.001ms)
    1. Strategy limits (local counter - <0.001ms)
    2. NewsFilter (cached datetime comparisons - <0.1ms)
    3. Market conditions (MT5 tick fetch - 1-5ms)
    4. Position sizing (MT5 symbol_info - 1-5ms)
    """
    
    DEFAULT_ACCOUNT_CACHE_TTL = 900  # 15min
    DEFAULT_SYMBOL_CACHE_TTL = 14400  # 4 hour
    DEFAULT_EQUITY_CACHE_TTL = 1.0  # 1 second

    def __init__(
        self,
        strategy_config: dict[str, Any],
        global_policy: dict[str, Any],
        shared_state: dict,
        global_trade_count: AtomicInt,
        global_position_count: AtomicInt,
        data_handler,
        broker_tz: pytz.tzinfo.BaseTzInfo,
        strategy_runner=None,
        account_cache_ttl: int | None = None,
        symbol_cache_ttl: int | None = None
    ):
        self.ACCOUNT_CACHE_TTL = account_cache_ttl or self.DEFAULT_ACCOUNT_CACHE_TTL
        self.SYMBOL_CACHE_TTL = symbol_cache_ttl or self.DEFAULT_SYMBOL_CACHE_TTL
        self.EQUITY_CACHE_TTL = self.DEFAULT_EQUITY_CACHE_TTL
        self.strategy_config = strategy_config
        self.risk_config = strategy_config['risk']
        self.global_policy = global_policy
        self.shared_state = shared_state
        self.global_trade_count = global_trade_count
        self.global_position_count = global_position_count
        self.strategy_name = strategy_config['name']
        self.data_handler = data_handler
        self.broker_tz = broker_tz
        self.strategy_runner = strategy_runner
        
        self.news_filter = self._create_news_filter()
        self.cost_calculator = MarketCostCalculator(
            max_spread_points=self.risk_config['max_spread_points'],
            max_slippage_points=self.risk_config['max_slippage_points']
        )
        
        initial_balance = global_policy['initial_balance']
        history_start = datetime(2025, 1, 1)
        today_midnight = self._midnight_today()
        self._max_dd_cache = DrawdownCache(
            peak_equity=initial_balance,
            current_equity=initial_balance,
            last_deal_time=history_start,
            initialized=False,
            last_cursor=self._datetime_to_cursor(history_start)
        )
        self._daily_dd_cache = DrawdownCache(
            peak_equity=initial_balance,
            current_equity=initial_balance,
            last_deal_time=today_midnight,
            cache_date=self._broker_today(),
            initialized=False,
            last_cursor=self._datetime_to_cursor(today_midnight)
        )
        
        self._account_cache = TTLCache()
        self._equity_cache = TTLCache()
        self._symbol_cache: dict[str, TTLCache] = {}

        logger.debug(f"RiskInit strat={self.strategy_name} | spr_max={self.cost_calculator.max_spread_points:.1f} | slip_max={self.cost_calculator.max_slippage_points:.1f} | tz={self.broker_tz}")

    def _midnight_today(self) -> datetime:
        """Return midnight in broker timezone (timezone-aware)."""
        now_broker = datetime.now(self.broker_tz)
        return now_broker.replace(hour=0, minute=0, second=0, microsecond=0)

    def _broker_today(self) -> date:
        return datetime.now(self.broker_tz).date()

    def _datetime_to_cursor(self, dt: datetime) -> tuple[int, int]:
        return int(dt.timestamp() * 1000), 0

    def _get_deal_cursor(self, deal) -> tuple[int, int]:
        deal_time_msc = int(getattr(deal, 'time_msc', 0) or 0)
        if deal_time_msc <= 0:
            deal_time_msc = int(getattr(deal, 'time', 0) or 0) * 1000
        deal_ticket = int(getattr(deal, 'ticket', 0) or 0)
        return deal_time_msc, deal_ticket

    def _create_news_filter(self) -> NewsFilter | None:
        news_config = self.strategy_config.get('filters', {}).get('news_filter', {})
        if not news_config.get('enabled', False):
            logger.info(f"NewsCfg strat={self.strategy_name} | enabled=0")
            return None

        nf = NewsFilter(
            data_handler=self.data_handler,
            strategy_name=self.strategy_name,
            shared_state=self.shared_state
        )
        logger.debug(f"NewsInit strat={self.strategy_name} | cur={nf.filter_currencies or 'ALL'} | buf_min={nf.buffer_minutes}")
        return nf

    def validate_trade(
        self,
        strategy_name: str,
        symbol: str,
        expected_price: float,
        sl_price: float,
        signal: int,
        skip_position_limit_check: bool = False
    ) -> ValidationResult:
        """Validate trade with layered risk checks (ordered by computational cost)."""
        checks = [
            self.check_global_risk,
            lambda: self.check_strategy_limits(strategy_name, skip_position_limit_check),
            lambda: self._check_news_filter(strategy_name),
            lambda: self._check_market_conditions(symbol, expected_price, signal),
        ]
        
        for check in checks:
            result = check()
            if not result.can_trade:
                return result

        volume = self.calculate_position_size(symbol, expected_price, sl_price, strategy_name)
        if volume <= 0:
            return ValidationResult(False, "Position size invalid")

        reserved_count = self._reserve_position_slot()
        logger.debug(f"PosSlotRes strat={self.strategy_name} | cnt={reserved_count}/{self.global_policy['max_total_positions']}")
        return ValidationResult(True, "All checks passed", volume=volume)

    def _check_news_filter(self, strategy_name: str) -> ValidationResult:
        if self.news_filter is None or self.news_filter.should_trade():
            return ValidationResult(True, "NewsFilter passed")
        
        next_event = self.news_filter.get_next_event()
        event_detail = self._format_news_event(next_event) if next_event else ""
        logger.info(f"Reject strat={strategy_name} | reason=news_filter{event_detail}")
        return ValidationResult(False, f"High-impact news within {self.news_filter.buffer_minutes}-min buffer")

    def _check_market_conditions(self, symbol: str, expected_price: float, signal: int) -> ValidationResult:
        if signal not in (1, -1):
            logger.debug(f"MktCheckSkip strat={self.strategy_name} | sym={symbol} | reason=bracket_order")
            return ValidationResult(True, "Bracket order")
        
        condition = self.cost_calculator.validate_market_conditions(
            symbol=symbol,
            expected_price=expected_price,
            is_buy=(signal == 1)
            )
        
        if not condition.is_valid:
            logger.warning(f"MktCheckFail strat={self.strategy_name} | sym={symbol} | reason={condition.reason} | spr_pts={condition.spread_points:.1f} | spr_px={condition.spread_price:.5f} | slip_pts={condition.slippage_points:.1f} | slip_px={condition.slippage_price:.5f}")
            return ValidationResult(False, condition.reason)
        
        logger.debug(f"MktCheckOK strat={self.strategy_name} | sym={symbol} | spr_pts={condition.spread_points:.1f} | slip_pts={condition.slippage_points:.1f}")
        return ValidationResult(True, "Market conditions OK")

    def _reserve_position_slot(self) -> int:
        with self.global_position_count.get_lock():
            self.global_position_count.value += 1
            return self.global_position_count.value

    def _format_news_event(self, event: dict[str, Any]) -> str:
        currency = event.get('currency', 'N/A')
        event_name = event.get('event_name', 'Unknown')
        time_val = event.get('time_strategy')
        time_str = time_val.strftime('%H:%M %Z') if time_val else 'N/A'
        return f" (Next: {currency} {event_name} at {time_str})"

    def release_position_reservation(self, reason: str = "execution_failed") -> None:
        with self.global_position_count.get_lock():
            if self.global_position_count.value > 0:
                self.global_position_count.value -= 1
                logger.debug(f"PosSlotRel strat={self.strategy_name} | reason={reason} | cnt={self.global_position_count.value}/{self.global_policy['max_total_positions']}")
            else:
                logger.warning(f"PosSlotRelWarn strat={self.strategy_name} | reason=counter_already_zero")

    def check_global_risk(self) -> ValidationResult:
        """Check account-level risk limits using atomic counters and cached drawdown."""
        limit_checks = [
            (self.global_position_count.value, self.global_policy['max_total_positions'],
             "Global position limit", "position"),
            (self.global_trade_count.value, self.global_policy['max_daily_trades'],
             "Daily trade limit", "trade"),
        ]
        
        for current, maximum, msg, kind in limit_checks:
            if current >= maximum:
                logger.warning(f"RiskLimitHit typ={kind} | cur={current} | max={maximum}")
                return ValidationResult(False, f"{msg} reached")
        
        dd_checks = [
            (self.get_daily_drawdown('portfolio', 0), self.global_policy['max_daily_drawdown_pct'], "Daily"),
            (self.get_drawdown('portfolio', 0), self.global_policy['max_drawdown_pct'], "Max"),
        ]
        
        for current_dd, max_dd, label in dd_checks:
            if current_dd > max_dd:
                logger.warning(f"RiskLimitHit typ={label.lower()}_drawdown | cur={current_dd*100:.2f}% | max={max_dd*100:.1f}%")
                return ValidationResult(False, f"{label} Drawdown {current_dd*100:.1f}%")
        
        return ValidationResult(True, "Global checks passed")

    def check_strategy_limits(
        self,
        strategy_name: str,
        skip_position_limit_check: bool = False
    ) -> ValidationResult:
        """Check per-strategy position and trade limits."""
        magic = self.strategy_config['execution']['magic_number']

        if not skip_position_limit_check:
            current_positions = self._get_strategy_position_count(strategy_name, magic)
            if current_positions is None:
                return ValidationResult(False, "MT5 API error: positions_get() returned None")

            max_positions = self.risk_config['max_positions']
            if current_positions >= max_positions:
                logger.warning(f"StratLimitHit strat={strategy_name} | typ=positions | cur={current_positions} | max={max_positions}")
                return ValidationResult(False, "Strategy position limit")
        else:
            current_positions = self.strategy_runner.local_position_count if self.strategy_runner else 0
            max_positions = self.risk_config['max_positions']
        
        strategy_trades = self.shared_state.get('daily_trade_counts', {}).get(strategy_name, 0)
        max_trades = self.risk_config['max_trades']
        
        if strategy_trades >= max_trades:
            logger.warning(f"StratLimitHit strat={strategy_name} | typ=daily_trades | cur={strategy_trades} | max={max_trades}")
            return ValidationResult(False, "Strategy daily trade limit")
        
        logger.debug(f"StratLimitOK strat={strategy_name} | pos={current_positions}/{max_positions} | tr={strategy_trades}/{max_trades} | m={magic}")
        return ValidationResult(True, "Strategy checks passed")

    def _get_strategy_position_count(self, strategy_name: str, magic: int) -> int | None:
        if self.strategy_runner is not None:
            logger.debug(f"PosCountSrc strat={strategy_name} | src=local_counter")
            return self.strategy_runner.local_position_count
        
        logger.warning(f"PosCountSrc strat={strategy_name} | src=mt5_query | reason=no_runner")
        positions = mt.positions_get()
        if positions is None:
            logger.error(f"PosCountFail strat={strategy_name} | op=positions_get | err={mt.last_error()}")
            return None
        return sum(1 for pos in positions if pos.magic == magic)

    def _get_cached(self, fetch_fn, cache: TTLCache, ttl: float):
        if cache.is_valid(ttl):
            return cache.value
        result = fetch_fn()
        if result is not None:
            cache.value = result
            cache.timestamp = time.time()
        return result

    def _get_account_info_cached(self):
        return self._get_cached(mt.account_info, self._account_cache, self.ACCOUNT_CACHE_TTL)

    def _get_equity_info_cached(self):
        return self._get_cached(mt.account_info, self._equity_cache, self.EQUITY_CACHE_TTL)

    def _get_symbol_info_cached(self, symbol: str):
        if symbol not in self._symbol_cache:
            self._symbol_cache[symbol] = TTLCache()
        return self._get_cached(lambda: mt.symbol_info(symbol), self._symbol_cache[symbol], self.SYMBOL_CACHE_TTL)

    def calculate_position_size(
        self,
        symbol: str,
        entry: float,
        sl: float,
        strategy_name: str
    ) -> float:
        r"""
        Calculate position size: volume = (R × multiplier) / (d_SL × v_tick)
        """
        account_info = self._get_account_info_cached()
        if account_info is None:
            logger.error("PosSizeFail reason=account_info_unavailable")
            return 0.0
        
        symbol_info = self._get_symbol_info_cached(symbol)
        if symbol_info is None:
            logger.error(f"PosSizeFail sym={symbol} | reason=symbol_info_unavailable")
            return 0.0
        
        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            logger.error(f"PosSizeFail strat={strategy_name} | reason=zero_sl_distance")
            return 0.0
        
        tick_value = symbol_info.trade_tick_value or (symbol_info.trade_tick_size * symbol_info.trade_contract_size)
        if tick_value == 0:
            logger.error(f"PosSizeFail strat={strategy_name} | sym={symbol} | reason=tick_value_zero | tick_size={symbol_info.trade_tick_size} | contract_size={symbol_info.trade_contract_size}")
            return 0.0
        
        risk_per_trade = self.global_policy['strategy_risk'][strategy_name]
        risk_multiplier = self._get_adaptive_risk_multiplier(strategy_name)
        adjusted_risk = account_info.balance * risk_per_trade * risk_multiplier
        
        ticks = sl_distance / symbol_info.trade_tick_size
        volume = adjusted_risk / (ticks * tick_value)
        
        return self._normalize_volume(volume, symbol_info)

    def _normalize_volume(self, volume: float, symbol_info) -> float:
        step = symbol_info.volume_step
        volume = round(volume / step) * step
        return max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

    def _get_adaptive_risk_multiplier(self, strategy_name: str) -> float:
        adaptive_config = self.global_policy.get('adaptive_sizing', {})
        if not adaptive_config.get('enabled', False):
            return 1.0
        
        scope = adaptive_config.get('scope', 'portfolio')
        magic = self.strategy_config['execution']['magic_number']
        current_drawdown = self.get_drawdown(scope=scope, magic=magic)
        
        thresholds = sorted(adaptive_config.get('drawdown_thresholds', []), key=lambda x: x['drawdown_pct'], reverse=True)
        
        for threshold in thresholds:
            if current_drawdown >= threshold['drawdown_pct']:
                multiplier = threshold['risk_multiplier']
                logger.info(f"AdaptiveRisk strat={strategy_name} | dd={current_drawdown*100:.2f}% | thr={threshold['drawdown_pct']*100:.1f}% | mul={multiplier:.2f}")
                return multiplier
        return 1.0

    def get_drawdown(self, scope: str = 'portfolio', magic: int = 0) -> float:
        """Calculate maximum drawdown since inception (cached, incremental)."""
        if not self._max_dd_cache.initialized:
            self._init_drawdown_cache(self._max_dd_cache, datetime(2025, 1, 1), scope, magic, is_daily=False)
        self._update_drawdown_cache(self._max_dd_cache, scope, magic)
        return self._calc_dd_pct(self._max_dd_cache)

    def get_daily_drawdown(self, scope: str = 'portfolio', magic: int = 0) -> float:
        """Calculate drawdown since midnight (cached, daily reset)."""
        current_date = self._broker_today()
        
        needs_reset = (
            not self._daily_dd_cache.initialized or
            self._daily_dd_cache.cache_date is None or
            self._daily_dd_cache.cache_date < current_date
        )
        
        if needs_reset:
            self._init_drawdown_cache(self._daily_dd_cache, self._midnight_today(), scope, magic, is_daily=True)
            self._daily_dd_cache.cache_date = current_date
            logger.debug(f"DDCacheReset strat={self.strategy_name} | day={current_date}")
        
        self._update_drawdown_cache(self._daily_dd_cache, scope, magic)
        return self._calc_dd_pct(self._daily_dd_cache)

    def _calc_dd_pct(self, cache: DrawdownCache) -> float:
        if cache.peak_equity <= 0:
            logger.warning(f"DDCalcWarn strat={self.strategy_name} | reason=peak_le_zero")
            return 0.0
        if cache.current_equity >= cache.peak_equity:
            return 0.0
        return (cache.peak_equity - cache.current_equity) / cache.peak_equity

    def validate_position_size(self, symbol: str, volume: float) -> float | None:
        symbol_info = self._get_symbol_info_cached(symbol)
        if symbol_info is None:
            logger.error(f"PosNormFail sym={symbol} | reason=symbol_info_unavailable")
            return None
        return self._normalize_volume(volume, symbol_info) if volume >= symbol_info.volume_min else None

    def _filter_trading_deals(
        self,
        deals,
        scope: str,
        magic: int,
        min_cursor: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, int, tuple[int, int | None, tuple[int, int]]]:
        """Extract and filter trading deals in a single pass."""
        if not deals:
            return None

        start_cursor = min_cursor or (0, 0)
        last_seen_cursor = start_cursor
        last_trade_cursor = start_cursor
        net_pnl = []
        append_pnl = net_pnl.append
        trade_count = 0
        is_portfolio_scope = scope == "portfolio"

        for deal in deals:
            deal_cursor = self._get_deal_cursor(deal)
            if deal_cursor > last_seen_cursor:
                last_seen_cursor = deal_cursor
            if min_cursor is not None and deal_cursor <= min_cursor:
                continue
            if deal.type not in (0, 1):
                continue
            if not is_portfolio_scope and deal.magic != magic:
                continue

            append_pnl(float(deal.profit + deal.commission + deal.swap + deal.fee))
            trade_count += 1
            if deal_cursor > last_trade_cursor:
                last_trade_cursor = deal_cursor

        if trade_count == 0:
            return np.array([], dtype=np.float64), 0, start_cursor, last_seen_cursor

        return np.asarray(net_pnl, dtype=np.float64), trade_count, last_trade_cursor, last_seen_cursor

    def _fetch_deals_since(
        self,
        start_cursor: tuple[int, int],
        scope: str,
        magic: int
    ) -> tuple[np.ndarray, int, tuple[int, int | None, tuple[int, int]]]:
        start_time = datetime.fromtimestamp(start_cursor[0] / 1000.0)
        query_end = datetime.now() + timedelta(seconds=1)
        deals = mt.history_deals_get(start_time, query_end)
        if deals is None:
            logger.warning(f"DDFetchWarn strat={self.strategy_name} | reason=history_deals_get_failed")
            return None
        result = self._filter_trading_deals(deals, scope, magic, min_cursor=start_cursor)
        if result is None:
            return np.array([], dtype=np.float64), 0, start_cursor, start_cursor
        return result

    def _update_drawdown_cache(self, cache: DrawdownCache, scope: str, magic: int) -> None:
        fetch_result = self._fetch_deals_since(cache.last_cursor, scope, magic)
        if fetch_result is None:
            cache.current_equity = self._get_current_equity(cache.current_equity, scope, magic)
            cache.peak_equity = max(cache.peak_equity, cache.current_equity)
            return

        new_pnl, new_trade_count, last_trade_cursor, last_seen_cursor = fetch_result
        if new_trade_count > 0:
            cache.current_equity += float(np.sum(new_pnl))
            cache.last_cursor = last_trade_cursor
        elif last_seen_cursor > cache.last_cursor:
            cache.last_cursor = last_seen_cursor

        cache.current_equity = self._get_current_equity(cache.current_equity, scope, magic)
        cache.peak_equity = max(cache.peak_equity, cache.current_equity)
        cache.last_deal_time = datetime.now()
        
        logger.debug(
            f"DDCacheUpd strat={self.strategy_name} | deals={new_trade_count} | "
            f"cur={cache.current_equity:.2f} | peak={cache.peak_equity:.2f}"
        )

    def _get_current_equity(self, realized_equity: float, scope: str, magic: int) -> float:
        if scope == "portfolio":
            account_info = self._get_equity_info_cached()
            return account_info.equity if account_info else realized_equity
        positions = mt.positions_get(magic=magic)
        unrealized = sum(pos.profit for pos in positions) if positions else 0.0
        return realized_equity + unrealized

    def _init_drawdown_cache(
        self,
        cache: DrawdownCache,
        start_time: datetime,
        scope: str,
        magic: int,
        is_daily: bool
    ) -> None:
        """Initialize drawdown cache from historical data."""
        init_time = datetime.now()
        deals = mt.history_deals_get(start_time, init_time + timedelta(seconds=1))
        
        if deals is None:
            logger.warning(f"DDInitWarn strat={self.strategy_name} | reason=history_deals_get_failed")
            current_equity = self._get_current_equity(self.global_policy['initial_balance'], scope, magic)
            cache.peak_equity = cache.current_equity = current_equity
            cache.last_deal_time = init_time
            cache.last_cursor = self._datetime_to_cursor(init_time)
            cache.initialized = True
            return
        
        if not deals:
            current_equity = self._get_current_equity(cache.current_equity, scope, magic)
            cache.current_equity = current_equity
            if is_daily and scope == "portfolio":
                cache.peak_equity = current_equity
            else:
                cache.peak_equity = max(cache.peak_equity, current_equity)
            cache.last_deal_time = init_time
            cache.last_cursor = self._datetime_to_cursor(init_time)
            cache.initialized = True
            logger.debug(f"DDInit strat={self.strategy_name} | deals=0")
            return
        
        result = self._filter_trading_deals(deals, scope, magic)
        net_pnl = result[0] if result else np.array([], dtype=np.float64)
        deal_count = result[1] if result else 0
        last_trade_cursor = result[2] if result else cache.last_cursor
        realized_pnl = float(np.sum(net_pnl)) if net_pnl.size else 0.0
        
        start_balance = self._get_start_balance(deals, scope, is_daily, realized_pnl)
        current_equity = self._get_current_equity(start_balance + realized_pnl, scope, magic)
        
        if scope == "portfolio" and is_daily:
            account_info = self._get_equity_info_cached()
            if account_info:
                current_equity = account_info.equity
        
        if net_pnl.size:
            equity_path = start_balance + np.cumsum(net_pnl)
            historical_peak = max(start_balance, float(np.max(equity_path)))
        else:
            historical_peak = start_balance
        peak_equity = max(historical_peak, current_equity)
        
        cache.peak_equity = peak_equity
        cache.current_equity = current_equity
        cache.last_deal_time = init_time
        cache.last_cursor = last_trade_cursor if deal_count > 0 else self._datetime_to_cursor(init_time)
        cache.initialized = True
        
        logger.debug(f"DDInit strat={self.strategy_name} | daily={int(is_daily)} | deals={deal_count} | peak={peak_equity:.2f} | cur={current_equity:.2f}")

    def _get_start_balance(self, deals, scope: str, is_daily: bool, realized_pnl: float) -> float:
        if is_daily and scope == "portfolio":
            account_info = self._get_account_info_cached()
            if account_info:
                return account_info.balance - realized_pnl
        
        if not is_daily:
            deal_types = np.array([d.type for d in deals], dtype=np.int8)
            deposit_mask = deal_types == 2
            if np.any(deposit_mask):
                return float(np.array([d.profit for d in deals], dtype=np.float64)[deposit_mask][0])
        
        return self.global_policy['initial_balance']
