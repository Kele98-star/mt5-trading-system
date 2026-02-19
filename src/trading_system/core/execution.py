import MetaTrader5 as mt
import logging
from typing import Callable, Any
from datetime import datetime
from collections import defaultdict
import time
import pytz
import pandas as pd
from src.trading_system.config.broker_config import (
    OrderType, TradeAction, TimeInForce, get_symbol_spec,
    SymbolSpec, TIMEFRAME_TO_MINUTES, TIMEFRAME_STRING_MAP
)
from src.trading_system.core.data_handler import DataHandler
from src.trading_system.utils.logging_utils import format_price_display
import threading
from src.trading_system.core.execution_requests import EntryRequest, ExitRequest, ModifyRequest, ExecutionResult

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, data_handler: DataHandler, broker_tz: pytz.tzinfo.BaseTzInfo):
        self._data_handler = data_handler
        self.broker_tz = broker_tz
        self.server_tz = broker_tz.zone
        self._strategy_tz_cache: dict[str, pytz.BaseTzInfo] = {}
        self._strategy_runtime_cache: dict[str, dict[str, str | int]] = {}

        self._entry_handlers: dict[str, Callable] = {
            'market': self._execute_market_entry,
            'bracket': self._execute_bracket_entry,
            'stop': self._execute_stop_order,
            'limit': self._execute_limit_order,
        }

        self._symbol_spec_cache: dict[str, tuple[SymbolSpec, int, float]] = {}
        self._spec_cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutes

        # Retry configuration
        self.max_retries = 3
        self.retry_delays = [0.025, 0.05, 0.10]
        self.retryable_codes = {10006, 10007, 10010, 10018, 10019}
        self._retcode_done = int(mt.TRADE_RETCODE_DONE)
        retcode_no_changes = getattr(mt, "TRADE_RETCODE_NO_CHANGES", None)
        self._retcode_no_changes = (int(retcode_no_changes) if isinstance(retcode_no_changes, int) and retcode_no_changes > 0 else 10025)

        logger.debug(f"ExecInit retry_max={self.max_retries}")

    def _order_send_with_retry(self, mt5_request: dict, success_codes: set[int] | None = None) -> Any | None:
        """Execute order_send with exponential backoff retry on transient failures."""
        symbol = mt5_request.get('symbol')
        if success_codes is None:
            success_codes = {self._retcode_done}

        for attempt in range(self.max_retries):
            result = mt.order_send(mt5_request)

            if result is not None and int(result.retcode) in success_codes:
                if attempt > 0:
                    logger.info(f"OrderRetryOK sym={symbol} | try={attempt}/{self.max_retries - 1}")
                return result

            if result is not None and result.retcode not in self.retryable_codes:
                logger.warning(f"OrderSendNoRetry sym={symbol} | rc={result.retcode} | err={result.comment}")
                return result

            if attempt < self.max_retries - 1:
                backoff_seconds = self.retry_delays[attempt]
                error_msg = result.comment if result else "MT5 returned None"
                logger.warning(f"OrderSendRetry sym={symbol} | try={attempt + 1}/{self.max_retries} | wait_ms={backoff_seconds * 1000:.0f} | err={error_msg}")
                time.sleep(backoff_seconds)
            else:
                logger.error(f"OrderSendFail sym={symbol} | tries={self.max_retries} | err={result.comment if result else 'MT5 returned None'}")

        return result

    def _get_cached_symbol_spec(self, symbol: str) -> tuple[SymbolSpec, int]:
        """Retrieve symbol specification from cache with TTL validation."""
        with self._spec_cache_lock:
            if symbol in self._symbol_spec_cache:
                cached_spec, cached_filling, cached_ts = self._symbol_spec_cache[symbol]
                if time.time() - cached_ts < self._cache_ttl:
                    return cached_spec, cached_filling

        # Cache miss or expired → fetch fresh data
        spec = get_symbol_spec(symbol)

        filling_modes = spec.filling_modes
        filling = filling_modes[0]

        with self._spec_cache_lock:
            self._symbol_spec_cache[symbol] = (spec, filling, time.time())
        return spec, filling

    def _get_strategy_runtime(self, strategy_name: str) -> dict[str, str | int]:
        """Get cached strategy runtime parameters (timezone and timeframe)."""
        cached_runtime = self._strategy_runtime_cache.get(strategy_name)
        if cached_runtime is not None:
            return cached_runtime

        config = self._data_handler._load_strategy_config(strategy_name)
        strategy_timezone = config.get("timezone")
        timeframe_name = config.get("timeframe")
        timeframe_key = TIMEFRAME_STRING_MAP.get(timeframe_name) if timeframe_name else None
        timeframe_minutes = TIMEFRAME_TO_MINUTES.get(timeframe_key)

        runtime = {
            "strategy_timezone": str(strategy_timezone),
            "timeframe_minutes": int(timeframe_minutes),
            "config": config,
        }
        self._strategy_runtime_cache[strategy_name] = runtime
        return runtime

    def _round_prices(self, digits: int, **prices: float | None) -> dict[str, float]:
        """Round multiple prices to symbol digits. Returns 0.0 for None values."""
        return {key: round(value, digits) if value is not None else 0.0
                for key, value in prices.items()}

    def _validate_order_result(self,result: Any | None) -> tuple[bool, str]:
        """Validate MT5 order result for success."""
        if result is None:
            return False, "MT5 returned None"

        if int(result.retcode) != self._retcode_done:
            return False, result.comment

        return True, ""

    def execute_entry(self, request: EntryRequest) -> ExecutionResult:
        handler = self._entry_handlers.get(request.order_type)

        if handler is None:
            error_msg = f"Unknown order type: {request.order_type}"
            logger.error(f"EntryFail reason=unknown_order_type | ot={request.order_type}")
            return ExecutionResult(
                success=False,
                error_message=error_msg,
                request_type='entry',
                symbol=request.symbol)

        return handler(request)

    def execute_exit(self, request: ExitRequest) -> ExecutionResult:
        positions = mt.positions_get(ticket=request.ticket)

        if not positions:
            logger.warning(f"ExitSkip t={request.ticket} | reason=already_closed")
            return ExecutionResult(
                success=False,
                ticket=request.ticket,
                executed_volume=None,
                error_message=f"Position {request.ticket} already closed",
                request_type="exit",
            )

        position = positions[0]
        original_volume = position.volume
        attempted_volume = original_volume * request.portion

        results = self.close_positions(
            tickets=[request.ticket],
            portions=[request.portion],
            preloaded_positions={request.ticket: position},
        )
        success, deal_id = results.get(request.ticket, (False, None))

        return ExecutionResult(
            success=success,
            ticket=request.ticket,
            executed_volume=attempted_volume if success else None,
            error_message="" if success else f"Failed to close {request.ticket}",
            request_type="exit",
            deal_id=deal_id,
        )

    def execute_modify(self, request: ModifyRequest) -> ExecutionResult:
        success = self.modify_position_sl_tp(
            ticket=request.ticket,
            sl=request.new_sl,
            tp=request.new_tp
        )

        return ExecutionResult(
            success=success,
            ticket=request.ticket,
            error_message="" if success else f"Failed to modify {request.ticket}",
            request_type='modify'
        )

    def _execute_market_entry(self, request: EntryRequest) -> ExecutionResult:
        runtime = self._get_strategy_runtime(request.strategy_name)
        strategy_config = runtime["config"]
        symbol = request.symbol

        symbol_spec, type_filling = self._get_cached_symbol_spec(symbol)
        order_type = OrderType.BUY if request.signal == 1 else OrderType.SELL

        prices = self._round_prices(symbol_spec.digits, sl=request.sl, tp=request.tp)

        tick = mt.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"EntryFail sym={symbol} | reason=tick_unavailable")
            return ExecutionResult(success=False, error_message=f"{symbol}: tick data unavailable", request_type='entry', symbol=symbol)
        price = tick.ask if request.signal == 1 else tick.bid

        mt5_request = {
            "action": TradeAction.DEAL,
            "symbol": symbol,
            "volume": request.volume,
            "type": int(order_type),
            "price": price,
            "sl": prices["sl"],
            "tp": prices["tp"],
            "deviation": strategy_config['deviation'],
            "magic": strategy_config['magic_number'],
            "comment": request.comment,
            "type_filling": type_filling,
            "type_time": int(TimeInForce.GTC)
        }

        result = self._order_send_with_retry(mt5_request)
        is_valid, error_msg = self._validate_order_result(result)

        if not is_valid:
            return ExecutionResult(
                success=False,
                error_message=error_msg,
                request_type='entry',
                symbol=symbol
            )

        price_display = format_price_display(price)
        logger.info(f"EntryOK typ={order_type.name} | sym={symbol} | vol={request.volume:.2f} | px={price_display} | sl={prices['sl']} | tp={prices['tp']} | t={result.order}")

        return ExecutionResult(
            success=True,
            ticket=result.order,
            request_type='entry',
            symbol=symbol
        )

    def _execute_pending_order(self, request: EntryRequest, order_type: OrderType) -> ExecutionResult:
        """Unified handler for stop and limit orders."""
        runtime = self._get_strategy_runtime(request.strategy_name)
        strategy_config = runtime["config"]
        symbol = request.symbol
        symbol_spec, type_filling = self._get_cached_symbol_spec(symbol)

        prices = self._round_prices(symbol_spec.digits, sl=request.sl, tp=request.tp, entry_price=request.entry_price)

        # Fetch market data once if expiration needed
        market_data = self._fetch_market_data_for_expiration(symbol, request)
        expiration_timestamp = self._process_expiration(request, symbol, market_data)

        if expiration_timestamp is False:
            return ExecutionResult(
                success=False,
                error_message=f"Expiration time has already passed",
                request_type='entry',
                symbol=symbol
            )

        type_time = int(TimeInForce.SPECIFIED) if expiration_timestamp else int(TimeInForce.GTC)

        mt5_request = {
            "action": int(TradeAction.PENDING),
            "symbol": symbol,
            "volume": float(request.volume),
            "type": int(order_type),
            "price": float(prices["entry_price"]),
            "sl": float(prices["sl"]),
            "tp": float(prices["tp"]),
            "deviation": strategy_config['deviation'],
            "magic": strategy_config['magic_number'],
            "comment": request.comment,
            "type_filling": int(type_filling),
            "type_time": type_time
        }

        if expiration_timestamp:
            mt5_request["expiration"] = int(expiration_timestamp)

        result = self._order_send_with_retry(mt5_request)
        is_valid, error_msg = self._validate_order_result(result)

        if not is_valid:
            return ExecutionResult(
                success=False,
                error_message=error_msg,
                request_type='entry',
                symbol=symbol
            )

        entry_price_display = format_price_display(prices["entry_price"])
        logger.info(f"EntryOK typ={order_type.name} | sym={symbol} | vol={request.volume:.2f} | px={entry_price_display} | sl={prices['sl']} | tp={prices['tp']} | t={result.order}")

        return ExecutionResult(
            success=True,
            ticket=result.order,
            request_type='entry',
            symbol=symbol
        )

    def _execute_stop_order(self, request: EntryRequest) -> ExecutionResult:
        """Execute stop order (BUY_STOP or SELL_STOP) with optional expiration."""
        order_type = OrderType.BUY_STOP if request.signal == 1 else OrderType.SELL_STOP
        return self._execute_pending_order(request, order_type)

    def _execute_limit_order(self, request: EntryRequest) -> ExecutionResult:
        """Execute limit order (BUY_LIMIT or SELL_LIMIT) with optional expiration."""
        order_type = OrderType.BUY_LIMIT if request.signal == 1 else OrderType.SELL_LIMIT
        return self._execute_pending_order(request, order_type)

    def _fetch_market_data_for_expiration(self, symbol: str, request: EntryRequest) -> dict[str, Any] | None:
        """
        Fetch tick and symbol info once if expiration is requested.
        Returns dict with tick_epoch and server_epoch, or None if no expiration.
        """
        if not getattr(request, "expiration_time", None):
            return None

        tick = mt.symbol_info_tick(symbol)
        symbol_info = mt.symbol_info(symbol)

        return {
            "tick": tick,
            "symbol_info": symbol_info,
            "tick_epoch": self._extract_tick_epoch(tick),
            "server_epoch": getattr(symbol_info, "time", None) if symbol_info else None,
        }

    def _extract_tick_epoch(self, tick: Any | None) -> int | None:
        tick_dict = tick._asdict()
        mapped_time = tick_dict.get("time")
        return int(mapped_time)

    def _process_expiration(self, request: EntryRequest, symbol: str, market_data: dict[str, Any] | None = None) -> int | None | bool:
        """
        Process expiration time and return timestamp.
        Returns:
            int: Valid expiration timestamp
            None: No expiration requested
            False: Expiration validation failed
        """
        if not getattr(request, "expiration_time", None):
            return None

        runtime = self._get_strategy_runtime(request.strategy_name)
        strategy_timezone = runtime["strategy_timezone"]

        # Use provided market data or fetch if not provided
        if market_data:
            tick_epoch = market_data["tick_epoch"]
            server_epoch = market_data["server_epoch"]
        else:
            tick_epoch = self._extract_tick_epoch(mt.symbol_info_tick(symbol))
            symbol_info = mt.symbol_info(symbol)
            server_epoch = getattr(symbol_info, "time", None) if symbol_info else None

        expiration_timestamp = self._convert_expiration_to_broker_time(
            request.strategy_name,
            strategy_timezone,
            request.expiration_time,
            tick_epoch,
        )

        if expiration_timestamp is None:
            logger.error(
                f"ExpFail hhmm={request.expiration_time} | tz={strategy_timezone} | reason=passed"
            )
            return False

        # Validate against MT5 server time
        if server_epoch and expiration_timestamp <= int(server_epoch):
            logger.error(f"ExpFail hhmm={request.expiration_time} | tz={strategy_timezone} | exp={datetime.fromtimestamp(expiration_timestamp, tz=self.broker_tz).strftime('%Y-%m-%d %H:%M:%S %Z')} | srv={datetime.fromtimestamp(server_epoch, tz=self.broker_tz).strftime('%Y-%m-%d %H:%M:%S %Z')} | reason=lte_server_time")
            return False

        return expiration_timestamp

    def _build_bracket_request(
        self,
        action: TradeAction,
        order_type: OrderType,
        symbol: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        strategy_config: dict,
        type_filling: int,
        expiration_timestamp: int | None
    ) -> dict:
        """Build MT5 request dictionary for bracket orders."""
        type_time = int(TimeInForce.SPECIFIED) if (expiration_timestamp and action == TradeAction.PENDING) else int(TimeInForce.GTC)

        request = {
            "action": int(action),
            "symbol": symbol,
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": strategy_config["deviation"],
            "magic": strategy_config["magic_number"],
            "comment": f"{strategy_config['comment_prefix']}",
            "type_filling": int(type_filling),
            "type_time": type_time,
        }

        if expiration_timestamp and action == TradeAction.PENDING:
            request["expiration"] = int(expiration_timestamp)

        return request

    def _determine_bracket_order_type(
        self,
        entry_price: float,
        current_price: float,
        stops_level_points: int,
        min_market_threshold_points: int,
        symbol_spec: SymbolSpec,
        is_buy: bool
    ) -> tuple[OrderType, float, TradeAction]:
        """
        Determine order type, execution price, and action for bracket leg.

        Returns: (order_type, price, action)
        """
        distance_price = entry_price - current_price if is_buy else current_price - entry_price
        distance_points = distance_price / symbol_spec.point

        if distance_points < min_market_threshold_points:
            order_type = OrderType.BUY if is_buy else OrderType.SELL
            price = current_price
            action = TradeAction.DEAL
            logger.warning(f"BrktLegMode side={'B' if is_buy else 'S'} | mode=market | reason=too_close_stoplimit")
        elif distance_points < stops_level_points:
            order_type = OrderType.BUY_LIMIT if is_buy else OrderType.SELL_LIMIT
            price = entry_price
            action = TradeAction.PENDING
            logger.warning(f"BrktLegMode side={'B' if is_buy else 'S'} | mode=limit | reason=stoplevel_violation | entry={entry_price:.{symbol_spec.digits}f} | cur={current_price:.{symbol_spec.digits}f}")
        else:
            order_type = OrderType.BUY_STOP if is_buy else OrderType.SELL_STOP
            price = entry_price
            action = TradeAction.PENDING
            logger.debug(f"BrktLegMode side={'B' if is_buy else 'S'} | mode=stop | entry={entry_price:.{symbol_spec.digits}f}")

        return order_type, price, action

    def _gather_bracket_market_data(self, symbol: str, request: EntryRequest) -> tuple[Any | None, Any | None, dict[str, Any] | None]:
        """
        Gather tick, symbol_info, and expiration data for bracket order.
        Returns: (tick, symbol_info, market_data_dict)
        """
        tick = mt.symbol_info_tick(symbol)

        symbol_info = mt.symbol_info(symbol)
        market_data = {
            "tick": tick,
            "symbol_info": symbol_info,
            "tick_epoch": self._extract_tick_epoch(tick),
            "server_epoch": getattr(symbol_info, "time", None),
        }

        return tick, symbol_info, market_data

    def _prepare_bracket_orders(
        self,
        request: EntryRequest,
        symbol_spec: SymbolSpec,
        type_filling: int,
        tick: Any,
        symbol_info: Any,
        expiration_timestamp: int | None,
        strategy_config: dict,
        min_market_threshold_points: int,
    ) -> tuple[dict, dict]:
        """
        Prepare buy and sell bracket order requests.
        Returns: (buy_request, sell_request)
        """
        prices = self._round_prices(
            symbol_spec.digits,
            buy_stop=request.buy_stop,
            sell_stop=request.sell_stop,
            buy_sl=request.buy_sl,
            sell_sl=request.sell_sl,
            buy_tp=request.buy_tp,
            sell_tp=request.sell_tp,
        )

        current_ask = tick.ask
        current_bid = tick.bid
        stops_level_points = symbol_info.trade_stops_level
        stops_level_price = stops_level_points * symbol_spec.point

        stops_level_price_display = format_price_display(stops_level_price)
        logger.debug(f"BrktMkt sym={request.symbol} | ask={current_ask:.{symbol_spec.digits}f} | bid={current_bid:.{symbol_spec.digits}f} | stp_pts={stops_level_points} | stp_px={stops_level_price_display}")

        # Determine order types for both legs
        buy_order_type, buy_price, buy_action = self._determine_bracket_order_type(
            prices["buy_stop"], current_ask, stops_level_points,
            min_market_threshold_points, symbol_spec, is_buy=True
        )

        sell_order_type, sell_price, sell_action = self._determine_bracket_order_type(
            prices["sell_stop"], current_bid, stops_level_points,
            min_market_threshold_points, symbol_spec, is_buy=False
        )

        # Build requests
        buy_request = self._build_bracket_request(
            buy_action, buy_order_type, request.symbol, request.volume, buy_price,
            prices["buy_sl"], prices["buy_tp"], strategy_config, type_filling, expiration_timestamp
        )

        sell_request = self._build_bracket_request(
            sell_action, sell_order_type, request.symbol, request.volume, sell_price,
            prices["sell_sl"], prices["sell_tp"], strategy_config, type_filling, expiration_timestamp
        )

        return buy_request, sell_request

    def _execute_bracket_orders(self, buy_request: dict, sell_request: dict, symbol: str) -> tuple[Any | None, Any | None]:
        """
        Execute both bracket orders with cleanup on failure.
        Returns: (buy_result, sell_result) or (None, None) on failure
        """
        buy_result = self._order_send_with_retry(buy_request)
        is_valid, error_msg = self._validate_order_result(buy_result)

        if not is_valid:
            logger.error(f"BrktBuyFail sym={symbol} | err={error_msg}")
            return None, None

        sell_result = self._order_send_with_retry(sell_request)
        is_valid, error_msg = self._validate_order_result(sell_result)

        if not is_valid:
            logger.error(f"BrktSellFail sym={symbol} | err={error_msg}")
            self._cleanup_failed_bracket(buy_result.order)
            return None, None

        return buy_result, sell_result

    def _cleanup_failed_bracket(self, buy_ticket: int) -> None:
        """Cancel buy order after sell order fails in bracket execution."""
        for cancel_attempt in range(3):
            cancel_ok = self._cancel_order(buy_ticket)
            if cancel_ok:
                logger.info(f"BrktCleanupOK buy_t={buy_ticket}")
                return
            logger.warning(f"BrktCleanupRetryFail try={cancel_attempt + 1}/3 | buy_t={buy_ticket}")
            time.sleep(0.05 * (cancel_attempt + 1))

        logger.error(f"BrktCleanupFail buy_t={buy_ticket} | tries=3")

    def _execute_bracket_entry(self, request: EntryRequest) -> ExecutionResult:
        """Execute bracket order with buy stop and sell stop."""
        runtime = self._get_strategy_runtime(request.strategy_name)
        strategy_config = runtime["config"]
        symbol = request.symbol
        symbol_spec, type_filling = self._get_cached_symbol_spec(symbol)
        min_market_threshold_points = strategy_config.get('min_market_threshold_points')

        # Gather market data
        tick, symbol_info, market_data = self._gather_bracket_market_data(symbol, request)
        if tick is None or symbol_info is None:
            return ExecutionResult(
                success=False,
                error_message=f"{symbol}: market data unavailable",
                request_type='entry',
                symbol=symbol
            )

        # Process expiration
        expiration_timestamp = self._process_expiration(request, symbol, market_data)
        if expiration_timestamp is False:
            return ExecutionResult(
                success=False,
                error_message="Expiration time has already passed.",
                request_type="entry",
                symbol=symbol
            )

        # Prepare orders
        buy_request, sell_request = self._prepare_bracket_orders(
            request, symbol_spec, type_filling, tick, symbol_info,
            expiration_timestamp, strategy_config, min_market_threshold_points
        )

        # Execute orders
        buy_result, sell_result = self._execute_bracket_orders(buy_request, sell_request, symbol)

        if buy_result is None or sell_result is None:
            error_msg = "Bracket execution failed"
            return ExecutionResult(success=False, error_message=error_msg, request_type="entry", symbol=symbol)

        logger.info(
            f"{request.strategy_name}: BrktOK sym={symbol} | buy={OrderType(buy_request['type']).name}@{request.buy_stop:.{symbol_spec.digits}f} | sell={OrderType(sell_request['type']).name}@{request.sell_stop:.{symbol_spec.digits}f} | exp={request.expiration_time if hasattr(request, 'expiration_time') and request.expiration_time else 'GTC'}"
        )
        logger.debug(
            f"BrktOKIds sym={symbol} | buy={buy_result.order} | sell={sell_result.order}"
        )

        return ExecutionResult(
            success=True,
            ticket=buy_result.order,
            order_tickets=[buy_result.order, sell_result.order],
            request_type="entry",
            symbol=symbol
        )

    def _get_broker_display_time(self, srv_epoch: int) -> pd.Timestamp:
        """Convert server epoch to broker display time."""
        time_utc = pd.to_datetime(srv_epoch, unit='s', utc=True)
        broker_offset_seconds = time_utc.tz_convert(self.broker_tz).utcoffset().total_seconds()
        return (time_utc - pd.Timedelta(seconds=broker_offset_seconds)).tz_convert(self.broker_tz)

    def _parse_expiration_time(self, expiration_hhmm: str, now_strategy: pd.Timestamp, tf_minutes: int, strategy_tz: pytz.BaseTzInfo) -> pd.Timestamp | None:
        """Parse HH:MM expiration time and localize to strategy timezone."""
        parts = expiration_hhmm.split(':')
        expiration_hour, expiration_minute = int(parts[0]), int(parts[1])

        naive_expiration = datetime(now_strategy.year, now_strategy.month, now_strategy.day, expiration_hour, expiration_minute) + pd.Timedelta(minutes=tf_minutes) - pd.Timedelta(seconds=1)

        try:
            expiration_strategy = strategy_tz.localize(naive_expiration, is_dst=None)
        except pytz.exceptions.AmbiguousTimeError:
            expiration_strategy = strategy_tz.localize(naive_expiration, is_dst=False)
            logger.warning(f"ExpDST hhmm={expiration_hhmm} | type=ambiguous | use_is_dst=False")
        except pytz.exceptions.NonExistentTimeError:
            expiration_strategy = strategy_tz.localize(naive_expiration, is_dst=True)
            logger.warning(f"ExpDST hhmm={expiration_hhmm} | type=non_existent | use_is_dst=True")

        return pd.Timestamp(expiration_strategy)

    def _convert_broker_display_to_utc(
        self,
        expiration_broker_display: pd.Timestamp
    ) -> int:
        """Convert broker display time to real UTC timestamp."""
        time_utc = pd.to_datetime(expiration_broker_display.value, unit='ns', utc=True)
        broker_offset_seconds = time_utc.tz_convert(self.broker_tz).utcoffset().total_seconds()
        expiration_as_utc_minus_offset = expiration_broker_display.tz_convert('UTC')
        expiration_real_utc = expiration_as_utc_minus_offset + pd.Timedelta(seconds=broker_offset_seconds)
        return int(expiration_real_utc.timestamp())

    def _convert_expiration_to_broker_time(
        self,
        strategy_name: str,
        strategy_tz_input: str | pytz.BaseTzInfo,
        expiration_hhmm: str,
        srv_epoch: int
    ) -> int | None:
        """Convert strategy-local expiration time to broker server epoch."""
        if strategy_name not in self._strategy_tz_cache:
            self._strategy_tz_cache[strategy_name] = (pytz.timezone(strategy_tz_input) if isinstance(strategy_tz_input, str) else strategy_tz_input)
        strategy_tz = self._strategy_tz_cache[strategy_name]

        now_broker_display = self._get_broker_display_time(srv_epoch)
        now_strategy = now_broker_display.tz_convert(strategy_tz)

        runtime = self._get_strategy_runtime(strategy_name)
        tf_minutes = runtime["timeframe_minutes"]

        expiration_pd = self._parse_expiration_time(expiration_hhmm, now_strategy, tf_minutes, strategy_tz)

        expiration_broker_display = expiration_pd.tz_convert(self.broker_tz)

        if expiration_broker_display <= now_broker_display:
            logger.warning(f"ExpPast hhmm={expiration_hhmm} | tf_min={tf_minutes} | tz={strategy_tz.zone} | now={now_broker_display.strftime('%Y-%m-%d %H:%M:%S %Z')} | exp={expiration_broker_display.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return None

        return self._convert_broker_display_to_utc(expiration_broker_display)

    def _handle_dual_fills(self, position_groups: dict[tuple[str, int], dict[str, list[Any]]]) -> set[tuple[str, int]]:
        """Detect and close dual-filled bracket positions."""
        dual_fill_keys = set()

        for (symbol, magic), group in position_groups.items():
            buy_positions = group['BUY']
            sell_positions = group['SELL']

            if buy_positions and sell_positions:
                logger.warning(f"DualFill sym={symbol} | m={magic} | buy={len(buy_positions)} | sell={len(sell_positions)}")

                all_tickets = [p.ticket for p in buy_positions + sell_positions]
                close_results = self.close_positions(tickets=all_tickets)
                successful_closes = sum(1 for success, _ in close_results.values() if success)

                if successful_closes > 0:
                    logger.info(f"DualFillResolved sym={symbol} | m={magic} | closed={successful_closes}/{len(all_tickets)}")
                else:
                    logger.error(f"DualFillCloseFail sym={symbol} | m={magic}")

                dual_fill_keys.add((symbol, magic))

        return dual_fill_keys

    def cancel_bracket_orders(self, symbols: list[str], magics: list[int]) -> dict[str, dict[int, int]]:
        """
        Cancel opposite bracket orders when one side fills.
        DUAL-FILL PROTECTION: If both BUY_STOP and SELL_STOP fill simultaneously, close both positions immediately.
        """
        positions = mt.positions_get()
        orders = mt.orders_get()

        if not positions or not orders:
            return {s: {m: 0 for m in magics} for s in symbols}

        symbol_set = set(symbols)
        magic_set = set(magics)

        # Group positions by (symbol, magic) to detect dual fills
        position_groups = defaultdict(lambda: {'BUY': [], 'SELL': []})

        for p in positions:
            if p.symbol in symbol_set and p.magic in magic_set:
                key = (p.symbol, p.magic)
                pos_type = 'BUY' if p.type == mt.POSITION_TYPE_BUY else 'SELL'
                position_groups[key][pos_type].append(p)

        # Handle dual fills
        dual_fill_keys = self._handle_dual_fills(position_groups)

        # Cancel remaining pending orders for dual-fill pairs
        dual_fill_cancelled = sum(1 for order in orders if (order.symbol, order.magic) in dual_fill_keys and order.symbol in symbol_set and order.magic in magic_set and self._cancel_order(order.ticket))

        if dual_fill_cancelled > 0:
            logger.info(f"DualFillCancel n={dual_fill_cancelled}")

        # Standard OCO cancellation for single-filled pairs
        position_map = {
            (symbol, magic): mt.POSITION_TYPE_BUY if group['BUY'] else mt.POSITION_TYPE_SELL
            for (symbol, magic), group in position_groups.items()
            if (symbol, magic) not in dual_fill_keys and (group['BUY'] or group['SELL'])
        }

        if not position_map:
            return {s: {m: 0 for m in magics} for s in symbols}

        results = {s: {m: 0 for m in magics} for s in symbols}

        for order in orders:
            if (order.symbol, order.magic) in dual_fill_keys:
                continue

            if order.symbol not in symbol_set or order.magic not in magic_set:
                continue

            position_key = (order.symbol, order.magic)
            if position_key not in position_map:
                continue

            position_type = position_map[position_key]

            # Cancel opposite order (OCO logic)
            should_cancel = (
                (position_type == mt.POSITION_TYPE_BUY and order.type == mt.ORDER_TYPE_SELL_STOP) or
                (position_type == mt.POSITION_TYPE_SELL and order.type == mt.ORDER_TYPE_BUY_STOP)
            )

            if should_cancel and self._cancel_order(order.ticket):
                results[order.symbol][order.magic] += 1
                logger.debug(f"OCOCancel sym={order.symbol} | m={order.magic} | ot={order.type} | t={order.ticket} | filled={'B' if position_type == mt.POSITION_TYPE_BUY else 'S'}")

        return results

    def _cancel_order(self, ticket: int) -> bool:
        """Cancel a pending order."""
        request = {"action": int(TradeAction.REMOVE), "order": ticket}
        result = self._order_send_with_retry(request)
        return bool(result and int(result.retcode) == self._retcode_done)

    def _rank_deal_match(self, deal: Any, target_type: int, close_volume: float, volume_tolerance: float, ticket: int, exit_entry_code: int | None) -> tuple[int, int, int, int] | None:
        """Calculate ranking tuple for deal matching quality."""
        deal_ticket = getattr(deal, "ticket", None)
        if deal_ticket in (None, 0):
            return None

        if getattr(deal, "type", None) != target_type:
            return None

        deal_volume = float(getattr(deal, "volume", 0.0) or 0.0)
        if abs(deal_volume - close_volume) > volume_tolerance:
            return None

        position_match = int(getattr(deal, "position_id", None) == ticket)
        entry_value = getattr(deal, "entry", None)
        entry_match = int(exit_entry_code is not None and entry_value == exit_entry_code)

        deal_time_msc = getattr(deal, "time_msc", None)
        if not deal_time_msc:
            deal_time_msc = int(getattr(deal, "time", 0) or 0) * 1000

        return (position_match, entry_match, int(deal_time_msc), int(deal_ticket))

    def _recover_close_deal_id(self, ticket: int, close_type: OrderType | int, close_volume: float, volume_step: float, max_retries: int = 3) -> int | None:
        """
        Recover missing deal_id after successful close execution.

        MT5 occasionally returns retcode DONE with result.deal=0/None for partial closes.
        This fallback runs only in that path to avoid extra MT5 history calls on normal exits.
        """
        target_type = int(close_type)
        volume_tolerance = max(float(volume_step) / 2.0, 1e-9)
        exit_entry_code = getattr(mt, "DEAL_ENTRY_OUT", None)

        for attempt in range(max_retries):
            deals = mt.history_deals_get(position=ticket)
            if deals:
                best_deal = None
                best_rank: tuple[int, int, int, int] | None = None

                for deal in deals:
                    rank = self._rank_deal_match(deal, target_type, close_volume, volume_tolerance, ticket, exit_entry_code)

                    if rank is not None and (best_rank is None or rank > best_rank):
                        best_rank = rank
                        best_deal = deal

                if best_deal is not None:
                    return int(getattr(best_deal, "ticket"))

            if attempt < max_retries - 1:
                time.sleep(0.02 * (attempt + 1))

        return None

    def _normalize_close_volume(self, close_volume_raw: float, symbol_spec: SymbolSpec, portion: float | None, position_volume: float, symbol: str) -> float:
        """Normalize close volume to symbol's volume step and valid range."""
        volume_step = symbol_spec.volume_step
        volume_min = symbol_spec.volume_min
        volume_max = symbol_spec.volume_max
        close_volume = round(close_volume_raw / volume_step) * volume_step

        if close_volume <= volume_min:
            if portion is not None:
                logger.warning(f"CloseVolAdj sym={symbol} | req={close_volume_raw:.4f} | portion={portion:.4f} | pos={position_volume:.4f} | min={volume_min:.4f} | action=use_min")
            close_volume = volume_min

        return min(close_volume, volume_max)

    def close_positions(
        self,
        tickets: list[int],
        portions: list[float] | None = None,
        preloaded_positions: dict[int, Any] | None = None,
    ) -> dict[int, tuple[bool, int | None]]:
        """Close positions with optional partial closing. Returns {ticket: (success, deal_id)}."""
        results = {}
        if portions is None:
            portions = [None] * len(tickets)
        if len(portions) != len(tickets):
            return {ticket: (False, None) for ticket in tickets}

        position_by_ticket: dict[int, Any] = dict(preloaded_positions or {})
        missing_tickets = [ticket for ticket in tickets if ticket not in position_by_ticket]
        if missing_tickets:
            if len(missing_tickets) == 1:
                position_rows = mt.positions_get(ticket=missing_tickets[0]) or []
                if position_rows:
                    position_by_ticket[missing_tickets[0]] = position_rows[0]
            else:
                all_positions = mt.positions_get() or []
                missing_ticket_set = set(missing_tickets)
                for position in all_positions:
                    if position.ticket in missing_ticket_set:
                        position_by_ticket[position.ticket] = position

        tick_cache: dict[str, Any] = {}
        for ticket, portion in zip(tickets, portions):
            pos = position_by_ticket.get(ticket)
            if pos is None:
                results[ticket] = (False, None)
                continue

            symbol = pos.symbol
            symbol_spec, filling_mode = self._get_cached_symbol_spec(symbol)

            position_volume = pos.volume
            close_volume_raw = pos.volume if portion is None else pos.volume * portion

            close_volume = self._normalize_close_volume(close_volume_raw, symbol_spec, portion, position_volume, symbol)

            tick = tick_cache.get(symbol)
            if tick is None:
                tick = mt.symbol_info_tick(symbol)
                if tick is not None:
                    tick_cache[symbol] = tick
            if tick is None:
                logger.error(f"CloseFail sym={symbol} | t={ticket} | reason=tick_unavailable")
                results[ticket] = (False, None)
                continue

            close_type = OrderType.SELL if pos.type == mt.POSITION_TYPE_BUY else OrderType.BUY
            close_price = tick.bid if close_type == OrderType.SELL else tick.ask

            request = {
                "action": int(TradeAction.DEAL),
                "symbol": symbol,
                "volume": float(close_volume),
                "type": int(close_type),
                "position": ticket,
                "price": float(close_price),
                "magic": pos.magic,
                "comment": f"Close {portion*100 if portion else 100:.0f}%",
                "type_filling": int(filling_mode),
                "type_time": int(TimeInForce.GTC)
            }

            result = self._order_send_with_retry(request)
            is_valid, error_msg = self._validate_order_result(result)

            if not is_valid:
                logger.error(f"CloseFail t={ticket} | err={error_msg}")
                results[ticket] = (False, None)
                continue

            deal_id = getattr(result, "deal", None)
            if deal_id in (0, None):
                deal_id = self._recover_close_deal_id(
                    ticket=ticket,
                    close_type=close_type,
                    close_volume=float(close_volume),
                    volume_step=float(symbol_spec.volume_step),
                )
                if deal_id is None:
                    logger.warning(f"CloseWarn sym={symbol} | t={ticket} | reason=deal_id_unavailable")
                else:
                    logger.info(f"CloseDealRecovered sym={symbol} | t={ticket} | deal={deal_id}")

            logger.info(f"CloseOK t={ticket} | vol={close_volume:.2f} | deal={deal_id}")
            results[ticket] = (True, deal_id)

        successful = sum(1 for success, _ in results.values() if success)
        if successful > 0:
            logger.info(f"CloseBatch ok={successful}/{len(tickets)}")

        return results

    def modify_position_sl_tp(self, ticket: int, sl: float | None, tp: float | None) -> bool:
        """Modify stop loss and take profit for a position."""
        position = mt.positions_get(ticket=ticket)
        if not position:
            logger.error(f"ModFail t={ticket} | reason=position_not_found")
            return False

        pos = position[0]
        symbol_spec, _ = self._get_cached_symbol_spec(pos.symbol)

        prices = self._round_prices(symbol_spec.digits, sl=sl, tp=tp)
        final_sl = prices["sl"] if sl is not None else pos.sl
        final_tp = prices["tp"] if tp is not None else pos.tp

        if final_sl == pos.sl and final_tp == pos.tp:
            logger.debug(f"ModSkip t={ticket} | reason=already_set")
            return True

        request = {
            "action": int(TradeAction.SLTP),
            "position": ticket,
            "symbol": pos.symbol,
            "sl": final_sl,
            "tp": final_tp,
        }

        result = self._order_send_with_retry(
            request,
            success_codes={self._retcode_done, self._retcode_no_changes}
        )

        if result is None:
            logger.error(f"ModFail t={ticket} | reason=no_response")
            return False

        if int(result.retcode) not in {self._retcode_done, self._retcode_no_changes}:
            logger.error(f"ModFail t={ticket} | err={result.comment}")
            return False

        if int(result.retcode) == self._retcode_no_changes:
            logger.debug(f"ModSkip t={ticket} | reason=mt5_no_changes")
            return True

        logger.info(f"ModOK t={ticket} | sl={final_sl} | tp={final_tp}")
        return True
