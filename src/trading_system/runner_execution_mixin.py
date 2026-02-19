"""Entry, exit, modify, and execution logging mixin for StrategyRunner."""

import logging
import time
from typing import Any
import MetaTrader5 as mt
import numpy as np
import pandas as pd

from src.trading_system.core.execution_requests import ExitRequest, ModifyRequest
from src.trading_system.core.trade_logger import CloseData, FillData, PartialCloseData
from src.trading_system.core.types import PositionSnapshot, snapshot_partial_close_position

logger = logging.getLogger(__name__)

TAKE_PROFIT_REASON_TOKENS = ("take_profit", "take profit", "tp")
STOP_LOSS_REASON_TOKENS = ("stop_loss", "stop loss", "sl")


class RunnerExecutionMixin:
    """Execution flow, exit resolution, and trade logging helpers."""

    def _build_fill_data(self, trade_id: int, position: PositionSnapshot, expected_entry_price: float, opening_sl: float, fill_time_ms: float | None = None, volume_multiplier: float | None = None) -> FillData:
        """Build typed fill payload for logger API."""
        return FillData(
            trade_id=trade_id,
            position=position,
            expected_entry_price=expected_entry_price,
            opening_sl=opening_sl,
            strategy_name=self.strategy_name,
            fill_time_ms=fill_time_ms,
            volume_multiplier=volume_multiplier,
        )

    @staticmethod
    def _build_close_data(trade_id: int, position: PositionSnapshot, expected_exit_price: float | None, opening_sl: float, exit_trigger: str, entry_price: float, expected_entry_price: float) -> CloseData:
        """Build typed close payload for logger API."""
        return CloseData(
            trade_id=trade_id,
            position=position,
            expected_exit_price=expected_exit_price,
            opening_sl=opening_sl,
            exit_trigger=exit_trigger,
            entry_price=entry_price,
            expected_entry_price=expected_entry_price,
        )

    def _build_partial_close_data(self, trade_id: int, position: Any, remaining_volume: float, data: Any) -> PartialCloseData:
        """Build typed partial-close payload for logger API."""
        return PartialCloseData(
            trade_id=trade_id,
            position=snapshot_partial_close_position(position),
            closed_volume=data.closed_volume,
            remaining_volume=remaining_volume,
            expected_exit_price=data.expected_exit_price,
            opening_sl=data.opening_sl,
            strategy_name=self.strategy_name,
            exit_trigger=data.exit_trigger,
            entry_price=data.entry_price,
            expected_entry_price=data.expected_entry_price,
            deal_id=data.deal_id,
        )

    def _get_cached_symbol_tick(self, symbol: str) -> Any | None:
        """Fetch symbol tick once per processing cycle and reuse it."""
        if symbol in self._symbol_tick_cache:
            return self._symbol_tick_cache[symbol]
        tick = mt.symbol_info_tick(symbol)
        if tick is None:
            return None
        self._symbol_tick_cache[symbol] = tick
        return tick

    def _log_partial_close_execution(self, data: Any) -> Any | None:
        """Log partial close execution with volume tracking."""
        self._invalidate_cache_for_ticket(data.ticket)
        updated_position = self._requery_position(data.ticket)
        if updated_position is None:
            logger.warning(f"{self.strategy_name:<9}: Partial close query failed for ticket {data.ticket}")
            return None
        remaining_volume = updated_position.volume
        partial_position = {'price_open': data.entry_price, 'sl': data.opening_sl, 'type': updated_position.type}
        trade_id, _, _, _ = self._resolve_entry_prices(data.ticket, partial_position)
        if trade_id is None:
            logger.error(f"{self.strategy_name:<9}: Cannot log partial close - trade_id not found for ticket {data.ticket}")
            return None
        partial_data = self._build_partial_close_data(trade_id=trade_id, position=updated_position, remaining_volume=remaining_volume, data=data)
        self.trade_logger.log_partial_close(partial_data)
        if trade_id in self.entry_metadata:
            self.entry_metadata[trade_id]['position_snapshot'] = self._create_position_snapshot(updated_position)
        logger.info(f"{self.strategy_name:<9}: PARTIAL CLOSE - Trade ID {trade_id} | Ticket {data.ticket} | Closed: {data.closed_volume:.2f} lots | Remaining: {remaining_volume:.2f} lots")
        return updated_position

    def _log_full_close_execution(self, pos_dict: dict[str, Any], data: Any) -> None:
        """Log full close execution and cleanup tracking structures."""
        ticket = pos_dict['ticket']
        trade_id, _, _, _ = self._resolve_entry_prices(ticket, pos_dict)
        if trade_id is None:
            logger.error(f"{self.strategy_name:<9}: Cannot log close - trade_id not found for ticket {ticket}")
        else:
            position_obj = PositionSnapshot(**pos_dict)
            close_data = self._build_close_data(trade_id=trade_id, position=position_obj, expected_exit_price=data.expected_exit_price, opening_sl=data.opening_sl, exit_trigger=data.exit_trigger, entry_price=data.entry_price, expected_entry_price=data.expected_entry_price)
            self.trade_logger.log_close(close_data)
        with self.position_state_lock:
            self.known_positions.discard(ticket)
            self.local_position_count = max(0, self.local_position_count - 1)
            self.ticket_to_trade_id.pop(ticket, None)
            if trade_id is not None:
                self.entry_metadata.pop(trade_id, None)
        global_count = self._atomic_decrement_global_positions(1, "full_close")
        self._invalidate_cache_for_ticket(ticket)
        self._log_close_summary(ticket, trade_id, data.exit_trigger, global_count)

    def _log_close_summary(self, ticket: int, trade_id: int | None, exit_trigger: str, global_count: int) -> None:
        """Log close summary with local/global position counters."""
        max_positions = self.global_risk_policy['max_total_positions']
        if trade_id is None:
            logger.warning(f"{self.strategy_name:<9}: CLOSED - Ticket {ticket} without trade_id metadata | Trigger: {exit_trigger} | Positions {self.local_position_count}/{max_positions} | Global {global_count}/{max_positions}")
            return
        logger.info(f"{self.strategy_name:<9}: CLOSED - Trade ID {trade_id} | Ticket {ticket} | Trigger: {exit_trigger} | Positions {self.local_position_count}/{max_positions} | Global {global_count}/{max_positions}")

    @staticmethod
    def _check_reason_match(reason: str, level: float | None, tokens: tuple[str, ...], source: str) -> tuple[float, str] | None:
        """Return level when exit reason contains one of the expected tokens."""
        if level is None or level == 0.0:
            return None
        if any(token in reason for token in tokens):
            return level, source
        return None

    def _get_current_side_price(self, symbol: str, pos_type: int) -> float | None:
        """Get current bid/ask price for position side."""
        tick = self._get_cached_symbol_tick(symbol)
        if tick is None:
            return None
        if pos_type not in (0, 1):
            raise ValueError(f"Invalid position type for side price: {pos_type}")
        return tick.bid if pos_type == 0 else tick.ask

    def _resolve_expected_exit_price(self, exit_request: ExitRequest, pos_dict: dict[str, Any]) -> tuple[float, str] | None:
        """Resolve expected exit price for strategy-triggered exits."""
        reason = (exit_request.exit_reason or "").lower()
        tp = pos_dict['tp']
        sl = pos_dict['sl']
        symbol = pos_dict['symbol']
        pos_type = pos_dict['type']
        tp_match = self._check_reason_match(reason=reason, level=tp, tokens=TAKE_PROFIT_REASON_TOKENS, source="take_profit_reason")
        if tp_match is not None:
            return tp_match
        sl_match = self._check_reason_match(reason=reason, level=sl, tokens=STOP_LOSS_REASON_TOKENS, source="stop_loss_reason")
        if sl_match is not None:
            return sl_match
        side_price = self._get_current_side_price(symbol, pos_type)
        if side_price is not None:
            return side_price, "tick_price"
        if exit_request.expected_exit_price is not None:
            return exit_request.expected_exit_price, "request_fallback"
        return None

    def _execute_and_log_exit(self, exit_request: ExitRequest, pos_dict: dict[str, Any], exit_context: str = "UNKNOWN") -> None:
        """Unified exit execution and logging for all exit types."""
        ticket = pos_dict['ticket']
        exit_price_result = self._resolve_expected_exit_price(exit_request, pos_dict)
        if exit_price_result is None:
            logger.error(f"{self.strategy_name:<9}: Cannot resolve expected exit price | Ticket {ticket} | Context: {exit_context}")
            return
        expected_exit_price, expected_source = exit_price_result
        logger.debug(f"{self.strategy_name:<9}: Exit expected price resolved | Ticket {ticket} | Source: {expected_source} | Price: {expected_exit_price:.5f}")
        _, expected_entry_price, opening_sl, entry_price = self._resolve_entry_prices(ticket, pos_dict)
        result = self.executor.execute_exit(exit_request)
        if not result.success:
            logger.error(f"{self.strategy_name:<9}: Exit failed - {result.error_message}")
            return
        exit_log_data = self.exit_log_data_cls(ticket=ticket, expected_exit_price=expected_exit_price, exit_trigger=getattr(exit_request, 'exit_reason', exit_context), expected_entry_price=expected_entry_price, opening_sl=opening_sl, entry_price=entry_price, deal_id=result.deal_id)
        if exit_request.portion < 1.0:
            exit_log_data.closed_volume = pos_dict['volume'] * exit_request.portion
            self._log_partial_close_execution(exit_log_data)
        else:
            self._log_full_close_execution(pos_dict, exit_log_data)

    def _apply_meta_labeling(self, data: pd.DataFrame, entry_request: Any) -> tuple[float | None, float]:
        """Apply meta-labeling to adjust position volume."""
        if self.meta_model is None:
            return None, entry_request.volume
        features = self.feature_extractor(data)
        if features.empty:
            logger.warning(f"{self.strategy_name:<9}: Meta-labeling skipped - feature extractor returned no rows")
            return None, entry_request.volume
        features = features.iloc[[-1]]
        volume_multiplier = self.meta_model.predict_proba(features)[0, 1]
        if self.calibration_model is not None:
            volume_multiplier = self.calibration_model.transform(np.array([[volume_multiplier]]))[0]
        if volume_multiplier < self.meta_min_confidence:
            return None, entry_request.volume
        original_volume = entry_request.volume
        adjusted_volume = self.risk_manager.validate_position_size(self.symbol, original_volume * volume_multiplier)
        logger.info(f"{self.strategy_name:<9}: Meta-labeling applied: volume {original_volume:.4f} -> {adjusted_volume:.4f} (multiplier={volume_multiplier:.3f})")
        return volume_multiplier, adjusted_volume

    def _process_entry_signal(self, data: pd.DataFrame) -> None:
        """Generate and submit entry orders via strategy and risk manager."""
        entry_request = self.strategy.generate_entry_signal(data)
        if entry_request is None:
            return
        submission_time = time.time()
        if self.order_type == 'bracket':
            sl_price = entry_request.buy_sl
            entry_price = entry_request.buy_stop
        else:
            sl_price = entry_request.sl
            entry_price = data["Close"].iloc[-1]
        validation = self.risk_manager.validate_trade(strategy_name=self.strategy_name, symbol=self.symbol, expected_price=entry_price, sl_price=sl_price, signal=entry_request.signal)
        if not validation.can_trade:
            logger.warning(f"{self.strategy_name:<9}: Trade rejected - {validation.reason}")
            return
        position_reserved = True
        execution_submitted = False
        try:
            entry_request.volume = validation.volume
            volume_multiplier, adjusted_volume = self._apply_meta_labeling(data, entry_request)
            if volume_multiplier is None and self.meta_model is not None:
                self.risk_manager.release_position_reservation(reason="meta_labeling_rejected")
                position_reserved = False
                logger.info(f"{self.strategy_name:<9}: Trade CANCELLED by meta-labeling: confidence < min={self.meta_min_confidence:.3f}")
                return
            entry_request.volume = adjusted_volume
            result = self.executor.execute_entry(entry_request)
            if not result.success:
                self.risk_manager.release_position_reservation(reason="execution_failed")
                position_reserved = False
                logger.error(f"{self.strategy_name:<9}: Entry failed - {result.error_message}")
                return
            execution_submitted = True
        finally:
            if position_reserved and not execution_submitted:
                self.risk_manager.release_position_reservation(reason="exception_during_execution")
                logger.error(f"{self.strategy_name:<9}: Position reservation released due to exception")
        trade_id = self._generate_trade_id()
        if self.order_type == 'bracket':
            buy_ticket, sell_ticket = result.order_tickets
            self._store_entry_metadata_bracket(trade_id, entry_request, volume_multiplier, submission_time, buy_ticket, sell_ticket)
            signal_str = 'BRACKET'
        else:
            expected_entry_price = entry_request.entry_price or data["Close"].iloc[-1]
            self._store_entry_metadata_standard(trade_id, result.ticket, entry_request, volume_multiplier, submission_time, expected_entry_price, entry_request.sl)
            signal_str = 'BUY' if entry_request.signal == 1 else 'SELL'
        logger.info(f"{self.strategy_name:<9}: ENTRY tid={trade_id} | dir={signal_str} | vol={entry_request.volume:.2f}")

    def _process_modify_signals(self, data: pd.DataFrame) -> None:
        """Generate and execute position modification signals on bar boundaries."""
        positions = self._get_strategy_positions(use_cache=True, include_unknown=False)
        if positions is None:
            error_msg = f"MT5 API critical failure: {mt.last_error()}"
            logger.error(f"{self.strategy_name:<9}: {error_msg}")
            raise RuntimeError(error_msg)
        if not positions:
            return
        for pos_dict in positions:
            request = self.strategy.generate_modify_signal(pos_dict, data)
            if request is None:
                continue
            if isinstance(request, ExitRequest):
                self._execute_and_log_exit(exit_request=request, pos_dict=pos_dict, exit_context="BAR_ALIGNED")
            elif isinstance(request, ModifyRequest):
                self._handle_modify_adjustment(request)
            else:
                logger.warning(f"{self.strategy_name:<9}: Unknown request type from generate_modify_signal: {type(request).__name__}")

    def _handle_modify_adjustment(self, request: ModifyRequest) -> None:
        """Handle SL/TP modification request."""
        result = self.executor.execute_modify(request)
        if result.success:
            self._invalidate_cache_for_ticket(request.ticket)
            self._patch_tracked_snapshot_levels(ticket=request.ticket, new_sl=request.new_sl, new_tp=request.new_tp)
            sl_str = f"{request.new_sl:.5f}" if request.new_sl is not None else "None"
            tp_str = f"{request.new_tp:.5f}" if request.new_tp is not None else "None"
            logger.debug(f"{self.strategy_name:<9}: MODIFIED (BAR-ALIGNED) - Ticket {request.ticket} | SL: {sl_str}, TP: {tp_str} | {request.comment}")
        else:
            logger.error(f"{self.strategy_name:<9}: Modify failed - {result.error_message}")

    def _monitor_exits(self, data: pd.DataFrame) -> None:
        """Monitor positions for exits (every minute)."""
        positions = self._get_strategy_positions(use_cache=False, include_unknown=True)
        if positions is None:
            logger.error(f"{self.strategy_name:<9}: MT5 API failure in _monitor_exits")
            return
        self._refresh_tracked_position_snapshots(positions)
        current_tickets = {pos['ticket'] for pos in positions}
        with self.position_state_lock:
            known_snapshot = set(self.known_positions)
        closed_tickets = [ticket for ticket in known_snapshot if ticket not in current_tickets]
        if closed_tickets:
            self._handle_closed_positions(closed_tickets)
        if not positions or data.empty:
            return
        for pos_dict in positions:
            ticket = pos_dict['ticket']
            if ticket not in known_snapshot:
                self._handle_new_fill(pos_dict)
                known_snapshot.add(ticket)
            self._check_and_execute_exit(pos_dict, data)

    def _check_and_execute_exit(self, pos_dict: dict[str, Any], data: pd.DataFrame) -> None:
        """Generate exit signal and execute if triggered."""
        exit_request = self.strategy.generate_exit_signal(pos_dict, data)
        if exit_request is None:
            return
        self._execute_and_log_exit(exit_request=exit_request, pos_dict=pos_dict, exit_context="SIGNAL_EXIT")

    def _cancel_opposite_bracket_orders(self) -> None:
        """Cancel opposite bracket orders after one side fills."""
        results = self.executor.cancel_bracket_orders(symbols=[self.symbol], magics=[self.magic_number])
        total_cancelled = sum(count for symbol_results in results.values() for count in symbol_results.values())
        if total_cancelled > 0:
            logger.info(f"{self.strategy_name:<9}: Cancelled {total_cancelled} bracket orders")
