"""Position and state lifecycle mixin for StrategyRunner."""

import logging
import time
from typing import Any
import MetaTrader5 as mt

from src.trading_system.config.risk_policies import SYSTEM_TIMINGS
from src.trading_system.core.types import PositionSnapshot, normalize_position
from src.trading_system.utils.logging_utils import format_price_display

logger = logging.getLogger(__name__)

POSITION_CACHE_STALENESS_THRESHOLD_SECONDS = SYSTEM_TIMINGS['cache_staleness_threshold']


class RunnerPositionMixin:
    """Position tracking, pending resolution, and counter helpers."""

    def _get_composite_key(self) -> tuple[str, int]:
        """Return composite key (symbol, magic) for pending ticket lookup."""
        return (self.symbol, self.magic_number)

    def _create_position_snapshot(self, position: Any) -> dict[str, Any]:
        """Create standardized snapshot from a position dict or MT5 position object."""
        is_dict = isinstance(position, dict)

        def required(field: str) -> Any:
            return position[field] if is_dict else getattr(position, field)

        def optional(field: str, default: Any) -> Any:
            return position.get(field, default) if is_dict else getattr(position, field, default)

        return {
            'ticket': required('ticket'),
            'type': required('type'),
            'volume': required('volume'),
            'profit': required('profit'),
            'swap': optional('swap', 0.0),
            'tp': required('tp'),
            'sl': required('sl'),
            'magic': optional('magic', self.magic_number),
            'symbol': required('symbol'),
            'price_open': required('price_open'),
            'time': optional('time', int(time.time())),
        }

    def _refresh_tracked_position_snapshots(self, positions: list[dict[str, Any]]) -> None:
        """Refresh tracked metadata snapshots from already-fetched live positions."""
        if not positions:
            return

        with self.position_state_lock:
            for pos_dict in positions:
                ticket = pos_dict['ticket']
                trade_id = self.ticket_to_trade_id.get(ticket)
                if trade_id is None:
                    continue

                metadata = self.entry_metadata.get(trade_id)
                if metadata is None:
                    continue

                metadata['position_snapshot'] = self._create_position_snapshot(pos_dict)

    def _patch_tracked_snapshot_levels(self, ticket: int, new_sl: float | None, new_tp: float | None) -> None:
        """Patch tracked snapshot SL/TP immediately after successful modify."""
        if new_sl is None and new_tp is None:
            return

        with self.position_state_lock:
            trade_id = self.ticket_to_trade_id.get(ticket)
            if trade_id is None:
                return

            metadata = self.entry_metadata.get(trade_id)
            if metadata is None:
                return

            snapshot = metadata.get('position_snapshot')
            if snapshot is None:
                return

            if new_sl is not None:
                snapshot['sl'] = new_sl
            if new_tp is not None:
                snapshot['tp'] = new_tp

    def _filter_strategy_positions(self, positions) -> list[dict]:
        """Filter positions by magic number and symbol (single pass)."""
        return [
            normalize_position(p) for p in positions
            if p.magic == self.magic_number and p.symbol == self.symbol
        ]

    def _seed_startup_position_tracking(
        self,
        trade_id: int,
        pos_dict: dict[str, Any],
        submission_time: float,
        expected_entry_price: float | None = None,
        opening_sl: float | None = None,
        volume_multiplier: float | None = None,
    ) -> None:
        """Seed startup position tracking maps for a known open ticket."""
        ticket = pos_dict['ticket']
        self.known_positions.add(ticket)
        self.entry_metadata[trade_id] = {
            'expected_entry_price': (
                expected_entry_price if expected_entry_price is not None else pos_dict['price_open']
            ),
            'opening_sl': opening_sl if opening_sl is not None else pos_dict['sl'],
            'submission_time': submission_time,
            'volume_multiplier': volume_multiplier,
            'ticket': ticket,
            'entry_request': None,
            'position_snapshot': self._create_position_snapshot(pos_dict),
        }
        self.ticket_to_trade_id[ticket] = trade_id

    def _init_known_positions(self) -> None:
        """Seed known_positions with current MT5 positions for this strategy."""
        positions = self._get_strategy_positions(use_cache=True, include_unknown=True)
        if positions is None:
            raise RuntimeError(f"{self.strategy_name}: Cannot initialize - MT5 API failure")
        if not positions:
            self.local_position_count = 0
            logger.debug(f"{self.strategy_name:<9}: Initialized with 0 positions")
            return

        position_count = len(positions)
        tracked_tickets = [pos['ticket'] for pos in positions]
        current_time = time.time()
        try:
            reconciled_map = self.trade_logger.get_open_trades_by_ticket_last_three(tracked_tickets)
        except Exception:
            logger.error(f"{self.strategy_name:<9}: Startup reconciliation failed - falling back to fresh fill logging for all startup positions", exc_info=True)
            reconciled_map = {}

        reconciled_count = 0
        missing_positions: list[dict[str, Any]] = []

        for pos_dict in positions:
            ticket = pos_dict['ticket']
            reconciled = reconciled_map.get(ticket)
            if reconciled is None:
                missing_positions.append(pos_dict)
                continue

            trade_id = reconciled['trade_id']
            self._seed_startup_position_tracking(
                trade_id=trade_id,
                pos_dict=pos_dict,
                submission_time=current_time,
                expected_entry_price=reconciled.get('expected_entry_price'),
                opening_sl=reconciled.get('opening_sl'),
                volume_multiplier=reconciled.get('volume_multiplier'),
            )
            reconciled_count += 1

        new_trade_ids: list[int] = []
        if missing_positions:
            new_trade_ids = self.trade_id_manager.generate_batch(len(missing_positions))
            for pos_dict, trade_id in zip(missing_positions, new_trade_ids):
                self._seed_startup_position_tracking(
                    trade_id=trade_id,
                    pos_dict=pos_dict,
                    submission_time=current_time,
                )
                metadata = self.entry_metadata[trade_id]
                position_obj = PositionSnapshot(**pos_dict)
                fill_data = self._build_fill_data(
                    trade_id=trade_id,
                    position=position_obj,
                    expected_entry_price=metadata['expected_entry_price'],
                    opening_sl=metadata['opening_sl'],
                    fill_time_ms=None,
                    volume_multiplier=metadata.get('volume_multiplier')
                )
                self.trade_logger.log_fill(fill_data)

        self.local_position_count = position_count
        ticket_str = (", ".join(map(str, tracked_tickets)) if position_count <= 5 else f"{tracked_tickets[0]}, {tracked_tickets[1]}, ... ({position_count} total)")
        new_ids_str = (f"{new_trade_ids[0]}..{new_trade_ids[-1]}" if new_trade_ids else "none")
        logger.info(f"{self.strategy_name:<9}: InitPos n={position_count} | rec={reconciled_count} | new={len(missing_positions)} | ids={new_ids_str} | t=[{ticket_str}]")

    def _get_positions_from_cache(self) -> list[dict] | None:
        """Read positions from shared cache with staleness check."""
        try:
            with self.position_cache_lock:
                cache_timestamp = self.shared_state.get('position_cache_timestamp', 0)
                cache_age = time.time() - cache_timestamp

                if cache_age > POSITION_CACHE_STALENESS_THRESHOLD_SECONDS:
                    logger.debug(f"{self.strategy_name:<9}: Cache stale ({cache_age:.1f}s > {POSITION_CACHE_STALENESS_THRESHOLD_SECONDS:.0f}s), using direct MT5 query")
                    return None

                cache_ref = self.shared_state.get('position_cache', {})
                positions = [pos for pos in cache_ref.values() if pos.get('magic') == self.magic_number and pos.get('symbol') == self.symbol]

            logger.debug(f"{self.strategy_name:<9}: Cache hit - {len(positions)} positions (age: {cache_age:.1f}s)")
            return positions

        except Exception as error:
            logger.warning(f"{self.strategy_name:<9}: Cache read failed: {error}, using direct query")
            return None

    def _invalidate_cache_for_ticket(self, ticket: int) -> None:
        """Log position modification (actual cache refresh happens in orchestrator heartbeat)."""
        logger.debug(f"{self.strategy_name:<9}: Position {ticket} modified (cache will refresh on next heartbeat)")

    def _get_strategy_positions(self, use_cache: bool = True, include_unknown: bool = False) -> list[dict[str, Any | None]]:
        """Centralized position query with cache-first fallback to MT5."""
        positions = self._get_positions_from_cache() if use_cache else None
        if positions is None:
            all_positions = mt.positions_get()
            if all_positions is None:
                logger.error(f"{self.strategy_name:<9}: MT5 positions_get() failed: {mt.last_error()}")
                return None
            if not all_positions:
                return []
            positions = self._filter_strategy_positions(all_positions)

        if include_unknown:
            return positions

        with self.position_state_lock:
            return [pos for pos in positions if pos['ticket'] in self.known_positions]

    def _get_positions_with_fallback(self, include_unknown: bool = True) -> list[dict[str, Any]]:
        """Get positions with cache fallback, raising on MT5 failure."""
        positions = self._get_strategy_positions(use_cache=True, include_unknown=include_unknown)
        if positions is None:
            positions = self._get_strategy_positions(use_cache=False, include_unknown=include_unknown)
            if positions is None:
                raise RuntimeError(f"MT5 API critical failure: {mt.last_error()}")
        return positions

    def _resolve_entry_prices(self, ticket: int, pos_dict: dict[str, Any]) -> tuple[int | None, float, float, float]:
        """Resolve entry prices and SL from metadata using ticket reverse lookup."""
        trade_id = self.ticket_to_trade_id.get(ticket)
        if trade_id is None:
            logger.error(f"{self.strategy_name:<9}: Orphaned position {ticket} - no metadata found")
            return None, pos_dict['price_open'], pos_dict['sl'], pos_dict['price_open']

        metadata = self.entry_metadata[trade_id]
        entry_request = metadata.get('entry_request')
        entry_price = pos_dict['price_open']
        if entry_request is None:
            expected_entry_price = metadata.get('expected_entry_price', entry_price)
            opening_sl = metadata.get('opening_sl', pos_dict['sl'])
            return trade_id, expected_entry_price, opening_sl, entry_price

        if entry_request.order_type == 'bracket':
            is_buy = pos_dict['type'] == 0
            expected_entry_price = entry_request.buy_stop if is_buy else entry_request.sell_stop
            opening_sl = entry_request.buy_sl if is_buy else entry_request.sell_sl
        else:
            expected_entry_price = entry_request.entry_price or entry_price
            opening_sl = entry_request.sl
        return trade_id, expected_entry_price, opening_sl, entry_price

    def _register_pending_trade(self, trade_id: int, pending_ticket: Any) -> None:
        """Register pending trade in all lookup structures."""
        self.pending_tickets[trade_id] = pending_ticket
        self.pending_by_key.setdefault(self._get_composite_key(), []).append(trade_id)
        if pending_ticket.order_type == 'standard' and hasattr(pending_ticket, 'ticket'):
            self.pending_by_ticket[pending_ticket.ticket] = trade_id

    def _store_entry_metadata_bracket(
        self,
        trade_id: int,
        entry_request: Any,
        volume_multiplier: float | None,
        submission_time: float,
        buy_ticket: int,
        sell_ticket: int,
    ) -> None:
        """Store metadata for bracket orders."""
        with self.position_state_lock:
            self.ticket_to_trade_id.update({buy_ticket: trade_id, sell_ticket: trade_id})
            self.entry_metadata[trade_id] = {
                'entry_request': entry_request,
                'submission_time': submission_time,
                'volume_multiplier': volume_multiplier,
                'ticket': None,
                'expected_buy_entry': entry_request.buy_stop,
                'expected_sell_entry': entry_request.sell_stop,
                'buy_sl': entry_request.buy_sl,
                'sell_sl': entry_request.sell_sl,
                'opening_sl': None,
            }
            self._register_pending_trade(
                trade_id,
                self.pending_ticket_cls(
                    order_type='bracket',
                    symbol=self.symbol,
                    magic=self.magic_number,
                    submission_time=submission_time,
                    buy_order_ticket=buy_ticket,
                    sell_order_ticket=sell_ticket,
                    expected_volume=entry_request.volume
                )
            )

    def _store_entry_metadata_standard(
        self,
        trade_id: int,
        ticket: int,
        entry_request: Any,
        volume_multiplier: float | None,
        submission_time: float,
        expected_entry_price: float,
        opening_sl: float,
    ) -> None:
        """Store metadata for standard orders."""
        with self.position_state_lock:
            self.entry_metadata[trade_id] = {
                'entry_request': entry_request,
                'submission_time': submission_time,
                'volume_multiplier': volume_multiplier,
                'ticket': ticket,
                'expected_entry_price': expected_entry_price,
                'opening_sl': opening_sl,
            }
            self.ticket_to_trade_id[ticket] = trade_id
            self._register_pending_trade(
                trade_id,
                self.pending_ticket_cls(
                    order_type='standard',
                    symbol=self.symbol,
                    magic=self.magic_number,
                    submission_time=submission_time,
                    ticket=ticket
                )
            )

    def _cleanup_pending(self, trade_id: int, composite_key: tuple[str, int]) -> None:
        """Remove pending tracking entries for a resolved trade_id."""
        with self.position_state_lock:
            pending_ticket = self.pending_tickets.pop(trade_id, None)
            if pending_ticket and pending_ticket.order_type == 'standard':
                self.pending_by_ticket.pop(getattr(pending_ticket, 'ticket', None), None)
            if composite_key in self.pending_by_key:
                try:
                    self.pending_by_key[composite_key].remove(trade_id)
                except ValueError:
                    pass
                if not self.pending_by_key[composite_key]:
                    del self.pending_by_key[composite_key]

    def _handle_closed_positions(self, closed_tickets: list[int]) -> None:
        """Handle positions that closed externally (SL/TP hit or manual close)."""
        close_logs = []
        cleanup_data = []

        with self.position_state_lock:
            for ticket in closed_tickets:
                trade_id = self.ticket_to_trade_id.get(ticket)
                metadata = self.entry_metadata.get(trade_id) if trade_id is not None else None
                snapshot = metadata.get('position_snapshot') if metadata is not None else None
                was_known = ticket in self.known_positions

                if trade_id is not None and metadata is not None and snapshot is not None:
                    close_logs.append((trade_id, ticket, snapshot, metadata))
                elif ticket in self.known_positions or trade_id is not None:
                    logger.warning(
                        f"{self.strategy_name}:{self.symbol} | "
                        f"EXTERNAL CLOSE - Ticket {ticket} missing metadata/snapshot | "
                        "Cleaning stale tracking only"
                    )

                cleanup_data.append((ticket, trade_id, was_known))

            closed_count = 0
            for ticket, trade_id, was_known in cleanup_data:
                if trade_id is not None:
                    self.entry_metadata.pop(trade_id, None)
                self.ticket_to_trade_id.pop(ticket, None)
                self.known_positions.discard(ticket)
                if was_known:
                    closed_count += 1

            self.local_position_count = max(0, self.local_position_count - closed_count)

        for trade_id, ticket, snapshot, metadata in close_logs:
            position_obj = PositionSnapshot(**snapshot)
            close_data = self._build_close_data(
                trade_id=trade_id,
                position=position_obj,
                expected_exit_price=None,
                opening_sl=metadata.get('opening_sl'),
                exit_trigger='EXTERNAL_CLOSE',
                entry_price=snapshot['price_open'],
                expected_entry_price=metadata.get('expected_entry_price'),
            )
            self.trade_logger.log_close(close_data)
            logger.info(
                f"{self.strategy_name}: EXTERNAL CLOSE | trade_id={trade_id} | ticket={ticket} | "
                # f"Logged from snapshot (SL/TP hit or manual)"
            )

        global_count = self._atomic_decrement_global_positions(closed_count, "external_close")

        if closed_count:
            logger.info(f"{self.strategy_name:<9}: PosClosedDetect | n={len(closed_tickets)} | local= {self.local_position_count}/{self.global_risk_policy['max_total_positions']} | global= {global_count}/{self.global_risk_policy['max_total_positions']}")
        elif closed_tickets:
            logger.warning(f"{self.strategy_name}:{self.symbol} | Detected {len(closed_tickets)} closed tickets but no tracked positions to decrement")

    def _handle_new_fill(self, pos_dict: dict[str, Any]) -> None:
        """Handle new position fill with direct mapping, lazy resolution, and orphan fallback."""
        ticket = pos_dict['ticket']
        trade_id = self.ticket_to_trade_id.get(ticket) or self._resolve_pending_ticket(pos_dict)
        if trade_id is None:
            trade_id = self._handle_orphaned_fill(ticket, pos_dict)
            if trade_id is None:
                logger.error(f"{self.strategy_name:<9}: Failed to handle orphan {ticket} - skipping")
                return

        metadata = self.entry_metadata.get(trade_id)
        if metadata is None:
            logger.error(f"{self.strategy_name:<9}: Missing metadata for trade_id {trade_id} - skipping")
            return

        with self.position_state_lock:
            self.local_position_count += 1
            self.known_positions.add(ticket)

        new_trades = self.atomic_increment_trade()

        position_snapshot = self._create_position_snapshot(pos_dict)
        metadata['position_snapshot'] = position_snapshot
        metadata['ticket'] = ticket

        if 'expected_entry_price' not in metadata and 'expected_buy_entry' in metadata:
            is_buy = pos_dict['type'] == 0
            metadata['expected_entry_price'] = metadata['expected_buy_entry' if is_buy else 'expected_sell_entry']
            metadata['opening_sl'] = metadata['buy_sl' if is_buy else 'sell_sl']

        submission_time = metadata.get('submission_time')
        fill_latency_ms = (time.time() - submission_time) * 1000 if submission_time else None

        position_obj = PositionSnapshot(**pos_dict)
        fill_data = self._build_fill_data(
            trade_id=trade_id,
            position=position_obj,
            expected_entry_price=metadata['expected_entry_price'],
            opening_sl=metadata['opening_sl'],
            fill_time_ms=fill_latency_ms,
            volume_multiplier=metadata.get('volume_multiplier')
        )
        self.trade_logger.log_fill(fill_data)
        fill_price_display = format_price_display(pos_dict['price_open'])
        logger.info(f"{self.strategy_name:<9}: Fill id={trade_id} | t={ticket} | {'B' if pos_dict['type'] == 0 else 'S'} {pos_dict['volume']:.2f}@{fill_price_display} | pos={self.local_position_count}/{self.global_risk_policy['max_total_positions']} | tr={new_trades}/{self.global_risk_policy['max_daily_trades']}")

    def _handle_orphaned_fill(self, ticket: int, pos_dict: dict[str, Any]) -> int | None:
        """Helper for true orphans: Generate new trade_id with fallback metadata."""
        logger.debug(f"{self.strategy_name:<9}: Orphaned position detected | Ticket {ticket} | Generating fallback trade_id")
        trade_id = self._generate_trade_id()
        if trade_id is None:
            return None

        with self.position_state_lock:
            self.entry_metadata[trade_id] = {
                'expected_entry_price': pos_dict['price_open'],
                'opening_sl': pos_dict['sl'],
                'submission_time': time.time(),
                'volume_multiplier': None,
                'ticket': ticket,
            }
            self.ticket_to_trade_id[ticket] = trade_id
        return trade_id

    def _matches_bracket_conditions(
        self,
        pos_dict: dict[str, Any],
        conditions: Any,
        submission_time: float,
    ) -> bool:
        """Check if position matches bracket order conditions."""
        expected_volume = conditions.expected_volume
        if expected_volume is None or expected_volume <= 0:
            return False
        actual_volume = pos_dict['volume']
        volume_match = abs(actual_volume - expected_volume) / expected_volume < 0.01
        timing_match = time.time() - submission_time < 10.0
        return volume_match and timing_match

    def _resolve_pending_ticket(self, pos_dict: dict[str, Any]) -> int | None:
        """Resolve ticket -> trade_id mapping from pending tickets using O(1) ticket index or composite key."""
        ticket = pos_dict['ticket']
        direct_match_trade_id: int | None = None

        with self.position_state_lock:
            composite_key = self._get_composite_key()
            direct_match_trade_id = self.pending_by_ticket.get(ticket)
            candidate_trade_ids = tuple(self.pending_by_key.get(composite_key, ()))
            pending_snapshot = {trade_id: self.pending_tickets.get(trade_id) for trade_id in candidate_trade_ids}

        if direct_match_trade_id is not None:
            return self._finalize_standard_resolution(direct_match_trade_id, composite_key)

        if not candidate_trade_ids:
            return None

        for pending_trade_id in candidate_trade_ids:
            conditions = pending_snapshot.get(pending_trade_id)
            if conditions is None or conditions.symbol != pos_dict['symbol']:
                continue
            if conditions.magic != pos_dict.get('magic', self.magic_number):
                continue

            if conditions.order_type == 'bracket':
                if self._matches_bracket_conditions(pos_dict, conditions, conditions.submission_time):
                    return self._finalize_bracket_resolution(
                        pending_trade_id,
                        pos_dict,
                        conditions,
                        composite_key,
                    )

        return None

    def _finalize_bracket_resolution(self, pending_trade_id: int, pos_dict: dict[str, Any], conditions: Any, composite_key: tuple[str, int]) -> int:
        """Finalize bracket order resolution, cancel opposite side, and update metadata."""
        ticket = pos_dict['ticket']
        is_buy = pos_dict['type'] == 0
        with self.position_state_lock:
            self.ticket_to_trade_id[ticket] = pending_trade_id
            metadata = self.entry_metadata[pending_trade_id]
            metadata['ticket'] = ticket
            entry_request = metadata['entry_request']
            metadata['expected_entry_price'] = entry_request.buy_stop if is_buy else entry_request.sell_stop
            metadata['opening_sl'] = entry_request.buy_sl if is_buy else entry_request.sell_sl

        opposite_ticket = conditions.sell_order_ticket if is_buy else conditions.buy_order_ticket
        if opposite_ticket and self.executor._cancel_order(opposite_ticket):
            logger.info(f"{self.strategy_name:<9}: Canceled opposite bracket order {opposite_ticket}")

        self._cleanup_pending(pending_trade_id, composite_key)
        logger.debug(f"{self.strategy_name:<9}: Resolved pending ticket | Trade ID {pending_trade_id} -> Ticket {ticket} | Bracket {'BUY' if is_buy else 'SELL'} side filled")
        return pending_trade_id

    def _finalize_standard_resolution(self, pending_trade_id: int, composite_key: tuple[str, int]) -> int:
        """Finalize standard order resolution."""
        self._cleanup_pending(pending_trade_id, composite_key)
        logger.debug(f"{self.strategy_name:<9}: Resolved pending ticket | Trade ID {pending_trade_id} -> Ticket | Standard order")
        return pending_trade_id

    def _requery_position(self, ticket: int) -> Any | None:
        """Re-query position from MT5 after partial close."""
        try:
            positions = mt.positions_get(ticket=ticket)
            return positions[0] if positions and len(positions) > 0 else None
        except Exception as error:
            logger.error(f"{self.strategy_name:<9}: Position re-query failed: {error}", exc_info=True)
            return None

    def atomic_increment_trade(self) -> int:
        """Atomically increment global trade counter."""
        with self.global_trade_count.get_lock():
            self.global_trade_count.value += 1
            return self.global_trade_count.value

    def _atomic_decrement_global_positions(self, count: int, reason: str) -> int:
        """Atomically decrement global position counter with underflow protection."""
        if count <= 0:
            return self.global_position_count.value

        with self.global_position_count.get_lock():
            current_count = self.global_position_count.value
            if current_count < count:
                logger.warning(
                    f"{self.strategy_name:<9}: Global position counter underflow prevented | "
                    f"Requested: -{count}, Current: {current_count} | Reason: {reason}"
                )
                self.global_position_count.value = 0
            else:
                self.global_position_count.value = current_count - count
            new_count = self.global_position_count.value

        logger.debug(f"{self.strategy_name:<9}: Global position counter decremented | Count: {new_count}/{self.global_risk_policy['max_total_positions']} | Delta: -{count} | Reason: {reason}")
        return new_count
