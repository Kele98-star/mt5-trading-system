"""OrderExecutor exit-path tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytz

from tests.util import MockDataHandler, ensure_mt5_stub

ensure_mt5_stub()

import MetaTrader5 as mt

from trading_system.config.broker_config import OrderType, TradeAction
from trading_system.core.execution import OrderExecutor
from trading_system.core.execution_requests import EntryRequest, ExitRequest


def _build_executor() -> OrderExecutor:
    """Create executor with lightweight mock data handler."""
    return OrderExecutor(data_handler=MockDataHandler(), broker_tz=pytz.UTC)


def _build_close_position(ticket: int) -> SimpleNamespace:
    """Create minimal position object required by close_positions."""
    return SimpleNamespace(
        ticket=ticket,
        symbol="GER40",
        type=mt.POSITION_TYPE_BUY,
        volume=0.45,
        magic=2001,
    )


def _build_symbol_spec() -> SimpleNamespace:
    """Create minimal symbol spec used for close volume normalization."""
    return SimpleNamespace(volume_step=0.01, volume_min=0.01, volume_max=100.0, digits=1)


def test_close_positions_recovers_deal_id_when_result_missing_deal():
    """Fallback should recover the latest matching close deal for partial exits."""
    executor = _build_executor()
    ticket = 235435404
    position = _build_close_position(ticket)
    symbol_spec = _build_symbol_spec()

    order_result = SimpleNamespace(retcode=mt.TRADE_RETCODE_DONE, deal=0, comment="done")

    older_match = SimpleNamespace(
        ticket=800001,
        type=int(OrderType.SELL),
        volume=0.27,
        position_id=ticket,
        entry=getattr(mt, "DEAL_ENTRY_OUT", None),
        time_msc=1000,
    )
    latest_match = SimpleNamespace(
        ticket=800002,
        type=int(OrderType.SELL),
        volume=0.27,
        position_id=ticket,
        entry=getattr(mt, "DEAL_ENTRY_OUT", None),
        time_msc=2000,
    )

    with (
        patch.object(executor, "_get_cached_symbol_spec", return_value=(symbol_spec, 0)),
        patch.object(executor, "_order_send_with_retry", return_value=order_result),
        patch("trading_system.core.execution.mt.positions_get", return_value=[position]),
        patch(
            "trading_system.core.execution.mt.symbol_info_tick",
            return_value=SimpleNamespace(bid=17000.0, ask=17000.5),
        ),
        patch(
            "trading_system.core.execution.mt.history_deals_get",
            return_value=[older_match, latest_match],
        ) as history_deals_get,
    ):
        results = executor.close_positions(tickets=[ticket], portions=[0.6])

    assert results[ticket] == (True, latest_match.ticket)
    history_deals_get.assert_called_once_with(position=ticket)


def test_close_positions_skips_history_lookup_when_deal_is_present():
    """Normal path should not issue extra history queries when result.deal exists."""
    executor = _build_executor()
    ticket = 235435405
    position = _build_close_position(ticket)
    symbol_spec = _build_symbol_spec()

    order_result = SimpleNamespace(retcode=mt.TRADE_RETCODE_DONE, deal=991122, comment="done")

    with (
        patch.object(executor, "_get_cached_symbol_spec", return_value=(symbol_spec, 0)),
        patch.object(executor, "_order_send_with_retry", return_value=order_result),
        patch("trading_system.core.execution.mt.positions_get", return_value=[position]),
        patch(
            "trading_system.core.execution.mt.symbol_info_tick",
            return_value=SimpleNamespace(bid=17000.0, ask=17000.5),
        ),
        patch("trading_system.core.execution.mt.history_deals_get") as history_deals_get,
    ):
        results = executor.close_positions(tickets=[ticket], portions=[0.6])

    assert results[ticket] == (True, 991122)
    history_deals_get.assert_not_called()


def test_modify_position_sl_tp_skips_mt5_call_when_values_unchanged():
    """Idempotent modify should not call MT5 when SL/TP already match requested values."""
    executor = _build_executor()
    ticket = 235435406
    position = SimpleNamespace(
        ticket=ticket,
        symbol="GER40",
        sl=17000.0,
        tp=17100.0,
    )

    with (
        patch.object(executor, "_get_cached_symbol_spec", return_value=(_build_symbol_spec(), 0)),
        patch("trading_system.core.execution.mt.positions_get", return_value=[position]),
        patch.object(executor, "_order_send_with_retry") as order_send_with_retry,
    ):
        success = executor.modify_position_sl_tp(ticket=ticket, sl=17000.0, tp=17100.0)

    assert success is True
    order_send_with_retry.assert_not_called()


def test_modify_position_sl_tp_treats_no_changes_retcode_as_success():
    """MT5 no-change retcode should be accepted to avoid false modify errors."""
    executor = _build_executor()
    ticket = 235435407
    position = SimpleNamespace(
        ticket=ticket,
        symbol="GER40",
        sl=17000.0,
        tp=17100.0,
    )
    order_result = SimpleNamespace(retcode=10025, comment="No changes")

    with (
        patch.object(executor, "_get_cached_symbol_spec", return_value=(_build_symbol_spec(), 0)),
        patch("trading_system.core.execution.mt.positions_get", return_value=[position]),
        patch.object(executor, "_order_send_with_retry", return_value=order_result) as order_send_with_retry,
    ):
        success = executor.modify_position_sl_tp(ticket=ticket, sl=17000.1, tp=17100.0)

    assert success is True
    _, kwargs = order_send_with_retry.call_args
    assert kwargs["success_codes"] == {executor._retcode_done, executor._retcode_no_changes}


def test_execute_exit_reuses_prefetched_position_without_second_lookup():
    """execute_exit should avoid a second positions_get call inside close_positions."""
    executor = _build_executor()
    ticket = 235435408
    position = _build_close_position(ticket)
    symbol_spec = _build_symbol_spec()
    order_result = SimpleNamespace(retcode=mt.TRADE_RETCODE_DONE, deal=991199, comment="done")

    with (
        patch("trading_system.core.execution.mt.positions_get", return_value=[position]) as positions_get,
        patch.object(executor, "_get_cached_symbol_spec", return_value=(symbol_spec, 0)),
        patch(
            "trading_system.core.execution.mt.symbol_info_tick",
            return_value=SimpleNamespace(bid=17000.0, ask=17000.5),
        ),
        patch.object(executor, "_order_send_with_retry", return_value=order_result),
    ):
        result = executor.execute_exit(ExitRequest(ticket=ticket, portion=0.5))

    assert result.success is True
    assert result.deal_id == 991199
    assert positions_get.call_count == 1


def test_close_positions_batches_position_lookup_and_tick_by_symbol():
    """Batch close should use one positions_get and one tick lookup per symbol."""
    executor = _build_executor()
    symbol_spec = _build_symbol_spec()

    positions = [
        SimpleNamespace(ticket=11, symbol="EURUSD", type=mt.POSITION_TYPE_BUY, volume=0.20, magic=1001),
        SimpleNamespace(ticket=12, symbol="EURUSD", type=mt.POSITION_TYPE_BUY, volume=0.25, magic=1001),
        SimpleNamespace(ticket=13, symbol="GBPUSD", type=mt.POSITION_TYPE_SELL, volume=0.30, magic=1002),
    ]
    tickets = [11, 12, 13]
    portions = [1.0, 1.0, 1.0]
    tick_calls = []

    def mock_positions_get(ticket=None):
        if ticket is not None:
            raise AssertionError("Per-ticket positions_get should not be used in batch mode")
        return positions

    def mock_symbol_info_tick(symbol):
        tick_calls.append(symbol)
        return SimpleNamespace(bid=1.1000, ask=1.1002)

    deal_counter = {"value": 2000}

    def mock_order_send(_request, _success_codes=None):
        deal_counter["value"] += 1
        return SimpleNamespace(retcode=mt.TRADE_RETCODE_DONE, deal=deal_counter["value"], comment="done")

    with (
        patch("trading_system.core.execution.mt.positions_get", side_effect=mock_positions_get) as positions_get,
        patch("trading_system.core.execution.mt.symbol_info_tick", side_effect=mock_symbol_info_tick),
        patch.object(executor, "_get_cached_symbol_spec", return_value=(symbol_spec, 0)),
        patch.object(executor, "_order_send_with_retry", side_effect=mock_order_send),
    ):
        results = executor.close_positions(tickets=tickets, portions=portions)

    assert all(results[ticket][0] for ticket in tickets)
    assert positions_get.call_count == 1
    assert sorted(set(tick_calls)) == ["EURUSD", "GBPUSD"]
    assert len(tick_calls) == 2


def test_process_expiration_accepts_mt5_tick_with_asdict_time():
    """Expiration processing assumes MT5-native tick payload with _asdict()['time']."""
    executor = _build_executor()
    request = EntryRequest(
        order_type="stop",
        symbol="EURUSD",
        volume=0.1,
        signal=1,
        entry_price=1.1000,
        sl=1.0950,
        tp=1.1050,
        strategy_name="test_strategy",
        expiration_time="23:59",
    )

    with (
        patch(
            "trading_system.core.execution.mt.symbol_info_tick",
            return_value=SimpleNamespace(_asdict=lambda: {"time": 1730000000}),
        ),
        patch("trading_system.core.execution.mt.symbol_info", return_value=SimpleNamespace(time=1729999999)),
        patch.object(executor, "_convert_expiration_to_broker_time", return_value=1730003600),
    ):
        expiration = executor._process_expiration(request=request, symbol="EURUSD")

    assert expiration == 1730003600


def test_bracket_sell_leg_uses_bid_for_distance_calculation():
    """Sell leg should classify using bid, allowing market execution when near bid."""
    executor = _build_executor()
    request = EntryRequest(
        order_type="bracket",
        symbol="EURUSD",
        volume=0.1,
        signal=2,
        buy_stop=1.1010,
        sell_stop=1.09998,
        buy_sl=1.0980,
        sell_sl=1.1020,
        buy_tp=1.1040,
        sell_tp=1.0960,
        strategy_name="test_strategy",
    )
    symbol_spec = SimpleNamespace(
        digits=5,
        point=0.00001,
        trade_stops_level=10,
    )
    submitted_requests = []

    def mock_order_send(request_payload, _success_codes=None):
        submitted_requests.append(dict(request_payload))
        order_ticket = 3000 + len(submitted_requests)
        return SimpleNamespace(retcode=mt.TRADE_RETCODE_DONE, order=order_ticket, comment="done")

    with (
        patch.object(executor, "_get_cached_symbol_spec", return_value=(symbol_spec, 0)),
        patch(
            "trading_system.core.execution.mt.symbol_info_tick",
            return_value=SimpleNamespace(
                bid=1.10000,
                ask=1.10020,
                _asdict=lambda: {"time": 1730000000},
            ),
        ),
        patch(
            "trading_system.core.execution.mt.symbol_info",
            return_value=SimpleNamespace(trade_stops_level=10, time=1730000000),
        ),
        patch.object(executor, "_order_send_with_retry", side_effect=mock_order_send),
    ):
        result = executor.execute_entry(request)

    assert result.success is True
    assert len(submitted_requests) == 2
    sell_request = submitted_requests[1]
    assert sell_request["action"] == int(TradeAction.DEAL)
    assert sell_request["type"] == int(OrderType.SELL)
    assert sell_request["price"] == 1.10000
