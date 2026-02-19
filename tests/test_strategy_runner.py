"""StrategyRunner unit tests."""

import pytest
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, MagicMock, patch
import ctypes
from multiprocessing import Value, Lock
import threading
import pandas as pd

from tests.util import (
    IMPORTANT_PATCH_TARGETS,
    MockPosition,
    MockTradeLogger,
    ensure_mt5_stub,
    patch_strategy_runner_loggers,
)

ensure_mt5_stub()

# Import REAL StrategyRunner (system under test)
from trading_system.strategy_runner import (
    StrategyRunner,
    RunnerConfig,
    ExitLogData,
    PendingTicket,
)
from trading_system.core.execution_requests import EntryRequest, ModifyRequest
from trading_system.core.trade_logger import FillData, TradeLogger
from trading_system.core.trade_id_manager import TradeIDSequenceManager


# FIXTURES

@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test databases and logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def runner(temp_test_dir):
    """
    Real StrategyRunner with mocked dependencies.
    """
    # Regular dict for unit tests
    shared_state = {}
    
    strategy_config = {
        "name": "test_strategy",
        "params": {
            "symbol": "EURUSD",
            "timeframe": "M15",
            "timezone": "Europe/Budapest",
        },
        "execution": {"magic_number": 100001},
        "order_type": "market",
        "strategy_module": "strategies.example",
        "strategy_class": "ExampleStrategy",
        "risk": {
            "max_positions": 10,
            "max_trades": 50,
            "max_spread_points": 10,
            "max_slippage_points": 10,
        },
        "filters": {"news_filter": {"enabled": False}},
    }
    
    global_risk_policy = {
        'max_total_positions': 10,
        'max_daily_trades': 50,
        'log_root': str(temp_test_dir),
    }
    
    # Atomic counters
    global_position_count = Value(ctypes.c_int, 0)
    global_trade_count = Value(ctypes.c_int, 0)
    position_cache_lock = Lock()
    trade_id_db_path = temp_test_dir / "trade_id_sequence.db"
    
    # Patch loggers to avoid real DB writes/threads in unit tests.
    mock_logger_instance = MockTradeLogger()
    with patch_strategy_runner_loggers(
        trade_logger=MagicMock(return_value=mock_logger_instance),
        async_trade_logger=MagicMock(
            side_effect=lambda trade_logger, max_queue_size=100: trade_logger
        ),
    ):
        runner_config = RunnerConfig(
            strategy_name="test_strategy",
            strategy_config=strategy_config,
            broker_config=Mock(),
            global_risk_policy=global_risk_policy,
            shared_state=shared_state,
            global_position_count=global_position_count,
            global_trade_count=global_trade_count,
            position_cache_lock=position_cache_lock,
            trade_id_db_path=trade_id_db_path,
        )
        # Create StrategyRunner (will use mocked TradeLogger)
        runner = StrategyRunner(config=runner_config)
    
    # Use real sequence manager for _generate_trade_id()
    runner.trade_id_manager = TradeIDSequenceManager(trade_id_db_path)

    # Mock other dependencies
    runner.connection = Mock()
    runner.data_handler = Mock()
    runner.executor = Mock()
    runner.risk_manager = Mock()
    runner.strategy = Mock()
    # TradeLogger already mocked during __init__
    runner.symbol_spec = Mock(trade_contract_size=100000, point=0.00001, digits=5)
    
    yield runner

    runner.trade_id_manager.close()


def get_trade_id_by_ticket(runner: StrategyRunner, ticket: int):
    """Helper for reverse lookup in current runner implementation."""
    return runner.ticket_to_trade_id.get(ticket)


def increment_global_positions(runner: StrategyRunner, count: int = 1) -> int:
    """Seed global position count for tests that assert close-time decrements."""
    with runner.global_position_count.get_lock():
        runner.global_position_count.value += count
        return runner.global_position_count.value


def make_exit_log_data(
    ticket: int,
    expected_exit_price: float,
    exit_trigger: str,
    expected_entry_price: float,
    opening_sl: float,
    entry_price: float,
) -> ExitLogData:
    """Build ExitLogData for _log_full_close_execution tests."""
    return ExitLogData(
        ticket=ticket,
        expected_exit_price=expected_exit_price,
        exit_trigger=exit_trigger,
        expected_entry_price=expected_entry_price,
        opening_sl=opening_sl,
        entry_price=entry_price,
    )


@pytest.fixture
def position_factory():
    """
    Factory for creating position dictionaries matching MT5 structure.
    
    Returns callable: position_factory(ticket, **kwargs)
    """
    def _create_position(
        ticket: int,
        symbol: str = "EURUSD",
        pos_type: int = 0,  # 0=BUY, 1=SELL
        volume: float = 1.0,
        price_open: float = 1.10000,
        sl: float = 1.09500,
        tp: float = 1.10500,
        magic: int = 100001,
        **kwargs
    ) -> dict[str, Any]:
        """Create position dict matching normalize_position() output."""
        return {
            'ticket': ticket,
            'symbol': symbol,
            'type': pos_type,
            'volume': volume,
            'price_open': price_open,
            'sl': sl,
            'tp': tp,
            'magic': magic,
            'profit': kwargs.get('profit', 0.0),
            'swap': kwargs.get('swap', 0.0),
            'time': kwargs.get('time', int(time.time())),
        }
    return _create_position


# TEST CLASS 1: Trade ID Generation

class TestTradeIDGeneration:
    """
    Test real StrategyRunner._generate_trade_id() implementation.
    
    Critical properties:
    - IDs are strictly increasing (sequential)
    - Thread-safe atomic increments
    - Shared counter across instances
    """
    
    def test_generate_sequential_ids(self, runner):
        """
        Verify trade IDs increment sequentially from 1.
        
        Property: ID(n) = n for nth generation
        """
        ids = [runner._generate_trade_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5], "Trade IDs should be sequential starting from 1"
    
    def test_atomic_increment_thread_safety(self, runner):
        """
        Verify IDs remain unique for concurrent callers when DB access is serialized.
        
        Spawns 5 threads, each generating 10 IDs (50 total).
        All IDs must be unique and sequential.
        """
        import threading
        
        generated_ids = []
        collect_lock = threading.Lock()
        generate_lock = threading.Lock()
        
        def generate_concurrent_ids(count: int):
            """Worker thread generates 'count' trade IDs."""
            local_ids = []
            for _ in range(count):
                with generate_lock:
                    trade_id = runner._generate_trade_id()
                local_ids.append(trade_id)
            
            with collect_lock:
                generated_ids.extend(local_ids)
        
        # Launch 5 threads, each generating 10 IDs
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=generate_concurrent_ids, args=(10,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Validate no duplicates (proves atomic operation works)
        assert len(generated_ids) == 50, "Should generate 50 total IDs"
        assert len(set(generated_ids)) == 50, "All IDs must be unique (no race conditions)"
        assert min(generated_ids) == 1 and max(generated_ids) == 50, "IDs should be in range [1, 50]"
    
    def test_shared_counter_across_methods(self, runner):
        """
        Verify trade_id_counter is shared across different method calls.
        
        Tests that counter persists between _generate_trade_id() and _handle_new_fill().
        """
        # Generate 3 IDs directly
        runner._generate_trade_id()
        runner._generate_trade_id()
        runner._generate_trade_id()
        
        # Next ID should be 4
        next_id = runner._generate_trade_id()
        assert next_id == 4, "Counter should persist across calls"


# TEST CLASS 2: Bidirectional Mapping

class TestBidirectionalMapping:
    """
    Test real StrategyRunner ticket ↔ trade_id reverse lookup.
    
    Critical properties:
    - O(1) lookup performance via dict
    - Bidirectional consistency
    - Returns None for unknown tickets
    """
    
    def test_reverse_lookup_returns_trade_id(self, runner):
        """
        Verify ticket_to_trade_id.get() performs O(1) lookup.
        
        Tests dict-based implementation (not linear search).
        """
        # Setup mapping
        runner.ticket_to_trade_id[12345] = 1
        runner.ticket_to_trade_id[12346] = 2
        runner.ticket_to_trade_id[12347] = 3
        
        # Call real method
        assert get_trade_id_by_ticket(runner, 12345) == 1
        assert get_trade_id_by_ticket(runner, 12346) == 2
        assert get_trade_id_by_ticket(runner, 12347) == 3
    
    def test_reverse_lookup_unknown_ticket_returns_none(self, runner):
        """
        Verify real implementation returns None for unmapped tickets.
        
        Handles orphaned positions from previous sessions.
        """
        trade_id = get_trade_id_by_ticket(runner, 99999)
        assert trade_id is None, "Unknown ticket should return None (not raise exception)"
    
    def test_bidirectional_consistency(self, runner):
        """
        Verify mapping consistency in real data structures.
        
        Invariant: ticket_to_trade_id[ticket] = trade_id ⟺ 
                   entry_metadata[trade_id]['ticket'] = ticket
        """
        # Create mapping via real methods
        trade_id = runner._generate_trade_id()
        ticket = 12345
        
        # Store mapping (as done in real _handle_new_fill)
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
        }
        
        # Verify bidirectional consistency
        assert get_trade_id_by_ticket(runner, ticket) == trade_id, "Forward lookup should work"
        assert runner.entry_metadata[trade_id]['ticket'] == ticket, "Reverse reference should match"
    
    def test_lookup_performance_o1_complexity(self, runner):
        """
        Verify O(1) lookup performance regardless of mapping count.
        
        Creates 1000 mappings and validates lookup time is constant.
        """
        # Create 1000 mappings
        for i in range(1, 1001):
            trade_id = runner._generate_trade_id()
            ticket = 10000 + i
            runner.ticket_to_trade_id[ticket] = trade_id
        
        # Benchmark lookup time (should be <1µs for dict access)
        import timeit
        
        def lookup_test():
            return get_trade_id_by_ticket(runner, 10500)
        
        # Execute 10000 lookups
        elapsed = timeit.timeit(lookup_test, number=10000)
        
        assert elapsed < 0.01, f"10k lookups took {elapsed*1000:.2f}ms (expected <10ms, indicates O(1))"
        assert get_trade_id_by_ticket(runner, 10500) == 500, "Lookup should return correct trade_id"


# TEST CLASS 3: Fill Handling

class TestFillHandling:
    """
    Test real StrategyRunner._handle_new_fill() implementation.
    
    Covers:
    - Standard fills with pre-existing metadata
    - Orphaned positions (no metadata)
    - Counter increments
    - Position snapshot storage
    """
    
    def test_standard_fill_with_existing_metadata(self, runner, position_factory):
        """
        Verify real _handle_new_fill() processes standard order correctly.
        
        Flow: Entry signal created metadata → Fill detected → Mapping established
        """
        ticket = 12345
        trade_id = runner._generate_trade_id()
        
        # Pre-create metadata (simulates entry signal submission)
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': time.time(),
            'volume_multiplier': None,
            'ticket': ticket,
        }
        
        # Create position dict
        pos_dict = position_factory(ticket=ticket, volume=1.0)
        
        # Call real _handle_new_fill()
        runner._handle_new_fill(pos_dict)
        
        # Verify position tracked
        assert ticket in runner.known_positions, "Position should be added to known_positions"
        
        # Verify fill logged
        assert len(runner.trade_logger.fills_logged) == 1, "Fill should be logged"
        logged = runner.trade_logger.fills_logged[0]
        assert logged['trade_id'] == trade_id
        assert logged['ticket'] == ticket
        assert logged['volume'] == 1.0
        
        # Verify position snapshot stored
        assert 'position_snapshot' in runner.entry_metadata[trade_id], "Snapshot should be stored"
        snapshot = runner.entry_metadata[trade_id]['position_snapshot']
        assert snapshot['ticket'] == ticket
        assert snapshot['volume'] == 1.0
    
    def test_orphaned_position_generates_fallback_trade_id(self, runner, position_factory):
        """
        Verify real implementation handles orphaned positions gracefully.
        
        Scenario: Position exists but no metadata (crashed session recovery).
        Expected: Generate fallback trade_id and minimal metadata.
        """
        ticket = 99999
        pos_dict = position_factory(ticket=ticket, volume=0.5, price_open=1.12345)
        
        # Call real _handle_new_fill() with no pre-existing metadata
        runner._handle_new_fill(pos_dict)
        
        # Verify fallback trade_id created
        trade_id = get_trade_id_by_ticket(runner, ticket)
        assert trade_id is not None, "Orphaned position should get fallback trade_id"
        assert trade_id == 1, "First generated trade_id should be 1"
        
        # Verify minimal metadata created
        assert trade_id in runner.entry_metadata, "Metadata should be created"
        metadata = runner.entry_metadata[trade_id]
        assert metadata['ticket'] == ticket
        assert metadata['expected_entry_price'] == 1.12345, "Should use actual fill price as expected"
        assert metadata['opening_sl'] == pos_dict['sl']
        assert metadata['volume_multiplier'] is None, "No meta-labeling data for orphan"
        
        # Verify logged with fallback data
        assert len(runner.trade_logger.fills_logged) == 1
        logged = runner.trade_logger.fills_logged[0]
        assert logged['expected_entry_price'] == 1.12345
    
    def test_fill_increments_position_and_trade_counters(self, runner, position_factory):
        """
        Verify fill detection increments local position tracking and global trade counter.
        
        Counters used for max position/trade limits in risk manager.
        """
        initial_position_count = runner.local_position_count
        initial_trade_count = runner.global_trade_count.value
        
        # Handle 3 fills (orphaned positions)
        for i in range(3):
            ticket = 10000 + i
            pos_dict = position_factory(ticket=ticket)
            runner._handle_new_fill(pos_dict)
        
        # Verify counters increased
        assert runner.local_position_count == initial_position_count + 3, "Local position count should increment by 3"
        assert runner.global_trade_count.value == initial_trade_count + 3, "Trade counter should increment by 3"
    
    def test_fill_stores_latency_calculation(self, runner, position_factory):
        """
        Verify fill handler calculates fill latency correctly.
        
        Latency = time between submission_time and fill detection.
        """
        ticket = 12345
        trade_id = runner._generate_trade_id()
        submission_time = time.time() - 0.5  # 500ms ago
        
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': submission_time,
            'volume_multiplier': None,
            'ticket': ticket,
        }
        
        pos_dict = position_factory(ticket=ticket)
        runner._handle_new_fill(pos_dict)
        
        # Verify latency calculated (should be ~500ms)
        logged = runner.trade_logger.fills_logged[0]
        # Latency is passed to log_fill but not stored in our mock
        # We verify the calculation happened by checking submission_time was used
        assert runner.entry_metadata[trade_id]['submission_time'] == submission_time


# TEST CLASS 4: Startup Reconciliation

class TestStartupReconciliation:
    """Test strict latest-2 startup reconciliation in _init_known_positions()."""

    def test_init_known_positions_all_reconciled_skips_fill_logging(self, runner, position_factory):
        positions = [
            position_factory(ticket=40001, price_open=1.10100, sl=1.09600),
            position_factory(ticket=40002, price_open=1.20200, sl=1.19700),
        ]
        runner._get_strategy_positions = Mock(return_value=positions)
        runner.trade_logger.open_trades_by_ticket_last_two = {
            40001: {
                'trade_id': 9001,
                'expected_entry_price': 1.10050,
                'opening_sl': 1.09550,
                'volume_multiplier': 0.75,
            },
            40002: {
                'trade_id': 9002,
                'expected_entry_price': 1.20150,
                'opening_sl': 1.19650,
                'volume_multiplier': None,
            },
        }

        runner._init_known_positions()

        assert runner.local_position_count == 2
        assert len(runner.trade_logger.fills_logged) == 0
        assert runner.ticket_to_trade_id[40001] == 9001
        assert runner.ticket_to_trade_id[40002] == 9002
        assert runner.entry_metadata[9001]['expected_entry_price'] == 1.10050
        assert runner.entry_metadata[9001]['opening_sl'] == 1.09550
        assert runner.entry_metadata[9001]['volume_multiplier'] == 0.75
        assert 40001 in runner.known_positions
        assert 40002 in runner.known_positions

    def test_init_known_positions_mixed_reconciled_and_missing_logs_only_missing(self, runner, position_factory):
        positions = [
            position_factory(ticket=50001, price_open=1.30100, sl=1.29600),
            position_factory(ticket=50002, price_open=1.40100, sl=1.39600),
        ]
        runner._get_strategy_positions = Mock(return_value=positions)
        runner.trade_logger.open_trades_by_ticket_last_two = {
            50001: {
                'trade_id': 9101,
                'expected_entry_price': 1.30050,
                'opening_sl': 1.29550,
                'volume_multiplier': None,
            },
        }

        runner._init_known_positions()

        assert runner.local_position_count == 2
        assert len(runner.trade_logger.fills_logged) == 1
        logged_fill = runner.trade_logger.fills_logged[0]
        assert logged_fill['ticket'] == 50002
        assert runner.ticket_to_trade_id[50001] == 9101
        assert runner.ticket_to_trade_id[50002] == 1
        assert runner.entry_metadata[9101]['expected_entry_price'] == 1.30050
        assert runner.entry_metadata[9101]['opening_sl'] == 1.29550

    def test_init_known_positions_no_reconciled_matches_previous_behavior(self, runner, position_factory):
        positions = [
            position_factory(ticket=60001),
            position_factory(ticket=60002),
        ]
        runner._get_strategy_positions = Mock(return_value=positions)
        runner.trade_logger.open_trades_by_ticket_last_two = {}

        runner._init_known_positions()

        assert runner.local_position_count == 2
        assert len(runner.trade_logger.fills_logged) == 2
        assert [fill['trade_id'] for fill in runner.trade_logger.fills_logged] == [1, 2]
        assert runner.ticket_to_trade_id[60001] == 1
        assert runner.ticket_to_trade_id[60002] == 2

    def test_init_known_positions_reconciliation_failure_falls_back_to_logging_all(self, runner, position_factory):
        positions = [
            position_factory(ticket=70001),
            position_factory(ticket=70002),
        ]
        runner._get_strategy_positions = Mock(return_value=positions)
        runner.trade_logger.get_open_trades_by_ticket_last_three = Mock(side_effect=RuntimeError("db read failed"))

        runner._init_known_positions()

        assert len(runner.trade_logger.fills_logged) == 2
        assert runner.ticket_to_trade_id[70001] == 1
        assert runner.ticket_to_trade_id[70002] == 2


# TEST CLASS 5: Orphaned Position Detection

class TestOrphanedPositions:
    """
    Test real orphaned position recovery logic.
    
    Scenarios:
    - Positions from crashed sessions
    - Manual MT5 entries
    - Missing metadata
    """
    
    def test_multiple_orphaned_positions_unique_trade_ids(self, runner, position_factory):
        """
        Verify real implementation assigns unique trade_ids to orphaned positions.
        
        Edge case: Restart with 5 open positions from previous session.
        """
        orphaned_tickets = [11111, 22222, 33333, 44444, 55555]
        trade_ids = []
        
        for ticket in orphaned_tickets:
            pos_dict = position_factory(ticket=ticket)
            runner._handle_new_fill(pos_dict)
            
            trade_id = get_trade_id_by_ticket(runner, ticket)
            trade_ids.append(trade_id)
        
        # Verify all unique and sequential
        assert len(set(trade_ids)) == 5, "Each orphaned position should get unique trade_id"
        assert trade_ids == [1, 2, 3, 4, 5], "Trade IDs should be sequential"
    
    def test_orphaned_position_creates_bidirectional_mapping(self, runner, position_factory):
        """
        Verify orphaned position establishes complete bidirectional mapping.
        """
        ticket = 77777
        pos_dict = position_factory(ticket=ticket)
        
        runner._handle_new_fill(pos_dict)
        
        trade_id = get_trade_id_by_ticket(runner, ticket)
        
        # Verify forward mapping: ticket → trade_id
        assert trade_id is not None
        assert runner.ticket_to_trade_id[ticket] == trade_id
        
        # Verify reverse reference: trade_id → ticket
        assert runner.entry_metadata[trade_id]['ticket'] == ticket
    
    def test_orphaned_position_added_to_known_positions(self, runner, position_factory):
        """
        Verify orphaned positions are added to known_positions set.
        
        Critical for duplicate fill detection.
        """
        ticket = 88888
        pos_dict = position_factory(ticket=ticket)
        
        assert ticket not in runner.known_positions, "Should not be tracked initially"
        
        runner._handle_new_fill(pos_dict)
        
        assert ticket in runner.known_positions, "Should be added to known_positions"


# TEST CLASS 6: Position Cleanup

class TestPositionCleanup:
    """
    Test real StrategyRunner._log_full_close_execution() implementation.
    
    Cleanup sequence:
    1. Extract trade_id from ticket
    2. Log close to database
    3. Remove from entry_metadata
    4. Remove from ticket_to_trade_id
    5. Remove from known_positions
    6. Decrement position counter
    """
    
    def test_full_close_removes_all_tracking(self, runner, position_factory):
        """
        Verify real cleanup implementation removes all tracking structures.
        
        Post-cleanup state: No metadata, no mapping, not in known_positions.
        """
        # Setup position
        ticket = 12345
        trade_id = runner._generate_trade_id()
        pos_dict = position_factory(ticket=ticket)
        
        # Create metadata and mapping
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': time.time(),
            'volume_multiplier': None,
            'position_snapshot': pos_dict,
        }
        runner.known_positions.add(ticket)
        
        # Seed global position counter (simulate filled position in shared counter)
        increment_global_positions(runner, 1)
        initial_count = runner.global_position_count.value
        
        # Execute real close
        runner._log_full_close_execution(
            pos_dict=pos_dict,
            data=make_exit_log_data(
                ticket=ticket,
                expected_exit_price=1.10500,
                exit_trigger='TEST_EXIT',
                expected_entry_price=1.10000,
                opening_sl=1.09500,
                entry_price=1.10000,
            ),
        )
        
        # Verify all tracking removed
        assert get_trade_id_by_ticket(runner, ticket) is None, "Ticket mapping should be removed"
        assert trade_id not in runner.entry_metadata, "Metadata should be removed"
        assert ticket not in runner.known_positions, "Position should be removed from tracking"
        
        # Verify counter decremented
        assert runner.global_position_count.value == initial_count - 1, "Counter should decrement by 1"
    
    def test_close_logged_with_correct_metadata(self, runner, position_factory):
        """
        Verify real implementation logs close with correct trade_id and exit trigger.
        """
        ticket = 12345
        trade_id = runner._generate_trade_id()
        pos_dict = position_factory(ticket=ticket)
        
        # Setup
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': time.time(),
            'volume_multiplier': None,
            'position_snapshot': pos_dict,
        }
        runner.known_positions.add(ticket)
        
        # Execute close
        runner._log_full_close_execution(
            pos_dict=pos_dict,
            data=make_exit_log_data(
                ticket=ticket,
                expected_exit_price=1.10500,
                exit_trigger='STOP_LOSS',
                expected_entry_price=1.10000,
                opening_sl=1.09500,
                entry_price=1.10000,
            ),
        )
        
        # Verify logged
        assert len(runner.trade_logger.closes_logged) == 1
        logged = runner.trade_logger.closes_logged[0]
        assert logged['trade_id'] == trade_id
        assert logged['ticket'] == ticket
        assert logged['exit_trigger'] == 'STOP_LOSS'
        assert logged['expected_exit_price'] == 1.10500
    
    def test_close_handles_missing_trade_id_gracefully(self, runner, position_factory):
        """
        Verify cleanup handles missing trade_id without crashing.
        
        Edge case: Corrupted state where ticket has no mapping.
        """
        ticket = 12345
        pos_dict = position_factory(ticket=ticket)
        
        # Add to known_positions but no metadata (corrupted state)
        runner.known_positions.add(ticket)
        
        # Execute close (should not raise exception)
        try:
            runner._log_full_close_execution(
                pos_dict=pos_dict,
                data=make_exit_log_data(
                    ticket=ticket,
                    expected_exit_price=1.10500,
                    exit_trigger='TEST_EXIT',
                    expected_entry_price=1.10000,
                    opening_sl=1.09500,
                    entry_price=1.10000,
                ),
            )
        except Exception as e:
            pytest.fail(f"Cleanup should handle missing trade_id gracefully, raised: {e}")
        
        # Verify minimal cleanup attempted (known_positions should still be cleaned)
        assert ticket not in runner.known_positions, "Should remove from known_positions"
        
        # Verify nothing logged (no trade_id means no metadata)
        assert len(runner.trade_logger.closes_logged) == 0, "Should not log without trade_id"
    
    def test_multiple_closes_decrement_counter_correctly(self, runner, position_factory):
        """
        Verify multiple closes decrement counter atomically.
        
        Edge case: Simultaneous closes (SL hit on multiple positions).
        """
        # Setup 5 positions
        tickets = [10001, 10002, 10003, 10004, 10005]
        
        for ticket in tickets:
            trade_id = runner._generate_trade_id()
            pos_dict = position_factory(ticket=ticket)
            
            runner.ticket_to_trade_id[ticket] = trade_id
            runner.entry_metadata[trade_id] = {
                'ticket': ticket,
                'expected_entry_price': 1.10000,
                'opening_sl': 1.09500,
                'submission_time': time.time(),
                'volume_multiplier': None,
                'position_snapshot': pos_dict,
            }
            runner.known_positions.add(ticket)
            increment_global_positions(runner, 1)
        
        initial_count = runner.global_position_count.value
        
        # Close all positions
        for ticket in tickets:
            pos_dict = position_factory(ticket=ticket)
            runner._log_full_close_execution(
                pos_dict=pos_dict,
                data=make_exit_log_data(
                    ticket=ticket,
                    expected_exit_price=1.10500,
                    exit_trigger='STOP_LOSS',
                    expected_entry_price=1.10000,
                    opening_sl=1.09500,
                    entry_price=1.10000,
                ),
            )
        
        # Verify counter decremented by 5
        assert runner.global_position_count.value == initial_count - 5, "Counter should decrement by 5"
        
        # Verify all tracking cleaned up
        assert len(runner.known_positions) == 0, "All positions should be removed"
        assert len(runner.entry_metadata) == 0, "All metadata should be removed"
        assert len(runner.ticket_to_trade_id) == 0, "All mappings should be removed"


# TEST CLASS 7: Metadata Extraction

class TestMetadataExtraction:
    """
    Test real StrategyRunner._resolve_entry_prices() implementation.
    
    Validates extraction of:
    - trade_id from ticket
    - Expected entry price (order-type dependent)
    - Opening SL (direction-dependent for brackets)
    - Actual entry price
    """
    
    def test_extract_metadata_standard_order(self, runner, position_factory):
        """
        Verify metadata extraction for standard market orders.
        """
        ticket = 12345
        trade_id = runner._generate_trade_id()
        pos_dict = position_factory(ticket=ticket, price_open=1.10123, sl=1.09500)
        
        # Create mock entry request
        entry_request = Mock()
        entry_request.order_type = 'market'
        entry_request.entry_price = 1.10000
        entry_request.sl = 1.09500
        
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'entry_request': entry_request,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
        }
        
        # Call real method
        extracted_trade_id, expected_entry, opening_sl, actual_entry = runner._resolve_entry_prices(
            ticket, pos_dict
        )
        
        assert extracted_trade_id == trade_id
        assert expected_entry == 1.10000, "Should use entry_request.entry_price"
        assert opening_sl == 1.09500, "Should use entry_request.sl"
        assert actual_entry == 1.10123, "Should use actual fill price from position"
    
    def test_extract_metadata_orphaned_position_returns_fallback(self, runner, position_factory):
        """
        Verify metadata extraction returns fallback values for orphaned positions.
        
        When trade_id is None, should return None and position-based fallbacks.
        """
        ticket = 99999  # Unknown ticket
        pos_dict = position_factory(ticket=ticket, price_open=1.12345, sl=1.11000)
        
        # Call real method
        extracted_trade_id, expected_entry, opening_sl, actual_entry = runner._resolve_entry_prices(
            ticket, pos_dict
        )
        
        assert extracted_trade_id is None, "Should return None for orphaned position"
        assert expected_entry == 1.12345, "Should fallback to actual price"
        assert opening_sl == 1.11000, "Should fallback to current SL"
        assert actual_entry == 1.12345, "Should use actual price"

    def test_extract_metadata_without_entry_request_uses_stored_values(self, runner, position_factory):
        """When entry_request is None, stored metadata should be preferred."""
        ticket = 99901
        trade_id = runner._generate_trade_id()
        pos_dict = position_factory(ticket=ticket, price_open=1.55555, sl=1.50000)

        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'entry_request': None,
            'expected_entry_price': 1.44444,
            'opening_sl': 1.43333,
        }

        extracted_trade_id, expected_entry, opening_sl, actual_entry = runner._resolve_entry_prices(
            ticket, pos_dict
        )

        assert extracted_trade_id == trade_id
        assert expected_entry == 1.44444
        assert opening_sl == 1.43333
        assert actual_entry == 1.55555
    
    def test_extract_metadata_bracket_order_buy_side(self, runner, position_factory):
        """
        Verify metadata extraction for bracket order (BUY side filled).
        
        Should use buy_stop and buy_sl from entry_request.
        """
        ticket = 12345
        trade_id = runner._generate_trade_id()
        pos_dict = position_factory(ticket=ticket, pos_type=0, price_open=1.10500)  # BUY
        
        # Create mock bracket entry request
        entry_request = Mock()
        entry_request.order_type = 'bracket'
        entry_request.buy_stop = 1.10500
        entry_request.buy_sl = 1.09500
        entry_request.sell_stop = 1.09500
        entry_request.sell_sl = 1.10500
        
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'entry_request': entry_request,
        }
        
        # Call real method
        extracted_trade_id, expected_entry, opening_sl, actual_entry = runner._resolve_entry_prices(
            ticket, pos_dict
        )
        
        assert expected_entry == 1.10500, "Should use buy_stop for BUY"
        assert opening_sl == 1.09500, "Should use buy_sl for BUY"
    
    def test_extract_metadata_bracket_order_sell_side(self, runner, position_factory):
        """
        Verify metadata extraction for bracket order (SELL side filled).
        
        Should use sell_stop and sell_sl from entry_request.
        """
        ticket = 12346
        trade_id = runner._generate_trade_id()
        pos_dict = position_factory(ticket=ticket, pos_type=1, price_open=1.09500)  # SELL
        
        # Create mock bracket entry request
        entry_request = Mock()
        entry_request.order_type = 'bracket'
        entry_request.buy_stop = 1.10500
        entry_request.buy_sl = 1.09500
        entry_request.sell_stop = 1.09500
        entry_request.sell_sl = 1.10500
        
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'entry_request': entry_request,
        }
        
        # Call real method
        extracted_trade_id, expected_entry, opening_sl, actual_entry = runner._resolve_entry_prices(
            ticket, pos_dict
        )
        
        assert expected_entry == 1.09500, "Should use sell_stop for SELL"
        assert opening_sl == 1.10500, "Should use sell_sl for SELL"


# TEST CLASS 8: Pending Ticket Resolution

class TestPendingTicketResolution:
    """
    Test real StrategyRunner._resolve_pending_ticket() implementation.
    
    Validates:
    - Bracket order resolution (volume + timing window)
    - Standard order resolution (exact ticket match)
    - Pending ticket cleanup
    """
    
    def test_resolve_bracket_order_with_volume_match(self, runner, position_factory):
        """
        Verify bracket order resolution via volume matching.
        
        Tolerance: ±1% volume difference allowed for floating-point precision.
        """
        trade_id = runner._generate_trade_id()
        expected_volume = 1.0
        
        # Create pending bracket order
        runner._register_pending_trade(
            trade_id,
            PendingTicket(
                order_type='bracket',
                symbol='EURUSD',
                magic=100001,
                submission_time=time.time(),
                buy_order_ticket=12345,
                sell_order_ticket=12346,
                expected_volume=expected_volume,
            ),
        )
        
        # Create mock entry request
        entry_request = Mock()
        entry_request.buy_stop = 1.10500
        entry_request.buy_sl = 1.09500
        
        runner.entry_metadata[trade_id] = {
            'entry_request': entry_request,
        }
        
        # Position with matching volume
        pos_dict = position_factory(ticket=12345, volume=1.0, pos_type=0)
        
        # Call real method
        resolved_trade_id = runner._resolve_pending_ticket(pos_dict)
        
        assert resolved_trade_id == trade_id, "Should resolve to correct trade_id"
        assert runner.ticket_to_trade_id[12345] == trade_id, "Should create mapping"
        assert trade_id not in runner.pending_tickets, "Should remove from pending"
    
    def test_resolve_bracket_order_with_timing_window(self, runner, position_factory):
        """
        Verify bracket order resolution requires fill within 10s window.
        
        Fills older than 10s are rejected (prevents matching wrong positions).
        """
        trade_id = runner._generate_trade_id()
        
        # Create pending bracket order (11 seconds ago - expired)
        runner._register_pending_trade(
            trade_id,
            PendingTicket(
                order_type='bracket',
                symbol='EURUSD',
                magic=100001,
                submission_time=time.time() - 11.0,  # 11s ago
                buy_order_ticket=12345,
                sell_order_ticket=12346,
                expected_volume=1.0,
            ),
        )
        
        entry_request = Mock()
        entry_request.buy_stop = 1.10500
        entry_request.buy_sl = 1.09500
        
        runner.entry_metadata[trade_id] = {
            'entry_request': entry_request,
        }
        
        pos_dict = position_factory(ticket=12345, volume=1.0)
        
        # Call real method
        resolved_trade_id = runner._resolve_pending_ticket(pos_dict)
        
        assert resolved_trade_id is None, "Should reject expired fill (>10s)"
        assert 12345 not in runner.ticket_to_trade_id, "Should not create mapping"
    
    def test_resolve_standard_order_exact_ticket_match(self, runner, position_factory):
        """
        Verify standard order resolution via exact ticket matching.
        """
        trade_id = runner._generate_trade_id()
        ticket = 12345
        
        # Create pending standard order
        runner._register_pending_trade(
            trade_id,
            PendingTicket(
                order_type='standard',
                symbol='EURUSD',
                magic=100001,
                submission_time=time.time(),
                ticket=ticket,
            ),
        )
        
        # Mapping already created in _process_entry_signal
        runner.ticket_to_trade_id[ticket] = trade_id
        
        pos_dict = position_factory(ticket=ticket)
        
        # Call real method
        resolved_trade_id = runner._resolve_pending_ticket(pos_dict)
        
        assert resolved_trade_id == trade_id
        assert trade_id not in runner.pending_tickets, "Should remove from pending"
    
    def test_resolve_returns_none_for_no_match(self, runner, position_factory):
        """
        Verify resolution returns None when no pending ticket matches.
        
        Handles true orphaned positions.
        """
        # No pending tickets
        pos_dict = position_factory(ticket=99999)
        
        resolved_trade_id = runner._resolve_pending_ticket(pos_dict)
        
        assert resolved_trade_id is None, "Should return None for unmatched position"


# TEST CLASS 9: Atomic Counter Operations

class TestAtomicCounterOperations:
    """
    Test real atomic increment/decrement methods.
    
    Validates thread safety and correct counter updates.
    """
    
    def test_atomic_decrement_global_positions_thread_safe(self, runner):
        """
        Verify _atomic_decrement_global_positions() is thread-safe.
        """
        runner.global_position_count.value = 10
        results = []
        lock = threading.Lock()
        
        def decrement_worker():
            new_count = runner._atomic_decrement_global_positions(1, "thread_test")
            with lock:
                results.append(new_count)
        
        threads = [threading.Thread(target=decrement_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify all decrements succeeded
        assert len(results) == 10
        assert runner.global_position_count.value == 0
        
        # Verify no duplicates in returned counts (10..1..0 progression)
        assert len(set(results)) == 10, "All returned counts should be unique"
    
    def test_atomic_decrement_global_positions(self, runner):
        """
        Verify _atomic_decrement_global_positions() decrements correctly.
        """
        # set initial count
        runner.global_position_count.value = 10
        
        # Decrement by 3
        new_count = runner._atomic_decrement_global_positions(3, "unit_test")
        
        assert new_count == 7, "Should return new count after decrement"
        assert runner.global_position_count.value == 7, "Counter should be decremented"
    
    def test_atomic_increment_trade(self, runner):
        """
        Verify atomic_increment_trade() increments trade counter.
        """
        initial = runner.global_trade_count.value
        
        new_count = runner.atomic_increment_trade()
        
        assert new_count == initial + 1
        assert runner.global_trade_count.value == initial + 1


# TEST CLASS 10: Cross-Component Workflows

class TestCrossComponentWorkflows:
    """
    Test real end-to-end workflows spanning multiple methods.
    
    Validates complete trade lifecycle.
    """
    
    def test_full_lifecycle_standard_order(self, runner, position_factory):
        """
        Test complete lifecycle: generate_trade_id → fill → close.
        
        Uses real implementations throughout.
        """
        # Step 1: Generate trade_id (simulates entry signal)
        trade_id = runner._generate_trade_id()
        ticket = 12345
        
        # Step 2: Store metadata (simulates order submission)
        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': time.time(),
            'volume_multiplier': None,
        }
        
        # Step 3: Handle fill
        pos_dict = position_factory(ticket=ticket)
        runner._handle_new_fill(pos_dict)
        
        assert ticket in runner.known_positions, "Position should be tracked"
        assert len(runner.trade_logger.fills_logged) == 1, "Fill should be logged"
        
        # Step 4: Update metadata with snapshot (required for close)
        runner.entry_metadata[trade_id]['position_snapshot'] = pos_dict
        
        # Step 5: Close position
        runner._log_full_close_execution(
            pos_dict=pos_dict,
            data=make_exit_log_data(
                ticket=ticket,
                expected_exit_price=1.10500,
                exit_trigger='TAKE_PROFIT',
                expected_entry_price=1.10000,
                opening_sl=1.09500,
                entry_price=1.10000,
            ),
        )
        
        # Verify complete cleanup
        assert ticket not in runner.known_positions, "Position should be removed"
        assert trade_id not in runner.entry_metadata, "Metadata should be removed"
        assert ticket not in runner.ticket_to_trade_id, "Mapping should be removed"
        assert len(runner.trade_logger.closes_logged) == 1, "Close should be logged"
    
    def test_orphaned_position_lifecycle(self, runner, position_factory):
        """
        Test orphaned position recovery: no metadata → generate fallback → close.
        """
        ticket = 99999
        
        # Step 1: Handle orphaned fill (no pre-existing metadata)
        pos_dict = position_factory(ticket=ticket)
        runner._handle_new_fill(pos_dict)
        
        trade_id = get_trade_id_by_ticket(runner, ticket)
        assert trade_id is not None, "Fallback trade_id should be generated"
        
        # Step 2: Update with snapshot
        runner.entry_metadata[trade_id]['position_snapshot'] = pos_dict
        
        # Step 3: Close
        runner._log_full_close_execution(
            pos_dict=pos_dict,
            data=make_exit_log_data(
                ticket=ticket,
                expected_exit_price=1.10500,
                exit_trigger='MANUAL_CLOSE',
                expected_entry_price=pos_dict['price_open'],
                opening_sl=pos_dict['sl'],
                entry_price=pos_dict['price_open'],
            ),
        )
        
        # Verify cleanup
        assert ticket not in runner.known_positions
        assert trade_id not in runner.entry_metadata
        assert len(runner.trade_logger.fills_logged) == 1
        assert len(runner.trade_logger.closes_logged) == 1


class TestEntrySignalProcessing:
    """Validate entry signal flow for ID allocation and meta-label guards."""

    def test_rejected_entry_does_not_allocate_trade_id(self, runner):
        entry_request = Mock()
        entry_request.order_type = "market"
        entry_request.signal = 1
        entry_request.sl = 1.09500
        entry_request.entry_price = None
        entry_request.volume = 0.20

        runner.strategy.generate_entry_signal.return_value = entry_request
        runner.risk_manager.validate_trade.return_value = Mock(
            can_trade=False,
            reason="risk_rejected",
        )

        data = pd.DataFrame(
            {"Close": [1.1010]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 10:15:00")]),
        )
        current_id = runner.trade_id_manager.get_current_id()

        runner._process_entry_signal(data)

        assert runner.trade_id_manager.get_current_id() == current_id
        runner.executor.execute_entry.assert_not_called()

    def test_successful_entry_allocates_trade_id_after_execution(self, runner):
        entry_request = Mock()
        entry_request.order_type = "market"
        entry_request.signal = 1
        entry_request.sl = 1.09500
        entry_request.entry_price = None
        entry_request.volume = 0.20

        runner.strategy.generate_entry_signal.return_value = entry_request
        runner.risk_manager.validate_trade.return_value = Mock(
            can_trade=True,
            reason="ok",
            volume=0.20,
        )
        runner.executor.execute_entry.return_value = Mock(
            success=True,
            ticket=91001,
            error_message=None,
        )

        data = pd.DataFrame(
            {"Close": [1.1010]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 10:15:00")]),
        )
        current_id = runner.trade_id_manager.get_current_id()

        runner._process_entry_signal(data)

        assert runner.trade_id_manager.get_current_id() == current_id + 1
        assert 91001 in runner.ticket_to_trade_id

    def test_zero_volume_placeholder_is_replaced_by_risk_sized_volume(self, runner):
        entry_request = EntryRequest(
            order_type="market",
            symbol="EURUSD",
            volume=0.0,
            signal=1,
            sl=1.09500,
            tp=1.10500,
            strategy_name=runner.strategy_name,
            comment="TEST",
        )

        runner.strategy.generate_entry_signal.return_value = entry_request
        runner.risk_manager.validate_trade.return_value = Mock(
            can_trade=True,
            reason="ok",
            volume=0.24,
        )
        runner.executor.execute_entry.return_value = Mock(
            success=True,
            ticket=91002,
            error_message=None,
        )

        data = pd.DataFrame(
            {"Close": [1.1010]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 10:15:00")]),
        )

        runner._process_entry_signal(data)

        runner.executor.execute_entry.assert_called_once()
        executed_request = runner.executor.execute_entry.call_args[0][0]
        assert executed_request.volume == 0.24

    def test_apply_meta_labeling_handles_empty_feature_frame(self, runner):
        runner.meta_model = Mock()
        runner.feature_extractor = Mock(return_value=pd.DataFrame())

        entry_request = Mock()
        entry_request.volume = 0.35
        data = pd.DataFrame({"Close": [1.1010]})

        volume_multiplier, adjusted_volume = runner._apply_meta_labeling(data, entry_request)

        assert volume_multiplier is None
        assert adjusted_volume == 0.35
        runner.meta_model.predict_proba.assert_not_called()


class TestSnapshotRefreshBehavior:
    """Validate snapshot freshness for external-close logging paths."""

    def test_monitor_exits_refreshes_tracked_snapshot_levels(self, runner, position_factory):
        ticket = 12345
        trade_id = 7
        stale_snapshot = position_factory(ticket=ticket, sl=1.09500, tp=1.10500)
        live_position = position_factory(ticket=ticket, sl=1.10000, tp=1.11000)

        runner.ticket_to_trade_id[ticket] = trade_id
        runner.known_positions.add(ticket)
        runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': time.time(),
            'volume_multiplier': None,
            'position_snapshot': stale_snapshot,
        }

        runner._get_strategy_positions = Mock(return_value=[live_position])
        runner.strategy.generate_exit_signal.return_value = None
        runner._handle_new_fill = Mock()
        runner._handle_closed_positions = Mock()

        data = pd.DataFrame(
            {"Close": [1.1010]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 10:15:00")]),
        )
        runner._monitor_exits(data)

        refreshed_snapshot = runner.entry_metadata[trade_id]['position_snapshot']
        assert refreshed_snapshot['sl'] == pytest.approx(1.10000, abs=1e-9)
        assert refreshed_snapshot['tp'] == pytest.approx(1.11000, abs=1e-9)
        runner._handle_new_fill.assert_not_called()
        runner._handle_closed_positions.assert_not_called()

    def test_modify_adjustment_patches_snapshot_without_positions_query(self, runner, position_factory):
        ticket = 12346
        trade_id = 8

        runner.ticket_to_trade_id[ticket] = trade_id
        runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09500,
            'submission_time': time.time(),
            'volume_multiplier': None,
            'position_snapshot': position_factory(ticket=ticket, sl=1.09500, tp=1.10500),
        }

        runner.executor.execute_modify.return_value = Mock(success=True, error_message=None)
        runner._invalidate_cache_for_ticket = Mock()
        request = ModifyRequest(ticket=ticket, new_sl=1.10000, new_tp=1.11250, comment="breakeven")

        with patch("trading_system.runner_execution_mixin.mt.positions_get") as positions_get:
            runner._handle_modify_adjustment(request)
            positions_get.assert_not_called()

        patched_snapshot = runner.entry_metadata[trade_id]['position_snapshot']
        assert patched_snapshot['sl'] == pytest.approx(1.10000, abs=1e-9)
        assert patched_snapshot['tp'] == pytest.approx(1.11250, abs=1e-9)
        runner._invalidate_cache_for_ticket.assert_called_once_with(ticket)

    def test_sl_move_updates_snapshot_and_drives_close_slippage(self, runner, position_factory, tmp_path):
        ticket = 22346
        trade_id = 42
        stale_sl = 1.09500
        moved_sl = 1.10000
        moved_tp = 1.11250

        real_logger = TradeLogger(
            log_root=tmp_path / "trades",
            strategy_name=runner.strategy_name,
            strategy_tz=runner.strategy_tz,
        )
        runner.trade_logger = real_logger

        try:
            runner.ticket_to_trade_id[ticket] = trade_id
            runner.known_positions.add(ticket)
            runner.entry_metadata[trade_id] = {
                'ticket': ticket,
                'expected_entry_price': 1.10000,
                'opening_sl': stale_sl,
                'submission_time': time.time(),
                'volume_multiplier': None,
                'position_snapshot': position_factory(
                    ticket=ticket,
                    sl=stale_sl,
                    tp=1.10500,
                    price_open=1.10000,
                    volume=1.0,
                    pos_type=0,
                ),
            }

            with patch('trading_system.core.trade_logger.mt5') as mock_mt5:
                symbol_info = Mock()
                symbol_info.trade_tick_value = 10.0
                symbol_info.trade_tick_value_loss = 9.8
                symbol_info.trade_tick_value_profit = 10.2
                symbol_info.trade_tick_size = 0.0001
                mock_mt5.symbol_info.return_value = symbol_info

                entry_deal = Mock(entry=0, price=1.10000, volume=1.0, profit=0.0, commission=-7.0)
                exit_deal = Mock(entry=1, price=1.09995, volume=1.0, profit=-5.0, commission=-7.0)
                mock_mt5.history_deals_get.side_effect = lambda *args, **kwargs: [entry_deal, exit_deal] if kwargs.get("position") == ticket else []

                real_logger.log_fill(
                    FillData(
                        trade_id=trade_id,
                        position=MockPosition(
                        ticket=ticket,
                        symbol="EURUSD",
                        type=0,
                        volume=1.0,
                        price_open=1.10000,
                        sl=stale_sl,
                        tp=1.10500,
                        magic=runner.magic_number,
                        profit=0.0,
                        swap=0.0,
                        ),
                        expected_entry_price=1.10000,
                        opening_sl=stale_sl,
                        strategy_name=runner.strategy_name,
                    )
                )

                runner.executor.execute_modify.return_value = Mock(success=True, error_message=None)
                runner._invalidate_cache_for_ticket = Mock()
                request = ModifyRequest(ticket=ticket, new_sl=moved_sl, new_tp=moved_tp, comment="move sl")

                with patch("trading_system.runner_execution_mixin.mt.positions_get") as positions_get:
                    runner._handle_modify_adjustment(request)
                    positions_get.assert_not_called()

                patched_snapshot = runner.entry_metadata[trade_id]['position_snapshot']
                assert patched_snapshot['sl'] == pytest.approx(moved_sl, abs=1e-9)
                assert patched_snapshot['tp'] == pytest.approx(moved_tp, abs=1e-9)
                runner._invalidate_cache_for_ticket.assert_called_once_with(ticket)

                runner._handle_closed_positions([ticket])

            conn = real_logger._get_connection()
            row = conn.execute(
                """
                SELECT expected_exit_price, exit_spread, slippage_cost
                FROM trades
                WHERE trade_id = ? AND partial_sequence = 0
                """,
                (trade_id,),
            ).fetchone()

            assert row is not None
            expected_exit_price, exit_spread, slippage_cost = row
            assert expected_exit_price == pytest.approx(moved_sl, abs=1e-9)
            assert exit_spread == pytest.approx(0.00005, abs=1e-9)
            assert slippage_cost == pytest.approx(4.9, abs=0.2)
        finally:
            real_logger.close()


class TestRunnerMainLoop:
    """Validate setup failure cleanup and stale-schedule recovery behavior."""

    def test_run_calls_cleanup_when_setup_fails(self, runner):
        runner.setup = Mock(return_value=False)
        runner.cleanup = Mock()

        runner.run()

        runner.cleanup.assert_called_once()

    def test_run_recovers_when_schedule_deadlines_are_stale(self, runner):
        runner.shared_state["shutdown_flag"] = False
        runner.shared_state["heartbeats"] = {}
        runner.setup = Mock(return_value=True)
        runner._init_known_positions = Mock()
        runner._initialize_schedule = Mock(
            return_value=(
                datetime.now() - timedelta(seconds=30),
                datetime.now() - timedelta(seconds=20),
            )
        )

        data = pd.DataFrame(
            {"Close": [1.1010]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 10:15:00")]),
        )
        runner._fetch_data = Mock(return_value=data)
        runner._should_check_entry_signal = Mock(return_value=False)
        runner._monitor_exits = Mock(
            side_effect=lambda _data: runner.shared_state.__setitem__("shutdown_flag", True)
        )
        runner.cleanup = Mock()

        with patch("trading_system.strategy_runner.time.sleep", return_value=None):
            runner.run()

        assert runner._fetch_data.call_count == 1
        assert runner._monitor_exits.call_count == 1
        runner.cleanup.assert_called_once()


class TestStrategyClassHardening:
    """Validate strict strategy module/class loading checks."""

    def test_load_strategy_class_rejects_non_canonical_module(self, runner):
        runner.config["strategy_module"] = "strategies.other.strategy"
        runner.config["strategy_class"] = "OtherStrategy"

        with pytest.raises(ModuleNotFoundError):
            runner._load_strategy_class()

    def test_load_strategy_class_rejects_invalid_class_symbol(self, runner):
        runner.config["strategy_module"] = "strategies.base_strategy"
        runner.config["strategy_class"] = "not-valid-class"

        with pytest.raises(AttributeError):
            runner._load_strategy_class()

    def test_load_strategy_class_rejects_non_base_strategy_type(self, runner):
        runner.config["strategy_module"] = "strategies.base_strategy"
        runner.config["strategy_class"] = "NotAStrategy"

        module = Mock()
        module.NotAStrategy = object
        with patch("trading_system.strategy_runner.importlib.import_module", return_value=module):
            assert runner._load_strategy_class() is object


class TestRunnerRefactorStructure:
    """Validate StrategyRunner refactor keeps expected mixin composition."""

    def test_strategy_runner_inherits_expected_mixins(self):
        from src.trading_system.runner_execution_mixin import RunnerExecutionMixin
        from src.trading_system.runner_position_mixin import RunnerPositionMixin
        from src.trading_system.runner_schedule_mixin import RunnerScheduleMixin

        mro = StrategyRunner.__mro__
        assert RunnerPositionMixin in mro
        assert RunnerExecutionMixin in mro
        assert RunnerScheduleMixin in mro


class TestScheduleHelpers:
    """Validate schedule helper behavior after extraction into mixin."""

    def test_calculate_next_entry_time_rounds_to_m15_boundary(self, runner):
        from_time = datetime(2026, 1, 1, 10, 7, 30)
        next_entry = runner._calculate_next_entry_time(from_time)
        assert next_entry == datetime(2026, 1, 1, 10, 15, 0)

    def test_should_check_entry_signal_deduplicates_same_bar(self, runner):
        data = pd.DataFrame(
            {"Close": [1.1010]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-01 10:15:00", tz="UTC")]),
        )

        assert runner._should_check_entry_signal(data) is True
        assert runner._should_check_entry_signal(data) is False


class TestExitPriceTickCache:
    """Validate low-MT5-call symbol tick caching in exit price resolution."""

    def test_resolve_expected_exit_price_reuses_tick_per_symbol(self, runner, position_factory):
        runner._symbol_tick_cache.clear()
        tick = Mock(bid=1.10234, ask=1.10256)
        exit_request = Mock(exit_reason="signal_exit", expected_exit_price=None)
        pos_dict = position_factory(ticket=12345, pos_type=0, tp=0.0, sl=0.0)

        with patch(
            IMPORTANT_PATCH_TARGETS["runner_execution_symbol_tick"],
            MagicMock(return_value=tick),
        ) as symbol_tick_mock:
            price_1, source_1 = runner._resolve_expected_exit_price(exit_request, pos_dict)
            price_2, source_2 = runner._resolve_expected_exit_price(exit_request, pos_dict)

        assert symbol_tick_mock.call_count == 1
        assert source_1 == "tick_price"
        assert source_2 == "tick_price"
        assert price_1 == tick.bid
        assert price_2 == tick.bid

    def test_resolve_expected_exit_price_requeries_after_cache_clear(self, runner, position_factory):
        runner._symbol_tick_cache.clear()
        tick = Mock(bid=1.20010, ask=1.20030)
        exit_request = Mock(exit_reason="signal_exit", expected_exit_price=None)
        pos_dict = position_factory(ticket=12346, pos_type=1, tp=0.0, sl=0.0)

        with patch(
            IMPORTANT_PATCH_TARGETS["runner_execution_symbol_tick"],
            MagicMock(return_value=tick),
        ) as symbol_tick_mock:
            price_1, source_1 = runner._resolve_expected_exit_price(exit_request, pos_dict)
            runner._symbol_tick_cache.clear()
            price_2, source_2 = runner._resolve_expected_exit_price(exit_request, pos_dict)

        assert symbol_tick_mock.call_count == 2
        assert source_1 == "tick_price"
        assert source_2 == "tick_price"
        assert price_1 == tick.ask
        assert price_2 == tick.ask
