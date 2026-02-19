"""
Unit tests for bidirectional trade_id ↔ ticket mapping.

Tests the O(1) reverse lookup mechanism used in StrategyRunner
to correlate MT5 position tickets with internal trade IDs.

Test Coverage:
1. Bidirectional mapping creation on fill
2. Reverse lookup (ticket -> trade_id)
3. Orphaned position handling (no metadata)
4. Cleanup on position close
5. Concurrent fills (multiple positions)
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.util import MockPosition, MockStrategyRunner


class TestBidirectionalMapping(unittest.TestCase):
    """Test suite for trade_id ↔ ticket bidirectional mapping."""
    
    def setUp(self):
        """Create fresh mock runner for each test."""
        self.runner = MockStrategyRunner(strategy_name="test_strategy")
    
    def tearDown(self):
        """Clean up temporary files."""
        self.runner.cleanup()
    
    # =========================================================================
    # Test 1: Basic Bidirectional Mapping
    # =========================================================================
    
    def test_bidirectional_mapping_creation(self):
        """
        Verify bidirectional mapping is created on position fill.
        
        Flow:
        1. Generate trade_id and store metadata
        2. Store ticket -> trade_id mapping
        3. Verify reverse lookup works
        """
        # Arrange
        trade_id = 42
        ticket = 12345
        
        # Act: Simulate entry metadata storage
        self.runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09800,
            'submission_time': 1234567890.0,
        }
        self.runner.ticket_to_trade_id[ticket] = trade_id
        
        # Assert: Reverse lookup returns correct trade_id
        result = self.runner._get_trade_id_by_ticket(ticket)
        self.assertEqual(result, trade_id)
        
        # Assert: Forward lookup works
        self.assertIn(trade_id, self.runner.entry_metadata)
        self.assertEqual(self.runner.entry_metadata[trade_id]['ticket'], ticket)
    
    def test_orphaned_position_returns_none(self):
        """
        Verify orphaned positions (no metadata) return None on lookup.
        
        Scenario: Position exists from previous session, no metadata available.
        """
        # Act: Lookup unknown ticket
        result = self.runner._get_trade_id_by_ticket(99999)
        
        # Assert
        self.assertIsNone(result)
    
    # =========================================================================
    # Test 2: Full Lifecycle (Fill -> Close)
    # =========================================================================
    
    def test_fill_close_lifecycle(self):
        """
        Test complete position lifecycle with mapping creation/cleanup.
        
        Flow:
        1. Generate trade_id
        2. Position fills (creates mapping)
        3. Verify mapping exists
        4. Close position (cleanup mapping)
        5. Verify mapping removed
        """
        # Step 1: Generate trade_id and store metadata
        trade_id = self.runner._generate_trade_id()
        self.assertEqual(trade_id, 1)  # First ID should be 1
        
        ticket = 12345
        self.runner.entry_metadata[trade_id] = {
            'ticket': ticket,
            'expected_entry_price': 1.10000,
            'opening_sl': 1.09800,
            'submission_time': 1234567890.0,
        }
        self.runner.ticket_to_trade_id[ticket] = trade_id
        
        # Step 2: Position fills (simulate via _handle_new_fill)
        position_dict = {
            'ticket': ticket,
            'symbol': 'EURUSD',
            'type': 0,  # BUY
            'volume': 1.0,
            'price_open': 1.10000,
            'sl': 1.09800,
            'tp': 1.10500,
            'magic': 100001,
        }
        
        # Manually add to known_positions (normally done in _handle_new_fill)
        self.runner.known_positions.add(ticket)
        
        # Step 3: Verify mapping created
        self.assertEqual(self.runner._get_trade_id_by_ticket(ticket), trade_id)
        self.assertIn(ticket, self.runner.known_positions)
        
        # Step 4: Close position
        self.runner._log_full_close_execution(position_dict)
        
        # Step 5: Verify cleanup
        self.assertNotIn(ticket, self.runner.ticket_to_trade_id)
        self.assertNotIn(trade_id, self.runner.entry_metadata)
        self.assertNotIn(ticket, self.runner.known_positions)
    
    # =========================================================================
    # Test 3: Orphaned Position Handling
    # =========================================================================
    
    def test_orphaned_position_generates_fallback_trade_id(self):
        """
        Verify orphaned positions get new trade_id on fill detection.
        
        Scenario: System restart with existing MT5 positions.
        Expected: Generate new trade_id, create mapping, log fill.
        """
        # Arrange: Position exists but no metadata
        orphaned_ticket = 99999
        
        position_dict = {
            'ticket': orphaned_ticket,
            'symbol': 'GBPUSD',
            'type': 1,  # SELL
            'volume': 0.5,
            'price_open': 1.25000,
            'sl': 1.25500,
            'tp': 1.24000,
            'magic': 100002,
        }
        
        # Act: Handle new fill (should generate fallback trade_id)
        self.runner._handle_new_fill(position_dict)
        
        # Assert: Mapping created
        trade_id = self.runner._get_trade_id_by_ticket(orphaned_ticket)
        self.assertIsNotNone(trade_id)
        self.assertEqual(trade_id, 1)  # First generated ID
        
        # Assert: Metadata created
        self.assertIn(trade_id, self.runner.entry_metadata)
        self.assertEqual(
            self.runner.entry_metadata[trade_id]['ticket'], 
            orphaned_ticket
        )
        
        # Assert: Fill logged
        self.assertEqual(len(self.runner.trade_logger.fills_logged), 1)
        logged_fill = self.runner.trade_logger.fills_logged[0]
        self.assertEqual(logged_fill['trade_id'], trade_id)
        self.assertEqual(logged_fill['ticket'], orphaned_ticket)
    
    # =========================================================================
    # Test 4: Multiple Concurrent Positions
    # =========================================================================
    
    def test_multiple_positions_independent_mappings(self):
        """
        Verify multiple positions maintain independent mappings.
        
        Tests that trade_id ↔ ticket mappings don't interfere with each other.
        """
        # Arrange: Create 3 positions
        positions = [
            (1, 11111, 'EURUSD'),
            (2, 22222, 'GBPUSD'),
            (3, 33333, 'USDJPY'),
        ]
        
        # Act: Create mappings
        for trade_id, ticket, symbol in positions:
            self.runner.entry_metadata[trade_id] = {
                'ticket': ticket,
                'expected_entry_price': 1.10000,
                'opening_sl': 1.09800,
            }
            self.runner.ticket_to_trade_id[ticket] = trade_id
        
        # Assert: Each ticket maps to correct trade_id
        for trade_id, ticket, _ in positions:
            result = self.runner._get_trade_id_by_ticket(ticket)
            self.assertEqual(result, trade_id)
        
        # Assert: Reverse mappings unique
        self.assertEqual(len(self.runner.ticket_to_trade_id), 3)
        self.assertEqual(len(self.runner.entry_metadata), 3)
    
    def test_close_one_position_preserves_others(self):
        """
        Verify closing one position doesn't affect other mappings.
        """
        # Arrange: Create 3 positions
        positions = [
            (1, 11111, 'EURUSD'),
            (2, 22222, 'GBPUSD'),
            (3, 33333, 'USDJPY'),
        ]
        
        for trade_id, ticket, symbol in positions:
            self.runner.entry_metadata[trade_id] = {
                'ticket': ticket,
                'expected_entry_price': 1.10000,
                'opening_sl': 1.09800,
            }
            self.runner.ticket_to_trade_id[ticket] = trade_id
        
        # Act: Close middle position (trade_id=2, ticket=22222)
        position_dict = {
            'ticket': 22222,
            'symbol': 'GBPUSD',
            'type': 0,
            'volume': 1.0,
            'price_open': 1.25000,
            'sl': 1.24500,
            'tp': 1.26000,
            'magic': 100001,
        }
        self.runner._log_full_close_execution(position_dict)
        
        # Assert: Closed position removed
        self.assertIsNone(self.runner._get_trade_id_by_ticket(22222))
        self.assertNotIn(2, self.runner.entry_metadata)
        
        # Assert: Other positions preserved
        self.assertEqual(self.runner._get_trade_id_by_ticket(11111), 1)
        self.assertEqual(self.runner._get_trade_id_by_ticket(33333), 3)
        self.assertIn(1, self.runner.entry_metadata)
        self.assertIn(3, self.runner.entry_metadata)
    
    # =========================================================================
    # Test 5: Trade ID Uniqueness
    # =========================================================================
    
    def test_trade_id_uniqueness_across_fills(self):
        """
        Verify each fill gets unique trade_id even with identical tickets.
        
        Edge case: MT5 ticket reuse after position close.
        """
        # Arrange: Same ticket used twice (MT5 recycles tickets)
        ticket = 12345
        
        # Act: First fill
        trade_id_1 = self.runner._generate_trade_id()
        self.runner.entry_metadata[trade_id_1] = {'ticket': ticket}
        self.runner.ticket_to_trade_id[ticket] = trade_id_1
        
        # Simulate close (cleanup mapping)
        self.runner.entry_metadata.pop(trade_id_1)
        self.runner.ticket_to_trade_id.pop(ticket)
        
        # Act: Second fill (same ticket reused)
        trade_id_2 = self.runner._generate_trade_id()
        self.runner.entry_metadata[trade_id_2] = {'ticket': ticket}
        self.runner.ticket_to_trade_id[ticket] = trade_id_2
        
        # Assert: Different trade_ids
        self.assertNotEqual(trade_id_1, trade_id_2)
        self.assertEqual(trade_id_2, trade_id_1 + 1)
        
        # Assert: Current mapping points to latest trade_id
        self.assertEqual(self.runner._get_trade_id_by_ticket(ticket), trade_id_2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)