"""
Shared pytest fixtures for trading system tests.
"""

import pytest

from tests.util import ensure_mt5_stub, patch_important_dependencies


@pytest.fixture(scope="session", autouse=True)
def install_mt5_stub():
    """
    Guarantee MetaTrader5 is importable in the test environment.
    """
    ensure_mt5_stub()


@pytest.fixture
def important_mocks():
    """
    One-stop patch bundle for high-value external dependencies.
    """
    with patch_important_dependencies() as mocks:
        yield mocks
