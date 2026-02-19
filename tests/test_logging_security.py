"""Security-focused logging tests."""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.util import ensure_mt5_stub

ensure_mt5_stub()

from trading_system import main as main_module
from trading_system.core.connection import MT5Connection


def test_main_uses_env_basename_and_redacted_startup_log(caplog):
    """Use env basename in startup print and avoid account/server disclosure logs."""

    class _FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            self.started = False

        def start(self):
            self.started = True

        def shutdown(self):
            return None

    caplog.set_level(logging.INFO)
    env_path = Path("/sensitive/path/.env.broker1")

    with patch("trading_system.main._parse_args", return_value=argparse.Namespace(env=".env.broker1")):
        with patch("trading_system.main.resolve_env_path", return_value=env_path):
            with patch("trading_system.main.load_env_file") as load_env_mock:
                with patch("trading_system.main.setup_logging"):
                    with patch("trading_system.main.WindowsInhibitor", side_effect=lambda *a, **k: nullcontext()):
                        with patch("trading_system.main.MT5Config.from_env", return_value=SimpleNamespace(login=123456, server="Demo")):
                            with patch("src.trading_system.orchestrator.Orchestrator", _FakeOrchestrator):
                                with patch("trading_system.main.signal.getsignal", return_value=MagicMock()):
                                    with patch("trading_system.main.signal.signal"):
                                        with patch("builtins.print") as print_mock:
                                            exit_code = main_module.main()

    assert exit_code == 0
    print_mock.assert_any_call("Loading environment from: .env.broker1")
    load_env_mock.assert_called_once_with(str(env_path), strict=True, override_existing=False)
    assert "/sensitive/path" not in caplog.text
    assert "Acct=" not in caplog.text
    assert "Srv=" not in caplog.text
    assert "MT5 config loaded" in caplog.text


def test_connection_logs_do_not_expose_sensitive_account_fields(caplog):
    """Log generic connection markers without login, balance, or terminal path."""
    caplog.set_level(logging.DEBUG)

    config = SimpleNamespace(
        login=123456,
        password="secret",
        server="DemoServer",
        path="/private/terminal/path",
    )
    account_info = SimpleNamespace(login=123456, server="DemoServer", balance=98765.43)
    terminal_info = SimpleNamespace(connected=True)
    connection = MT5Connection(config=config)

    with patch("trading_system.core.connection.mt5.initialize", return_value=True):
        with patch("trading_system.core.connection.mt5.shutdown"):
            with patch("trading_system.core.connection.mt5.account_info", return_value=account_info):
                with patch(
                    "trading_system.core.connection.mt5.terminal_info",
                    side_effect=[None, terminal_info],
                ):
                    assert connection.connect(max_retries=1, retry_delay=0.0) is True
                    assert connection.is_connected(use_cache=False) is True

    logs = caplog.text
    assert "ConnOK" in logs
    assert "ConnCheckOK" in logs
    assert "123456" not in logs
    assert "98765.43" not in logs
    assert "/private/terminal/path" not in logs
