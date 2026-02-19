# FILE: trading_system/main.py
# python scripts/run.py --env .env.broker1
# python scripts/run.py --env .env.broker2

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import signal
import logging

CONFIG_DIR = Path(os.environ.get("broker_config_DIR", str(Path.home() / ".config" / "mt5-trading")))

from src.trading_system.utils.system import WindowsInhibitor
from src.trading_system.utils.logging_utils import log_section_header
from src.trading_system.config.broker_config import MT5Config, load_env_file

logger = logging.getLogger(__name__)


class _ExcludeHeartbeatFromFileFilter(logging.Filter):
    """Exclude orchestrator heartbeat status lines from file logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "HB t=" not in record.getMessage()


def setup_logging(log_root: Path, clean_env_name: str) -> None:
    """Setup logging with environment-specific filenames to avoid conflicts."""
    log_root.mkdir(parents=True, exist_ok=True)
    log_filename = f"orchestrator_{clean_env_name}.log"
    file_handler = logging.FileHandler(log_root / log_filename)
    file_handler.addFilter(_ExcludeHeartbeatFromFileFilter())
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            stream_handler
        ],
        force=True,
    )

def resolve_env_path(env_arg: str) -> Path | None:
    """Resolve env file path: absolute → CONFIG_DIR → project root."""
    candidate = Path(env_arg).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for search_dir in (CONFIG_DIR, project_root):
        path = search_dir / env_arg
        if path.is_file():
            return path
    return None

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT5 Trading System Orchestrator")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Path to environment file. Searched in: ~/.config/mt5-trading/, then project root."
    )
    return parser.parse_args()

def _clean_env_name(env_path: Path) -> str:
    env_filename = env_path.name
    return env_filename.replace(".", "").replace("env", "")

def main() -> int:
    """Main entry point supporting multiple broker instances via CLI arguments."""
    args = _parse_args()

    env_path = resolve_env_path(args.env)
    if env_path is None:
        print(f"CRITICAL: Environment file '{args.env}' not found. Searched:")
        print(f"  1. {CONFIG_DIR}")
        print(f"  2. {project_root}")
        return 1
    
    print(f"Loading environment from: {env_path.name}")
    load_env_file(str(env_path), strict=True, override_existing=False)

    from src.trading_system.orchestrator import Orchestrator

    clean_env_name = _clean_env_name(env_path)
    log_root = project_root / "logs" / clean_env_name
    setup_logging(log_root, clean_env_name)

    log_section_header(logger, f"TRADING SYSTEM STARTING | Config: {env_path.name} | Log dir: {log_root}", level=logging.DEBUG)

    orchestrator = None
    shutdown_initiated = False

    def shutdown_once() -> None:
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return
        shutdown_initiated = True
        if orchestrator:
            orchestrator.shutdown()

    def signal_handler(sig, frame):
        """Handle Ctrl+C by delegating shutdown to finally block."""
        logger.info("Signal sig=SIGINT | action=shutdown")
        raise KeyboardInterrupt

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        account_type = clean_env_name.upper()
        logger.info(f"MainStart acct={account_type}")

        broker_config = MT5Config.from_env()
        logger.info("MT5 config loaded")

        orchestrator = Orchestrator(broker_config=broker_config, account_type=account_type, log_root=log_root)

        with WindowsInhibitor(keep_display=False, away_mode=True, logger=logger):
            orchestrator.start()

    except KeyboardInterrupt:
        logger.debug("MainStop reason=keyboard_interrupt")
    except Exception as e:
        logger.exception(f"MainCrash err={e}")
        return 1
    finally:
        shutdown_once()
        signal.signal(signal.SIGINT, previous_sigint)
        log_section_header(logger, "TRADING SYSTEM STOPPED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
