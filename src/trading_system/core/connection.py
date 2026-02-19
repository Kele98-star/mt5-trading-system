import time
import logging
import MetaTrader5 as mt5
from enum import Enum

from src.trading_system.config.broker_config import MT5Config

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MT5 connection states for reconnection management."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


class MT5Connection:
    MAX_BACKOFF_SECONDS = 30

    def __init__(self, config: MT5Config, max_reconnection_attempts: int = 5,
                 connection_check_ttl: int = 60):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.reconnection_attempts = 0
        self.max_reconnection_attempts = max_reconnection_attempts
        self.connection_check_ttl = connection_check_ttl
        self._cached_connection_state: bool | None = None
        self._cache_timestamp_monotonic: float = 0.0

    def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """Establish MT5 connection with retry logic."""
        if self.is_connected(use_cache=False):
            logger.debug("ConnSkip state=already_connected")
            return True

        init_kwargs = {
            'login': self.config.login,
            'password': self.config.password,
            'server': self.config.server,
            'path': self.config.path,
        }

        for attempt in range(max_retries):
            if attempt > 0:
                mt5.shutdown()

            if not mt5.initialize(**init_kwargs):
                logger.warning(f"ConnInitFail n={attempt+1}/{max_retries} | err={mt5.last_error()}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue

            account_info = mt5.account_info()
            if not self._validate_account(account_info):
                mt5.shutdown()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue

            logger.debug("ConnOK")
            self._set_connected()
            return True

        logger.error(f"ConnFail tries={max_retries}")
        self._set_disconnected()
        return False
    
    def reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff."""
        if self.state == ConnectionState.RECONNECTING:
            logger.warning("ReconnSkip reason=in_progress")
            return False

        self.state = ConnectionState.RECONNECTING
        self.reconnection_attempts += 1

        if self.reconnection_attempts > self.max_reconnection_attempts:
            logger.critical(f"ReconnFail tries={self.max_reconnection_attempts} | action=manual")
            self._set_disconnected()
            return False

        backoff_seconds = min(2 ** (self.reconnection_attempts - 1), self.MAX_BACKOFF_SECONDS)
        logger.warning(f"ReconnStart n={self.reconnection_attempts}/{self.max_reconnection_attempts} | backoff={backoff_seconds}s")
        time.sleep(backoff_seconds)

        try:
            connected = self.connect(max_retries=3, retry_delay=1.0)
        except Exception:
            self._set_disconnected()
            raise

        if connected:
            logger.debug(f"ReconnOK n={self.reconnection_attempts}")
            return True

        logger.error(f"ReconnFail n={self.reconnection_attempts}")
        self._set_disconnected()
        return False

    def disconnect(self) -> None:
        """Graceful disconnection."""
        try:
            mt5.shutdown()
            logger.debug("ConnClosed")
            self._set_disconnected()
        except (OSError, RuntimeError) as e:
            logger.error(f"ConnCloseErr err={e}", exc_info=True)
            self._set_disconnected()
        except Exception as e:
            logger.critical(f"ConnCloseFatal err={e}", exc_info=True)
            self._set_disconnected()
            raise
    
    def is_connected(self, use_cache: bool = True) -> bool:
        """Check if MT5 terminal is alive and authenticated. Uses cached result within TTL."""
        if use_cache and self._is_cache_valid():
            return self._cached_connection_state

        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            self._set_disconnected()
            return False

        account_info = mt5.account_info()
        if not self._validate_account(account_info):
            self._set_disconnected()
            return False

        logger.debug("ConnCheckOK")
        self._set_connected()
        return True

    def ensure_connected(self) -> bool:
        """Ensure MT5 connection is active, attempt reconnection if disconnected."""
        if self.is_connected():
            return True

        logger.warning("ConnLost action=reconnect")
        return self.reconnect()

    def _validate_account(self, account_info) -> bool:
        """Verify connection is to the correct account."""
        if account_info is None or account_info.login != self.config.login:
            logger.error("ConnAcctErr reason=account_mismatch")
            return False
        if getattr(account_info, "server", None) != self.config.server:
            logger.error("ConnAcctErr reason=server_mismatch")
            return False
        return True

    def _is_cache_valid(self) -> bool:
        """Check if cached connection state is still valid."""
        if self._cached_connection_state is None:
            return False
        cache_age = time.monotonic() - self._cache_timestamp_monotonic
        return cache_age < self.connection_check_ttl

    def _set_connected(self) -> None:
        """Update state to connected and refresh cache."""
        self.state = ConnectionState.CONNECTED
        self.reconnection_attempts = 0
        self._update_cache(True)

    def _set_disconnected(self) -> None:
        """Update state to disconnected and invalidate cache."""
        self.state = ConnectionState.DISCONNECTED
        self.invalidate_cache()

    def _update_cache(self, connection_state: bool) -> None:
        """Update the connection state cache with current timestamp."""
        self._cached_connection_state = connection_state
        self._cache_timestamp_monotonic = time.monotonic()

    def invalidate_cache(self) -> None:
        """Manually invalidate the connection state cache."""
        self._cached_connection_state = None
        self._cache_timestamp_monotonic = 0.0
