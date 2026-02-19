import os
import logging
import pytz
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any
from collections import OrderedDict
import threading

import MetaTrader5 as mt

logger = logging.getLogger(__name__)

MT5_ENV_ALLOWED_KEYS = {
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "MT5_PATH",
    "BROKER_TIMEZONE",
    "MT5_CALENDAR_PATH",
    "MT5_CONFIG_DIR",
    "RISK_PROFILE",
}

MT5_ENV_ALLOWED_PREFIXES = ("SYMBOL_",)

class TimeFrame(IntEnum):
    M1 = mt.TIMEFRAME_M1
    M5 = mt.TIMEFRAME_M5
    M15 = mt.TIMEFRAME_M15
    M30 = mt.TIMEFRAME_M30
    H1 = mt.TIMEFRAME_H1
    H4 = mt.TIMEFRAME_H4
    D1 = mt.TIMEFRAME_D1
    W1 = mt.TIMEFRAME_W1
    MN1 = mt.TIMEFRAME_MN1

class OrderType(IntEnum):
    BUY = mt.ORDER_TYPE_BUY
    SELL = mt.ORDER_TYPE_SELL
    BUY_LIMIT = mt.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt.ORDER_TYPE_SELL_STOP
    BUY_STOP_LIMIT = mt.ORDER_TYPE_BUY_STOP_LIMIT
    SELL_STOP_LIMIT = mt.ORDER_TYPE_SELL_STOP_LIMIT

class OrderFilling(IntEnum):
    FOK = mt.ORDER_FILLING_FOK
    IOC = mt.ORDER_FILLING_IOC
    RETURN = mt.ORDER_FILLING_RETURN
    BOC = mt.ORDER_FILLING_BOC

class TradeAction(IntEnum):
    DEAL = mt.TRADE_ACTION_DEAL
    PENDING = mt.TRADE_ACTION_PENDING
    SLTP = mt.TRADE_ACTION_SLTP
    MODIFY = mt.TRADE_ACTION_MODIFY
    REMOVE = mt.TRADE_ACTION_REMOVE
    CLOSE_BY = mt.TRADE_ACTION_CLOSE_BY

class TimeInForce(IntEnum):
    GTC = mt.ORDER_TIME_GTC
    DAY = mt.ORDER_TIME_DAY
    SPECIFIED = mt.ORDER_TIME_SPECIFIED
    SPECIFIED_DAY = mt.ORDER_TIME_SPECIFIED_DAY

TIMEFRAME_TO_MINUTES = {
    TimeFrame.M1: 1,
    TimeFrame.M5: 5,
    TimeFrame.M15: 15,
    TimeFrame.M30: 30,
    TimeFrame.H1: 60,
    TimeFrame.H4: 240,
    TimeFrame.D1: 1440,
    TimeFrame.W1: 10080,
    TimeFrame.MN1: 43200,
}

TIMEFRAME_STRING_MAP = {
    'M1': TimeFrame.M1,
    'M5': TimeFrame.M5,
    'M15': TimeFrame.M15,
    'M30': TimeFrame.M30,
    'H1': TimeFrame.H1,
    'H4': TimeFrame.H4,
    'D1': TimeFrame.D1,
    'W1': TimeFrame.W1,
    'MN1': TimeFrame.MN1,
}

def string_to_timeframe(tf_str: str) -> TimeFrame | None:
    return TIMEFRAME_STRING_MAP.get(tf_str.upper())

def load_env_file(filepath: str = ".env",allowed_keys: set[str] | None = None,strict: bool = True,override_existing: bool = False) -> None:
    """
    Load environment key/value pairs with optional strict key validation.

    Strict mode:
    1. Ignore keys outside allowlist.
    2. Optionally preserve existing process-level env values.
    """
    env_path = Path(filepath)
    if not env_path.exists():
        return
    effective_allowed_keys = allowed_keys if allowed_keys is not None else MT5_ENV_ALLOWED_KEYS

    def is_allowed_key(key: str) -> bool:
        if key in effective_allowed_keys:
            return True
        return any(key.startswith(prefix) for prefix in MT5_ENV_ALLOWED_PREFIXES)

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized_key = key.strip()
            if strict and not is_allowed_key(normalized_key):
                logger.warning(f"EnvSkip key={normalized_key} | reason=not_allowed")
                continue
            if strict and not override_existing and normalized_key in os.environ:
                logger.warning(f"EnvSkip key={normalized_key} | reason=already_set")
                continue
            os.environ[normalized_key] = value.strip(' "\'')

@dataclass(frozen=True)
class MT5Config:
    login: int
    password: str
    server: str
    path: str
    broker_tz: Any

    @classmethod
    def from_env(cls, prefix: str = "MT5") -> "MT5Config":
        login = int(os.getenv(f"{prefix}_LOGIN"))
        password = os.getenv(f"{prefix}_PASSWORD")
        server = os.getenv(f"{prefix}_SERVER")
        path = os.getenv(f"{prefix}_PATH")
        broker_tz_str = os.getenv("BROKER_TIMEZONE")
        broker_tz = pytz.timezone(broker_tz_str)
        return cls(login=login, password=password, server=server, path=path, broker_tz=broker_tz)

@dataclass
class SymbolSpec:
    symbol: str
    description: str
    contract_size: float
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    bid: float
    ask: float
    spread: int
    spread_float: bool
    tick_size: float
    tick_value: float
    tick_value_profit: float
    tick_value_loss: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    trade_mode: int
    filling_mode: int
    stops_level: int
    freeze_level: int
    swap_long: float
    swap_short: float
    swap_mode: int
    asset_class: str = 'unknown'

    @classmethod
    def from_mt5(cls, symbol: str, asset_class: str = 'unknown') -> "SymbolSpec | None":
        info = mt.symbol_info(symbol)
        if info is None:
            logger.error(f"SymSpecFail sym={symbol} | reason=symbol_info_none")
            return None
        return cls(
            symbol=symbol,
            description=info.description,
            contract_size=info.trade_contract_size,
            point=info.point,
            digits=info.digits,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            bid=info.bid,
            ask=info.ask,
            spread=info.spread,
            spread_float=info.spread_float,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            tick_value_profit=info.trade_tick_value_profit,
            tick_value_loss=info.trade_tick_value_loss,
            currency_base=info.currency_base,
            currency_profit=info.currency_profit,
            currency_margin=info.currency_margin,
            trade_mode=info.trade_mode,
            filling_mode=info.filling_mode,
            stops_level=info.trade_stops_level,
            freeze_level=info.trade_freeze_level,
            swap_long=info.swap_long,
            swap_short=info.swap_short,
            swap_mode=info.swap_mode,
            asset_class=asset_class,
        )

    @property
    def filling_modes(self) -> list[OrderFilling]:
        bit_map = [(1, OrderFilling.FOK), (2, OrderFilling.IOC),
                   (4, OrderFilling.RETURN), (8, OrderFilling.BOC)]
        return [mode for bit, mode in bit_map if self.filling_mode & bit]

class SymbolRegistry:
    def __init__(self, max_cache_size: int = 2048):
        self._cache: OrderedDict[str, SymbolSpec | None] = OrderedDict()
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._max_cache_size = max_cache_size

    def get(self, symbol: str, asset_class: str = 'unknown') -> SymbolSpec | None:
        while True:
            with self._lock:
                if symbol in self._cache:
                    spec = self._cache[symbol]
                    self._cache.move_to_end(symbol)
                    return spec

                wait_event = self._inflight.get(symbol)
                if wait_event is None:
                    wait_event = threading.Event()
                    self._inflight[symbol] = wait_event
                    should_fetch = True
                else:
                    should_fetch = False

            if should_fetch:
                try:
                    spec = SymbolSpec.from_mt5(symbol, asset_class)
                    with self._lock:
                        self._cache[symbol] = spec
                        self._cache.move_to_end(symbol)
                        while len(self._cache) > self._max_cache_size:
                            self._cache.popitem(last=False)
                finally:
                    with self._lock:
                        done_event = self._inflight.pop(symbol, None)
                        if done_event is not None:
                            done_event.set()
                return spec

            wait_event.wait()

    def refresh(self, symbol: str, asset_class: str = 'unknown') -> SymbolSpec | None:
        with self._lock:
            self._cache.pop(symbol, None)
        return self.get(symbol, asset_class)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
        logger.info("SymCacheClear")

    def preload_symbols(self, symbols: list[str], asset_class: str = 'unknown'):
        for symbol in symbols:
            self.get(symbol, asset_class)

SYMBOL_REGISTRY = SymbolRegistry()

def get_symbol_spec(symbol: str, asset_class: str = 'unknown') -> SymbolSpec | None:
    return SYMBOL_REGISTRY.get(symbol, asset_class)
