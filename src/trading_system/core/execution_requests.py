import math
from dataclasses import dataclass
from datetime import datetime


VALID_ORDER_TYPES = {'market', 'bracket', 'stop', 'limit'}


@dataclass
class EntryRequest:
    """
    Declarative entry order specification.

    Order type usage:
    - market: Uses signal, sl, tp
    - limit/stop: Uses signal, entry_price, sl, tp
    - bracket: Uses buy_stop, sell_stop, buy_sl, sell_sl (ignores signal, sl, tp)
    """
    order_type: str
    symbol: str
    volume: float
    signal: int  # 1=long, -1=short, 2=bracket (both directions)

    # Single-direction orders (market/limit/stop)
    entry_price: float | None = None
    sl: float | None = None
    tp: float | None = None

    # Bracket orders (OCO) - reuses entry_price fields differently
    buy_stop: float | None = None   # Buy side trigger
    sell_stop: float | None = None  # Sell side trigger
    buy_sl: float | None = None     # Buy side stop loss
    sell_sl: float | None = None    # Sell side stop loss
    buy_tp: float | None = None
    sell_tp: float | None = None

    # Metadata
    comment: str = ""
    strategy_name: str | None = None
    expiration_time: str | None = None

    def __post_init__(self):
        if self.order_type not in VALID_ORDER_TYPES:
            raise ValueError(f"Invalid order_type '{self.order_type}', must be one of {VALID_ORDER_TYPES}")
        if self.signal not in {1, -1, 2}:
            raise ValueError(f"Invalid signal {self.signal}, must be 1 (long), -1 (short), or 2 (bracket)")
        if self.volume < 0 or (not math.isfinite(self.volume)):
            raise ValueError(f"Invalid volume {self.volume}, must be >= 0 and finite")


@dataclass
class ExitRequest:
    """
    Declarative exit order specification. Supports full and partial closes.

    Note:
        expected_exit_price is optional. StrategyRunner derives expected exit from live tick
        price and uses this field as fallback when tick data is unavailable.
    """
    ticket: int
    portion: float = 1.0  # 1.0 = full close, 0.5 = half close
    comment: str = ""
    expected_exit_price: float | None = None

    # Optional SL/TP modification before exit
    new_sl: float | None = None
    new_tp: float | None = None

    # Metadata
    strategy_name: str | None = None
    exit_reason: str = ""

    def __post_init__(self):
        if self.ticket <= 0:
            raise ValueError(f"Invalid ticket {self.ticket}, must be > 0")
        if not (0 < self.portion <= 1.0):
            raise ValueError(f"Invalid portion {self.portion}, must be in (0, 1.0]")


@dataclass
class ModifyRequest:
    """Position modification request (SL/TP adjustment). Used for trailing stops and breakeven."""
    ticket: int
    new_sl: float | None = None
    new_tp: float | None = None
    comment: str = ""


@dataclass
class ExecutionResult:
    """Result of order execution attempt with fill details for slippage analysis."""
    success: bool
    ticket: int | None = None
    order_tickets: list[int] | None = None
    error_code: int | None = None
    error_message: str = ""

    # Execution details
    executed_price: float | None = None
    executed_volume: float | None = None
    timestamp: datetime | None = None
    deal_id: int | None = None
    # Request metadata (for logging)
    request_type: str = ""
    symbol: str = ""
