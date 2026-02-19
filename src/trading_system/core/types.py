"""
Type definitions for multiprocessing primitives.
"""
from dataclasses import dataclass
from typing import Any, Protocol
__all__ = [
    'AtomicInt',
    'ProcessLock',
    'PositionSnapshot',
    'PartialClosePositionSnapshot',
    'normalize_position',
    'snapshot_partial_close_position',
]


class AtomicInt(Protocol):
    """Protocol for multiprocessing.Value('i', default). All .value ops are serialized via lock."""
    value: int

    def get_lock(self) -> 'ProcessLock': ...


class ProcessLock(Protocol):
    """Protocol for multiprocessing.Lock(). Supports context manager usage."""
    def acquire(self, block: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> bool: ...
    def __exit__(self, *args) -> None: ...


_POSITION_FIELDS = ('ticket', 'symbol', 'type', 'volume', 'price_open',
                    'sl', 'tp', 'profit', 'swap', 'magic', 'time')
_PARTIAL_CLOSE_REQUIRED_FIELDS = ('ticket', 'symbol', 'type')


@dataclass
class PositionSnapshot:
    ticket: int
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float
    magic: int
    symbol: str
    time: int


@dataclass(slots=True, frozen=True)
class PartialClosePositionSnapshot:
    """Lightweight position snapshot for partial-close logging paths."""
    ticket: int
    type: int
    symbol: str
    swap: float = 0.0


def normalize_position(
    position: dict[str, Any] | Any,
    required_fields: tuple = _POSITION_FIELDS
) -> dict[str, Any]:
    """Convert MT5 position object or dict to standardized dictionary format."""
    if isinstance(position, dict):
        return position
    return {field: getattr(position, field) for field in required_fields}


def snapshot_partial_close_position(
    position: dict[str, Any] | Any | PartialClosePositionSnapshot
) -> PartialClosePositionSnapshot:
    """Convert MT5 position object/dict to minimal partial-close snapshot."""
    if isinstance(position, PartialClosePositionSnapshot):
        return position

    if isinstance(position, dict):
        swap = position.get('swap', 0.0)
        return PartialClosePositionSnapshot(
            ticket=int(position['ticket']),
            type=int(position['type']),
            symbol=str(position['symbol']),
            swap=float(0.0 if swap is None else swap),
        )

    swap = getattr(position, 'swap', 0.0)
    return PartialClosePositionSnapshot(
        ticket=int(getattr(position, 'ticket')),
        type=int(getattr(position, 'type')),
        symbol=str(getattr(position, 'symbol')),
        swap=float(0.0 if swap is None else swap),
    )
