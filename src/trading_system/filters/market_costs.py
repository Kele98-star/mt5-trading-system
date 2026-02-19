from dataclasses import dataclass
import MetaTrader5 as mt


@dataclass
class MarketCondition:
    """Market condition validation result."""
    spread_points: float
    slippage_points: float
    spread_price: float
    slippage_price: float
    is_valid: bool
    reason: str


class MarketCostCalculator:
    """
    Validate spread and slippage with automatic normalization per symbol.
    
    Normalization Formula:
        Points = Price Difference / Symbol Point Size
    
    Example:
        - EURUSD: spread = 0.00020, point = 0.00001 -> 20 points
        - US100: spread = 0.50, point = 0.01 -> 50 points
    """
    
    def __init__(
        self,
        max_spread_points: float = 30.0,
        max_slippage_points: float = 20.0
    ):
        self.max_spread_points = max_spread_points
        self.max_slippage_points = max_slippage_points
    
    def validate_market_conditions(
        self,
        symbol: str,
        expected_price: float,
        is_buy: bool,
        cached_symbol_info=None
    ) -> MarketCondition:
        tick = mt.symbol_info_tick(symbol)
        symbol_info = cached_symbol_info if cached_symbol_info else mt.symbol_info(symbol)
        point = symbol_info.point

        spread_price = tick.ask - tick.bid
        spread_points = spread_price / point
        slippage_price = (tick.ask if is_buy else tick.bid) - expected_price
        slippage_points = abs(slippage_price / point)

        if spread_points > self.max_spread_points:
            is_valid = False
            reason = f"Spread too wide: {spread_points:.1f} points (max {self.max_spread_points:.1f})"
        elif slippage_points > self.max_slippage_points:
            is_valid = False
            reason = f"Slippage too high: {slippage_points:.1f} points (max {self.max_slippage_points:.1f})"
        else:
            is_valid = True
            reason = "Market conditions acceptable"

        return MarketCondition(
            spread_points=spread_points,
            slippage_points=slippage_points,
            spread_price=spread_price,
            slippage_price=slippage_price,
            is_valid=is_valid,
            reason=reason
        )