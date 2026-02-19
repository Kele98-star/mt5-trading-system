import logging


def log_section_header(
    logger: logging.Logger,
    title: str,
    width: int = 87,
    level: int = logging.INFO,
) -> None:
    """Log a title wrapped by separator lines."""
    separator = "=" * max(1, int(width))
    logger.log(level, separator)
    logger.log(level, title)
    logger.log(level, separator)


def format_price_display(price: float, min_decimals: int = 2, max_decimals: int = 5) -> str:
    """Format price with adaptive decimals, bounded by min/max precision."""
    fixed = f"{price:.{max_decimals}f}"
    integer_part, fractional_part = fixed.split(".")
    trimmed_fractional = fractional_part.rstrip("0")
    if len(trimmed_fractional) < min_decimals:
        trimmed_fractional = trimmed_fractional.ljust(min_decimals, "0")
    return f"{integer_part}.{trimmed_fractional}"
