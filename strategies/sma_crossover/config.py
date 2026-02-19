from typing import Any


def get_config() -> dict[str, Any]:
    """Lazy config factory."""

    return {
        'name': 'sma_crossover',
        'order_type': 'market',
        'strategy_module': 'trading_system.strategies.sma_crossover.strategy',
        'strategy_class': 'SMACrossover',

        'params': {
            'symbol': 'EURUSD',
            'strategy_name': 'sma_crossover',
            'fast_period': 20,
            'slow_period': 50,
            'sl_points': 50,
            'timeframe': 'H1',
            'timezone': 'Europe/Berlin',
        },

        'execution': {
            'magic_number': 99,
            'deviation': 15,
            'comment_prefix': 'SMA_XO',
            'min_market_threshold_points': 15,
        },

        'filters': {
            'news_filter': {
                'enabled': False,
                'currencies': ['EUR', 'USD'],
                'buffer_minutes': 15,
            },
        },

        'trading_hours': {
            'timezone': 'Europe/Berlin',
            'enabled': False,
            'sessions': [
                {'start': '00:00', 'end': '23:00'},
            ],
        },

        'risk': {
            'position_sizing_method': 'fractional',
            'max_positions': 1,
            'max_trades': 1,
            'max_spread_points': 10,
            'max_slippage_points': 10,
            'max_order_fill_time_seconds': 1,
        },
    }