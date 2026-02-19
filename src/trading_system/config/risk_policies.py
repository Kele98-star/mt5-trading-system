from typing import Any
SYSTEM_TIMINGS = {
    'heartbeat_interval': 300,  # Heartbeat interval in seconds (5 minutes)
    'heartbeat_log_interval': 900,  # Heartbeat interval in seconds (15 minutes)
    'cache_staleness_threshold': 60,
}

BASE_RISK_VALUES_BROKER1 = {
    "sma_crossover": 1,
}

BASE_RISK_VALUES_BROKER2 = {
    "sma_crossover": 1,
}

STRATEGY_NAMES = sorted(set(BASE_RISK_VALUES_BROKER1) | set(BASE_RISK_VALUES_BROKER2))

BROKER1_ENABLED_STRATEGIES = {name: True for name in BASE_RISK_VALUES_BROKER1}
BROKER2_ENABLED_STRATEGIES = {name: True for name in BASE_RISK_VALUES_BROKER2}

STRATEGY_RISK_BROKER1 = {k: v * 0.01 / 4 for k, v in BASE_RISK_VALUES_BROKER1.items()}
STRATEGY_RISK_BROKER2 = {k: v * 0.01 for k, v in BASE_RISK_VALUES_BROKER2.items()}

ACCOUNT_STRATEGY_CONFIGS = {
    "BROKER1": BROKER1_ENABLED_STRATEGIES,
    "BROKER2": BROKER2_ENABLED_STRATEGIES,
}

ACCOUNT_RISK_POLICIES = {
    "BROKER1": STRATEGY_RISK_BROKER1,
    "BROKER2": STRATEGY_RISK_BROKER2,
}

ADAPTIVE_SIZING_TEMPLATE = {
    'enabled': False,
    'scope': 'portfolio',
    'drawdown_thresholds': [
        {'drawdown_pct': 0.05, 'risk_multiplier': 0.5},
        {'drawdown_pct': 0.10, 'risk_multiplier': 0.25},
    ],
}


def _build_adaptive_sizing_config() -> dict[str, Any]:
    """Return per-policy adaptive sizing config to avoid cross-policy mutation."""
    return {
        'enabled': ADAPTIVE_SIZING_TEMPLATE['enabled'],
        'scope': ADAPTIVE_SIZING_TEMPLATE['scope'],
        'drawdown_thresholds': [
            threshold.copy() for threshold in ADAPTIVE_SIZING_TEMPLATE['drawdown_thresholds']
        ],
    }


GLOBAL_POLICY_BROKER1 = {
    'max_total_positions': 7,
    'max_daily_drawdown_pct': 0.03,
    'max_drawdown_pct': 0.10,
    'max_daily_trades': 10,
    'initial_balance': 50_000,
    'adaptive_sizing': _build_adaptive_sizing_config(),
}

GLOBAL_POLICY_BROKER2 = {
    'max_total_positions': 7,
    'max_daily_drawdown_pct': 0.1,
    'max_drawdown_pct': 0.30,
    'max_daily_trades': 10,
    'initial_balance': 10_000,
    'adaptive_sizing': _build_adaptive_sizing_config(),
}

ACCOUNT_GLOBAL_POLICIES = {
    'BROKER1': GLOBAL_POLICY_BROKER1,
    'BROKER2': GLOBAL_POLICY_BROKER2,
}

def inject_risk_per_trade(strategy_config: dict[str, Any], strategy_name: str, account_type: str):
    if account_type not in ACCOUNT_RISK_POLICIES:
        raise KeyError(f"Unknown account_type '{account_type}'. Valid: {list(ACCOUNT_RISK_POLICIES.keys())}")

    risk_config = strategy_config.get('risk')
    if not isinstance(risk_config, dict):
        raise KeyError("strategy_config missing required 'risk' dictionary")

    account_policy = ACCOUNT_RISK_POLICIES[account_type]
    if strategy_name not in account_policy:
        raise KeyError(
            f"Unknown strategy_name '{strategy_name}' for account '{account_type}'. "
            f"Available: {list(account_policy.keys())}"
        )

    risk_config['risk_per_trade'] = account_policy[strategy_name]
    return strategy_config
