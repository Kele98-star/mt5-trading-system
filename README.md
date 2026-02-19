# MT5 Trading System

Multiprocessing framework for MetaTrader 5. The system runs isolated strategy processes, enforces account and strategy risk limits, minimizes MT5 calls through shared caching, and records trades to SQLite.

## Notes
- **Not suitable for HFT** — designed for swing/intraday strategies with moderate trade frequency.
- **Test on demo first** — always validate against a demo account before deploying live.
- **Strategy conflicts** — running many strategies with overlapping entry conditions may cause unexpected interactions; not designed for 20+ identical-signal strategies.

## Limitations
- **No cross-strategy risk netting** — risk limits are enforced per strategy. The orchestrator does not aggregate or net exposure across strategies trading the same instrument.
- **MT5 API coupling** — the entire data and execution layer is built around the MetaTrader 5 Python package. Porting to a different broker API or exchange requires rewriting the core layer.
- **No live parameter adjustment** — configuration is loaded at startup from env and config files. Changing parameters requires restarting the system.

## Architecture
- `scripts/run.py`: canonical CLI launcher wrapper.
- `src/trading_system/main.py`: runtime entry implementation (env resolution, logging bootstrap, orchestrator lifecycle).
- `src/trading_system/orchestrator.py`: strategy discovery, process supervision, shared position cache, heartbeat loop, daily resets, graceful shutdown.
- `src/trading_system/strategy_runner.py`: per-strategy loop for entry/exit/modify signals and execution coordination.
- `src/trading_system/core/`: MT5 connection, data caching, execution, risk, indicators, trade ID sequencing, logging.
- `src/trading_system/filters/`: news filter and meta-labeling loader.
- `strategies/`: strategy implementations and configs (lives at project root, not in `src/`).
- `tests/`: core unit tests.

## Project Structure
```
MT5/
├── scripts/
│   └── run.py                  # CLI entry point
├── src/
│   └── trading_system/
│       ├── main.py             # Orchestrator bootstrap
│       ├── orchestrator.py     # Process supervisor
│       ├── strategy_runner.py  # Per-strategy execution loop
│       ├── config/
│       │   ├── broker_config.py
│       │   └── risk_policies.py
│       ├── core/               # MT5 connection, execution, risk, data, indicators
│       ├── filters/            # News filter, meta-labeling
│       └── utils/
├── strategies/                 # Strategy implementations and configs
│   ├── base_strategy.py
│   └── sma_crossover/
├── tests/
├── requirements.txt
└── LICENSE
```

## Requirements
- Python 3.10+ (uses `X | Y` union type syntax throughout)
- `MetaTrader5` Python package and MT5 terminal
- Broker credentials and server access
- Windows host for live trading (MT5 runtime dependency)

## Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuration
Store broker env files outside the repo:

```bash
mkdir -p ~/.config/mt5-trading
chmod 700 ~/.config/mt5-trading
nano ~/.config/mt5-trading/.env.<broker>
chmod 600 ~/.config/mt5-trading/.env.<broker>
```

Required variables:
```bash
MT5_LOGIN=12345678
MT5_PASSWORD="your-password"
MT5_SERVER="YourBroker-Demo"
MT5_PATH="C:\\Path\\To\\terminal64.exe"
BROKER_TIMEZONE="Europe/Berlin"
```

Required when using the news filter:
```bash
MT5_CALENDAR_PATH="C:\\Path\\To\\MQL5\\Files\\calendar.txt"
```

This path points to the `calendar.txt` file exported by the MT5 economic calendar indicator into the terminal's `MQL5\Files\` directory. If this variable is not set and no `calendar_path` is provided explicitly, the news filter will be unable to load the calendar and will block all trading by default (`fail_open=False`).

Optional symbol overrides (used by some configs):
```bash
SYMBOL_DAX=GER40.cash
SYMBOL_NQ=US100.cash
SYMBOL_DOW=US30.cash
```

`--env` resolution order:
1. Absolute path passed to `--env`
2. `~/.config/mt5-trading/` (or `MT5_CONFIG_DIR`)
3. Project root

Account type is derived from the env filename, uppercased after cleanup (e.g. `.env.broker1` → `BROKER1`) and must match a key defined in `src/trading_system/config/risk_policies.py` (`ACCOUNT_STRATEGY_CONFIGS` and `ACCOUNT_RISK_POLICIES`).

## Run
```bash
python scripts/run.py --env .env.<broker>
```

## Tests
```bash
python -m pytest tests/
```

## Logs and State
- Orchestrator log: `logs/<env_name>/orchestrator_<env_name>.log`
- Strategy logs: `logs/<env_name>/<strategy_name>.log`
- Trade logs DB: `logs/<env_name>/trades/trades_<strategy_name>.db`
- Trade ID sequence DB: `logs/<env_name>/trade_id_sequence.db`

## Adding a Strategy
1. Create `strategies/<strategy_name>/config.py` and `strategy.py`.
2. Implement a strategy class inheriting `strategies/base_strategy.py`.
3. Register enablement in `ACCOUNT_STRATEGY_CONFIGS` in `src/trading_system/config/risk_policies.py`. The key must be the uppercased env filename (e.g. `BROKER1`).
4. Register risk allocation in `ACCOUNT_RISK_POLICIES` in `src/trading_system/config/risk_policies.py`.
5. Add strategy tests under `tests/`.

## Contributing
1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -r requirements.txt pytest`
3. Run the test suite: `python -m pytest tests/`
4. Ensure all tests pass before opening a pull request.
5. Open a PR with a clear description of the change and why it's needed.

## Risk Warning
Automated trading involves substantial financial risk. Past performance does not guarantee future results. This software is provided as-is with no warranty of any kind — see the [LICENSE](LICENSE) for details. Use at your own risk. Always test thoroughly on a demo account before deploying to a live account.
