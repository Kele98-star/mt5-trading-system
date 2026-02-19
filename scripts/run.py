import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.trading_system.main import main

# python scripts/run.py --env .env.{broker_name}
if __name__ == "__main__":
    sys.exit(main())
