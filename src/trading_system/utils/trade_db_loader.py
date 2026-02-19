import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd


def get_project_root() -> Path:
    """Return repository root directory from module location."""
    return Path(__file__).resolve().parents[3]


def resolve_broker_trades_dir(broker_name: str, project_root: Path | None = None) -> Path:
    """Resolve broker trades directory under logs."""
    root = project_root if project_root is not None else get_project_root()
    return root / "logs" / broker_name / "trades"


def discover_strategy_trade_dbs(trades_dir: Path) -> list[Path]:
    """Discover strategy trade databases in broker trades directory."""
    if not trades_dir.exists():
        return []
    return sorted(path for path in trades_dir.glob("trades_*.db") if path.is_file())


def _build_select_query(columns: Sequence[str] | None) -> str:
    """Build SELECT query for trade table."""
    if columns is None:
        return "SELECT * FROM trades"
    if not columns:
        raise ValueError("columns must contain at least one column name")
    projected = ", ".join(f'"{column}"' for column in columns)
    return f"SELECT {projected} FROM trades"


def _trade_table_exists(connection: sqlite3.Connection) -> bool:
    """Return whether trades table exists in database."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades' LIMIT 1"
    ).fetchone()
    return row is not None


def load_trade_db(
    db_path: Path,
    columns: Sequence[str] | None = None,
    include_source_metadata: bool = True,
) -> pd.DataFrame:
    """
    Load one strategy trade database into a pandas DataFrame.

    Use SQLite read-only mode to avoid interfering with live writer processes.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    try:
        if not _trade_table_exists(connection):
            return pd.DataFrame()

        query = _build_select_query(columns)
        frame = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    if frame.empty:
        return frame

    if include_source_metadata:
        strategy_name = db_path.stem.removeprefix("trades_")
        frame.insert(0, "source_strategy", strategy_name)
        frame.insert(1, "source_db_path", str(db_path))

    return frame


def load_broker_strategy_trades(
    broker_name: str,
    project_root: Path | None = None,
    columns: Sequence[str] | None = None,
    include_source_metadata: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate all strategy trade databases for one broker.

    Concatenate once at the end to keep memory overhead low.
    """
    trades_dir = resolve_broker_trades_dir(broker_name=broker_name, project_root=project_root)
    db_paths = discover_strategy_trade_dbs(trades_dir)
    if not db_paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for db_path in db_paths:
        frame = load_trade_db(
            db_path=db_path,
            columns=columns,
            include_source_metadata=include_source_metadata,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, copy=False)
