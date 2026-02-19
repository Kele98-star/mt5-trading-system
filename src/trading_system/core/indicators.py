import pandas as pd
import numpy as np


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA)."""
    return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_sma(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    return df[column].rolling(window=period, min_periods=period).mean()


def calculate_ema(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    return df[column].ewm(span=period, min_periods=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    close = df[column].to_numpy(dtype=float, copy=False)
    delta = np.diff(close, prepend=np.nan)
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)
    avg_gain = _rma(pd.Series(gain, index=df.index, copy=False), period)
    avg_loss = _rma(pd.Series(loss, index=df.index, copy=False), period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return rs.ffill().fillna(50)


def calculate_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['High'].to_numpy(dtype=float, copy=False)
    low = df['Low'].to_numpy(dtype=float, copy=False)
    close = df['Close'].to_numpy(dtype=float, copy=False)

    up = np.diff(high, prepend=np.nan)
    dn = -np.diff(low, prepend=np.nan)
    pDM = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index, copy=False)
    mDM = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index, copy=False)

    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = _rma(pd.Series(tr, index=df.index, copy=False), period)
    atr_safe = atr.replace(0, np.nan)
    pDI = 100 * _rma(pDM, period) / atr_safe
    mDI = 100 * _rma(mDM, period) / atr_safe
    denom_raw = pDI + mDI
    denom = denom_raw.where(denom_raw != 0, 1)
    dx = (pDI - mDI).abs() / denom
    return 100 * _rma(dx, period)


def add_session_ranges(df: pd.DataFrame, start_time: str = "03:00", end_time: str = "04:30") -> pd.DataFrame:
    session = df.between_time(start_time, end_time)

    daily = (session.resample("1D").agg(RangeHigh=("High", "max"), RangeLow=("Low", "min")).assign(SessionRange=lambda x: ((x["RangeHigh"] - x["RangeLow"]) / x["RangeHigh"] * 100.0)))

    ranges = daily.reindex(df.index, method="ffill")

    return df.assign(RangeHigh=ranges["RangeHigh"].to_numpy(copy=False),RangeLow=ranges["RangeLow"].to_numpy(copy=False),SessionRange=ranges["SessionRange"].to_numpy(copy=False))
