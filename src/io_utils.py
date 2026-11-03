"""Save/load tables as Parquet when pyarrow is installed, else compressed CSV."""
from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, stem: Path) -> Path:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = stem.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except ImportError:
        p = stem.with_suffix(".csv.gz")
        df.to_csv(p, index=False)
        return p


def load_table(stem: Path, date_cols=("date", "expiry")) -> pd.DataFrame:
    stem = Path(stem)
    p = stem.with_suffix(".parquet")
    if p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    k = stem.with_suffix(".pkl")
    if k.exists():
        return pd.read_pickle(k)
    c = stem.with_suffix(".csv.gz")
    if c.exists():
        df = pd.read_csv(c)
        for col in date_cols:
            if col in df:
                df[col] = pd.to_datetime(df[col])
        return df
    raise FileNotFoundError(f"neither {p} nor {c} found")
