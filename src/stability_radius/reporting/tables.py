from pathlib import Path
import pandas as pd


def write_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_topk_csv(
    df: pd.DataFrame, out_path: Path, by: str, k: int = 20, ascending: bool = True
):
    """
    Save top-k rows by a column.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    small = df.sort_values(by, ascending=ascending).head(k)
    small.to_csv(out_path, index=False)
