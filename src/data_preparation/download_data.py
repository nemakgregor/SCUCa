# src/data_preparation/download_data.py
import requests
from pathlib import Path
from tqdm import tqdm


def download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    """Stream a file to *dst* with a progress bar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            dst.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, disable=total == 0) as bar,
        ):
            for chunk_data in r.iter_content(chunk_size=chunk):
                size = fh.write(chunk_data)
                bar.update(size)
