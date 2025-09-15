from pathlib import Path
import requests
from tqdm import tqdm


def download(
    url: str, dst: Path, chunk: int = 1 << 20, show_progress: bool = True
) -> None:
    """Stream a file to *dst* with an optional progress bar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        # Silence the progress bar if show_progress=False or length is unknown
        disable_bar = (not show_progress) or (total == 0)

        with dst.open("wb") as fh:
            if disable_bar:
                for chunk_data in r.iter_content(chunk_size=chunk):
                    if chunk_data:
                        fh.write(chunk_data)
            else:
                with tqdm(total=total, unit="B", unit_scale=True) as bar:
                    for chunk_data in r.iter_content(chunk_size=chunk):
                        if chunk_data:
                            size = fh.write(chunk_data)
                            bar.update(size)
