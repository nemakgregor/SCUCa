import os
import time
from pathlib import Path
import requests
from tqdm import tqdm


def download(
    url: str,
    dst: Path,
    chunk: int = 1 << 20,
    show_progress: bool = True,
    attempts: int = 3,
    timeout: int = 60,
) -> None:
    """
    Stream a file to *dst* with an optional progress bar.

    Retries transient network failures up to ``attempts`` times.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tries = max(1, int(attempts))
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=max(1, int(timeout))) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))

                # Silence the progress bar if show_progress=False or length is unknown
                disable_bar = (not show_progress) or (total == 0)

                with tmp.open("wb") as fh:
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
                    fh.flush()
                    os.fsync(fh.fileno())
            os.replace(tmp, dst)
            return
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt >= tries:
                raise
            # Exponential backoff capped to keep long runs responsive.
            time.sleep(min(30.0, 2.0 ** float(attempt - 1)))
