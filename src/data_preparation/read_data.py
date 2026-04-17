from typing import Union, Sequence, Optional
import gzip
import json
import os
import re
from pathlib import Path

from src.data_preparation.params import DataParams
from src.data_preparation.download_data import download
from src.data_preparation.data_structure import (
    UnitCommitmentInstance,
    UnitCommitmentScenario,
)
from src.data_preparation.utils import (
    from_json,
    repair_scenario_names_and_probabilities,
    migrate,
)


def _sanitize_identifier(s: str) -> str:
    """
    Turn an instance name like 'matpower/case300/2017-06-24'
    into a filesystem/log-friendly id: 'matpower_case300_2017-06-24'.
    """
    s = s.strip().strip("/\\")
    s = s.replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


def read_benchmark(name: str, *, quiet: bool = False) -> UnitCommitmentInstance:
    """
    Download (if necessary) a benchmark instance and load it.

    Example
    -------
    inst = read_benchmark("matpower/case3375wp/2017-02-01")
    """
    gz_name = f"{name}.json.gz"
    local_path = DataParams._CACHE / gz_name
    url = f"{DataParams.INSTANCES_URL}/{gz_name}"

    if not local_path.is_file():
        if not quiet:
            print(f"Downloading  {url}")
        # Hide the download progress bar when quiet=True (experiments mode)
        download(url, local_path, show_progress=not quiet)

    instance = _read(str(local_path), scenario_id_hint=name, quiet=quiet)

    if not quiet:
        print(f"→ Loaded instance '{name}' with {len(instance.scenarios)} scenarios.")
        print("Path to instance:", local_path)

    return instance


def _read(
    path_or_paths: Union[str, Sequence[str]],
    scenario_id_hint: Optional[str] = None,
    quiet: bool = False,
) -> UnitCommitmentInstance:
    """
    Generic loader.  Accepts:
      • single path (JSON or JSON.GZ) ➜ deterministic instance
      • list / tuple of paths           ➜ stochastic instance

    scenario_id_hint:
      When a single path is passed (deterministic case), use this hint to
      label the scenario name in a log-friendly way, so logs clearly show
      which dataset was solved.
    """
    if isinstance(path_or_paths, (list, tuple)):
        scenarios = [
            _read_scenario(p, quiet=quiet) for p in path_or_paths if isinstance(p, str)
        ]
        repair_scenario_names_and_probabilities(scenarios, list(path_or_paths))
    else:
        scenarios = [_read_scenario(path_or_paths, quiet=quiet)]
        # Name scenario using the original "name" hint if available; otherwise
        # fall back to a sanitized path-based id.
        if scenario_id_hint:
            scenarios[0].name = _sanitize_identifier(scenario_id_hint)
        else:
            try:
                rel = (
                    Path(path_or_paths)
                    .resolve()
                    .relative_to(DataParams._CACHE.resolve())
                )
                base = rel.as_posix()
                if base.endswith(".json.gz"):
                    base = base[: -len(".json.gz")]
                elif base.endswith(".json"):
                    base = base[: -len(".json")]
                scenarios[0].name = _sanitize_identifier(base)
            except Exception:
                scenarios[0].name = "scenario"
        scenarios[0].probability = 1.0

    return UnitCommitmentInstance(time=scenarios[0].time, scenarios=scenarios)


def _read_scenario(path: str, quiet: bool = False) -> UnitCommitmentScenario:
    raw = _read_json(path)
    migrate(raw)
    return from_json(raw, quiet=quiet)


class _JsonTextStream:
    def __init__(self, fh, chunk_size: int = 1 << 20):
        self._fh = fh
        self._chunk_size = chunk_size
        self._buf = ""
        self._pos = 0
        self._eof = False

    def _compact(self) -> None:
        if self._pos > 0:
            self._buf = self._buf[self._pos :]
            self._pos = 0

    def _fill(self, need: int = 1) -> None:
        while len(self._buf) - self._pos < need and not self._eof:
            chunk = self._fh.read(self._chunk_size)
            if not chunk:
                self._eof = True
                break
            if self._pos > 0 and (self._pos > (1 << 20) or self._pos > len(self._buf) // 2):
                self._compact()
            self._buf += chunk

    def peek(self) -> str:
        self._fill(1)
        if self._pos >= len(self._buf):
            return ""
        return self._buf[self._pos]

    def get(self) -> str:
        self._fill(1)
        if self._pos >= len(self._buf):
            raise EOFError("Unexpected end of JSON stream.")
        ch = self._buf[self._pos]
        self._pos += 1
        return ch

    def skip_ws(self) -> None:
        while True:
            ch = self.peek()
            if not ch or not ch.isspace():
                return
            self._pos += 1


def _read_json_string_text(stream: _JsonTextStream) -> str:
    if stream.get() != '"':
        raise ValueError("Expected JSON string.")
    parts = ['"']
    escaped = False
    while True:
        ch = stream.get()
        parts.append(ch)
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            return "".join(parts)


def _read_json_value_text(stream: _JsonTextStream) -> str:
    first = stream.peek()
    if not first:
        raise EOFError("Unexpected end of JSON stream while reading value.")
    if first == '"':
        return _read_json_string_text(stream)
    if first in "{[":
        parts = []
        depth = 0
        in_string = False
        escaped = False
        while True:
            ch = stream.get()
            parts.append(ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return "".join(parts)
        raise EOFError("Unbalanced JSON structure.")
    parts = []
    while True:
        ch = stream.peek()
        if not ch or ch.isspace() or ch in ",}]":
            return "".join(parts)
        parts.append(stream.get())


def _read_large_json_object(fh) -> dict:
    stream = _JsonTextStream(fh)
    stream.skip_ws()
    if stream.get() != "{":
        raise ValueError("Expected top-level JSON object.")
    result = {}
    stream.skip_ws()
    if stream.peek() == "}":
        stream.get()
        return result
    while True:
        stream.skip_ws()
        key = json.loads(_read_json_string_text(stream))
        stream.skip_ws()
        if stream.get() != ":":
            raise ValueError("Expected ':' after JSON object key.")
        stream.skip_ws()
        result[key] = json.loads(_read_json_value_text(stream))
        stream.skip_ws()
        sep = stream.get()
        if sep == "}":
            break
        if sep != ",":
            raise ValueError(f"Expected ',' or '}}' in top-level JSON object, got {sep!r}.")
    stream.skip_ws()
    if stream.peek():
        raise ValueError("Unexpected trailing data after JSON object.")
    return result


def _read_json(path: str) -> dict:
    """Open JSON or JSON.GZ transparently."""
    size_bytes = os.path.getsize(path)
    use_streaming = size_bytes >= 32 * (1 << 20)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            if use_streaming:
                return _read_large_json_object(fh)
            try:
                return json.load(fh)
            except MemoryError:
                fh.close()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return _read_large_json_object(fh)
    with open(path, "r", encoding="utf-8") as fh:
        if use_streaming:
            return _read_large_json_object(fh)
        try:
            return json.load(fh)
        except MemoryError:
            fh.close()
    with open(path, "r", encoding="utf-8") as fh:
        return _read_large_json_object(fh)
