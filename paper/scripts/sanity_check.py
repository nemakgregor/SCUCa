"""Quick LaTeX sanity check — no compilation, just structural consistency."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper.tex"


def main() -> int:
    src = PAPER.read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{([^}]+)\}", src)
    # include labels from \input'd files, because LaTeX will see them
    full_src = src
    for fname in inputs:
        fp = PAPER.parent / fname
        if fp.exists():
            full_src += "\n" + fp.read_text(encoding="utf-8")

    labels = set(re.findall(r"\\label\{([^}]+)\}", full_src))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
    eqrefs = set(re.findall(r"\\eqref\{([^}]+)\}", src))
    refs |= eqrefs
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)

    broken_refs = sorted(refs - labels)
    unused_labels = sorted(labels - refs)

    print(f"labels        : {len(labels):2d}  {sorted(labels)}")
    print(f"refs          : {len(refs):2d}  {sorted(refs)}")
    print(f"broken refs   : {broken_refs or 'OK'}")
    print(f"unused labels : {unused_labels or 'OK'}")
    print(f"\\input files : {inputs}")
    print(f"figures       : {figs}")

    begins = len(re.findall(r"\\begin\{", src))
    ends = len(re.findall(r"\\end\{", src))
    print(f"begin/end env : begin={begins}  end={ends}  match={'OK' if begins==ends else 'MISMATCH'}")

    root = PAPER.parent
    missing_inputs = [f for f in inputs if not (root / f).exists()]
    missing_figs = [
        f for f in figs
        if not any((root / (f + ext)).exists() for ext in ["", ".png", ".pdf", ".jpg"])
    ]
    print(f"missing inputs: {missing_inputs or 'OK'}")
    print(f"missing figs  : {missing_figs or 'OK'}")

    # cite keys
    cites = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", src):
        for key in m.group(1).split(","):
            cites.add(key.strip())
    bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", src))
    uncited = sorted(bib_keys - cites)
    missing_cites = sorted(cites - bib_keys)
    print(f"\\cite keys   : {len(cites)}")
    print(f"\\bibitem keys: {len(bib_keys)}")
    print(f"missing cites : {missing_cites or 'OK'}")
    print(f"unused bib    : {uncited or 'OK'}")

    ok = not (broken_refs or missing_inputs or missing_figs or missing_cites) and begins == ends
    print("\nRESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
