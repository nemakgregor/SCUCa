import gurobipy as gp


def apply_branching_hints_from_starts(model: gp.Model) -> int:
    """
    Use current Start values as branching hints for commitment variables:
      - Set VarHintVal to Start (0/1)
      - Set VarHintPri proportional to temporal importance (earlier t higher)
      - Set BranchPriority to prefer early-time commitment vars

    Returns the number of variables with hints applied.
    """
    commit = getattr(model, "commit", None)
    if commit is None:
        return 0

    # Infer time horizon from keys
    times = set()
    for gname, t in commit.keys():
        times.add(int(t))
    if not times:
        return 0
    T = 1 + max(times)

    applied = 0
    for gname, t in commit.keys():
        var = commit[gname, t]
        try:
            st = float(var.Start)
        except Exception:
            st = None
        # Only hint when we have a Start (warm-start likely set it)
        if st is None:
            continue

        try:
            # VarHintVal must be numeric 0/1 for binaries
            var.VarHintVal = 1.0 if st >= 0.5 else 0.0
        except Exception:
            pass

        try:
            # Higher priority for earlier periods
            # Scale priority in [10, 10+T], always >=10 to stand out
            pri = 10 + (T - int(t))
            var.VarHintPri = float(pri)
        except Exception:
            pass

        try:
            # BranchPriority uses integer scale (0 default). Use same scheme.
            var.BranchPriority = int(10 + (T - int(t)))
        except Exception:
            pass

        applied += 1

    return applied
