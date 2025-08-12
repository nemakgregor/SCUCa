def compute_total_load(buses, T):
    """Compute total system load at each time period."""

    total_load = [sum(b.load[t] for b in buses) for t in range(T)]
    return total_load
