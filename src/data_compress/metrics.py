from __future__ import annotations

import math

EPS = 1e-12


def _flatten(values):
    if isinstance(values, list):
        for v in values:
            yield from _flatten(v)
    else:
        yield float(values)


def error_stats(original, restored) -> dict[str, float]:
    o = list(_flatten(original))
    r = list(_flatten(restored))
    diff = [abs(a - b) for a, b in zip(o, r)]
    mean = sum(diff) / max(len(diff), 1)
    rmse = math.sqrt(sum(d * d for d in diff) / max(len(diff), 1))
    p99_idx = min(len(diff) - 1, int(0.99 * len(diff))) if diff else 0
    diff_sorted = sorted(diff) if diff else [0.0]
    max_rel = max((d / max(abs(a), 1e-3) for d, a in zip(diff, o)), default=0.0)
    return {
        "mae": mean,
        "rmse": rmse,
        "maxae": max(diff, default=0.0),
        "p99ae": diff_sorted[p99_idx],
        "max_rel": max_rel,
    }
