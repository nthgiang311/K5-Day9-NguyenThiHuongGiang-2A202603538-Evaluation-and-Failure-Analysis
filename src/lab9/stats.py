"""Statistics students implement (M2 judge calibration, M4 paired bootstrap).

Rules: a judge is a classifier -- measure it like one. A higher mean is not
"better" until the paired CI of the difference excludes 0.
Reference values to self-check are in tests/test_lab9_stats.py.
"""

from __future__ import annotations

from collections.abc import Sequence

POS = "FAIL"   # the judge's job is to FIND failures, so FAIL is the positive class


def confusion(human: Sequence[str], judge: Sequence[str]) -> dict[str, int]:
    """tp/fp/fn/tn with FAIL as positive."""
    if len(human) != len(judge):
        raise ValueError("human and judge labels must have the same length")
    if any(str(x).upper() not in {"PASS", "FAIL"} for x in (*human, *judge)):
        raise ValueError("labels must be PASS or FAIL")
    h = [str(x).upper() for x in human]
    j = [str(x).upper() for x in judge]
    return {
        "tp": sum(a == POS and b == POS for a, b in zip(h, j)),
        "fp": sum(a != POS and b == POS for a, b in zip(h, j)),
        "fn": sum(a == POS and b != POS for a, b in zip(h, j)),
        "tn": sum(a != POS and b != POS for a, b in zip(h, j)),
    }


def precision_recall(human: Sequence[str], judge: Sequence[str]) -> tuple[float, float]:
    c = confusion(human, judge)
    return (c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0,
            c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0)


def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). Raise ValueError when undefined (p_e == 1):
    a constant judge has no signal and must FAIL calibration, not pass on a technicality."""
    c = confusion(human, judge)
    n = sum(c.values())
    if not n:
        raise ValueError("kappa is undefined for an empty label set")
    observed = (c["tp"] + c["tn"]) / n
    human_fail = c["tp"] + c["fn"]
    judge_fail = c["tp"] + c["fp"]
    expected = (human_fail * judge_fail + (n - human_fail) * (n - judge_fail)) / (n * n)
    if expected == 1:
        raise ValueError("kappa is undefined when expected agreement is 1")
    return (observed - expected) / (1 - expected)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed failure rate for judge error; clip to [0, 1]."""
    if not all(0 <= x <= 1 for x in (observed_rate, sensitivity, specificity)):
        raise ValueError("rates must be between 0 and 1")
    denominator = sensitivity + specificity - 1
    if denominator == 0:
        raise ValueError("Rogan-Gladen correction is undefined when sensitivity + specificity = 1")
    return min(1.0, max(0.0, (observed_rate + specificity - 1) / denominator))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                     rng: int = 2027, confidence: float = 0.95) -> tuple[float, float, float]:
    """a[i], b[i] = per-item mean over seeds, SAME items, SAME order.
    Return (mean(a)-mean(b), lo, hi) with scipy.stats.bootstrap(paired=True, method="percentile",
    rng=numpy.random.default_rng(rng)). If a == b everywhere return (0.0, 0.0, 0.0)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must have equal, nonzero lengths")
    if n_resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be positive and confidence must be in (0, 1)")
    import numpy as np
    from scipy.stats import bootstrap

    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError("paired samples must be finite")
    delta = float(np.mean(aa) - np.mean(bb))
    if np.array_equal(aa, bb):
        return (0.0, 0.0, 0.0)
    result = bootstrap((aa, bb), lambda x, y, axis: np.mean(x - y, axis=axis),
                       paired=True, vectorized=True, n_resamples=n_resamples,
                       confidence_level=confidence, method="percentile",
                       rng=np.random.default_rng(rng))
    return delta, float(result.confidence_interval.low), float(result.confidence_interval.high)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that AT LEAST ONE of k trials drawn from n (c of them passing) passes.
    Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k). Raise ValueError if k > n."""
    _validate_combination(n, c, k)
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench, Yao et al. 2024): probability that ALL k trials drawn from n pass.
    Unbiased estimator: C(c, k) / C(n, k). Raise ValueError if k > n.
    pass^k <= pass@1 <= pass@k; a reliable agent keeps pass^k close to pass@1."""
    _validate_combination(n, c, k)
    from math import comb
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def _validate_combination(n: int, c: int, k: int) -> None:
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, c, k)):
        raise ValueError("n, c, and k must be integers")
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError("require n > 0, 0 <= c <= n, and 1 <= k <= n")
