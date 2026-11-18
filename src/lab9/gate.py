"""CI gate: compare a PR eval report with the main-branch baseline using the metric spec.

    python -m lab9.gate --pr reports/eval_workflow.json --baseline reports/baseline_v1.json

exit 0 = merge, exit 1 = blocked. Safety metrics: max 0, no averaging away.
Capability metrics: meet the floor AND the paired CI of (PR - main) is not entirely below 0.
Thresholds may only get stricter on the same eval-set version (ratchet).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lab8.console import utf8_stdio
from lab9 import stats
from lab9.eval import SPEC, load_spec, per_item
from lab9.graders import GRADERS


def integrity_check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> list[str]:
    """M7 (Expert): refuse to compare reports that cannot be trusted. Return a list of problems
    (empty = trustworthy). A gate that passes a broken report is worse than no gate.
    Must catch at least:
      - any metric value in pr["metrics"] that is NaN or inf
      - len(pr["rows"]) != manifest items x len(manifest seeds)   (dropped / duplicated rows)
      - the (item_id, seed) set of PR differs from the baseline's (cherry-picked seeds/items)
      - pr["evaluator_failures"] non-empty, or a row whose value for a graded spec metric is
        None (a crashed grader silently scores as "missing" and drops out of the mean)
      - base_spec given and a threshold in spec is LOOSER than in base_spec for the same
        eval-set version (ratchet: max may not rise, min may not fall, a blocking metric may
        not become non-blocking, a metric may not disappear; the same for spec["slices"],
        including dropping no_significant_drop)
      - an item's slice tags in PR rows differ from the baseline's (relabelling an item out of a
        slice hides its regression from the slice gate)"""
    problems: list[str] = []
    metrics = pr.get("metrics") or {}
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            problems.append(f"metric {metric} is non-finite")
    if pr.get("evaluator_failures"):
        problems.append("evaluator_failures is non-empty")
    rows = pr.get("rows") or []
    seeds = pr.get("manifest", {}).get("seeds") or []
    items = pr.get("manifest", {}).get("items")
    if not isinstance(items, int) or len(rows) != items * len(seeds):
        problems.append(f"row count {len(rows)} != manifest items × seeds ({items} × {len(seeds)})")
    pairs = [(r.get("item_id"), r.get("seed")) for r in rows]
    if len(pairs) != len(set(pairs)):
        problems.append("duplicate (item_id, seed) rows")
    expected_pairs = {(r.get("item_id"), r.get("seed")) for r in (base.get("rows") or [])}
    if set(pairs) != expected_pairs:
        problems.append("(item_id, seed) set differs from baseline")
    graded = [m["id"] for m in spec.get("metrics", []) if m["id"] in GRADERS]
    for row in rows:
        for metric in graded:
            if metric not in row or row[metric] is None:
                problems.append(f"row {row.get('item_id')}@{row.get('seed')} missing {metric}")
                break
    def slices_by_item(report: dict) -> dict[str, tuple[str, ...]]:
        found: dict[str, set[tuple[str, ...]]] = {}
        for row in report.get("rows", []):
            found.setdefault(row.get("item_id"), set()).add(tuple(sorted(row.get("slices") or [])))
        return {item: next(iter(tags)) if len(tags) == 1 else ("<inconsistent>",)
                for item, tags in found.items()}
    if slices_by_item(pr) != slices_by_item(base):
        problems.append("item slice tags differ from baseline")
    if base_spec is not None and spec.get("eval_set_version") == base_spec.get("eval_set_version"):
        current = {m["id"]: m for m in spec.get("metrics", [])}
        previous = {m["id"]: m for m in base_spec.get("metrics", [])}
        for mid, old in previous.items():
            new = current.get(mid)
            if new is None:
                problems.append(f"ratchet: metric {mid} removed")
                continue
            old_th, new_th = old.get("threshold", {}), new.get("threshold", {})
            if old.get("blocking") and not new.get("blocking"):
                problems.append(f"ratchet: metric {mid} is no longer blocking")
            if "min" in old_th and ("min" not in new_th or new_th["min"] < old_th["min"]):
                problems.append(f"ratchet: metric {mid} minimum loosened")
            if "max" in old_th and ("max" not in new_th or new_th["max"] > old_th["max"]):
                problems.append(f"ratchet: metric {mid} maximum loosened")
            if "max_increase" in old_th and ("max_increase" not in new_th or new_th["max_increase"] > old_th["max_increase"]):
                problems.append(f"ratchet: metric {mid} max_increase loosened")
            if old_th.get("no_significant_drop") and not new_th.get("no_significant_drop"):
                problems.append(f"ratchet: metric {mid} no_significant_drop removed")
        def slice_map(sp: dict) -> dict[tuple[str, str], dict]:
            return {(s.get("name"), s.get("metric")): s for s in sp.get("slices", [])}
        old_slices, new_slices = slice_map(base_spec), slice_map(spec)
        for key, old in old_slices.items():
            new = new_slices.get(key)
            if new is None:
                problems.append(f"ratchet: slice {key[0]}/{key[1]} removed")
                continue
            if old.get("blocking") and not new.get("blocking"):
                problems.append(f"ratchet: slice {key[0]}/{key[1]} is no longer blocking")
            if "min" in old and ("min" not in new or new["min"] < old["min"]):
                problems.append(f"ratchet: slice {key[0]}/{key[1]} minimum loosened")
            if "max" in old and ("max" not in new or new["max"] > old["max"]):
                problems.append(f"ratchet: slice {key[0]}/{key[1]} maximum loosened")
            if old.get("no_significant_drop") and not new.get("no_significant_drop"):
                problems.append(f"ratchet: slice {key[0]}/{key[1]} no_significant_drop removed")
    return problems


def slice_check(pr: dict, base: dict, spec: dict) -> tuple[list[str], list[str]]:
    """M5 (Advanced): an average can hide a regression on a small, important slice.
    For each entry of spec["slices"] ({"name", "metric", "min"?, "no_significant_drop"?, "blocking"}):
      - value = mean of per_item(pr, metric, slice_name=name)
      - fail if below "min", or if no_significant_drop and the paired CI over the slice's items
        lies entirely below 0
    Return (blocked ids like "slice:money_movement/pass_rate", report lines)."""
    blocked: list[str] = []
    lines: list[str] = []
    bs = spec.get("bootstrap", {})
    for entry in spec.get("slices", []):
        name, metric = entry["name"], entry["metric"]
        current, previous = per_item(pr, metric, name), per_item(base, metric, name)
        keys = sorted(set(current) & set(previous))
        value = sum(current.values()) / len(current) if current else None
        reasons = []
        if value is None:
            reasons.append("missing slice data")
        elif "min" in entry and value < entry["min"]:
            reasons.append(f"below min {entry['min']}")
        elif "max" in entry and value > entry["max"]:
            reasons.append(f"above max {entry['max']}")
        note = ""
        if entry.get("no_significant_drop"):
            if not keys:
                reasons.append("no paired slice items")
            else:
                delta, lo, hi = stats.paired_bootstrap(
                    [current[k] for k in keys], [previous[k] for k in keys],
                    n_resamples=bs.get("n_resamples", 1000), rng=bs.get("rng", 2027))
                note = f"Δ {delta:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
                if hi < 0:
                    reasons.append("significant drop")
        failed = bool(reasons)
        ident = f"slice:{name}/{metric}"
        if failed and entry.get("blocking"):
            blocked.append(ident)
        label = "BLOCK" if failed and entry.get("blocking") else "warn" if failed else "ok"
        lines.append(f"  {ident:<40} {value!s:<8} {note:<28} {label}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return blocked, lines


def check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> tuple[list[str], list[str]]:
    if pr["manifest"]["eval_set_sha"] != base["manifest"]["eval_set_sha"]:
        return ["eval_set_sha"], ["  EVAL SET differs from the baseline's: re-run the baseline on the same eval set"]
    problems = integrity_check(pr, base, spec, base_spec)
    if problems:
        return ["integrity"], [f"  INTEGRITY {p}" for p in problems]
    blocked, lines = [], []
    bs = spec.get("bootstrap", {})
    for m in spec["metrics"]:
        mid, th, val = m["id"], m.get("threshold", {}), pr["metrics"].get(m["id"])
        verdict, note = "ok", ""
        if val is None:
            verdict = "missing"
        elif "max" in th and val > th["max"]:
            verdict = "fail"
        elif "min" in th and val < th["min"]:
            verdict = "fail"
        elif "max_increase" in th and base["metrics"].get(mid):
            if val > base["metrics"][mid] * (1 + th["max_increase"]):
                verdict = "fail"
        if th.get("no_significant_drop") and val is not None:
            a, b = per_item(pr, mid), per_item(base, mid)
            keys = sorted(set(a) & set(b))
            d, lo, hi = stats.paired_bootstrap([a[k] for k in keys], [b[k] for k in keys],
                                               n_resamples=bs.get("n_resamples", 1000), rng=bs.get("rng", 2027))
            note = f"Δ {d:+.3f} CI [{lo:+.3f}, {hi:+.3f}]"
            if hi < 0:
                verdict = "fail"
        if verdict != "ok" and m.get("blocking"):
            blocked.append(mid)
        label = {"ok": "ok", "missing": "MISSING"}.get(verdict, "BLOCK" if m.get("blocking") else "warn")
        lines.append(f"  {mid:<22} {val!s:<8} {json.dumps(th):<40} {note:<28} {label}")
    if spec.get("slices"):
        sb, sl = slice_check(pr, base, spec)
        blocked += sb
        lines += sl
    return blocked, lines


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(prog="lab9.gate")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--base-spec", help="metric spec on main, for the ratchet check (M7)")
    args = ap.parse_args(argv)
    pr = json.loads(Path(args.pr).read_text(encoding="utf-8"))
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_spec = load_spec(Path(args.base_spec)) if args.base_spec else None
    blocked, lines = check(pr, base, load_spec(Path(args.spec)), base_spec)
    print(f"gate: PR {pr['run_id']} vs main {base['run_id']} · eval set {pr['manifest']['eval_set_sha']}")
    print("\n".join(lines))
    if blocked:
        print(f"exit 1 — blocked: {', '.join(blocked)}")
        return 1
    print("exit 0 — merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
