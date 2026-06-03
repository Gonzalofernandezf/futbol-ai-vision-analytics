"""
perf_monitor — Per-chunk timing and output sanity checks.

Used to spot regressions when stacking multiple performance changes in a
single branch: if a phase suddenly takes 5x longer or a sanity metric goes
red, we know which optimisation broke something without having to bisect.

Two pieces:
  - PhaseTimer: context manager that accumulates wall-clock time per named
    phase, prints a sorted report at the end.
  - run_sanity_checks: reads the exported JSON + tracks and validates
    output ranges (player count, ball detection rate, max speed, etc.).
    Returns the number of issues but never raises — we'd rather finish the
    chunk and surface warnings than abort 1h of compute.
"""

import json
import time
from contextlib import contextmanager


class PhaseTimer:
    def __init__(self):
        self.phases = {}
        self.order  = []

    @contextmanager
    def phase(self, name):
        if name not in self.phases:
            self.phases[name] = 0.0
            self.order.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] += time.perf_counter() - start

    def report(self):
        total = sum(self.phases.values())
        if total <= 0:
            return
        print("═" * 55)
        print("⏱️  CHUNK TIMING REPORT")
        print("─" * 55)
        for name in self.order:
            t   = self.phases[name]
            pct = t / total * 100
            print(f"  {name:<22}: {t:7.1f}s  ({pct:5.1f}%)")
        print("─" * 55)
        print(f"  {'Total':<22}: {total:7.1f}s  ({total/60:.2f} min)")
        print("═" * 55)


def _check(label, ok, value_str, expected_str, severity="warn"):
    if ok:
        print(f"  ✅ {label:<22}: {value_str}  (expected {expected_str})")
        return 0
    icon = "⚠️ " if severity == "warn" else "❌"
    print(f"  {icon} {label:<22}: {value_str}  (expected {expected_str})")
    return 1


def run_sanity_checks(tracks, json_path):
    """
    Validate per-chunk output against expected ranges.

    Prints a report and returns the number of warnings/errors raised.
    Designed to never raise — we never want a check to abort a long run.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"🔍 SANITY CHECKS — could not read {json_path}: {e}")
        return 1

    players = data.get("players", {})
    meta    = data.get("match_meta", {})

    print("🔍 SANITY CHECKS")
    print("─" * 55)
    issues = 0

    # 1. Player count
    n = len(players)
    issues += _check(
        "Players detected", 14 <= n <= 30, str(n), "14-30",
    )

    # 2. Ball detection rate (post all filters)
    if tracks and "ball" in tracks and len(tracks["ball"]) > 0:
        total = len(tracks["ball"])
        with_ball = sum(1 for f in tracks["ball"] if f and 1 in f)
        rate = with_ball / total * 100
        issues += _check(
            "Ball detection rate", rate >= 40, f"{rate:.0f}%", ">40%",
            severity="warn" if rate >= 20 else "error",
        )

    # 3. Homography coverage (transformed position present)
    pos_covered = 0
    pos_total   = 0
    for p in players.values():
        for pos in p.get("position_history", []) or []:
            pos_total += 1
            if pos is not None:
                pos_covered += 1
    if pos_total > 0:
        cov = pos_covered / pos_total * 100
        issues += _check(
            "Homography coverage", cov >= 60, f"{cov:.0f}%", ">60%",
            severity="warn" if cov >= 30 else "error",
        )

    # 4. Max speed plausibility
    max_speed = max((p.get("max_speed_kmh", 0) for p in players.values()), default=0)
    issues += _check(
        "Max speed (km/h)", max_speed <= 35, f"{max_speed:.1f}", "<35",
    )

    # 5. Possession sum ~ 100
    home = meta.get("home_possession", 0)
    away = meta.get("away_possession", 0)
    total_poss = home + away
    issues += _check(
        "Possession sum", 95 <= total_poss <= 105, f"{total_poss:.1f}%", "~100%",
    )

    # 6. Average distance per player
    distances = [p.get("total_distance_m", 0) for p in players.values()]
    avg_dist  = sum(distances) / len(distances) if distances else 0
    issues += _check(
        "Avg distance per pl.", 20 <= avg_dist <= 2000, f"{avg_dist:.0f} m", "20-2000",
    )

    print("═" * 55)
    if issues == 0:
        print("✅ All sanity checks passed.")
    else:
        print(f"⚠️  {issues} sanity check(s) outside expected range — review output.")
    print("═" * 55)
    return issues
