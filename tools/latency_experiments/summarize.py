"""Summarize complete, common-frame windows; exclude startup/transition samples."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("scenario", type=int)
parser.add_argument("duration", type=int, nargs="?", default=40)
parser.add_argument(
    "--evidence-dir", type=Path, default=Path("evidence/latency-experiments-2026-09-08")
)
args = parser.parse_args()
root = args.evidence_dir
if not list(root.glob("lanes-*.csv")):
    sys.exit("WAIT: no lanes CSV files in " + str(root))
p = max(root.glob("lanes-*.csv"), key=lambda p: p.stat().st_mtime)
s = p.read_text()
rows = list(csv.DictReader(s[: s.rfind("\n") + 1].splitlines()))
scenario = args.scenario
duration = args.duration
rows = [r for r in rows if r["scenario"] == str(scenario)]
if not rows:
    print(f"WAIT: no rows for scenario {scenario}")
    sys.exit(2)
end = max(int(r["epoch_ms"]) for r in rows) - 1000
start = end - duration * 1000
first = min(int(r["epoch_ms"]) for r in rows)
if start < first + 5000:
    print(f"WAIT: only {(end-first)/1000:.1f}s available; need {duration+5}s")
    sys.exit(2)
rows = [r for r in rows if start <= int(r["epoch_ms"]) < end]
if not rows:
    print("WAIT: no rows in selected window")
    sys.exit(2)


def stat(values: list[float]) -> dict[str, float | int]:
    """Return count and interpolated latency statistics for the samples."""
    sorted_values = sorted(values)

    def quantile(probability: float) -> float:
        """Interpolate the requested quantile within sorted samples."""
        position = (len(sorted_values) - 1) * probability
        index = int(position)
        return sorted_values[index] + (
            sorted_values[min(index + 1, len(sorted_values) - 1)] - sorted_values[index]
        ) * (position - index)

    return (
        dict(
            n=len(sorted_values),
            mean=statistics.mean(sorted_values),
            p50=quantile(0.5),
            p95=quantile(0.95),
            p99=quantile(0.99),
            min=sorted_values[0],
            max=sorted_values[-1],
        )
        if sorted_values
        else dict(n=0)
    )


lanes = {}
for r in rows:
    k = r["lane"] + "/" + r["kind"]
    lanes.setdefault(k, {})[int(r["seq"])] = r
# Compare identical sequence numbers to avoid attributing boundary/loss differences to latency.
comparison_keys = [
    "robot20/joined",
    "scheduled5/joined",
    "thread5/joined",
    "thread1/joined",
    "event/joined",
    "udp/frame",
]
common = set.intersection(*(set(lanes.get(k, {})) for k in comparison_keys))
result = {
    "source": p.name,
    "scenario": scenario,
    "duration_s": duration,
    "start_ms": start,
    "end_ms": end,
    "common_frames": len(common),
    "lanes": {},
}
for k, rr in lanes.items():
    selected = (
        [r for seq, r in rr.items() if seq in common]
        if k in comparison_keys
        else list(rr.values())
    )
    result["lanes"][k] = {
        "all_unique_frames": len(rr),
        "capture_to_consume_ms": stat(
            [(int(r["consume_us"]) - int(r["capture_us"])) / 1000 for r in selected]
        ),
        "ready_to_consume_ms": stat(
            [(int(r["consume_us"]) - int(r["ready_us"])) / 1000 for r in selected]
        ),
    }
base = [lanes["robot20/joined"][seq] for seq in common]
result["capture_to_ready_ms"] = stat(
    [(int(r["ready_us"]) - int(r["capture_us"])) / 1000 for r in base]
)
(root / f"scenario-{scenario}.json").write_text(json.dumps(result, indent=2) + "\n")
with (root / f"scenario-{scenario}.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(
    "scenario",
    scenario,
    "common",
    len(common),
    "Pi capture-to-ready",
    result["capture_to_ready_ms"],
)
for k in comparison_keys:
    v = result["lanes"].get(k, {})
    a = v.get("capture_to_consume_ms", {})
    b = v.get("ready_to_consume_ms", {})
    print(
        k,
        "total mean/p95",
        round(a.get("mean", 0), 2),
        round(a.get("p95", 0), 2),
        "post mean/p95",
        round(b.get("mean", 0), 2),
        round(b.get("p95", 0), 2),
        "frames",
        v.get("all_unique_frames"),
    )
print(
    "anomalies",
    {k: len(v) for k, v in lanes.items() if "mismatch" in k or "reordered" in k},
)
