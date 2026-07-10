import argparse
import json
import math
import statistics


def geomean(values):
    vals = [v for v in values if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0


def percentile(values, pct):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = pct * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def rows_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "rows" in payload:
        return payload["rows"]
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(type(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--mode", choices=("prefill", "decode", "generation", "auto"), default="auto")
    parser.add_argument("--prefill-gate", type=float, default=1.20)
    parser.add_argument("--decode-gate", type=float, default=1.15)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()

    rows = []
    payloads = []
    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payloads.append({"path": path, "matrix": payload.get("matrix") if isinstance(payload, dict) else None})
        rows.extend(rows_from_payload(payload))

    speedups = [row.get("speedup_vs_sdpa", row.get("speedup_vs_baseline")) for row in rows]
    speedups = [value for value in speedups if value is not None]
    native_backends = [row.get("backend", "") for row in rows if row.get("backend")]
    by_family = {}
    for row in rows:
        family = row.get("family", row.get("api", "unknown"))
        by_family.setdefault(family, []).append(row)

    summary = {
        "inputs": payloads,
        "row_count": len(rows),
        "geomean_speedup_vs_baseline": geomean(speedups),
        "median_speedup_vs_baseline": statistics.median(speedups) if speedups else 0.0,
        "p10_speedup_vs_baseline": percentile(speedups, 0.10),
        "min_speedup_vs_baseline": min(speedups) if speedups else 0.0,
        "families": {
            family: {
                "row_count": len(family_rows),
                "geomean_speedup_vs_baseline": geomean([r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) for r in family_rows]),
                "median_speedup_vs_baseline": statistics.median([r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) for r in family_rows if r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) is not None]),
                "p10_speedup_vs_baseline": percentile([r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) for r in family_rows if r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) is not None], 0.10),
                "min_speedup_vs_baseline": min([r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) for r in family_rows if r.get("speedup_vs_sdpa", r.get("speedup_vs_baseline")) is not None], default=0.0),
            }
            for family, family_rows in sorted(by_family.items())
        },
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))

    if native_backends and any(not backend.startswith("sm75_native") for backend in native_backends):
        raise SystemExit("At least one benchmark row did not use an SM75 native backend")
    if args.no_gate:
        return
    mode = args.mode
    if mode == "auto":
        families = set(by_family)
        if families <= {"prefill", "varlen"}:
            mode = "prefill"
        elif families <= {"kvcache_decode"}:
            mode = "decode"
    if mode == "prefill":
        gates = {
            "geomean_speedup_vs_baseline": args.prefill_gate,
            "median_speedup_vs_baseline": 1.10,
            "p10_speedup_vs_baseline": 0.90,
            "min_speedup_vs_baseline": 0.85,
        }
    elif mode == "decode":
        gates = {
            "geomean_speedup_vs_baseline": args.decode_gate,
            "median_speedup_vs_baseline": 1.05,
            "p10_speedup_vs_baseline": 0.90,
            "min_speedup_vs_baseline": 0.85,
        }
    else:
        gates = {}
    failures = [
        f"{metric} {summary[metric]:.4f} < {threshold:.4f}"
        for metric, threshold in gates.items()
        if summary[metric] < threshold
    ]
    if failures:
        raise SystemExit("SM75 performance gates were not met: " + "; ".join(failures))


if __name__ == "__main__":
    main()
