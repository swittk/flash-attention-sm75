import argparse
import json
import math


def geomean(values):
    vals = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--prefill-gate", type=float, default=1.20)
    parser.add_argument("--decode-gate", type=float, default=1.10)
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload if isinstance(payload, list) else [payload]
    speedups = [row["speedup_vs_sdpa"] for row in rows]
    native_backends = [row.get("backend", "") for row in rows]
    summary = {"rows": rows}
    if args.mode == "prefill":
        summary["geomean_speedup_vs_sdpa"] = geomean(speedups)
    else:
        summary["min_speedup_vs_sdpa"] = min(speedups) if speedups else 0.0
    print(json.dumps(summary, indent=2))
    if any(not backend.startswith("sm75_native") for backend in native_backends):
        raise SystemExit("At least one benchmark row did not use an SM75 native backend")
    if args.mode == "prefill" and summary["geomean_speedup_vs_sdpa"] < args.prefill_gate:
        raise SystemExit("SM75 prefill geomean gate was not met")
    if args.mode == "decode" and summary["min_speedup_vs_sdpa"] < args.decode_gate:
        raise SystemExit("SM75 decode speedup gate was not met")


if __name__ == "__main__":
    main()
