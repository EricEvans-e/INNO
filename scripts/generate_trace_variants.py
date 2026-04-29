from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_json, save_json


def _to_python_traces(arr: List[List[int]]) -> List[List[int]]:
    return [[int(x) for x in row] for row in arr]


def _make_hotspot(traces: List[List[int]], num_experts: int, top_k: int, rng: np.random.RandomState) -> List[List[int]]:
    # bias requests toward head experts to emulate burst hotspot traffic
    hot_count = max(1, int(0.15 * num_experts))
    hot = list(range(hot_count))
    out: List[List[int]] = []
    for row in traces:
        if rng.rand() < 0.6:
            picked = rng.choice(hot, size=min(top_k, len(hot)), replace=False).tolist()
            out.append([int(x) for x in picked])
        else:
            out.append([int(x) for x in row[:top_k]])
    return out


def _make_balanced(traces: List[List[int]], num_experts: int, top_k: int, rng: np.random.RandomState) -> List[List[int]]:
    # flatten popularity to emulate evenly distributed expert calls
    out: List[List[int]] = []
    experts = np.arange(num_experts)
    for _ in traces:
        picked = rng.choice(experts, size=min(top_k, num_experts), replace=False).tolist()
        out.append([int(x) for x in picked])
    return out


def _make_bursty(traces: List[List[int]], num_experts: int, top_k: int, rng: np.random.RandomState) -> List[List[int]]:
    # maintain a short burst window where selected experts persist
    out: List[List[int]] = []
    current = rng.choice(np.arange(num_experts), size=min(top_k, num_experts), replace=False).tolist()
    burst_left = 0
    for _ in traces:
        if burst_left <= 0:
            current = rng.choice(np.arange(num_experts), size=min(top_k, num_experts), replace=False).tolist()
            burst_left = int(rng.randint(3, 10))
        elif rng.rand() < 0.25:
            idx = int(rng.randint(0, len(current)))
            current[idx] = int(rng.randint(0, num_experts))
            current = list(sorted(set(current)))
            while len(current) < min(top_k, num_experts):
                current.append(int(rng.randint(0, num_experts)))
                current = list(sorted(set(current)))
        burst_left -= 1
        out.append([int(x) for x in current[:top_k]])
    return out


def generate_variants(input_trace: Path, output_dir: Path, seed: int = 42) -> Dict[str, str]:
    data = load_json(input_trace)
    traces = data.get("traces", [])
    num_experts = int(data.get("num_experts", 0))
    top_k = int(data.get("top_k", 1))
    if not traces or num_experts <= 0:
        raise ValueError("invalid trace file for variant generation")

    ensure_dir(output_dir)
    rng = np.random.RandomState(seed)

    variants = {
        "base": _to_python_traces(traces),
        "hotspot": _make_hotspot(traces, num_experts, top_k, rng),
        "balanced": _make_balanced(traces, num_experts, top_k, rng),
        "bursty": _make_bursty(traces, num_experts, top_k, rng),
    }

    out_map: Dict[str, str] = {}
    for name, vtraces in variants.items():
        payload = {
            "num_experts": num_experts,
            "top_k": top_k,
            "n_inferences": int(len(vtraces)),
            "traces": vtraces,
            "shared_experts": list(data.get("shared_experts", [])),
            "variant": name,
            "source": str(input_trace),
        }
        out_path = output_dir / f"trace_{name}.json"
        save_json(out_path, payload)
        out_map[name] = str(out_path)

    save_json(output_dir / "trace_variants_index.json", out_map)
    return out_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple workload variants from one trace")
    parser.add_argument("--input", type=Path, default=Path("data/sample_trace.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trace_variants"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = generate_variants(args.input, args.output, seed=args.seed)
    print("==== Trace Variants Generated ====")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
