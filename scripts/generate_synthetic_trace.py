#!/usr/bin/env python3
"""Generate synthetic MoE activation traces for pipeline testing.

Produces a JSON file compatible with `src.model_parser.parse_activation_trace`:
{
  "num_experts": int,
  "top_k": int,
  "n_inferences": int,
  "traces": [[expert_id, ...], ...],
  "shared_experts": [...]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def generate_traces(num_experts: int, n_inferences: int, top_k: int, hot_ratio: float, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    # power-law popularity over experts (rank-based)
    ranks = np.arange(1, num_experts + 1)
    weights = 1.0 / (ranks ** alpha)
    # boost head (hot) experts to create a hot-tail distribution
    n_hot = max(1, int(num_experts * hot_ratio))
    weights[:n_hot] *= (1.0 + 5.0)
    probs = weights / weights.sum()

    traces: List[List[int]] = []
    for _ in range(n_inferences):
        k = min(top_k, num_experts)
        # weighted sample without replacement
        raw = list(rng.choice(num_experts, size=k, replace=False, p=probs))
        active = [int(x) for x in raw]
        traces.append(active)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MoE activation traces")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/synthetic_trace.json"))
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--n-inferences", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--hot-ratio", type=float, default=0.1, help="fraction of experts that are "
                        "considered hot")
    parser.add_argument("--alpha", type=float, default=1.1, help="power-law exponent for popularity")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    traces = generate_traces(
        args.num_experts, args.n_inferences, args.top_k, args.hot_ratio, args.alpha, args.seed
    )

    payload = {
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "n_inferences": args.n_inferences,
        "traces": traces,
        "shared_experts": [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved synthetic trace to {args.output}")


if __name__ == "__main__":
    main()
