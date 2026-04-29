from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_trace import generate_traces
from src.utils import save_json


def build_synthetic_moe_model(
    num_experts: int,
    hidden_size: int,
    expert_width: int,
    router_width: int,
    dtype: str,
) -> Dict[str, Any]:
    operators: List[Dict[str, Any]] = [
        {"id": "embed_proj", "type": "linear", "shape": [hidden_size, hidden_size], "deps": []},
        {"id": "shared_attn", "type": "linear", "shape": [hidden_size, hidden_size], "deps": ["embed_proj"]},
        {"id": "shared_mlp", "type": "linear", "shape": [hidden_size, hidden_size], "deps": ["shared_attn"]},
        {"id": "moe_router", "type": "linear", "shape": [hidden_size, router_width], "deps": ["shared_attn"]},
    ]

    for expert_id in range(num_experts):
        operators.append(
            {
                "id": f"expert_{expert_id}",
                "type": "moe_expert",
                "expert_id": expert_id,
                "shape": [hidden_size, expert_width],
                "deps": ["moe_router"],
                "parallel_group": "experts",
            }
        )

    operators.append(
        {
            "id": "moe_merge",
            "type": "elementwise",
            "op": "add",
            "deps": ["shared_mlp"] + [f"expert_{i}" for i in range(num_experts)],
        }
    )
    operators.append(
        {
            "id": "lm_head",
            "type": "linear",
            "shape": [hidden_size, hidden_size * 2],
            "deps": ["moe_merge"],
        }
    )
    return {
        "model_name": f"deepseek_moe_synthetic_e{num_experts}",
        "dtype": dtype,
        "operators": operators,
        "notes": {
            "weights_are_metadata_only": True,
            "contest_track": "track_2_problem_2",
            "purpose": "large-scale metadata-only MoE mapping/scheduling proof",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DeepSeek/MoE-style model and trace metadata")
    parser.add_argument("--model-output", type=Path, default=Path("outputs/large_scale_v5/synthetic_model.json"))
    parser.add_argument("--trace-output", type=Path, default=Path("outputs/large_scale_v5/synthetic_trace.json"))
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--expert-width", type=int, default=1024)
    parser.add_argument("--router-width", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="int8")
    parser.add_argument("--n-inferences", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--shared-experts", type=str, default="0")
    parser.add_argument("--hot-ratio", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    model = build_synthetic_moe_model(
        num_experts=max(1, int(args.num_experts)),
        hidden_size=max(1, int(args.hidden_size)),
        expert_width=max(1, int(args.expert_width)),
        router_width=max(1, int(args.router_width)),
        dtype=str(args.dtype),
    )
    traces = generate_traces(
        num_experts=max(1, int(args.num_experts)),
        n_inferences=max(1, int(args.n_inferences)),
        top_k=max(1, int(args.top_k)),
        hot_ratio=max(0.0, min(1.0, float(args.hot_ratio))),
        alpha=max(0.01, float(args.alpha)),
        seed=int(args.seed),
    )
    shared = [int(x.strip()) for x in str(args.shared_experts).split(",") if x.strip()]
    trace = {
        "num_experts": max(1, int(args.num_experts)),
        "top_k": max(1, int(args.top_k)),
        "shared_experts": shared,
        "n_inferences": len(traces),
        "traces": traces,
        "synthetic": True,
    }

    save_json(args.model_output, model)
    save_json(args.trace_output, trace)
    print(f"Saved model to {args.model_output}")
    print(f"Saved trace to {args.trace_output}")


if __name__ == "__main__":
    main()
