from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.moe_config import DEFAULT_MOE_CONFIG
from main import run_pipeline
from src.utils import ensure_dir, save_json


def _run_one(run_id: int, model_path: Path, trace_path: Path, output_root: Path) -> Dict[str, Any]:
    cfg = replace(
        DEFAULT_MOE_CONFIG,
        local_search_restarts=4 + (run_id % 4),
        local_search_max_iters=25 + (run_id % 3) * 10,
        parallel_trials=6 + (run_id % 5),
        hot_subgraph_top_k=4 + (run_id % 4),
        replication_volume_budget_ratio=0.15 + 0.05 * (run_id % 5),
        enable_simulated_annealing=(run_id % 2 == 0),
        sa_steps=60 + (run_id % 4) * 20,
        cold_quant_bits=5 + (run_id % 3),
        cold_sparsity_ratio=0.10 + 0.05 * (run_id % 4),
    )

    out_dir = output_root / f"run_{run_id:03d}"
    ensure_dir(out_dir)
    comparison = run_pipeline(
        model_path=model_path,
        trace_path=trace_path,
        output_dir=out_dir,
        moe_cfg_override=cfg,
        export_profile=True,
    )

    opt = comparison["optimized"]
    return {
        "run": run_id,
        "latency": float(opt["latency"]),
        "temporal_utilization": float(opt["temporal_utilization"]),
        "space_utilization": float(opt["space_utilization"]),
        "conflict_score": float(opt["conflict_score"]),
        "pipeline_bubble_cycles": float(opt["pipeline_bubble_cycles"]),
    }


def run_parallel_benchmark(
    model_path: Path,
    trace_path: Path,
    output_root: Path,
    runs: int,
    workers: int,
) -> Dict[str, Any]:
    ensure_dir(output_root)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(_run_one, i, model_path, trace_path, output_root)
            for i in range(max(1, runs))
        ]
        rows = [f.result() for f in futures]

    df = pd.DataFrame(rows).sort_values(by="latency", ascending=True)
    csv_path = output_root / "parallel_benchmark.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "runs": len(rows),
        "workers": max(1, workers),
        "best": df.iloc[0].to_dict() if not df.empty else {},
        "avg_latency": float(df["latency"].mean()) if not df.empty else 0.0,
        "csv": str(csv_path),
    }
    save_json(output_root / "parallel_benchmark_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel benchmark runner for distributed evaluation")
    parser.add_argument("--model", type=Path, default=Path("data/sample_model.json"))
    parser.add_argument("--trace", type=Path, default=Path("data/sample_trace.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/parallel_eval"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    summary = run_parallel_benchmark(args.model, args.trace, args.output, args.runs, args.workers)
    print("==== Parallel Benchmark Done ====")
    print(f"runs: {summary['runs']}, workers: {summary['workers']}")
    print(f"best: {summary['best']}")


if __name__ == "__main__":
    main()
