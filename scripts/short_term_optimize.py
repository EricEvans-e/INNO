from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
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


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_bool_list(raw: str) -> List[bool]:
    mapping = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }
    values: List[bool] = []
    for x in raw.split(","):
        t = x.strip().lower()
        if not t:
            continue
        if t not in mapping:
            raise ValueError(f"Invalid bool token: {x}")
        values.append(mapping[t])
    return values


def _parse_path_list(raw: str) -> List[Path]:
    return [Path(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _score_metrics(metrics: Dict[str, float], w_latency: float, w_conflict: float, w_space: float, w_bubble: float) -> float:
    latency = float(metrics.get("latency", 1e12))
    conflict = float(metrics.get("conflict_score", 1e9))
    space = float(metrics.get("space_utilization", 0.0))
    bubble = float(metrics.get("pipeline_bubble_cycles", 1e12))
    # lower is better
    return latency * w_latency + conflict * w_conflict + bubble * w_bubble - space * w_space


def run_short_term_optimization(
    model_path: Path,
    trace_paths: List[Path],
    output_root: Path,
    replica_ratios: List[float],
    cold_quant_bits: List[int],
    cold_sparsity: List[float],
    overlap_flags: List[bool],
    overlap_alphas: List[float],
    lb_weights: List[float],
    dispatch_policies: List[str],
    criticality_weights: List[float],
    w_latency: float,
    w_conflict: float,
    w_space: float,
    w_bubble: float,
    robust_worst_weight: float,
) -> Dict[str, Any]:
    ensure_dir(output_root)

    rows: List[Dict[str, Any]] = []
    trial_id = 0
    for rep_ratio, c_bits, c_sp, overlap, overlap_alpha, lb_weight, dispatch_policy, criticality_weight in product(
        replica_ratios,
        cold_quant_bits,
        cold_sparsity,
        overlap_flags,
        overlap_alphas,
        lb_weights,
        dispatch_policies,
        criticality_weights,
    ):
        cfg = replace(
            DEFAULT_MOE_CONFIG,
            replication_volume_budget_ratio=float(rep_ratio),
            cold_quant_bits=int(c_bits),
            cold_sparsity_ratio=float(c_sp),
            enable_blockwise_quantization=True,
            enable_sparse_compression=True,
        )

        trial_dir = output_root / f"trial_{trial_id:03d}"
        ensure_dir(trial_dir)
        per_trace_opt: List[Dict[str, float]] = []
        per_trace_imp: List[Dict[str, float]] = []
        per_trace_latency: Dict[str, float] = {}
        for tidx, trace_path in enumerate(trace_paths):
            trace_out = trial_dir / f"trace_{tidx:02d}_{trace_path.stem}"
            ensure_dir(trace_out)
            comparison = run_pipeline(
                model_path=model_path,
                trace_path=trace_path,
                output_dir=trace_out,
                moe_cfg_override=cfg,
                export_profile=True,
                overlap_transfer_compute=bool(overlap),
                overlap_alpha=float(overlap_alpha),
                load_balance_weight=float(lb_weight),
                dispatch_policy=str(dispatch_policy),
                criticality_weight=float(criticality_weight),
            )
            opt = comparison["optimized"]
            imp = comparison.get("improvement", {})
            per_trace_opt.append(opt)
            per_trace_imp.append(imp)
            per_trace_latency[trace_path.stem] = float(opt.get("latency", 0.0))

        def _mean(key: str, default: float = 0.0) -> float:
            vals = [float(x.get(key, default)) for x in per_trace_opt]
            return float(sum(vals) / max(len(vals), 1))

        def _mean_imp(key: str, default: float = 0.0) -> float:
            vals = [float(x.get(key, default)) for x in per_trace_imp]
            return float(sum(vals) / max(len(vals), 1))

        mean_metrics = {
            "latency": _mean("latency"),
            "conflict_score": _mean("conflict_score"),
            "space_utilization": _mean("space_utilization"),
            "temporal_utilization": _mean("temporal_utilization"),
            "pipeline_bubble_cycles": _mean("pipeline_bubble_cycles"),
        }
        worst_latency = max(per_trace_latency.values()) if per_trace_latency else 0.0
        score = _score_metrics(
            mean_metrics,
            w_latency=w_latency,
            w_conflict=w_conflict,
            w_space=w_space,
            w_bubble=w_bubble,
        ) + float(robust_worst_weight) * float(worst_latency)

        row = {
            "trial": trial_id,
            "replication_volume_budget_ratio": float(rep_ratio),
            "cold_quant_bits": int(c_bits),
            "cold_sparsity_ratio": float(c_sp),
            "overlap_transfer_compute": bool(overlap),
            "overlap_alpha": float(overlap_alpha),
            "load_balance_weight": float(lb_weight),
            "dispatch_policy": str(dispatch_policy),
            "criticality_weight": float(criticality_weight),
            "score": float(score),
            "latency": float(mean_metrics.get("latency", 0.0)),
            "worst_latency": float(worst_latency),
            "conflict_score": float(mean_metrics.get("conflict_score", 0.0)),
            "space_utilization": float(mean_metrics.get("space_utilization", 0.0)),
            "temporal_utilization": float(mean_metrics.get("temporal_utilization", 0.0)),
            "pipeline_bubble_cycles": float(mean_metrics.get("pipeline_bubble_cycles", 0.0)),
            "latency_improvement": _mean_imp("latency", 0.0),
            "space_improvement": _mean_imp("space_utilization", 0.0),
            "temporal_improvement": _mean_imp("temporal_utilization", 0.0),
            "trace_count": int(len(trace_paths)),
            "trace_latency": per_trace_latency,
        }
        rows.append(row)
        trial_id += 1

    df = pd.DataFrame(rows).sort_values(by=["score", "latency"], ascending=[True, True])
    csv_path = output_root / "short_term_grid_results.csv"
    df.to_csv(csv_path, index=False)

    best = df.iloc[0].to_dict() if not df.empty else {}
    summary = {
        "n_trials": int(len(rows)),
        "weights": {
            "latency": float(w_latency),
            "conflict": float(w_conflict),
            "space": float(w_space),
            "bubble": float(w_bubble),
            "robust_worst": float(robust_worst_weight),
        },
        "best": best,
        "csv": str(csv_path),
    }
    save_json(output_root / "short_term_best.json", summary)

    lines = [
        "# Short-Term Optimization Summary",
        "",
        f"- Trials: {summary['n_trials']}",
        f"- Best trial: {int(best.get('trial', -1))}",
        f"- Best score: {float(best.get('score', 0.0)):.4f}",
        f"- Latency: {float(best.get('latency', 0.0)):.2f}",
        f"- Worst Latency: {float(best.get('worst_latency', 0.0)):.2f}",
        f"- Conflict score: {float(best.get('conflict_score', 0.0)):.2f}",
        f"- Space utilization: {float(best.get('space_utilization', 0.0)):.4f}",
        f"- Bubble cycles: {float(best.get('pipeline_bubble_cycles', 0.0)):.2f}",
        f"- overlap_transfer_compute: {bool(best.get('overlap_transfer_compute', False))}",
        f"- overlap_alpha: {best.get('overlap_alpha')}",
        f"- load_balance_weight: {best.get('load_balance_weight')}",
        f"- dispatch_policy: {best.get('dispatch_policy')}",
        f"- criticality_weight: {best.get('criticality_weight')}",
        "",
        "## Best Params",
        f"- replication_volume_budget_ratio: {best.get('replication_volume_budget_ratio')}",
        f"- cold_quant_bits: {best.get('cold_quant_bits')}",
        f"- cold_sparsity_ratio: {best.get('cold_sparsity_ratio')}",
    ]
    report_path = output_root / "short_term_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-term optimizer: replica/compression/overlap grid scan")
    parser.add_argument("--model", type=Path, default=Path("data/sample_model.json"))
    parser.add_argument("--trace", type=Path, default=Path("data/sample_trace.json"))
    parser.add_argument("--traces", type=str, default="", help="Optional comma-separated trace paths for robust optimization")
    parser.add_argument("--output", type=Path, default=Path("outputs/short_term_opt"))
    parser.add_argument("--replica-ratios", type=str, default="0.10,0.20,0.30")
    parser.add_argument("--cold-quant-bits", type=str, default="5,6")
    parser.add_argument("--cold-sparsity", type=str, default="0.20,0.30")
    parser.add_argument("--overlap-flags", type=str, default="false,true")
    parser.add_argument("--overlap-alphas", type=str, default="0.7,1.0")
    parser.add_argument("--lb-weights", type=str, default="0.0,0.05")
    parser.add_argument("--dispatch-policies", type=str, default="fifo,criticality")
    parser.add_argument("--criticality-weights", type=str, default="0.0,0.3")
    parser.add_argument("--w-latency", type=float, default=1.0)
    parser.add_argument("--w-conflict", type=float, default=0.25)
    parser.add_argument("--w-space", type=float, default=800.0)
    parser.add_argument("--w-bubble", type=float, default=0.0003)
    parser.add_argument("--robust-worst-weight", type=float, default=0.15)
    args = parser.parse_args()

    trace_paths = _parse_path_list(args.traces) if args.traces.strip() else [args.trace]

    summary = run_short_term_optimization(
        model_path=args.model,
        trace_paths=trace_paths,
        output_root=args.output,
        replica_ratios=_parse_float_list(args.replica_ratios),
        cold_quant_bits=_parse_int_list(args.cold_quant_bits),
        cold_sparsity=_parse_float_list(args.cold_sparsity),
        overlap_flags=_parse_bool_list(args.overlap_flags),
        overlap_alphas=_parse_float_list(args.overlap_alphas),
        lb_weights=_parse_float_list(args.lb_weights),
        dispatch_policies=_parse_str_list(args.dispatch_policies),
        criticality_weights=_parse_float_list(args.criticality_weights),
        w_latency=float(args.w_latency),
        w_conflict=float(args.w_conflict),
        w_space=float(args.w_space),
        w_bubble=float(args.w_bubble),
        robust_worst_weight=float(args.robust_worst_weight),
    )

    print("==== Short-Term Optimization Done ====")
    print(f"trials: {summary['n_trials']}")
    print(f"best: {summary['best']}")


if __name__ == "__main__":
    main()
