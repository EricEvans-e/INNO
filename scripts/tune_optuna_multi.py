from __future__ import annotations

import argparse
from dataclasses import replace
import multiprocessing
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
EXPECTED_VENV_PYTHON = WORKSPACE_ROOT / "saidao2" / ".venv" / "Scripts" / "python.exe"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.cube_config import DEFAULT_CUBE_CONFIG, CubeConfig
from config.moe_config import DEFAULT_MOE_CONFIG
from main import run_pipeline
from src.utils import ensure_dir, load_json, save_json


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _ensure_project_venv_python() -> None:
    expected = EXPECTED_VENV_PYTHON.resolve()
    actual = Path(sys.executable).resolve()
    if _normalized_path(actual) != _normalized_path(expected):
        raise RuntimeError(
            "Project venv Python required. "
            f"Expected {expected}, got {actual}. "
            "Run with: & $PY scripts/tune_optuna_multi.py ..."
        )
    multiprocessing.set_executable(str(expected))


def _parse_path_list(raw: str) -> List[Path]:
    return [Path(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_trace_weight_map(raw: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not raw.strip():
        return mapping

    for item in raw.split(","):
        token = item.strip()
        if not token or "=" not in token:
            continue
        name, weight_raw = token.split("=", 1)
        name = name.strip()
        if not name:
            continue
        try:
            weight = float(weight_raw.strip())
        except Exception:
            continue
        mapping[name] = max(1e-6, weight)
    return mapping


def _resolve_trace_weights(
    trace_paths: List[Path],
    weight_map: Dict[str, float],
    auto_weight: bool,
) -> List[float]:
    weights: List[float] = []
    for trace_path in trace_paths:
        stem = trace_path.stem
        stem_l = stem.lower()
        if stem in weight_map:
            w = float(weight_map[stem])
        elif auto_weight:
            if "balanced" in stem_l:
                w = 1.20
            elif "hotspot" in stem_l:
                w = 1.10
            elif "bursty" in stem_l:
                w = 1.00
            elif "base" in stem_l:
                w = 0.95
            else:
                w = 1.00
        else:
            w = 1.00
        weights.append(max(1e-6, float(w)))

    total = float(sum(weights))
    if total <= 0.0:
        return [1.0 / max(len(trace_paths), 1) for _ in trace_paths]
    return [float(w / total) for w in weights]


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    if not weights or len(weights) != len(values):
        return float(np.mean(values))
    arr_w = np.array(weights, dtype=np.float64)
    w_sum = float(np.sum(arr_w))
    if w_sum <= 0.0:
        return float(np.mean(values))
    arr_w /= w_sum
    return float(np.dot(np.array(values, dtype=np.float64), arr_w))


def _tail_metrics(profile_json_path: Path) -> Dict[str, float]:
    if not profile_json_path.exists():
        return {"p95_latency": 0.0, "p99_latency": 0.0}
    profile = load_json(profile_json_path)
    lat_list = profile.get("latency_by_inference", [])
    if not lat_list:
        return {"p95_latency": 0.0, "p99_latency": 0.0}
    arr = np.array(lat_list, dtype=np.float64)
    return {
        "p95_latency": float(np.percentile(arr, 95)),
        "p99_latency": float(np.percentile(arr, 99)),
    }


def _build_cfg_from_params(params: Dict[str, Any]):
    valid_fields = set(DEFAULT_MOE_CONFIG.to_dict().keys())
    cfg_kwargs = {k: v for k, v in params.items() if k in valid_fields}
    return replace(DEFAULT_MOE_CONFIG, **cfg_kwargs)


def _evaluate_on_traces(
    model_path: Path,
    trace_paths: List[Path],
    output_dir: Path,
    params: Dict[str, Any],
    trace_weights: List[float],
    cube_cfg: CubeConfig,
    capacity_max_ratio: float,
    strict_capacity: bool,
    overlap_transfer_compute: bool,
    robust_worst_weight: float,
    tail_p95_weight: float,
    tail_p99_weight: float,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    moe_cfg = _build_cfg_from_params(params)

    dispatch_policy = str(params.get("dispatch_policy", "fifo"))
    criticality_weight = float(params.get("criticality_weight", 0.0))
    resource_pressure_weight = float(params.get("resource_pressure_weight", 0.0))
    overlap_alpha = float(params.get("overlap_alpha", 1.0))
    overlap_model_mode = str(params.get("overlap_model_mode", "linear"))
    overlap_bw_power_law_alpha = float(params.get("overlap_bw_power_law_alpha", 0.7))
    overlap_z_depth_penalty = float(params.get("overlap_z_depth_penalty", 0.1))
    load_balance_weight = float(params.get("load_balance_weight", 0.0))

    latencies: List[float] = []
    conflicts: List[float] = []
    spaces: List[float] = []
    temporals: List[float] = []
    p95_list: List[float] = []
    p99_list: List[float] = []
    per_trace: Dict[str, Any] = {}

    for tidx, trace_path in enumerate(trace_paths):
        trace_out = output_dir / f"trace_{tidx:02d}_{trace_path.stem}"
        ensure_dir(trace_out)
        comparison = run_pipeline(
            model_path=model_path,
            trace_path=trace_path,
            output_dir=trace_out,
            cube_cfg_override=cube_cfg,
            moe_cfg_override=moe_cfg,
            export_profile=True,
            overlap_transfer_compute=overlap_transfer_compute,
            overlap_alpha=overlap_alpha,
            overlap_model_mode=overlap_model_mode,
            overlap_bw_power_law_alpha=max(1e-6, overlap_bw_power_law_alpha),
            overlap_z_depth_penalty=max(0.0, overlap_z_depth_penalty),
            load_balance_weight=load_balance_weight,
            dispatch_policy=dispatch_policy,
            criticality_weight=criticality_weight,
            resource_pressure_weight=resource_pressure_weight,
            capacity_max_ratio=capacity_max_ratio,
            strict_capacity=strict_capacity,
        )
        opt = comparison["optimized"]
        latency = float(opt.get("latency", 1e12))
        conflict = float(opt.get("conflict_score", 1e9))
        space = float(opt.get("space_utilization", 0.0))
        temporal = float(opt.get("temporal_utilization", 0.0))
        tail = _tail_metrics(trace_out / "optimized_profile.json")

        latencies.append(latency)
        conflicts.append(conflict)
        spaces.append(space)
        temporals.append(temporal)
        p95_list.append(float(tail["p95_latency"]))
        p99_list.append(float(tail["p99_latency"]))
        per_trace[trace_path.stem] = {
            "latency": latency,
            "conflict": conflict,
            "space": space,
            "temporal": temporal,
            "p95": float(tail["p95_latency"]),
            "p99": float(tail["p99_latency"]),
        }

    mean_latency = _weighted_mean(latencies, trace_weights) if latencies else 1e12
    worst_latency = float(np.max(latencies)) if latencies else 1e12
    mean_conflict = _weighted_mean(conflicts, trace_weights) if conflicts else 1e9
    mean_space = _weighted_mean(spaces, trace_weights) if spaces else 0.0
    mean_temporal = _weighted_mean(temporals, trace_weights) if temporals else 0.0
    mean_p95 = _weighted_mean(p95_list, trace_weights) if p95_list else 0.0
    mean_p99 = _weighted_mean(p99_list, trace_weights) if p99_list else 0.0
    robust_latency = (
        mean_latency
        + float(robust_worst_weight) * worst_latency
        + float(tail_p95_weight) * mean_p95
        + float(tail_p99_weight) * mean_p99
    )

    return {
        "trace_count": int(len(trace_paths)),
        "trace_weights": {
            trace_paths[i].stem: float(trace_weights[i])
            for i in range(min(len(trace_paths), len(trace_weights)))
        },
        "per_trace": per_trace,
        "robust_latency": robust_latency,
        "mean_latency": mean_latency,
        "worst_latency": worst_latency,
        "mean_conflict": mean_conflict,
        "mean_space": mean_space,
        "mean_temporal": mean_temporal,
        "mean_p95": mean_p95,
        "mean_p99": mean_p99,
    }


def _objective_factory(
    model_path: Path,
    trace_paths: List[Path],
    trace_weights: List[float],
    output_root: Path,
    cube_cfg: CubeConfig,
    capacity_max_ratio: float,
    strict_capacity: bool,
    overlap_transfer_compute: bool,
    robust_worst_weight: float,
    tail_p95_weight: float,
    tail_p99_weight: float,
    enable_shared_operator_replication: bool = True,
):
    def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
        dispatch_policy = trial.suggest_categorical("dispatch_policy", ["fifo", "criticality"])
        criticality_weight = trial.suggest_float("criticality_weight", 0.0, 0.8)
        resource_pressure_weight = trial.suggest_float("resource_pressure_weight", 0.0, 0.35)
        replica_pressure_low = trial.suggest_float("replica_pressure_low_threshold", 0.35, 0.65)
        replica_pressure_high = trial.suggest_float(
            "replica_pressure_high_threshold",
            max(replica_pressure_low + 0.05, 0.55),
            0.95,
        )
        dynamic_ratio_min = trial.suggest_float("dynamic_hot_subgraph_min_ratio", 0.10, 0.25)
        dynamic_ratio_max = trial.suggest_float(
            "dynamic_hot_subgraph_max_ratio",
            max(dynamic_ratio_min + 0.02, 0.18),
            0.45,
        )
        moe_cfg = replace(
            DEFAULT_MOE_CONFIG,
            enable_shared_operator_replication=bool(enable_shared_operator_replication),
            local_search_restarts=trial.suggest_int("local_search_restarts", 3, 8),
            local_search_max_iters=trial.suggest_int("local_search_max_iters", 20, 70),
            parallel_trials=trial.suggest_int("parallel_trials", 4, 14),
            grouping_multi_start_trials=trial.suggest_int("grouping_multi_start_trials", 3, 10),
            hot_subgraph_top_k=trial.suggest_int("hot_subgraph_top_k", 3, 9),
            enable_dynamic_hot_subgraph_top_k=trial.suggest_categorical(
                "enable_dynamic_hot_subgraph_top_k",
                [True, False],
            ),
            dynamic_hot_subgraph_min_ratio=dynamic_ratio_min,
            dynamic_hot_subgraph_max_ratio=dynamic_ratio_max,
            replication_volume_budget_ratio=trial.suggest_float("replication_volume_budget_ratio", 0.1, 0.45),
            enable_simulated_annealing=trial.suggest_categorical("enable_simulated_annealing", [True, False]),
            sa_steps=trial.suggest_int("sa_steps", 40, 160),
            sa_init_temp=trial.suggest_float("sa_init_temp", 0.6, 2.4),
            sa_cooling=trial.suggest_float("sa_cooling", 0.90, 0.99),
            cold_quant_bits=trial.suggest_int("cold_quant_bits", 4, 8),
            cold_sparsity_ratio=trial.suggest_float("cold_sparsity_ratio", 0.05, 0.45),
            hot_sparsity_ratio=trial.suggest_float("hot_sparsity_ratio", 0.0, 0.25),
            replica_pressure_low_threshold=replica_pressure_low,
            replica_pressure_high_threshold=replica_pressure_high,
            placement_conflict_weight=trial.suggest_float("placement_conflict_weight", 0.2, 1.5),
            transition_conflict_weight=trial.suggest_float("transition_conflict_weight", 0.0, 1.2),
            placement_load_weight=trial.suggest_float("placement_load_weight", 0.0, 1.0),
            placement_group_penalty=trial.suggest_float("placement_group_penalty", 0.0, 0.8),
            replica_diversity_penalty=trial.suggest_float("replica_diversity_penalty", 0.0, 0.8),
            enable_aspect_aware_packing=trial.suggest_categorical("enable_aspect_aware_packing", [True, False]),
            aspect_aware_weight=trial.suggest_float("aspect_aware_weight", 0.1, 1.2),
            fragmentation_penalty_weight=trial.suggest_float("fragmentation_penalty_weight", 0.05, 0.8),
            conflict_propagation_weight=trial.suggest_float("conflict_propagation_weight", 0.0, 1.2),
            capacity_peak_weight=trial.suggest_float("capacity_peak_weight", 0.05, 0.6),
        )

        trial_dir = output_root / f"trial_{trial.number:03d}"
        ensure_dir(trial_dir)

        overlap_alpha = trial.suggest_float("overlap_alpha", 0.5, 1.0)
        overlap_model_mode = trial.suggest_categorical(
            "overlap_model_mode",
            ["linear", "nonlinear_bandwidth_aware"],
        )
        overlap_bw_power_law_alpha = trial.suggest_float("overlap_bw_power_law_alpha", 0.4, 1.4)
        overlap_z_depth_penalty = trial.suggest_float("overlap_z_depth_penalty", 0.0, 0.35)
        load_balance_weight = trial.suggest_float("load_balance_weight", 0.0, 0.15)

        latencies: List[float] = []
        conflicts: List[float] = []
        spaces: List[float] = []
        temporals: List[float] = []
        p95_list: List[float] = []
        p99_list: List[float] = []

        for tidx, trace_path in enumerate(trace_paths):
            trace_out = trial_dir / f"trace_{tidx:02d}_{trace_path.stem}"
            ensure_dir(trace_out)
            comparison = run_pipeline(
                model_path=model_path,
                trace_path=trace_path,
                output_dir=trace_out,
                cube_cfg_override=cube_cfg,
                moe_cfg_override=moe_cfg,
                export_profile=True,
                overlap_transfer_compute=overlap_transfer_compute,
                overlap_alpha=overlap_alpha,
                overlap_model_mode=overlap_model_mode,
                overlap_bw_power_law_alpha=overlap_bw_power_law_alpha,
                overlap_z_depth_penalty=overlap_z_depth_penalty,
                load_balance_weight=load_balance_weight,
                dispatch_policy=dispatch_policy,
                criticality_weight=criticality_weight,
                resource_pressure_weight=resource_pressure_weight,
                capacity_max_ratio=capacity_max_ratio,
                strict_capacity=strict_capacity,
            )
            opt = comparison["optimized"]
            latencies.append(float(opt.get("latency", 1e12)))
            conflicts.append(float(opt.get("conflict_score", 1e9)))
            spaces.append(float(opt.get("space_utilization", 0.0)))
            temporals.append(float(opt.get("temporal_utilization", 0.0)))

            tail = _tail_metrics(trace_out / "optimized_profile.json")
            p95_list.append(float(tail["p95_latency"]))
            p99_list.append(float(tail["p99_latency"]))

        mean_latency = _weighted_mean(latencies, trace_weights) if latencies else 1e12
        worst_latency = float(np.max(latencies)) if latencies else 1e12
        mean_conflict = _weighted_mean(conflicts, trace_weights) if conflicts else 1e9
        mean_space = _weighted_mean(spaces, trace_weights) if spaces else 0.0
        mean_temporal = _weighted_mean(temporals, trace_weights) if temporals else 0.0
        mean_p95 = _weighted_mean(p95_list, trace_weights) if p95_list else 0.0
        mean_p99 = _weighted_mean(p99_list, trace_weights) if p99_list else 0.0

        robust_latency = (
            mean_latency
            + float(robust_worst_weight) * worst_latency
            + float(tail_p95_weight) * mean_p95
            + float(tail_p99_weight) * mean_p99
        )

        trial.set_user_attr("mean_temporal_utilization", mean_temporal)
        trial.set_user_attr("mean_latency", mean_latency)
        trial.set_user_attr("worst_latency", worst_latency)
        trial.set_user_attr("mean_p95_latency", mean_p95)
        trial.set_user_attr("mean_p99_latency", mean_p99)
        trial.set_user_attr("robust_latency", robust_latency)
        return robust_latency, mean_conflict, mean_space

    return objective


def run_multi_tuning(
    model_path: Path,
    trace_paths: List[Path],
    output_root: Path,
    n_trials: int,
    trace_weight_map: Dict[str, float],
    auto_trace_weight: bool,
    overlap_transfer_compute: bool,
    robust_worst_weight: float,
    tail_p95_weight: float,
    tail_p99_weight: float,
    enable_two_stage_tuning: bool,
    stage1_ratio: float,
    stage2_warmstart_topk: int,
    n_jobs: int,
    seed: int,
    holdout_paths: List[Path],
    holdout_topk: int,
    cube_cfg: CubeConfig,
    capacity_max_ratio: float,
    strict_capacity: bool,
    enable_shared_operator_replication: bool = True,
) -> Dict[str, Any]:
    ensure_dir(output_root)
    trace_weights = _resolve_trace_weights(trace_paths, trace_weight_map, auto_trace_weight)
    holdout_weights = _resolve_trace_weights(holdout_paths, trace_weight_map, auto_trace_weight) if holdout_paths else []
    objective = _objective_factory(
        model_path=model_path,
        trace_paths=trace_paths,
        trace_weights=trace_weights,
        output_root=output_root,
        cube_cfg=cube_cfg,
        capacity_max_ratio=capacity_max_ratio,
        strict_capacity=strict_capacity,
        overlap_transfer_compute=overlap_transfer_compute,
        robust_worst_weight=robust_worst_weight,
        tail_p95_weight=tail_p95_weight,
        tail_p99_weight=tail_p99_weight,
        enable_shared_operator_replication=bool(enable_shared_operator_replication),
    )

    def _create_study(tag: str, seed_offset: int) -> optuna.Study:
        return optuna.create_study(
            directions=["minimize", "minimize", "maximize"],
            study_name=f"cim_moe_multiobjective_{tag}",
            sampler=optuna.samplers.NSGAIISampler(seed=int(seed) + int(seed_offset)),
        )

    staged_studies: List[Tuple[str, optuna.Study]] = []
    use_two_stage = bool(enable_two_stage_tuning) and int(n_trials) >= 4
    if use_two_stage:
        ratio = max(0.2, min(0.8, float(stage1_ratio)))
        stage1_trials = max(2, min(int(n_trials) - 1, int(round(float(n_trials) * ratio))))
        stage2_trials = max(1, int(n_trials) - stage1_trials)

        study_stage1 = _create_study("stage1", 0)
        study_stage1.optimize(objective, n_trials=stage1_trials, n_jobs=max(1, int(n_jobs)))
        staged_studies.append(("stage1", study_stage1))

        complete_stage1 = [
            tr
            for tr in study_stage1.trials
            if tr.values is not None and tr.state == optuna.trial.TrialState.COMPLETE
        ]
        warm_trials = sorted(
            complete_stage1,
            key=lambda tr: float(list(tr.values)[0]),
        )[: max(1, int(stage2_warmstart_topk))]

        study_stage2 = _create_study("stage2", 31)
        for tr in warm_trials:
            try:
                study_stage2.enqueue_trial(tr.params)
            except Exception:
                continue
        study_stage2.optimize(objective, n_trials=stage2_trials, n_jobs=max(1, int(n_jobs)))
        staged_studies.append(("stage2", study_stage2))
    else:
        study_single = _create_study("single", 0)
        study_single.optimize(objective, n_trials=max(1, n_trials), n_jobs=max(1, int(n_jobs)))
        staged_studies.append(("single", study_single))

    rows: List[Dict[str, Any]] = []
    merged_trials: List[optuna.trial.FrozenTrial] = []
    for stage_name, stage_study in staged_studies:
        for tr in stage_study.trials:
            values = list(tr.values) if tr.values is not None else [None, None, None]
            row = {
                "stage": stage_name,
                "trial": tr.number,
                "robust_latency": values[0],
                "conflict_score": values[1],
                "space_utilization": values[2],
                "state": str(tr.state),
                "mean_temporal_utilization": tr.user_attrs.get("mean_temporal_utilization"),
                "mean_latency": tr.user_attrs.get("mean_latency"),
                "worst_latency": tr.user_attrs.get("worst_latency"),
                "mean_p95_latency": tr.user_attrs.get("mean_p95_latency"),
                "mean_p99_latency": tr.user_attrs.get("mean_p99_latency"),
                "enable_shared_operator_replication": bool(enable_shared_operator_replication),
            }
            row.update(tr.params)
            rows.append(row)
            merged_trials.append(tr)

    df = pd.DataFrame(rows)
    csv_path = output_root / "multiobjective_trials.csv"
    df.to_csv(csv_path, index=False)

    pareto_rows: List[Dict[str, Any]] = []
    complete_trials = [
        tr
        for tr in merged_trials
        if tr.values is not None and tr.state == optuna.trial.TrialState.COMPLETE
    ]
    for tr in complete_trials:
        vals = list(tr.values)
        params = dict(tr.params)
        params["enable_shared_operator_replication"] = bool(enable_shared_operator_replication)
        pareto_rows.append(
            {
                "trial": tr.number,
                "robust_latency": vals[0],
                "conflict_score": vals[1],
                "space_utilization": vals[2],
                "mean_temporal_utilization": tr.user_attrs.get("mean_temporal_utilization"),
                "mean_latency": tr.user_attrs.get("mean_latency"),
                "worst_latency": tr.user_attrs.get("worst_latency"),
                "mean_p95_latency": tr.user_attrs.get("mean_p95_latency"),
                "mean_p99_latency": tr.user_attrs.get("mean_p99_latency"),
                "params": params,
            }
        )

    pareto_rows.sort(key=lambda item: (float(item.get("robust_latency", 1e18)), float(item.get("conflict_score", 1e18)), -float(item.get("space_utilization", 0.0))))
    pareto_rows = pareto_rows[: max(1, min(len(pareto_rows), int(max(3, stage2_warmstart_topk * 2))))]

    holdout_eval: List[Dict[str, Any]] = []
    if holdout_paths and pareto_rows:
        ranked = sorted(pareto_rows, key=lambda x: float(x.get("robust_latency", 1e18)))
        for rank, row in enumerate(ranked[: max(1, int(holdout_topk))], start=1):
            trial_id = int(row.get("trial", -1))
            eval_out = output_root / "holdout_eval" / f"pareto_rank_{rank:02d}_trial_{trial_id:03d}"
            metrics = _evaluate_on_traces(
                model_path=model_path,
                trace_paths=holdout_paths,
                output_dir=eval_out,
                params=row.get("params", {}),
                trace_weights=holdout_weights,
                cube_cfg=cube_cfg,
                capacity_max_ratio=capacity_max_ratio,
                strict_capacity=strict_capacity,
                overlap_transfer_compute=overlap_transfer_compute,
                robust_worst_weight=robust_worst_weight,
                tail_p95_weight=tail_p95_weight,
                tail_p99_weight=tail_p99_weight,
            )
            holdout_eval.append(
                {
                    "rank": rank,
                    "trial": trial_id,
                    "train_objectives": {
                        "robust_latency": row.get("robust_latency"),
                        "conflict_score": row.get("conflict_score"),
                        "space_utilization": row.get("space_utilization"),
                    },
                    "holdout_metrics": metrics,
                    "params": row.get("params", {}),
                }
            )

    save_json(
        output_root / "multiobjective_pareto.json",
        {
            "n_trials": int(len(merged_trials)),
            "pareto_size": int(len(pareto_rows)),
            "two_stage": {
                "enabled": bool(use_two_stage),
                "stage_count": len(staged_studies),
                "stage_names": [name for name, _ in staged_studies],
                "stage1_ratio": float(max(0.2, min(0.8, float(stage1_ratio)))),
                "stage2_warmstart_topk": int(max(1, stage2_warmstart_topk)),
            },
            "trace_count": int(len(trace_paths)),
            "trace_weights": {
                trace_paths[i].stem: float(trace_weights[i])
                for i in range(min(len(trace_paths), len(trace_weights)))
            },
            "robust_weights": {
                "worst": float(robust_worst_weight),
                "p95": float(tail_p95_weight),
                "p99": float(tail_p99_weight),
            },
            "pareto": pareto_rows,
            "holdout_eval": holdout_eval,
            "csv": str(csv_path),
            "overlap_transfer_compute": bool(overlap_transfer_compute),
            "enable_shared_operator_replication": bool(enable_shared_operator_replication),
            "n_jobs": int(max(1, n_jobs)),
            "seed": int(seed),
            "cube": cube_cfg.to_dict(),
            "capacity_max_ratio": float(capacity_max_ratio),
            "strict_capacity": bool(strict_capacity),
        },
    )

    return {
        "n_trials": int(len(merged_trials)),
        "pareto_size": int(len(pareto_rows)),
        "csv": str(csv_path),
    }


def main() -> None:
    _ensure_project_venv_python()
    parser = argparse.ArgumentParser(description="Robust multi-objective Optuna tuning for latency/conflict/space")
    parser.add_argument("--model", type=Path, default=Path("data/sample_model.json"))
    parser.add_argument("--trace", type=Path, default=Path("data/sample_trace.json"))
    parser.add_argument("--traces", type=str, default="", help="Optional comma-separated trace paths for robust tuning")
    parser.add_argument("--output", type=Path, default=Path("outputs/tuning_multi"))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--cube-n", type=int, default=DEFAULT_CUBE_CONFIG.n)
    parser.add_argument("--cube-d", type=int, default=DEFAULT_CUBE_CONFIG.d)
    parser.add_argument("--cube-h", type=int, default=DEFAULT_CUBE_CONFIG.h)
    parser.add_argument("--cube-w", type=int, default=DEFAULT_CUBE_CONFIG.w)
    parser.add_argument("--max-parallel-subcubes", type=int, default=DEFAULT_CUBE_CONFIG.max_parallel_subcubes)
    parser.add_argument("--capacity-max-ratio", type=float, default=2.0)
    parser.add_argument("--strict-capacity", action="store_true")
    parser.add_argument("--overlap-transfer-compute", action="store_true")
    parser.add_argument(
        "--disable-shared-replication",
        action="store_true",
        help="Disable shared/non-expert operator replication during tuning and holdout replay",
    )
    parser.add_argument("--enable-two-stage-tuning", action="store_true", help="Enable two-stage tuning with stage2 warm-start")
    parser.add_argument("--stage1-ratio", type=float, default=0.45, help="Stage1 trial ratio in two-stage tuning")
    parser.add_argument("--stage2-warmstart-topk", type=int, default=3, help="Warm-start top-k trials from stage1")
    parser.add_argument(
        "--trace-weights",
        type=str,
        default="",
        help="Optional per-trace weights, format: stem1=1.2,stem2=0.8",
    )
    parser.add_argument(
        "--auto-trace-weight",
        action="store_true",
        help="Enable filename-based automatic trace weighting",
    )
    parser.add_argument("--robust-worst-weight", type=float, default=0.15)
    parser.add_argument("--tail-p95-weight", type=float, default=0.05)
    parser.add_argument("--tail-p99-weight", type=float, default=0.08)
    parser.add_argument("--holdout-traces", type=str, default="", help="Optional comma-separated holdout traces")
    parser.add_argument("--holdout-topk", type=int, default=2, help="Evaluate top-k pareto candidates on holdout traces")
    parser.add_argument("--n-jobs", type=int, default=1, help="Optuna parallel workers")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    trace_paths = _parse_path_list(args.traces) if args.traces.strip() else [args.trace]
    holdout_paths = _parse_path_list(args.holdout_traces) if args.holdout_traces.strip() else []
    cube_cfg = CubeConfig(
        n=max(2, min(4, int(args.cube_n))),
        d=max(1, int(args.cube_d)),
        h=max(4096, min(16384, int(args.cube_h))),
        w=max(4096, min(16384, int(args.cube_w))),
        max_parallel_subcubes=max(1, int(args.max_parallel_subcubes)),
        switching_penalty=DEFAULT_CUBE_CONFIG.switching_penalty,
        z_access_penalty=DEFAULT_CUBE_CONFIG.z_access_penalty,
    )

    result = run_multi_tuning(
        model_path=args.model,
        trace_paths=trace_paths,
        output_root=args.output,
        n_trials=args.trials,
        trace_weight_map=_parse_trace_weight_map(args.trace_weights),
        auto_trace_weight=bool(args.auto_trace_weight),
        overlap_transfer_compute=args.overlap_transfer_compute,
        robust_worst_weight=float(args.robust_worst_weight),
        tail_p95_weight=float(args.tail_p95_weight),
        tail_p99_weight=float(args.tail_p99_weight),
        enable_two_stage_tuning=bool(args.enable_two_stage_tuning),
        stage1_ratio=float(args.stage1_ratio),
        stage2_warmstart_topk=max(1, int(args.stage2_warmstart_topk)),
        n_jobs=max(1, int(args.n_jobs)),
        seed=int(args.seed),
        holdout_paths=holdout_paths,
        holdout_topk=max(1, int(args.holdout_topk)),
        cube_cfg=cube_cfg,
        capacity_max_ratio=max(1e-9, float(args.capacity_max_ratio)),
        strict_capacity=bool(args.strict_capacity),
        enable_shared_operator_replication=not bool(args.disable_shared_replication),
    )
    print("==== Multi-objective Tuning Done ====")
    print(f"trials: {result['n_trials']}, pareto: {result['pareto_size']}")


if __name__ == "__main__":
    main()
