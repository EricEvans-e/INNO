from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
from pathlib import Path
import platform
import random
from typing import Any, Dict

import numpy as np

from config.cube_config import DEFAULT_CUBE_CONFIG, CubeConfig
from config.moe_config import DEFAULT_MOE_CONFIG
from src.mapping_solver import solve_mapping
from src.model_parser import parse_activation_trace, parse_model
from src.simulator import save_simulation, simulate
from src.utils import (
    ensure_dir,
    plot_cooccurrence_heatmap,
    plot_latency_distribution,
    plot_mapping_slices,
    plot_schedule_gantt,
    plot_subcube_contention,
    save_json,
)


def _comparison_dict(base: Dict[str, float], opt: Dict[str, float]) -> Dict[str, Any]:
    def improve_ratio(k: str, lower_is_better: bool) -> float:
        b = base.get(k, 0.0)
        o = opt.get(k, 0.0)
        if b == 0:
            return 0.0
        if lower_is_better:
            return (b - o) / b
        return (o - b) / b

    return {
        "baseline": base,
        "optimized": opt,
        "improvement": {
            "latency": improve_ratio("latency", lower_is_better=True),
            "space_utilization": improve_ratio("space_utilization", lower_is_better=False),
            "temporal_utilization": improve_ratio("temporal_utilization", lower_is_better=False),
            "switching_penalty_cycles": improve_ratio("switching_penalty_cycles", lower_is_better=True),
            "pipeline_bubble_cycles": improve_ratio("pipeline_bubble_cycles", lower_is_better=True),
        },
    }


def _variant_improvement(base: Dict[str, float], cur: Dict[str, float]) -> Dict[str, float]:
    def safe_ratio(k: str, lower_is_better: bool) -> float:
        b = base.get(k, 0.0)
        c = cur.get(k, 0.0)
        if b == 0:
            return 0.0
        if lower_is_better:
            return (b - c) / b
        return (c - b) / b

    return {
        "latency": safe_ratio("latency", lower_is_better=True),
        "space_utilization": safe_ratio("space_utilization", lower_is_better=False),
        "temporal_utilization": safe_ratio("temporal_utilization", lower_is_better=False),
        "switching_penalty_cycles": safe_ratio("switching_penalty_cycles", lower_is_better=True),
        "pipeline_bubble_cycles": safe_ratio("pipeline_bubble_cycles", lower_is_better=True),
    }


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _capacity_summary(parsed_model: Any, cube_cfg: Any, max_ratio: float) -> Dict[str, Any]:
    model_volume = float(sum(c.h * c.w * c.d for c in parsed_model.weight_cubes.values()))
    cube_volume = float(cube_cfg.total_volume)
    ratio = cube_volume / max(model_volume, 1.0)
    return {
        "model_logical_volume": model_volume,
        "cube_total_volume": cube_volume,
        "capacity_ratio": ratio,
        "capacity_max_ratio": float(max_ratio),
        "within_capacity_limit": bool(ratio <= float(max_ratio) + 1e-9),
    }


def _build_solution_dict(
    variant: str,
    cube_cfg: Any,
    mapping: Any,
    sim: Any,
) -> Dict[str, Any]:
    weight_cubes = [
        {
            "id": p.physical_cube_id,
            "op": p.operator_id,
            "logical_cube_id": p.logical_cube_id,
            "physical_cube_id": p.physical_cube_id,
            "operator_id": p.operator_id,
            "expert_id": p.expert_id,
            "replica_id": p.replica_id,
            "subcube": p.subcube,
            "coords": [p.x, p.y, p.z],
            "shape": [p.h, p.w, p.d],
            "replicas": len(mapping.cube_to_placements.get(p.logical_cube_id, [])),
            "compression_ratio": p.compression_ratio,
            "strategy": p.strategy,
        }
        for p in mapping.placements
    ]

    schedule = [
        {
            "op_id": rec.get("operator_id"),
            "task_id": rec.get("task_id"),
            "inference_id": rec.get("inference_id"),
            "operator_id": rec.get("operator_id"),
            "logical_cube_id": rec.get("cube_id"),
            "physical_cube_id": rec.get("physical_cube_id"),
            "subcube": rec.get("subcube"),
            "z": rec.get("z"),
            "start_cycle": rec.get("start"),
            "end_cycle": rec.get("end"),
            "duration": int(rec.get("end", 0)) - int(rec.get("start", 0)),
        }
        for rec in sim.schedule
    ]

    return {
        "variant": variant,
        "hardware": {
            "N": cube_cfg.n,
            "D": cube_cfg.d,
            "subcube_size": [cube_cfg.h, cube_cfg.w],
            "num_subcubes": cube_cfg.num_subcubes,
            "max_parallel_subcubes": cube_cfg.max_parallel_subcubes,
            "switching_penalty": cube_cfg.switching_penalty,
            "z_access_penalty": cube_cfg.z_access_penalty,
        },
        "spatial_mapping": {
            "weight_cubes": weight_cubes,
            "unplaced_cubes": mapping.unplaced_cubes,
            "mapping_rate": mapping.metrics.get("mapping_rate", 0.0),
            "space_utilization": mapping.metrics.get("space_utilization", 0.0),
        },
        "static_schedule": schedule,
        "weight_cubes": weight_cubes,
        "schedule": schedule,
        "metrics": sim.metrics,
        "metadata": {
            "mapping": mapping.metadata,
            "simulation": sim.metadata,
        },
    }


def _write_run_manifest(
    output_dir: Path,
    model_path: Path,
    trace_path: Path,
    cube_cfg: Any,
    moe_cfg: Any,
    overlap_transfer_compute: bool,
    overlap_alpha: float,
    overlap_model_mode: str,
    overlap_bw_power_law_alpha: float,
    overlap_z_depth_penalty: float,
    load_balance_weight: float,
    dispatch_policy: str,
    criticality_weight: float,
    resource_pressure_weight: float,
    capacity: Dict[str, Any],
    strict_capacity: bool,
    seed: int | None,
    comparison: Dict[str, Any],
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model_path": str(model_path),
            "trace_path": str(trace_path),
            "model_sha256": _file_sha256(model_path),
            "trace_sha256": _file_sha256(trace_path),
        },
        "configs": {
            "cube": cube_cfg.to_dict(),
            "moe": moe_cfg.to_dict(),
            "simulator": {
                "overlap_transfer_compute": bool(overlap_transfer_compute),
                "overlap_alpha": float(overlap_alpha),
                "overlap_model_mode": str(overlap_model_mode),
                "overlap_bw_power_law_alpha": float(overlap_bw_power_law_alpha),
                "overlap_z_depth_penalty": float(overlap_z_depth_penalty),
                "load_balance_weight": float(load_balance_weight),
                "dispatch_policy": str(dispatch_policy),
                "criticality_weight": float(criticality_weight),
                "resource_pressure_weight": float(resource_pressure_weight),
            },
            "seed": int(seed) if seed is not None else None,
            "capacity": capacity,
            "strict_capacity": bool(strict_capacity),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "result_summary": {
            "baseline_latency": float(comparison["baseline"].get("latency", 0.0)),
            "optimized_latency": float(comparison["optimized"].get("latency", 0.0)),
            "latency_improvement": float(comparison["improvement"].get("latency", 0.0)),
        },
    }
    save_json(output_dir / "run_manifest.json", manifest)


def run_pipeline(
    model_path: Path,
    trace_path: Path,
    output_dir: Path,
    cube_cfg_override: Any = None,
    moe_cfg_override: Any = None,
    export_profile: bool = True,
    overlap_transfer_compute: bool = False,
    overlap_alpha: float = 1.0,
    overlap_model_mode: str = "linear",
    overlap_bw_power_law_alpha: float = 0.7,
    overlap_z_depth_penalty: float = 0.1,
    load_balance_weight: float = 0.0,
    dispatch_policy: str = "fifo",
    criticality_weight: float = 0.0,
    resource_pressure_weight: float = 0.0,
    capacity_max_ratio: float = 2.0,
    strict_capacity: bool = False,
    seed: int | None = None,
) -> Dict[str, Any]:
    cube_cfg = cube_cfg_override if cube_cfg_override is not None else DEFAULT_CUBE_CONFIG
    moe_cfg = moe_cfg_override if moe_cfg_override is not None else DEFAULT_MOE_CONFIG
    cube_cfg.validate()
    moe_cfg.validate()

    ensure_dir(output_dir)

    parsed_model = parse_model(model_path, cube_cfg)
    capacity = _capacity_summary(parsed_model, cube_cfg, max(1e-9, float(capacity_max_ratio)))
    if strict_capacity and not capacity["within_capacity_limit"]:
        raise ValueError(
            "Cube capacity exceeds strict contest limit: "
            f"ratio={capacity['capacity_ratio']:.4f}, max={capacity['capacity_max_ratio']:.4f}"
        )
    activation_trace = parse_activation_trace(trace_path)

    variants = {
        "baseline": {
            "packing_policy": "first_fit",
            "enable_moe_optimization": False,
            "enable_adaptive_replication": False,
        },
        "ablation_no_moe": {
            "packing_policy": "best_fit",
            "enable_moe_optimization": False,
            "enable_adaptive_replication": True,
        },
        "ablation_no_replication": {
            "packing_policy": "best_fit",
            "enable_moe_optimization": True,
            "enable_adaptive_replication": False,
        },
        "optimized": {
            "packing_policy": "best_fit",
            "enable_moe_optimization": True,
            "enable_adaptive_replication": True,
        },
    }

    mapping_results = {}
    simulation_results = {}
    for name, cfg in variants.items():
        mapping = solve_mapping(
            parsed_model=parsed_model,
            activation_trace=activation_trace,
            cube_cfg=cube_cfg,
            moe_cfg=moe_cfg,
            packing_policy=cfg["packing_policy"],
            enable_moe_optimization=cfg["enable_moe_optimization"],
            enable_adaptive_replication=cfg["enable_adaptive_replication"],
        )
        sim = simulate(
            parsed_model,
            mapping,
            activation_trace,
            cube_cfg,
            overlap_transfer_compute=overlap_transfer_compute,
            overlap_alpha=overlap_alpha,
            overlap_model_mode=overlap_model_mode,
            overlap_bw_power_law_alpha=overlap_bw_power_law_alpha,
            overlap_z_depth_penalty=overlap_z_depth_penalty,
            load_balance_weight=load_balance_weight,
            dispatch_policy=dispatch_policy,
            criticality_weight=criticality_weight,
            resource_pressure_weight=resource_pressure_weight,
        )

        mapping_results[name] = mapping
        simulation_results[name] = sim
        save_json(output_dir / f"{name}_mapping.json", mapping.to_dict())
        save_simulation(output_dir / f"{name}_simulation.json", sim)
        save_json(
            output_dir / f"{name}_solution.json",
            _build_solution_dict(name, cube_cfg, mapping, sim),
        )

    baseline_sim = simulation_results["baseline"]
    optimized_sim = simulation_results["optimized"]

    comparison = _comparison_dict(baseline_sim.metrics, optimized_sim.metrics)
    comparison["ablation"] = {
        name: {
            "metrics": sim.metrics,
            "vs_baseline": _variant_improvement(baseline_sim.metrics, sim.metrics),
        }
        for name, sim in simulation_results.items()
    }
    save_json(output_dir / "comparison_metrics.json", comparison)
    save_json(
        output_dir / "solution.json",
        _build_solution_dict("optimized", cube_cfg, mapping_results["optimized"], optimized_sim),
    )

    co_matrix = activation_trace["cooccurrence_matrix"]
    plot_cooccurrence_heatmap(co_matrix, output_dir / "cooccurrence_heatmap.png")
    plot_mapping_slices(
        placements=[p.to_dict() for p in mapping_results["optimized"].placements],
        n_subcubes=cube_cfg.num_subcubes,
        d=cube_cfg.d,
        h=cube_cfg.h,
        w=cube_cfg.w,
        output_path=output_dir / "optimized_mapping_slices.png",
        max_layers=3,
    )
    plot_schedule_gantt(optimized_sim.schedule, output_dir / "optimized_schedule_gantt.png")

    profiling = optimized_sim.metadata.get("profiling", {})
    if profiling:
        plot_latency_distribution(
            profiling.get("latency_by_inference", []),
            output_dir / "optimized_latency_distribution.png",
        )
        plot_subcube_contention(
            profiling.get("subcube_contention", []),
            output_dir / "optimized_subcube_contention.png",
        )

    if export_profile:
        save_json(
            output_dir / "baseline_profile.json",
            simulation_results["baseline"].metadata.get("profiling", {}),
        )
        save_json(
            output_dir / "optimized_profile.json",
            optimized_sim.metadata.get("profiling", {}),
        )

    _write_run_manifest(
        output_dir=output_dir,
        model_path=model_path,
        trace_path=trace_path,
        cube_cfg=cube_cfg,
        moe_cfg=moe_cfg,
        overlap_transfer_compute=overlap_transfer_compute,
        overlap_alpha=overlap_alpha,
        overlap_model_mode=overlap_model_mode,
        overlap_bw_power_law_alpha=overlap_bw_power_law_alpha,
        overlap_z_depth_penalty=overlap_z_depth_penalty,
        load_balance_weight=load_balance_weight,
        dispatch_policy=dispatch_policy,
        criticality_weight=criticality_weight,
        resource_pressure_weight=resource_pressure_weight,
        capacity=capacity,
        strict_capacity=strict_capacity,
        seed=seed,
        comparison=comparison,
    )

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="3D heterogeneous CIM scheduler for MoE LLM inference")
    parser.add_argument("--model", type=Path, default=Path("data/sample_model.json"), help="Path to model JSON/ONNX")
    parser.add_argument("--trace", type=Path, default=Path("data/sample_trace.json"), help="Path to activation trace JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--profile", action="store_true", help="Export baseline/optimized profiling JSON reports")
    parser.add_argument("--cube-n", type=int, default=DEFAULT_CUBE_CONFIG.n, help="Cube grid side N in [2,4]")
    parser.add_argument("--cube-d", type=int, default=DEFAULT_CUBE_CONFIG.d, help="Cube depth D")
    parser.add_argument("--cube-h", type=int, default=DEFAULT_CUBE_CONFIG.h, help="Sub-Cube height H")
    parser.add_argument("--cube-w", type=int, default=DEFAULT_CUBE_CONFIG.w, help="Sub-Cube width W")
    parser.add_argument("--capacity-max-ratio", type=float, default=2.0, help="Strict Cube/model capacity ratio upper bound")
    parser.add_argument("--strict-capacity", action="store_true", help="Fail if Cube capacity exceeds --capacity-max-ratio")
    parser.add_argument("--search-trials", type=int, default=DEFAULT_MOE_CONFIG.parallel_trials, help="Parallel search trials")
    parser.add_argument("--parallel-workers", type=int, default=DEFAULT_MOE_CONFIG.parallel_workers, help="Parallel search workers")
    parser.add_argument("--local-restarts", type=int, default=DEFAULT_MOE_CONFIG.local_search_restarts, help="Local search restarts")
    parser.add_argument("--local-iters", type=int, default=DEFAULT_MOE_CONFIG.local_search_max_iters, help="Local search max iterations")
    parser.add_argument("--hot-subgraph-topk", type=int, default=DEFAULT_MOE_CONFIG.hot_subgraph_top_k, help="Exact hot subgraph top-k groups")
    parser.add_argument("--disable-sa", action="store_true", help="Disable simulated annealing")
    parser.add_argument("--disable-parallel-search", action="store_true", help="Disable parallel search")
    parser.add_argument("--disable-compression", action="store_true", help="Disable quantization/sparsity compression")
    parser.add_argument(
        "--overlap-transfer-compute",
        action="store_true",
        help="Allow transfer and compute to overlap in simulator duration model",
    )
    parser.add_argument(
        "--overlap-alpha",
        type=float,
        default=1.0,
        help="Overlap strength in [0,1] when --overlap-transfer-compute is enabled",
    )
    parser.add_argument(
        "--overlap-model-mode",
        type=str,
        default="linear",
        choices=["linear", "nonlinear_bandwidth_aware"],
        help="Overlap duration model: linear or bandwidth/z-aware nonlinear",
    )
    parser.add_argument(
        "--overlap-bw-power-law-alpha",
        type=float,
        default=0.7,
        help="Power-law alpha for nonlinear bandwidth pressure term (>0)",
    )
    parser.add_argument(
        "--overlap-z-depth-penalty",
        type=float,
        default=0.1,
        help="Depth penalty coefficient for nonlinear overlap model (>=0)",
    )
    parser.add_argument(
        "--load-balance-weight",
        type=float,
        default=0.0,
        help="Penalty weight for selecting overloaded subcubes in scheduling",
    )
    parser.add_argument(
        "--replication-volume-budget-ratio",
        type=float,
        default=DEFAULT_MOE_CONFIG.replication_volume_budget_ratio,
        help="Extra replication volume budget ratio in [0,1]",
    )
    parser.add_argument(
        "--hot-quant-bits",
        type=int,
        default=DEFAULT_MOE_CONFIG.hot_quant_bits,
        help="Quantization bits for hot experts",
    )
    parser.add_argument(
        "--cold-quant-bits",
        type=int,
        default=DEFAULT_MOE_CONFIG.cold_quant_bits,
        help="Quantization bits for cold experts",
    )
    parser.add_argument(
        "--hot-sparsity-ratio",
        type=float,
        default=DEFAULT_MOE_CONFIG.hot_sparsity_ratio,
        help="Sparsity ratio for hot experts in [0,1)",
    )
    parser.add_argument(
        "--cold-sparsity-ratio",
        type=float,
        default=DEFAULT_MOE_CONFIG.cold_sparsity_ratio,
        help="Sparsity ratio for cold experts in [0,1)",
    )
    parser.add_argument(
        "--sa-steps",
        type=int,
        default=DEFAULT_MOE_CONFIG.sa_steps,
        help="Simulated annealing steps",
    )
    parser.add_argument(
        "--sa-init-temp",
        type=float,
        default=DEFAULT_MOE_CONFIG.sa_init_temp,
        help="Simulated annealing initial temperature",
    )
    parser.add_argument(
        "--sa-cooling",
        type=float,
        default=DEFAULT_MOE_CONFIG.sa_cooling,
        help="Simulated annealing cooling factor in (0,1)",
    )
    parser.add_argument(
        "--placement-conflict-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.placement_conflict_weight,
        help="Dynamic placement penalty weight for co-occurrence conflict",
    )
    parser.add_argument(
        "--placement-load-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.placement_load_weight,
        help="Dynamic placement penalty weight for subcube load",
    )
    parser.add_argument(
        "--placement-group-penalty",
        type=float,
        default=DEFAULT_MOE_CONFIG.placement_group_penalty,
        help="Dynamic placement penalty when mixing groups in a subcube",
    )
    parser.add_argument(
        "--replica-diversity-penalty",
        type=float,
        default=DEFAULT_MOE_CONFIG.replica_diversity_penalty,
        help="Dynamic placement penalty for replica concentration on same subcube",
    )
    parser.add_argument(
        "--disable-aspect-aware-packing",
        action="store_true",
        help="Disable aspect/fragmentation-aware packing policy promotion",
    )
    parser.add_argument(
        "--aspect-aware-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.aspect_aware_weight,
        help="Aspect-ratio penalty weight used by aspect-aware packing",
    )
    parser.add_argument(
        "--fragmentation-penalty-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.fragmentation_penalty_weight,
        help="Fragmentation penalty weight used by aspect-aware packing",
    )
    parser.add_argument(
        "--conflict-propagation-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.conflict_propagation_weight,
        help="Conflict propagation weight for subcube imbalance scoring",
    )
    parser.add_argument(
        "--capacity-peak-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.capacity_peak_weight,
        help="Peak capacity pressure weight for placement scoring",
    )
    parser.add_argument(
        "--transition-conflict-weight",
        type=float,
        default=DEFAULT_MOE_CONFIG.transition_conflict_weight,
        help="Dynamic placement penalty for experts with high transition affinity",
    )
    parser.add_argument(
        "--grouping-multi-start-trials",
        type=int,
        default=DEFAULT_MOE_CONFIG.grouping_multi_start_trials,
        help="Multi-start trials for mutually exclusive grouping",
    )
    parser.add_argument(
        "--replica-pressure-low-threshold",
        type=float,
        default=DEFAULT_MOE_CONFIG.replica_pressure_low_threshold,
        help="Replica pressure threshold for enabling 2-way replication",
    )
    parser.add_argument(
        "--replica-pressure-high-threshold",
        type=float,
        default=DEFAULT_MOE_CONFIG.replica_pressure_high_threshold,
        help="Replica pressure threshold for enabling 3-way replication",
    )
    parser.add_argument(
        "--disable-dynamic-hot-subgraph-topk",
        action="store_true",
        help="Disable dynamic hot-subgraph top-k sizing",
    )
    parser.add_argument(
        "--dynamic-hot-subgraph-min-ratio",
        type=float,
        default=DEFAULT_MOE_CONFIG.dynamic_hot_subgraph_min_ratio,
        help="Minimum ratio used by dynamic hot-subgraph top-k sizing",
    )
    parser.add_argument(
        "--dynamic-hot-subgraph-max-ratio",
        type=float,
        default=DEFAULT_MOE_CONFIG.dynamic_hot_subgraph_max_ratio,
        help="Maximum ratio used by dynamic hot-subgraph top-k sizing",
    )
    parser.add_argument(
        "--dispatch-policy",
        type=str,
        default="fifo",
        choices=["fifo", "criticality"],
        help="Task dispatch policy in simulator",
    )
    parser.add_argument(
        "--criticality-weight",
        type=float,
        default=0.0,
        help="Priority weight for critical-path operators when dispatch-policy=criticality",
    )
    parser.add_argument(
        "--resource-pressure-weight",
        type=float,
        default=0.0,
        help="Priority weight for subcube resource pressure when dispatch-policy=criticality",
    )
    parser.add_argument(
        "--max-parallel-subcubes",
        type=int,
        default=DEFAULT_CUBE_CONFIG.max_parallel_subcubes,
        help="Upper bound of simultaneously active subcubes in simulator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MOE_CONFIG.random_seed,
        help="Global random seed used by mapping search and runtime reproducibility logs",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic execution by disabling parallel search workers",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    replica_low = max(0.0, float(args.replica_pressure_low_threshold))
    replica_high = max(replica_low, float(args.replica_pressure_high_threshold))
    dynamic_ratio_min = max(1e-6, min(1.0, float(args.dynamic_hot_subgraph_min_ratio)))
    dynamic_ratio_max = max(dynamic_ratio_min, min(1.0, float(args.dynamic_hot_subgraph_max_ratio)))

    moe_cfg = replace(
        DEFAULT_MOE_CONFIG,
        parallel_trials=max(1, args.search_trials),
        parallel_workers=max(1, args.parallel_workers),
        local_search_restarts=max(1, args.local_restarts),
        local_search_max_iters=max(1, args.local_iters),
        hot_subgraph_top_k=max(0, args.hot_subgraph_topk),
        enable_simulated_annealing=(not args.disable_sa),
        enable_parallel_search=(not args.disable_parallel_search),
        enable_blockwise_quantization=(not args.disable_compression),
        enable_sparse_compression=(not args.disable_compression),
        replication_volume_budget_ratio=max(0.0, min(1.0, args.replication_volume_budget_ratio)),
        hot_quant_bits=max(2, min(8, args.hot_quant_bits)),
        cold_quant_bits=max(2, min(8, args.cold_quant_bits)),
        hot_sparsity_ratio=max(0.0, min(0.99, args.hot_sparsity_ratio)),
        cold_sparsity_ratio=max(0.0, min(0.99, args.cold_sparsity_ratio)),
        sa_steps=max(1, int(args.sa_steps)),
        sa_init_temp=max(1e-6, float(args.sa_init_temp)),
        sa_cooling=min(0.9999, max(1e-6, float(args.sa_cooling))),
        placement_conflict_weight=max(0.0, args.placement_conflict_weight),
        transition_conflict_weight=max(0.0, args.transition_conflict_weight),
        placement_load_weight=max(0.0, args.placement_load_weight),
        placement_group_penalty=max(0.0, args.placement_group_penalty),
        replica_diversity_penalty=max(0.0, args.replica_diversity_penalty),
        enable_aspect_aware_packing=(not args.disable_aspect_aware_packing),
        aspect_aware_weight=max(0.0, args.aspect_aware_weight),
        fragmentation_penalty_weight=max(0.0, args.fragmentation_penalty_weight),
        conflict_propagation_weight=max(0.0, args.conflict_propagation_weight),
        capacity_peak_weight=max(0.0, args.capacity_peak_weight),
        grouping_multi_start_trials=max(1, int(args.grouping_multi_start_trials)),
        replica_pressure_low_threshold=replica_low,
        replica_pressure_high_threshold=replica_high,
        enable_dynamic_hot_subgraph_top_k=(not args.disable_dynamic_hot_subgraph_topk),
        dynamic_hot_subgraph_min_ratio=dynamic_ratio_min,
        dynamic_hot_subgraph_max_ratio=dynamic_ratio_max,
        random_seed=max(0, int(args.seed)),
    )

    if args.deterministic:
        moe_cfg = replace(
            moe_cfg,
            enable_parallel_search=False,
            parallel_workers=1,
        )

    cube_cfg = CubeConfig(
        n=max(2, min(4, int(args.cube_n))),
        d=max(1, int(args.cube_d)),
        h=max(4096, min(16384, int(args.cube_h))),
        w=max(4096, min(16384, int(args.cube_w))),
        max_parallel_subcubes=max(1, int(args.max_parallel_subcubes)),
        switching_penalty=DEFAULT_CUBE_CONFIG.switching_penalty,
        z_access_penalty=DEFAULT_CUBE_CONFIG.z_access_penalty,
    )

    comparison = run_pipeline(
        args.model,
        args.trace,
        args.output,
        cube_cfg_override=cube_cfg,
        moe_cfg_override=moe_cfg,
        export_profile=args.profile,
        overlap_transfer_compute=args.overlap_transfer_compute,
        overlap_alpha=args.overlap_alpha,
        overlap_model_mode=args.overlap_model_mode,
        overlap_bw_power_law_alpha=max(1e-6, float(args.overlap_bw_power_law_alpha)),
        overlap_z_depth_penalty=max(0.0, float(args.overlap_z_depth_penalty)),
        load_balance_weight=max(0.0, args.load_balance_weight),
        dispatch_policy=args.dispatch_policy,
        criticality_weight=max(0.0, args.criticality_weight),
        resource_pressure_weight=max(0.0, args.resource_pressure_weight),
        capacity_max_ratio=max(1e-9, float(args.capacity_max_ratio)),
        strict_capacity=bool(args.strict_capacity),
        seed=max(0, int(args.seed)),
    )

    base = comparison["baseline"]
    opt = comparison["optimized"]
    imp = comparison["improvement"]

    print("==== 3D CIM Scheduler Evaluation ====")
    print(f"Baseline latency: {base['latency']:.2f} cycles")
    print(f"Optimized latency: {opt['latency']:.2f} cycles")
    print(f"Latency improvement: {imp['latency'] * 100:.2f}%")
    print(f"Baseline space utilization: {base['space_utilization']:.4f}")
    print(f"Optimized space utilization: {opt['space_utilization']:.4f}")
    print(f"Space utilization improvement: {imp['space_utilization'] * 100:.2f}%")
    print(f"Temporal utilization improvement: {imp['temporal_utilization'] * 100:.2f}%")
    print(f"Optimized conflict score: {opt['conflict_score']:.2f}")
    print(f"Optimized memory utilization: {opt['memory_utilization']:.4f}")
    print(f"Optimized avg bandwidth utilization: {opt['avg_bandwidth_utilization']:.4f}")
    print(f"Seed: {int(args.seed)}")
    print(f"Deterministic mode: {bool(args.deterministic)}")
    print("\nAblation latency (cycles):")
    for key in ["baseline", "ablation_no_moe", "ablation_no_replication", "optimized"]:
        print(f"  - {key}: {comparison['ablation'][key]['metrics']['latency']:.2f}")


if __name__ == "__main__":
    main()
