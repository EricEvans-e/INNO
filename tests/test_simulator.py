from pathlib import Path

from config.cube_config import DEFAULT_CUBE_CONFIG
from config.moe_config import DEFAULT_MOE_CONFIG
from src.mapping_solver import solve_mapping
from src.model_parser import parse_activation_trace, parse_model
from src.simulator import simulate


ROOT = Path(__file__).resolve().parents[1]


def test_simulator_outputs_metrics() -> None:
    model = parse_model(ROOT / "data" / "sample_model.json", DEFAULT_CUBE_CONFIG)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")

    baseline_mapping = solve_mapping(
        model,
        trace,
        DEFAULT_CUBE_CONFIG,
        DEFAULT_MOE_CONFIG,
        packing_policy="first_fit",
        enable_moe_optimization=False,
        enable_adaptive_replication=False,
    )
    optimized_mapping = solve_mapping(
        model,
        trace,
        DEFAULT_CUBE_CONFIG,
        DEFAULT_MOE_CONFIG,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    baseline_sim = simulate(model, baseline_mapping, trace, DEFAULT_CUBE_CONFIG)
    optimized_sim = simulate(model, optimized_mapping, trace, DEFAULT_CUBE_CONFIG)

    assert baseline_sim.metrics["latency"] > 0
    assert optimized_sim.metrics["latency"] > 0
    assert optimized_sim.metrics["space_utilization"] >= baseline_sim.metrics["space_utilization"] * 0.8
    assert optimized_sim.metrics["switching_penalty_cycles"] >= 0
    assert optimized_sim.metrics["pipeline_bubble_cycles"] >= 0
    assert optimized_sim.metrics["memory_utilization"] >= 0
    assert optimized_sim.metrics["avg_bandwidth_utilization"] >= 0
    assert optimized_sim.metrics["subcube_busy_imbalance"] >= 0
    assert optimized_sim.metrics["effective_bandwidth_bytes_per_cycle"] >= 0
    assert optimized_sim.metrics["parallel_limit_wait_cycles"] >= 0
    assert "inter_subcube_transfer_penalty_cycles" in optimized_sim.metrics
    assert optimized_sim.metrics["inter_subcube_transfer_penalty_cycles"] >= 0
    assert "parameter_density" in optimized_sim.metrics
    assert optimized_sim.metrics["parameter_density"] >= 0

    profiling = optimized_sim.metadata.get("profiling", {})
    assert "latency_by_inference" in profiling
    assert "expert_call_frequency" in profiling
    assert "subcube_contention" in profiling

    constraints = optimized_sim.metadata.get("constraints", {})
    assert "parallel_limit_violations" in constraints
    assert "subcube_exclusivity_violations" in constraints
    assert "dependency_violations" in constraints
    assert "valid" in constraints


def test_simulator_inter_subcube_transfer_penalty() -> None:
    from config.cube_config import CubeConfig

    cube_cfg = CubeConfig(inter_subcube_transfer_penalty=5)
    model = parse_model(ROOT / "data" / "sample_model.json", cube_cfg)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")

    mapping = solve_mapping(
        model,
        trace,
        cube_cfg,
        DEFAULT_MOE_CONFIG,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    sim = simulate(model, mapping, trace, cube_cfg)

    assert sim.metrics["latency"] > 0
    assert "inter_subcube_transfer_penalty_cycles" in sim.metrics
    assert sim.metrics["inter_subcube_transfer_penalty_cycles"] >= 0
    assert sim.metadata["inter_subcube_transfer_penalty"] == 5


def test_simulator_criticality_dispatch_mode() -> None:
    model = parse_model(ROOT / "data" / "sample_model.json", DEFAULT_CUBE_CONFIG)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")

    mapping = solve_mapping(
        model,
        trace,
        DEFAULT_CUBE_CONFIG,
        DEFAULT_MOE_CONFIG,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    sim = simulate(
        model,
        mapping,
        trace,
        DEFAULT_CUBE_CONFIG,
        overlap_transfer_compute=True,
        overlap_alpha=0.8,
        overlap_model_mode="nonlinear_bandwidth_aware",
        overlap_bw_power_law_alpha=0.9,
        overlap_z_depth_penalty=0.12,
        load_balance_weight=0.05,
        dispatch_policy="criticality",
        criticality_weight=0.3,
        resource_pressure_weight=0.2,
    )

    assert sim.metrics["latency"] > 0
    assert sim.metrics["subcube_busy_imbalance"] >= 0
    hardware = sim.metadata.get("profiling", {}).get("hardware", {})
    assert hardware.get("dispatch_policy") == "criticality"
    assert float(hardware.get("criticality_weight", 0.0)) >= 0.0
    assert float(hardware.get("resource_pressure_weight", 0.0)) >= 0.0
    assert hardware.get("overlap_model_mode") == "nonlinear_bandwidth_aware"
    assert float(hardware.get("overlap_bw_power_law_alpha", 0.0)) > 0.0
    assert float(hardware.get("overlap_z_depth_penalty", -1.0)) >= 0.0
