from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from config.cube_config import CubeConfig
from config.cube_config import DEFAULT_CUBE_CONFIG
from config.moe_config import DEFAULT_MOE_CONFIG
from src.mapping_solver import solve_mapping
from src.model_parser import parse_activation_trace, parse_model
from src.utils import check_2d_non_overlap


ROOT = Path(__file__).resolve().parents[1]


def test_mapping_validity_and_optimization_signal() -> None:
    model = parse_model(ROOT / "data" / "sample_model.json", DEFAULT_CUBE_CONFIG)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")

    baseline = solve_mapping(
        model,
        trace,
        DEFAULT_CUBE_CONFIG,
        DEFAULT_MOE_CONFIG,
        packing_policy="first_fit",
        enable_moe_optimization=False,
        enable_adaptive_replication=False,
    )
    optimized = solve_mapping(
        model,
        trace,
        DEFAULT_CUBE_CONFIG,
        DEFAULT_MOE_CONFIG,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    assert baseline.metrics["mapping_rate"] > 0.9
    assert optimized.metrics["mapping_rate"] > 0.9

    # Verify no overlap per (subcube, z) in optimized mapping.
    grouped = defaultdict(list)
    for p in optimized.placements:
        grouped[(p.subcube, p.z)].append((p.x, p.y, p.w, p.h))

    for rects in grouped.values():
        assert check_2d_non_overlap(rects)

    # MoE-aware grouping should not increase conflict drastically.
    assert optimized.metrics["conflict_score"] <= baseline.metrics["conflict_score"] * 1.2

    assert "group_local_search" in optimized.metadata
    assert "exact_hot_subgraph" in optimized.metadata
    assert optimized.metadata["group_local_search"].get("enabled", False)
    assert "compression" in optimized.metadata
    assert optimized.metrics["compression_ratio"] > 0
    assert optimized.metadata["group_local_search"]["parallel"]["trials"] >= 1
    assert "placement_dynamic_weights" in optimized.metadata
    assert "packing_dynamic_weights" in optimized.metadata
    assert "effective_packing_policy" in optimized.metadata
    assert optimized.metadata["packing_dynamic_weights"]["aspect_aware_weight"] >= 0
    assert optimized.metadata["packing_dynamic_weights"]["conflict_propagation_weight"] >= 0
    assert "grouping_strategy" in optimized.metadata
    assert optimized.metadata["grouping_strategy"]["best_group_count"] >= 1
    assert "hot_subgraph_topk" in optimized.metadata
    assert "replication_pressure_thresholds" in optimized.metadata
    assert (
        optimized.metadata["replication_pressure_thresholds"]["low"]
        <= optimized.metadata["replication_pressure_thresholds"]["high"]
    )
    assert "constraints" in optimized.metadata
    assert optimized.metadata["constraints"]["weight_stationary"]


def test_shared_operator_replication_uses_extra_budget() -> None:
    cube_cfg = CubeConfig(n=3, d=4, h=4096, w=4096)
    model = parse_model(ROOT / "data" / "sample_model.json", cube_cfg)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")
    moe_cfg = replace(
        DEFAULT_MOE_CONFIG,
        enable_shared_operator_replication=True,
        shared_replication_operator_types=("linear",),
        shared_replication_min_volume=4096 * 4096,
        shared_replication_max_replicas=2,
        replication_volume_budget_ratio=0.5,
    )

    mapping = solve_mapping(
        model,
        trace,
        cube_cfg,
        moe_cfg,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    replicated = {
        cube_id: placements
        for cube_id, placements in mapping.cube_to_placements.items()
        if cube_id.startswith(("embed_proj", "shared_attn", "shared_mlp", "lm_head"))
        and len(placements) > 1
    }
    assert replicated
    assert mapping.metadata["shared_operator_replication"]["enabled"] is True
    assert mapping.metadata["shared_operator_replication"]["replicated_logical_cubes"] >= 1
    assert mapping.metadata["extra_replica_used"] <= mapping.metadata["extra_replica_budget"]


def test_shared_operator_replication_can_be_disabled() -> None:
    cube_cfg = CubeConfig(n=3, d=4, h=4096, w=4096)
    model = parse_model(ROOT / "data" / "sample_model.json", cube_cfg)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")
    moe_cfg = replace(
        DEFAULT_MOE_CONFIG,
        enable_shared_operator_replication=False,
        shared_replication_operator_types=("linear",),
        shared_replication_min_volume=4096 * 4096,
        shared_replication_max_replicas=2,
        replication_volume_budget_ratio=0.5,
    )

    mapping = solve_mapping(
        model,
        trace,
        cube_cfg,
        moe_cfg,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    assert len(mapping.cube_to_placements["embed_proj__s0"]) == 1
    assert mapping.metadata["shared_operator_replication"]["enabled"] is False
    assert mapping.metadata["shared_operator_replication"]["replicated_logical_cubes"] == 0


def test_shared_operator_replication_reports_requests_when_budget_denies_replicas() -> None:
    cube_cfg = CubeConfig(n=3, d=4, h=4096, w=4096)
    model = parse_model(ROOT / "data" / "sample_model.json", cube_cfg)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")
    moe_cfg = replace(
        DEFAULT_MOE_CONFIG,
        enable_shared_operator_replication=True,
        shared_replication_operator_types=("linear",),
        shared_replication_min_volume=4096 * 4096,
        shared_replication_max_replicas=2,
        replication_volume_budget_ratio=0.0,
    )

    mapping = solve_mapping(
        model,
        trace,
        cube_cfg,
        moe_cfg,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    shared_meta = mapping.metadata["shared_operator_replication"]
    assert shared_meta["requested_logical_cubes"] >= 1
    assert shared_meta["extra_physical_replicas"] == 0
    assert shared_meta["replicated_logical_cubes"] == 0
