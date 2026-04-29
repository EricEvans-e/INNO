from collections import defaultdict
from pathlib import Path

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
