from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from config.cube_config import CubeConfig, DEFAULT_CUBE_CONFIG
from config.moe_config import DEFAULT_MOE_CONFIG
from main import run_pipeline
from scripts.generate_synthetic_model import build_synthetic_moe_model
from scripts.validate_submission import validate_output_dir
from src.model_parser import parse_model
from src.utils import load_json, save_json


ROOT = Path(__file__).resolve().parents[1]


def _fast_moe_cfg():
    return replace(
        DEFAULT_MOE_CONFIG,
        enable_parallel_search=False,
        parallel_workers=1,
        parallel_trials=1,
        local_search_restarts=1,
        local_search_max_iters=5,
        enable_simulated_annealing=False,
    )


def test_strict_capacity_rejects_default_oversized_cube(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="capacity exceeds"):
        run_pipeline(
            model_path=ROOT / "data" / "sample_model.json",
            trace_path=ROOT / "data" / "sample_trace.json",
            output_dir=tmp_path / "oversized",
            moe_cfg_override=_fast_moe_cfg(),
            export_profile=False,
            strict_capacity=True,
            capacity_max_ratio=2.0,
        )


def test_solution_aliases_and_validator_pass_under_strict_capacity(tmp_path: Path) -> None:
    output_dir = tmp_path / "strict_ok"
    cube_cfg = replace(DEFAULT_CUBE_CONFIG, d=2)
    run_pipeline(
        model_path=ROOT / "data" / "sample_model.json",
        trace_path=ROOT / "data" / "sample_trace.json",
        output_dir=output_dir,
        cube_cfg_override=cube_cfg,
        moe_cfg_override=_fast_moe_cfg(),
        export_profile=False,
        strict_capacity=True,
        capacity_max_ratio=2.0,
        seed=2026,
    )

    solution = load_json(output_dir / "solution.json")
    assert "weight_cubes" in solution
    assert "schedule" in solution
    assert solution["weight_cubes"] == solution["spatial_mapping"]["weight_cubes"]
    assert solution["schedule"] == solution["static_schedule"]
    assert "replicas" in solution["weight_cubes"][0]

    report = validate_output_dir(output_dir, strict_capacity=True, capacity_max_ratio=2.0)
    assert report["ok"], report


def test_validator_detects_out_of_bounds_mapping(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad_solution"
    cube_cfg = replace(DEFAULT_CUBE_CONFIG, d=2)
    run_pipeline(
        model_path=ROOT / "data" / "sample_model.json",
        trace_path=ROOT / "data" / "sample_trace.json",
        output_dir=output_dir,
        cube_cfg_override=cube_cfg,
        moe_cfg_override=_fast_moe_cfg(),
        export_profile=False,
        strict_capacity=True,
        capacity_max_ratio=2.0,
    )

    solution = load_json(output_dir / "solution.json")
    solution["weight_cubes"][0]["coords"][0] = solution["hardware"]["subcube_size"][1] + 1
    solution["spatial_mapping"]["weight_cubes"] = solution["weight_cubes"]
    save_json(output_dir / "solution.json", solution)

    report = validate_output_dir(output_dir, strict_capacity=True, capacity_max_ratio=2.0)
    assert not report["ok"]
    assert any("bounds" in err for err in report["errors"])


def test_synthetic_model_can_be_parsed(tmp_path: Path) -> None:
    model_path = tmp_path / "synthetic_model.json"
    model = build_synthetic_moe_model(
        num_experts=32,
        hidden_size=4096,
        expert_width=1024,
        router_width=256,
        dtype="int8",
    )
    save_json(model_path, model)
    parsed = parse_model(model_path, CubeConfig(n=4, d=2, h=4096, w=4096))
    assert parsed.model_name == "deepseek_moe_synthetic_e32"
    assert len(parsed.weight_cubes) >= 32
    assert "moe_merge" in parsed.operators
