from pathlib import Path

from main import run_pipeline
from src.utils import load_json


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_exports_solution_and_manifest(tmp_path: Path) -> None:
	output_dir = tmp_path / "pipeline_outputs"
	comparison = run_pipeline(
		model_path=ROOT / "data" / "sample_model.json",
		trace_path=ROOT / "data" / "sample_trace.json",
		output_dir=output_dir,
		export_profile=False,
		seed=2026,
	)

	assert "optimized" in comparison
	assert (output_dir / "solution.json").exists()
	assert (output_dir / "optimized_solution.json").exists()
	assert (output_dir / "run_manifest.json").exists()

	solution = load_json(output_dir / "solution.json")
	assert "hardware" in solution
	assert "spatial_mapping" in solution
	assert "static_schedule" in solution
	assert "metrics" in solution

	manifest = load_json(output_dir / "run_manifest.json")
	assert manifest["configs"]["seed"] == 2026
	assert "model_sha256" in manifest["inputs"]
	assert "trace_sha256" in manifest["inputs"]


def test_pipeline_exports_clear_ablation_variants(tmp_path: Path) -> None:
	output_dir = tmp_path / "ablation_outputs"
	comparison = run_pipeline(
		model_path=ROOT / "data" / "sample_model.json",
		trace_path=ROOT / "data" / "sample_trace.json",
		output_dir=output_dir,
		export_profile=False,
		seed=2026,
	)

	assert "ablation_best_fit_replication_only" in comparison["ablation"]
	assert "ablation_moe_without_replication" in comparison["ablation"]
	assert "ablation_no_moe" not in comparison["ablation"]
	assert (output_dir / "ablation_best_fit_replication_only_mapping.json").exists()
	assert (output_dir / "ablation_moe_without_replication_mapping.json").exists()
	assert (output_dir / "ablation_best_fit_replication_only_solution.json").exists()
	assert (output_dir / "ablation_moe_without_replication_solution.json").exists()
	mapping = load_json(output_dir / "optimized_mapping.json")
	assert "shared_operator_replication" in mapping["metadata"]


def test_pipeline_exports_lightweight_internal_ir(tmp_path: Path) -> None:
	output_dir = tmp_path / "internal_ir_outputs"
	result = run_pipeline(
		model_path=ROOT / "data" / "sample_model.json",
		trace_path=ROOT / "data" / "sample_trace.json",
		output_dir=output_dir,
		export_profile=False,
		export_internal_ir=True,
		seed=2026,
	)

	ir_path = output_dir / "internal_ir.json"
	assert ir_path.exists()
	assert (output_dir / "run_manifest.json").exists()

	internal_ir = load_json(ir_path)
	assert internal_ir["schema_version"] == "cim_3d_scheduler.ir.v1"
	assert internal_ir["pipeline"] == [
		"model_parse",
		"activation_trace_parse",
		"weight_cube_partition",
		"spatial_mapping",
		"static_schedule",
	]
	assert internal_ir["hardware"]["subcube_count"] == 9
	assert internal_ir["operators"]
	assert isinstance(internal_ir["operators"], list)
	assert [op["id"] for op in internal_ir["operators"]] == internal_ir["model"]["topological_order"]
	assert internal_ir["weight_cubes"]
	assert isinstance(internal_ir["weight_cubes"], list)
	cube_ids = [cube["id"] for cube in internal_ir["weight_cubes"]]
	assert cube_ids == sorted(cube_ids)
	assert "activation_trace" in internal_ir
	assert internal_ir["spatial_mapping"]["placement_count"] > 0
	assert "mapping_rate" in internal_ir["spatial_mapping"]
	assert "space_utilization" in internal_ir["spatial_mapping"]
	assert "parameter_density" in internal_ir["spatial_mapping"]
	assert internal_ir["static_schedule"]["task_count"] > 0
	assert internal_ir["static_schedule"]["latency"] > 0

	manifest = load_json(output_dir / "run_manifest.json")
	assert manifest["artifacts"]["internal_ir"] == "internal_ir.json"
	assert result["artifacts"]["internal_ir"] == str(ir_path.resolve())
