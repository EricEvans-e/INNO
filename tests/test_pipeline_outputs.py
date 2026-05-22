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
