from pathlib import Path

from config.cube_config import DEFAULT_CUBE_CONFIG
from src.model_parser import parse_activation_trace, parse_model


ROOT = Path(__file__).resolve().parents[1]


def test_parse_model_and_trace() -> None:
    model = parse_model(ROOT / "data" / "sample_model.json", DEFAULT_CUBE_CONFIG)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")

    assert model.model_name == "deepseek_moe_toy"
    assert "lm_head" in model.operators
    # lm_head=[4096, 8192] should be split into 2 sections under 4096x4096 HxW limit.
    assert len(model.operators["lm_head"]["weight_cubes"]) == 2
    assert len(model.weight_cubes) > 0

    assert trace["num_experts"] == 16
    assert trace["cooccurrence_matrix"].shape == (16, 16)
    assert trace["expert_frequency"][0] == trace["n_inferences"]
