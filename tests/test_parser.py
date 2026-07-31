from pathlib import Path

import pytest

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


def test_parse_model_preserves_operator_metadata(tmp_path: Path) -> None:
    model_path = tmp_path / "metadata_model.json"
    model_path.write_text(
        """
        {
          "model_name": "metadata_unit",
          "dtype": "int8",
          "operators": [
            {
              "id": "linear0",
              "type": "linear",
              "shape": [64, 32],
              "deps": [],
              "bias_shape": [32],
              "inputs": ["x"],
              "outputs": ["y"],
              "metadata": {"source": "unit"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    parsed = parse_model(model_path, DEFAULT_CUBE_CONFIG)
    op = parsed.operators["linear0"]

    assert op["metadata"]["source"] == "unit"
    assert op["metadata"]["bias_shape"] == [32]
    assert op["metadata"]["inputs"] == ["x"]
    assert op["metadata"]["outputs"] == ["y"]
    assert "metadata" not in op["attrs"]
    assert "bias_shape" not in op["attrs"]
    assert "inputs" not in op["attrs"]
    assert "outputs" not in op["attrs"]


def test_parse_onnx_preserves_operator_metadata(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto

    weight = helper.make_tensor(
        name="weight0",
        data_type=tensor_proto.FLOAT,
        dims=[32, 64],
        vals=[0.0] * (32 * 64),
    )
    node = helper.make_node("MatMul", inputs=["x", "weight0"], outputs=["y"], name="matmul0")
    graph = helper.make_graph(
        [node],
        "metadata_graph",
        [helper.make_tensor_value_info("x", tensor_proto.FLOAT, [1, 32])],
        [helper.make_tensor_value_info("y", tensor_proto.FLOAT, [1, 64])],
        [weight],
    )
    model = helper.make_model(graph)
    model_path = tmp_path / "metadata_model.onnx"
    onnx.save(model, str(model_path))

    parsed = parse_model(model_path, DEFAULT_CUBE_CONFIG)
    op = parsed.operators["matmul_0"]

    assert op["metadata"]["inputs"] == ["x", "weight0"]
    assert op["metadata"]["outputs"] == ["y"]
    assert op["metadata"]["onnx_op_type"] == "MatMul"
    assert op["metadata"]["initializer_name"] == "weight0"
    assert op["weight_cubes"]


def test_parse_activation_trace_accepts_records_format(tmp_path: Path) -> None:
    trace_path = tmp_path / "records_trace.json"
    trace_path.write_text(
        """
        {
          "num_experts": 4,
          "records": [
            {"inference_id": 0, "active_experts": [0, 2]},
            {"inference_id": 1, "experts": [1, 3]}
          ]
        }
        """,
        encoding="utf-8",
    )

    parsed = parse_activation_trace(trace_path)

    assert parsed["traces"] == [[0, 2], [1, 3]]
    assert parsed["n_inferences"] == 2
    assert parsed["trace_format"]["source_field"] == "records"


def test_parse_activation_trace_accepts_activations_format(tmp_path: Path) -> None:
    trace_path = tmp_path / "activations_trace.json"
    trace_path.write_text(
        """
        {
          "num_experts": 3,
          "activations": [
            {"step": 0, "expert_ids": [2]},
            {"step": 1, "expert_ids": [1]}
          ]
        }
        """,
        encoding="utf-8",
    )

    parsed = parse_activation_trace(trace_path)

    assert parsed["traces"] == [[2], [1]]
    assert parsed["n_inferences"] == 2
    assert parsed["trace_format"]["source_field"] == "activations"
