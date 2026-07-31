from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from config.cube_config import CubeConfig
from src.utils import (
    build_cooccurrence_matrix,
    build_transition_matrix,
    compute_expert_burstiness,
    compute_expert_frequency,
    compute_transition_influence,
    load_json,
)

try:
    import onnx
except Exception:  # pragma: no cover - optional dependency
    onnx = None


@dataclass
class WeightCube:
    cube_id: str
    operator_id: str
    section_id: int
    h: int
    w: int
    d: int
    elements: int
    bytes_size: int
    expert_id: Optional[int] = None
    is_shared_expert: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedModel:
    model_name: str
    dtype: str
    operators: Dict[str, Dict[str, Any]]
    topological_order: List[str]
    dependency_graph: nx.DiGraph
    weight_cubes: Dict[str, WeightCube]
    raw_model: Dict[str, Any]


def _dtype_size(dtype: str) -> int:
    return {"int8": 1, "i8": 1, "int16": 2, "i16": 2, "int32": 4, "i32": 4}.get(dtype.lower(), 1)


def _split_matrix(rows: int, cols: int, max_h: int, max_w: int) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    for r0 in range(0, rows, max_h):
        r = min(max_h, rows - r0)
        for c0 in range(0, cols, max_w):
            c = min(max_w, cols - c0)
            chunks.append((r, c))
    return chunks


def _build_dependency_graph(operators: List[Dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for op in operators:
        graph.add_node(op["id"])
    for op in operators:
        for dep in op.get("deps", []):
            graph.add_edge(dep, op["id"])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Model dependency graph is not a DAG")
    return graph


def _build_operator_record(op: Dict[str, Any]) -> Dict[str, Any]:
    promoted_metadata_keys = {"bias_shape", "inputs", "outputs", "onnx_op_type", "initializer_name"}
    structural_keys = {"id", "type", "shape", "deps", "expert_id", "parallel_group", "metadata"}
    attrs_excluded_keys = structural_keys | promoted_metadata_keys
    metadata = dict(op.get("metadata", {}))
    for key in promoted_metadata_keys:
        if key in op:
            metadata[key] = op[key]

    return {
        "id": op["id"],
        "type": op["type"],
        "shape": op.get("shape"),
        "deps": list(op.get("deps", [])),
        "expert_id": op.get("expert_id"),
        "parallel_group": op.get("parallel_group"),
        "weight_cubes": [],
        "metadata": metadata,
        "attrs": {k: v for k, v in op.items() if k not in attrs_excluded_keys},
    }


def _normalize_trace_rows(trace_json: Dict[str, Any]) -> Tuple[List[List[int]], Dict[str, Any]]:
    row_expert_fields = ("active_experts", "expert_ids", "experts", "topk_experts")

    if "traces" in trace_json:
        source_field = "traces"
        rows = trace_json["traces"]
        traces = [[int(expert_id) for expert_id in row] for row in rows]
    elif "records" in trace_json:
        source_field = "records"
        traces = _normalize_record_trace_rows(trace_json[source_field], source_field, row_expert_fields)
    elif "activations" in trace_json:
        source_field = "activations"
        traces = _normalize_record_trace_rows(trace_json[source_field], source_field, row_expert_fields)
    else:
        raise ValueError("Activation trace must contain 'traces', 'records', or 'activations'")

    return traces, {"source_field": source_field}


def _normalize_record_trace_rows(
    records: List[Dict[str, Any]],
    source_field: str,
    row_expert_fields: Tuple[str, ...],
) -> List[List[int]]:
    traces: List[List[int]] = []
    for record in records:
        expert_ids = None
        for expert_field in row_expert_fields:
            if expert_field in record:
                expert_ids = record[expert_field]
                break
        if expert_ids is None:
            raise ValueError(f"Activation trace '{source_field}' record lacks expert ids: {record}")
        traces.append([int(expert_id) for expert_id in expert_ids])
    return traces


def parse_model(model_path: Path, cube_config: CubeConfig) -> ParsedModel:
    """Parse model.json or ONNX and produce dependency graph + split weight cubes."""
    cube_config.validate()

    if model_path.suffix.lower() == ".onnx":
        model_json = _parse_onnx(model_path)
    else:
        model_json = load_json(model_path)

    operators_raw = model_json.get("operators", [])
    if not operators_raw:
        raise ValueError("Model file has no operators")

    dependency_graph = _build_dependency_graph(operators_raw)
    topo = list(nx.topological_sort(dependency_graph))

    dtype = model_json.get("dtype", "int8")
    dtype_size = _dtype_size(dtype)

    operators: Dict[str, Dict[str, Any]] = {}
    weight_cubes: Dict[str, WeightCube] = {}

    for op in operators_raw:
        record = _build_operator_record(op)
        op_shape = op.get("shape")
        if op_shape and op["type"] in {"linear", "moe_expert", "matmul", "gemm", "router"}:
            if len(op_shape) != 2:
                raise ValueError(f"Only 2D weight matrices are supported, got {op_shape} for {op['id']}")
            rows, cols = int(op_shape[0]), int(op_shape[1])
            chunks = _split_matrix(rows, cols, cube_config.h, cube_config.w)
            for section_id, (h, w) in enumerate(chunks):
                cube_id = f"{op['id']}__s{section_id}"
                weight_cube = WeightCube(
                    cube_id=cube_id,
                    operator_id=op["id"],
                    section_id=section_id,
                    h=h,
                    w=w,
                    d=1,
                    elements=h * w,
                    bytes_size=h * w * dtype_size,
                    expert_id=op.get("expert_id"),
                    is_shared_expert=(op.get("expert_id") == 0),
                )
                weight_cubes[cube_id] = weight_cube
                record["weight_cubes"].append(cube_id)

        operators[op["id"]] = record

    return ParsedModel(
        model_name=model_json.get("model_name", model_path.stem),
        dtype=dtype,
        operators=operators,
        topological_order=topo,
        dependency_graph=dependency_graph,
        weight_cubes=weight_cubes,
        raw_model=model_json,
    )


def parse_activation_trace(trace_path: Path) -> Dict[str, Any]:
    trace_json = load_json(trace_path)
    num_experts = int(trace_json["num_experts"])
    traces, trace_format = _normalize_trace_rows(trace_json)

    matrix = build_cooccurrence_matrix(traces, num_experts)
    transition_matrix = build_transition_matrix(traces, num_experts)
    freq = compute_expert_frequency(traces, num_experts)
    burstiness = compute_expert_burstiness(traces, num_experts)
    transition_influence = compute_transition_influence(transition_matrix)

    return {
        "num_experts": num_experts,
        "top_k": int(trace_json.get("top_k", 1)),
        "shared_experts": list(trace_json.get("shared_experts", [])),
        "n_inferences": int(trace_json.get("n_inferences", len(traces))),
        "traces": traces,
        "trace_format": trace_format,
        "cooccurrence_matrix": matrix,
        "transition_matrix": transition_matrix,
        "expert_frequency": freq,
        "expert_burstiness": burstiness,
        "expert_transition_influence": transition_influence,
        "raw": trace_json,
    }


def _parse_onnx(model_path: Path) -> Dict[str, Any]:
    if onnx is None:
        raise ImportError("onnx is not installed, cannot parse ONNX model")

    model = onnx.load(str(model_path))
    graph = model.graph

    initializer_shapes = {init.name: tuple(init.dims) for init in graph.initializer}
    operators: List[Dict[str, Any]] = []
    produced_by: Dict[str, str] = {}

    for idx, node in enumerate(graph.node):
        op_type = node.op_type.lower()
        node_id = f"{op_type}_{idx}"
        deps = [produced_by[x] for x in node.input if x in produced_by]

        shape = None
        matched_initializer_name = None
        for inp in node.input:
            if inp in initializer_shapes and len(initializer_shapes[inp]) >= 2:
                dims = initializer_shapes[inp]
                shape = [int(dims[-2]), int(dims[-1])]
                matched_initializer_name = inp
                break

        mapped_type = "elementwise"
        if op_type in {"matmul", "gemm"}:
            mapped_type = "linear"
        elif op_type in {"add", "sub", "mul", "div"}:
            mapped_type = "elementwise"

        operators.append(
            {
                "id": node_id,
                "type": mapped_type,
                "shape": shape,
                "deps": deps,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "onnx_op_type": node.op_type,
                "initializer_name": matched_initializer_name,
            }
        )

        for out_name in node.output:
            produced_by[out_name] = node_id

    return {
        "model_name": model_path.stem,
        "dtype": "int8",
        "operators": operators,
    }
