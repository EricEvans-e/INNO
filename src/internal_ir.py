from __future__ import annotations

from typing import Any, Dict

import numpy as np


SCHEMA_VERSION = "cim_3d_scheduler.ir.v1"

PIPELINE_STAGES = [
    "model_parse",
    "activation_trace_parse",
    "weight_cube_partition",
    "spatial_mapping",
    "static_schedule",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _activation_trace_summary(activation_trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_experts": int(activation_trace.get("num_experts", 0)),
        "top_k": int(activation_trace.get("top_k", 0)),
        "shared_experts": _json_safe(activation_trace.get("shared_experts", [])),
        "n_inferences": int(activation_trace.get("n_inferences", 0)),
        "trace_count": len(activation_trace.get("traces", [])),
        "expert_frequency": _json_safe(activation_trace.get("expert_frequency", {})),
        "expert_burstiness": _json_safe(activation_trace.get("expert_burstiness", {})),
        "expert_transition_influence": _json_safe(activation_trace.get("expert_transition_influence", {})),
    }


def _operator_records(parsed_model: Any) -> list[Dict[str, Any]]:
    records = []
    for op_id in parsed_model.topological_order:
        op = parsed_model.operators[op_id]
        metadata = op.get("metadata", op.get("attrs", {}))
        records.append(
            {
                "id": op.get("id", op_id),
                "type": op.get("type"),
                "deps": _json_safe(op.get("deps", [])),
                "shape": _json_safe(op.get("shape")),
                "expert_id": _json_safe(op.get("expert_id")),
                "weight_cubes": _json_safe(op.get("weight_cubes", [])),
                "metadata": _json_safe(metadata),
            }
        )
    return records


def _weight_cube_records(parsed_model: Any) -> list[Dict[str, Any]]:
    records = []
    for cube_id in sorted(parsed_model.weight_cubes):
        cube = parsed_model.weight_cubes[cube_id]
        cube_dict = _json_safe(cube.to_dict())
        records.append(
            {
                "id": cube_id,
                "cube_id": cube_dict.get("cube_id", cube_id),
                "operator_id": cube_dict["operator_id"],
                "section_id": cube_dict["section_id"],
                "shape": [cube_dict["h"], cube_dict["w"], cube_dict["d"]],
                "elements": cube_dict["elements"],
                "bytes_size": cube_dict["bytes_size"],
                "expert_id": cube_dict.get("expert_id"),
                "is_shared_expert": cube_dict["is_shared_expert"],
                "partitioning": {
                    "h": cube_dict["h"],
                    "w": cube_dict["w"],
                    "d": cube_dict["d"],
                },
            }
        )
    return records


def _spatial_mapping_summary(mapping: Any) -> Dict[str, Any]:
    metrics = getattr(mapping, "metrics", {})
    return {
        "mapping_rate": _json_safe(metrics.get("mapping_rate")),
        "space_utilization": _json_safe(metrics.get("space_utilization")),
        "parameter_density": _json_safe(metrics.get("parameter_density")),
        "placement_count": len(getattr(mapping, "placements", [])),
        "unplaced_count": len(getattr(mapping, "unplaced_cubes", [])),
    }


def _static_schedule_summary(simulation: Any) -> Dict[str, Any]:
    metrics = getattr(simulation, "metrics", {})
    metadata = getattr(simulation, "metadata", {})
    constraints = metadata.get("constraints", {}) if isinstance(metadata, dict) else {}
    schedule = getattr(simulation, "schedule", [])
    return {
        "latency": _json_safe(metrics.get("latency")),
        "task_count": len(schedule),
        "constraint_valid": _json_safe(constraints.get("valid")),
    }


def build_internal_ir(
    parsed_model: Any,
    activation_trace: Dict[str, Any],
    cube_cfg: Any,
    mapping: Any = None,
    simulation: Any = None,
) -> Dict[str, Any]:
    internal_ir = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_STAGES,
        "model": {
            "name": parsed_model.model_name,
            "dtype": parsed_model.dtype,
            "operator_count": len(parsed_model.operators),
            "weight_cube_count": len(parsed_model.weight_cubes),
            "topological_order": list(parsed_model.topological_order),
        },
        "hardware": {
            "n": int(cube_cfg.n),
            "d": int(cube_cfg.d),
            "h": int(cube_cfg.h),
            "w": int(cube_cfg.w),
            "subcube_count": int(cube_cfg.num_subcubes),
            "subcube_volume": int(cube_cfg.subcube_volume),
            "total_volume": int(cube_cfg.total_volume),
            "max_parallel_subcubes": int(cube_cfg.max_parallel_subcubes),
            "switching_penalty": int(cube_cfg.switching_penalty),
            "z_access_penalty": int(cube_cfg.z_access_penalty),
            "inter_subcube_transfer_penalty": int(cube_cfg.inter_subcube_transfer_penalty),
        },
        "activation_trace": _activation_trace_summary(activation_trace),
        "operators": _operator_records(parsed_model),
        "weight_cubes": _weight_cube_records(parsed_model),
    }
    if mapping is not None:
        internal_ir["spatial_mapping"] = _spatial_mapping_summary(mapping)
    if simulation is not None:
        internal_ir["static_schedule"] = _static_schedule_summary(simulation)
    return internal_ir
