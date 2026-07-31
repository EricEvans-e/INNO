from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.cube_config import CubeConfig
from src.mapping_solver import MappingResult
from src.model_parser import ParsedModel
from src.utils import save_json


@dataclass
class Task:
    task_id: str
    inference_id: int
    operator_id: str
    logical_cube_id: Optional[str]
    deps: List[str]
    ready_time: int = 0


@dataclass
class SimulationResult:
    metrics: Dict[str, float]
    schedule: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "schedule": self.schedule,
            "metadata": self.metadata,
        }


def _active_expert_set(trace: Dict[str, Any], inference_id: int) -> set:
    traces = trace.get("traces", [])
    shared = set(trace.get("shared_experts", []))
    if not traces:
        return shared
    return set(traces[inference_id % len(traces)]) | shared


def _build_tasks(parsed_model: ParsedModel, activation_trace: Dict[str, Any]) -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}
    n_inferences = int(activation_trace.get("n_inferences", 1))

    for inf_id in range(n_inferences):
        active_experts = _active_expert_set(activation_trace, inf_id)
        op_to_tasks: Dict[str, List[str]] = {}

        for op_id in parsed_model.topological_order:
            op = parsed_model.operators[op_id]
            if op["type"] == "moe_expert" and op.get("expert_id") not in active_experts:
                continue

            task_ids: List[str] = []
            cubes = op.get("weight_cubes", [])
            if cubes:
                for cube in cubes:
                    task_id = f"i{inf_id}:{op_id}:{cube}"
                    task_ids.append(task_id)
                    tasks[task_id] = Task(
                        task_id=task_id,
                        inference_id=inf_id,
                        operator_id=op_id,
                        logical_cube_id=cube,
                        deps=[],
                    )
            else:
                task_id = f"i{inf_id}:{op_id}:ctrl"
                task_ids.append(task_id)
                tasks[task_id] = Task(
                    task_id=task_id,
                    inference_id=inf_id,
                    operator_id=op_id,
                    logical_cube_id=None,
                    deps=[],
                )

            op_to_tasks[op_id] = task_ids

            dep_task_ids: List[str] = []
            for dep in op.get("deps", []):
                dep_task_ids.extend(op_to_tasks.get(dep, []))

            for task_id in task_ids:
                tasks[task_id].deps = dep_task_ids.copy()

    return tasks


def _dtype_bytes(dtype: str) -> int:
    return {"int8": 1, "i8": 1, "int16": 2, "i16": 2, "int32": 4, "i32": 4}.get(dtype.lower(), 1)


def _hardware_vector(raw: Dict[str, Any], key: str, default: float, length: int) -> List[float]:
    values = raw.get(key, [])
    if not isinstance(values, list) or len(values) != length:
        return [default for _ in range(length)]
    normalized: List[float] = []
    for v in values:
        try:
            normalized.append(float(v))
        except Exception:
            normalized.append(default)
    return normalized


def _build_operator_criticality(parsed_model: ParsedModel) -> Dict[str, int]:
    """Longest-path depth from each operator to sink in DAG (>=1)."""
    criticality: Dict[str, int] = {}
    graph = parsed_model.dependency_graph

    for op_id in reversed(parsed_model.topological_order):
        succ = list(graph.successors(op_id))
        if not succ:
            criticality[op_id] = 1
        else:
            criticality[op_id] = 1 + max(criticality[s] for s in succ)
    return criticality


def _window_peak_concurrency(
    intervals: List[Tuple[int, int]],
    window_start: int,
    window_end: int,
) -> int:
    events: List[Tuple[int, int]] = []
    for start, end in intervals:
        if end <= window_start or start >= window_end:
            continue
        events.append((max(start, window_start), 1))
        events.append((min(end, window_end), -1))

    if not events:
        return 0

    # Process interval-end (-1) before interval-start (+1) at the same timestamp.
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def _apply_parallel_limit(
    start: int,
    duration: int,
    intervals: List[Tuple[int, int]],
    max_parallel_subcubes: int,
) -> Tuple[int, int]:
    if max_parallel_subcubes <= 0 or duration <= 0 or not intervals:
        return start, 0

    cur_start = int(start)
    guard = 0
    while guard < 100000:
        guard += 1
        cur_end = cur_start + duration
        peak = _window_peak_concurrency(intervals, cur_start, cur_end)
        if peak < max_parallel_subcubes:
            return cur_start, max(0, cur_start - start)

        overlapping_ends = [
            end
            for begin, end in intervals
            if not (end <= cur_start or begin >= cur_end) and end > cur_start
        ]
        if not overlapping_ends:
            return cur_start, max(0, cur_start - start)
        cur_start = min(overlapping_ends)

    return cur_start, max(0, cur_start - start)


def _validate_subcube_exclusivity(schedule: List[Dict[str, Any]]) -> int:
    by_subcube: Dict[int, List[Tuple[int, int]]] = {}
    for rec in schedule:
        sc = int(rec.get("subcube", -1))
        if sc < 0:
            continue
        by_subcube.setdefault(sc, []).append((int(rec["start"]), int(rec["end"])))

    violations = 0
    for entries in by_subcube.values():
        entries.sort(key=lambda item: (item[0], item[1]))
        prev_end = -1
        for start, end in entries:
            if start < prev_end:
                violations += 1
            prev_end = max(prev_end, end)
    return violations


def _validate_dependency_constraints(tasks: Dict[str, Task], schedule: List[Dict[str, Any]]) -> int:
    rec_by_id = {str(rec["task_id"]): rec for rec in schedule}
    violations = 0
    for task_id, task in tasks.items():
        rec = rec_by_id.get(task_id)
        if rec is None:
            continue
        start = int(rec["start"])
        dep_end_max = 0
        missing_dep = False
        for dep in task.deps:
            dep_rec = rec_by_id.get(dep)
            if dep_rec is None:
                missing_dep = True
                continue
            dep_end_max = max(dep_end_max, int(dep_rec["end"]))
        if missing_dep or start < dep_end_max:
            violations += 1
    return violations


def _validate_parallel_limit(schedule: List[Dict[str, Any]], max_parallel_subcubes: int) -> Tuple[int, int]:
    if max_parallel_subcubes <= 0:
        return 0, 0

    events: List[Tuple[int, int]] = []
    for rec in schedule:
        sc = int(rec.get("subcube", -1))
        if sc < 0:
            continue
        events.append((int(rec["start"]), 1))
        events.append((int(rec["end"]), -1))

    if not events:
        return 0, 0

    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    violations = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
        if active > max_parallel_subcubes:
            violations += 1
    return peak, violations


def _estimate_bandwidth_pressure(
    intervals: List[Tuple[int, int]],
    start: int,
    end: int,
    max_parallel_subcubes: int,
) -> float:
    if end <= start:
        return 0.0
    peak = _window_peak_concurrency(intervals, start, end)
    if max_parallel_subcubes <= 1:
        return 0.0
    return float(max(0.0, min(1.0, (float(peak) - 1.0) / float(max_parallel_subcubes - 1))))


def _compute_overlap_gain(
    compute_cycles: int,
    transfer_cycles: int,
    overlap_alpha: float,
    z: int,
    overlap_model_mode: str,
    bandwidth_pressure: float,
    overlap_bw_power_law_alpha: float,
    overlap_z_depth_penalty: float,
) -> Tuple[int, float]:
    base = int(min(compute_cycles, transfer_cycles))
    if base <= 0:
        return 0, 1.0

    ratio = max(0.0, min(1.0, float(overlap_alpha)))
    if overlap_model_mode == "linear":
        return int(np.ceil(ratio * float(base))), 1.0

    # For nonlinear model, higher pressure and deeper z reduce useful overlap.
    bw_power = max(1e-6, float(overlap_bw_power_law_alpha))
    bw_factor = max(0.0, min(1.0, 1.0 - float(bandwidth_pressure) ** bw_power))
    z_factor = 1.0 / (1.0 + max(0.0, float(overlap_z_depth_penalty)) * float(max(z, 0)))
    effective_ratio = ratio * bw_factor * z_factor
    gain = int(np.ceil(effective_ratio * float(base)))
    return max(0, gain), float(z_factor)


def simulate(
    parsed_model: ParsedModel,
    mapping: MappingResult,
    activation_trace: Dict[str, Any],
    cube_cfg: CubeConfig,
    overlap_transfer_compute: bool = False,
    overlap_alpha: float = 1.0,
    load_balance_weight: float = 0.0,
    dispatch_policy: str = "fifo",
    criticality_weight: float = 0.0,
    resource_pressure_weight: float = 0.0,
    overlap_model_mode: str = "linear",
    overlap_bw_power_law_alpha: float = 0.7,
    overlap_z_depth_penalty: float = 0.1,
) -> SimulationResult:
    """Static list scheduling with Sub-Cube mutual exclusion and switching penalties."""
    cube_cfg.validate()
    if dispatch_policy not in {"fifo", "criticality"}:
        raise ValueError("dispatch_policy must be 'fifo' or 'criticality'")
    if resource_pressure_weight < 0:
        raise ValueError("resource_pressure_weight must be >= 0")
    if overlap_model_mode not in {"linear", "nonlinear_bandwidth_aware"}:
        raise ValueError("overlap_model_mode must be 'linear' or 'nonlinear_bandwidth_aware'")
    if overlap_bw_power_law_alpha <= 0:
        raise ValueError("overlap_bw_power_law_alpha must be > 0")
    if overlap_z_depth_penalty < 0:
        raise ValueError("overlap_z_depth_penalty must be >= 0")

    tasks = _build_tasks(parsed_model, activation_trace)
    op_criticality = _build_operator_criticality(parsed_model)
    if not tasks:
        return SimulationResult(
            metrics={
                "latency": 0.0,
                "space_utilization": float(mapping.metrics.get("space_utilization", 0.0)),
                "parameter_density": float(mapping.metrics.get("parameter_density", 0.0)),
                "mapping_rate": float(mapping.metrics.get("mapping_rate", 0.0)),
                "temporal_utilization": 0.0,
                "pipeline_bubble_cycles": 0.0,
                "pipeline_bubble_ratio": 0.0,
                "switching_penalty_cycles": 0.0,
                "parallel_limit_wait_cycles": 0.0,
                "conflict_score": float(mapping.metrics.get("conflict_score", 0.0)),
                "memory_utilization": 0.0,
                "avg_bandwidth_utilization": 0.0,
                "max_bandwidth_utilization": 0.0,
                "effective_bandwidth_bytes_per_cycle": 0.0,
            },
            schedule=[],
            metadata={
                "constraints": {
                    "subcube_exclusivity_violations": 0,
                    "dependency_violations": 0,
                    "parallel_limit_violations": 0,
                    "peak_active_subcubes": 0,
                    "max_parallel_subcubes": int(cube_cfg.max_parallel_subcubes),
                    "valid": True,
                },
                "profiling": {},
            },
        )

    hardware = activation_trace.get("raw", {}).get("hardware", {})
    n_sc = cube_cfg.num_subcubes
    speed_factors = _hardware_vector(hardware, "subcube_speed_factors", default=1.0, length=n_sc)
    bandwidth_bpc = _hardware_vector(hardware, "subcube_bandwidth_bytes_per_cycle", default=32768.0, length=n_sc)

    succ: Dict[str, List[str]] = {task_id: [] for task_id in tasks}
    dep_left: Dict[str, int] = {task_id: len(task.deps) for task_id, task in tasks.items()}

    for task_id, task in tasks.items():
        for dep in task.deps:
            succ[dep].append(task_id)

    ready_times: Dict[str, int] = {task_id: 0 for task_id in tasks}
    finished_at: Dict[str, int] = {}

    subcube_available = {sc: 0 for sc in range(cube_cfg.num_subcubes)}
    subcube_last_cube: Dict[int, Optional[str]] = {sc: None for sc in range(cube_cfg.num_subcubes)}
    subcube_wait_cycles = {sc: 0 for sc in range(cube_cfg.num_subcubes)}
    subcube_busy_cycles = {sc: 0 for sc in range(cube_cfg.num_subcubes)}
    subcube_task_count = {sc: 0 for sc in range(cube_cfg.num_subcubes)}
    subcube_bytes = {sc: 0.0 for sc in range(cube_cfg.num_subcubes)}
    expert_call_frequency: Dict[int, int] = {}
    subcube_active_intervals: List[Tuple[int, int]] = []

    ready_heap: List[Tuple[Tuple[Any, ...], int, str]] = []
    push_seq = 0
    subcube_ready_pressure = {sc: 0 for sc in range(cube_cfg.num_subcubes)}
    task_anchor_subcube: Dict[str, int] = {}

    def _task_candidate_subcubes(task_obj: Task) -> List[int]:
        if task_obj.logical_cube_id is None:
            return []
        placement_ids = mapping.cube_to_placements.get(task_obj.logical_cube_id, [])
        candidates = {
            int(mapping.placement_by_id[pid].subcube)
            for pid in placement_ids
            if pid in mapping.placement_by_id
        }
        return sorted(candidates)

    def _estimate_task_resource_pressure(task_obj: Task, ready_t: int) -> Tuple[float, Optional[int]]:
        candidate_subcubes = _task_candidate_subcubes(task_obj)
        if not candidate_subcubes:
            return 0.0, None

        best_score = float("inf")
        best_sc: Optional[int] = None
        for sc in candidate_subcubes:
            availability_delay = max(0.0, float(subcube_available[sc] - ready_t))
            queue_pressure = float(subcube_ready_pressure[sc])
            busy_pressure = float(subcube_busy_cycles[sc]) / 1000.0
            score = availability_delay + 0.5 * queue_pressure + busy_pressure
            if score < best_score:
                best_score = score
                best_sc = sc

        if best_sc is None:
            return 0.0, None
        return float(best_score), int(best_sc)

    def push_ready_task(task_id: str) -> int:
        nonlocal push_seq
        t = tasks[task_id]
        ready_t = int(ready_times[task_id])
        if dispatch_policy == "criticality":
            crit = float(op_criticality.get(t.operator_id, 1))
            pressure, anchor_sc = _estimate_task_resource_pressure(t, ready_t)
            if anchor_sc is not None:
                subcube_ready_pressure[anchor_sc] += 1
                task_anchor_subcube[task_id] = anchor_sc
            key = (
                float(ready_t) + float(resource_pressure_weight) * float(pressure) - float(criticality_weight) * crit,
                ready_t,
                float(pressure),
                -int(crit),
                t.inference_id,
                t.operator_id,
            )
        else:
            key = (ready_t, t.inference_id, t.operator_id)

        heapq.heappush(ready_heap, (key, push_seq, task_id))
        push_seq += 1
        return push_seq

    for task_id, left in dep_left.items():
        if left == 0:
            push_ready_task(task_id)

    schedule: List[Dict[str, Any]] = []

    total_bubble = 0
    total_compute_cycles = 0
    total_switch_cycles = 0
    total_transfer_penalty_cycles = 0
    parallel_limit_wait_cycles = 0
    task_to_subcube: Dict[str, int] = {}

    infer_start: Dict[int, int] = {}
    infer_end: Dict[int, int] = {}

    while ready_heap:
        _, _, task_id = heapq.heappop(ready_heap)
        anchor_sc = task_anchor_subcube.pop(task_id, None)
        if anchor_sc is not None and anchor_sc in subcube_ready_pressure:
            subcube_ready_pressure[anchor_sc] = max(0, int(subcube_ready_pressure[anchor_sc]) - 1)
        task = tasks[task_id]
        earliest_ready = ready_times[task_id]

        if task.logical_cube_id is None:
            start = earliest_ready
            duration = 1
            end = start + duration
            schedule.append(
                {
                    "task_id": task_id,
                    "inference_id": task.inference_id,
                    "operator_id": task.operator_id,
                    "cube_id": None,
                    "physical_cube_id": None,
                    "subcube": -1,
                    "z": -1,
                    "start": start,
                    "end": end,
                    "switch_penalty": 0,
                    "bubble": max(0, start - earliest_ready),
                    "compute_cycles": duration,
                    "transfer_cycles": 0,
                    "bytes_size": 0.0,
                }
            )
            total_compute_cycles += duration
            total_bubble += max(0, start - earliest_ready)
            finished_at[task_id] = end
            task_to_subcube[task_id] = -1
            infer_start[task.inference_id] = min(infer_start.get(task.inference_id, start), start)
            infer_end[task.inference_id] = max(infer_end.get(task.inference_id, end), end)
        else:
            candidate_ids = mapping.cube_to_placements.get(task.logical_cube_id, [])
            if not candidate_ids:
                raise RuntimeError(f"No mapped placement for cube {task.logical_cube_id}")

            best = None
            for pid in candidate_ids:
                p = mapping.placement_by_id[pid]
                sc = p.subcube
                switch_penalty = 0
                if subcube_last_cube[sc] is not None and subcube_last_cube[sc] != pid:
                    switch_penalty = cube_cfg.switching_penalty
                transfer_penalty = 0
                if cube_cfg.inter_subcube_transfer_penalty > 0:
                    for dep_id in task.deps:
                        dep_sc = task_to_subcube.get(dep_id)
                        if dep_sc is not None and dep_sc >= 0 and dep_sc != sc:
                            transfer_penalty = cube_cfg.inter_subcube_transfer_penalty
                            break
                start = max(earliest_ready, subcube_available[sc] + switch_penalty + transfer_penalty)
                logical_bytes = float(parsed_model.weight_cubes[task.logical_cube_id].bytes_size)
                bytes_size = logical_bytes * float(getattr(p, "compression_ratio", 1.0))
                base_compute = 1 + p.d + cube_cfg.z_access_penalty * p.z
                speed = max(float(speed_factors[sc]), 1e-6)
                compute_cycles = int(np.ceil(base_compute / speed))
                transfer_cycles = int(np.ceil(bytes_size / max(float(bandwidth_bpc[sc]), 1.0)))
                overlap_ratio = 0.0
                overlap_gain = 0
                overlap_bw_pressure = 0.0
                overlap_z_factor = 1.0
                raw_duration = max(1, compute_cycles + transfer_cycles)
                if overlap_transfer_compute:
                    overlap_ratio = max(0.0, min(1.0, float(overlap_alpha)))
                if overlap_ratio > 0.0:
                    if overlap_model_mode == "nonlinear_bandwidth_aware":
                        overlap_bw_pressure = _estimate_bandwidth_pressure(
                            subcube_active_intervals,
                            start,
                            start + raw_duration,
                            max(2, cube_cfg.max_parallel_subcubes),
                        )
                    overlap_gain, overlap_z_factor = _compute_overlap_gain(
                        compute_cycles=compute_cycles,
                        transfer_cycles=transfer_cycles,
                        overlap_alpha=overlap_ratio,
                        z=p.z,
                        overlap_model_mode=overlap_model_mode,
                        bandwidth_pressure=overlap_bw_pressure,
                        overlap_bw_power_law_alpha=overlap_bw_power_law_alpha,
                        overlap_z_depth_penalty=overlap_z_depth_penalty,
                    )
                    duration = max(1, raw_duration - overlap_gain)
                else:
                    duration = raw_duration

                parallel_wait = 0
                if cube_cfg.max_parallel_subcubes < cube_cfg.num_subcubes:
                    start, parallel_wait = _apply_parallel_limit(
                        start=start,
                        duration=duration,
                        intervals=subcube_active_intervals,
                        max_parallel_subcubes=cube_cfg.max_parallel_subcubes,
                    )
                end = start + duration
                balance_penalty = float(load_balance_weight) * float(subcube_busy_cycles[sc])
                # Keep comparison key fully orderable; do not include Placement object.
                candidate_key = (
                    end + balance_penalty,
                    end,
                    start,
                    duration,
                    switch_penalty + transfer_penalty,
                    parallel_wait,
                    p.subcube,
                    p.z,
                    pid,
                )
                candidate = (
                    candidate_key,
                    p,
                    compute_cycles,
                    transfer_cycles,
                    bytes_size,
                    parallel_wait,
                    overlap_gain,
                    overlap_bw_pressure,
                    overlap_z_factor,
                    transfer_penalty,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

            assert best is not None
            (
                _candidate_key,
                placement,
                compute_cycles,
                transfer_cycles,
                bytes_size,
                parallel_wait,
                overlap_gain,
                overlap_bw_pressure,
                overlap_z_factor,
                transfer_penalty,
            ) = best
            (
                _,
                end,
                start,
                duration,
                _combined_penalty,
                _,
                _,
                _,
                _,
            ) = _candidate_key
            switch_penalty = max(0, _combined_penalty - transfer_penalty)
            sc = placement.subcube

            subcube_available[sc] = end
            subcube_last_cube[sc] = placement.physical_cube_id
            task_to_subcube[task_id] = placement.subcube
            subcube_active_intervals.append((start, end))

            bubble = max(0, start - earliest_ready)
            total_compute_cycles += duration
            total_switch_cycles += switch_penalty
            total_transfer_penalty_cycles += transfer_penalty
            total_bubble += bubble
            parallel_limit_wait_cycles += parallel_wait
            subcube_wait_cycles[sc] += bubble
            subcube_busy_cycles[sc] += duration
            subcube_task_count[sc] += 1
            subcube_bytes[sc] += bytes_size

            if placement.expert_id is not None:
                expert_call_frequency[placement.expert_id] = expert_call_frequency.get(placement.expert_id, 0) + 1

            schedule.append(
                {
                    "task_id": task_id,
                    "inference_id": task.inference_id,
                    "operator_id": task.operator_id,
                    "cube_id": task.logical_cube_id,
                    "physical_cube_id": placement.physical_cube_id,
                    "subcube": placement.subcube,
                    "z": placement.z,
                    "start": start,
                    "end": end,
                    "switch_penalty": switch_penalty,
                    "transfer_penalty": transfer_penalty,
                    "parallel_limit_wait": parallel_wait,
                    "bubble": bubble,
                    "compute_cycles": compute_cycles,
                    "transfer_cycles": transfer_cycles,
                    "overlap_gain": int(overlap_gain),
                    "overlap_bw_pressure": float(overlap_bw_pressure),
                    "overlap_z_factor": float(overlap_z_factor),
                    "overlap_model_mode": str(overlap_model_mode),
                    "bytes_size": bytes_size,
                    "speed_factor": float(speed_factors[sc]),
                    "bandwidth_bpc": float(bandwidth_bpc[sc]),
                }
            )
            finished_at[task_id] = end
            infer_start[task.inference_id] = min(infer_start.get(task.inference_id, start), start)
            infer_end[task.inference_id] = max(infer_end.get(task.inference_id, end), end)

        done_time = finished_at[task_id]
        for nxt in succ[task_id]:
            dep_left[nxt] -= 1
            ready_times[nxt] = max(ready_times[nxt], done_time)
            if dep_left[nxt] == 0:
                push_ready_task(nxt)

    latency = max(finished_at.values()) if finished_at else 0
    temporal_util = 0.0
    if latency > 0:
        temporal_util = total_compute_cycles / (latency * cube_cfg.num_subcubes)

    bubble_ratio = total_bubble / max((total_bubble + total_compute_cycles), 1)

    dtype_bytes = _dtype_bytes(parsed_model.dtype)
    memory_occupancy = float(
        sum(
            parsed_model.weight_cubes[p.logical_cube_id].bytes_size * float(getattr(p, "compression_ratio", 1.0))
            for p in mapping.placements
        )
    )
    memory_capacity = float(cube_cfg.total_volume * dtype_bytes)
    memory_utilization = memory_occupancy / max(memory_capacity, 1.0)

    total_bytes_processed = float(sum(subcube_bytes.values()))
    bw_util_list = []
    for sc in range(cube_cfg.num_subcubes):
        denom = max(float(latency) * max(float(bandwidth_bpc[sc]), 1.0), 1.0)
        bw_util_list.append(subcube_bytes[sc] / denom)
    avg_bw_util = float(np.mean(bw_util_list)) if bw_util_list else 0.0
    max_bw_util = float(np.max(bw_util_list)) if bw_util_list else 0.0
    effective_bw = total_bytes_processed / max(float(latency), 1.0)
    subcube_busy_arr = np.array([subcube_busy_cycles[sc] for sc in range(cube_cfg.num_subcubes)], dtype=np.float64)
    busy_mean = float(np.mean(subcube_busy_arr)) if len(subcube_busy_arr) > 0 else 0.0
    busy_imbalance = float(np.std(subcube_busy_arr) / max(busy_mean, 1.0)) if busy_mean > 0 else 0.0

    metrics = {
        "latency": float(latency),
        "space_utilization": float(mapping.metrics.get("space_utilization", 0.0)),
        "parameter_density": float(mapping.metrics.get("parameter_density", 0.0)),
        "mapping_rate": float(mapping.metrics.get("mapping_rate", 0.0)),
        "temporal_utilization": float(temporal_util),
        "pipeline_bubble_cycles": float(total_bubble),
        "pipeline_bubble_ratio": float(bubble_ratio),
        "switching_penalty_cycles": float(total_switch_cycles),
        "inter_subcube_transfer_penalty_cycles": float(total_transfer_penalty_cycles),
        "parallel_limit_wait_cycles": float(parallel_limit_wait_cycles),
        "conflict_score": float(mapping.metrics.get("conflict_score", 0.0)),
        "memory_utilization": float(memory_utilization),
        "avg_bandwidth_utilization": float(avg_bw_util),
        "max_bandwidth_utilization": float(max_bw_util),
        "effective_bandwidth_bytes_per_cycle": float(effective_bw),
        "subcube_busy_imbalance": float(busy_imbalance),
    }

    subcube_exclusivity_violations = _validate_subcube_exclusivity(schedule)
    dependency_violations = _validate_dependency_constraints(tasks, schedule)
    peak_active_subcubes, parallel_limit_violations = _validate_parallel_limit(
        schedule,
        cube_cfg.max_parallel_subcubes,
    )

    latency_by_inference = []
    for inf_id in sorted(infer_end.keys()):
        start = infer_start.get(inf_id, 0)
        end = infer_end[inf_id]
        latency_by_inference.append(int(max(0, end - start)))

    subcube_contention = [
        {
            "subcube": sc,
            "task_count": int(subcube_task_count[sc]),
            "wait_cycles": int(subcube_wait_cycles[sc]),
            "busy_cycles": int(subcube_busy_cycles[sc]),
            "bytes_processed": float(subcube_bytes[sc]),
            "bandwidth_utilization": float(bw_util_list[sc]) if sc < len(bw_util_list) else 0.0,
        }
        for sc in range(cube_cfg.num_subcubes)
    ]

    metadata = {
        "n_tasks": len(tasks),
        "n_inferences": int(activation_trace.get("n_inferences", 1)),
        "switching_penalty": cube_cfg.switching_penalty,
        "z_access_penalty": cube_cfg.z_access_penalty,
        "inter_subcube_transfer_penalty": cube_cfg.inter_subcube_transfer_penalty,
        "subcube_count": cube_cfg.num_subcubes,
        "max_parallel_subcubes": cube_cfg.max_parallel_subcubes,
        "constraints": {
            "subcube_exclusivity_violations": int(subcube_exclusivity_violations),
            "dependency_violations": int(dependency_violations),
            "parallel_limit_violations": int(parallel_limit_violations),
            "peak_active_subcubes": int(peak_active_subcubes),
            "max_parallel_subcubes": int(cube_cfg.max_parallel_subcubes),
            "valid": bool(
                subcube_exclusivity_violations == 0
                and dependency_violations == 0
                and parallel_limit_violations == 0
            ),
        },
        "mapping_metadata": mapping.metadata,
        "profiling": {
            "latency_by_inference": latency_by_inference,
            "expert_call_frequency": expert_call_frequency,
            "subcube_contention": subcube_contention,
            "memory_occupancy_bytes": memory_occupancy,
            "memory_capacity_bytes": memory_capacity,
            "total_processed_bytes": total_bytes_processed,
            "hardware": {
                "subcube_speed_factors": speed_factors,
                "subcube_bandwidth_bytes_per_cycle": bandwidth_bpc,
                "overlap_transfer_compute": bool(overlap_transfer_compute),
                "overlap_alpha": float(overlap_alpha),
                "overlap_model_mode": str(overlap_model_mode),
                "overlap_bw_power_law_alpha": float(overlap_bw_power_law_alpha),
                "overlap_z_depth_penalty": float(overlap_z_depth_penalty),
                "load_balance_weight": float(load_balance_weight),
                "dispatch_policy": str(dispatch_policy),
                "criticality_weight": float(criticality_weight),
                "resource_pressure_weight": float(resource_pressure_weight),
                "max_parallel_subcubes": int(cube_cfg.max_parallel_subcubes),
            },
            "operator_criticality": op_criticality,
        },
    }

    return SimulationResult(metrics=metrics, schedule=schedule, metadata=metadata)


def save_simulation(path: Path, sim: SimulationResult) -> None:
    save_json(path, sim.to_dict())
