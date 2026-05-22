from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.cube_config import CubeConfig
from config.moe_config import MoEConfig
from src.model_parser import ParsedModel, WeightCube
from src.utils import save_json


@dataclass
class Placement:
    logical_cube_id: str
    physical_cube_id: str
    operator_id: str
    expert_id: Optional[int]
    replica_id: int
    subcube: int
    z: int
    x: int
    y: int
    h: int
    w: int
    d: int
    strategy: str
    compression_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingResult:
    placements: List[Placement]
    cube_to_placements: Dict[str, List[str]]
    placement_by_id: Dict[str, Placement]
    unplaced_cubes: List[str]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": [p.to_dict() for p in self.placements],
            "cube_to_placements": self.cube_to_placements,
            "unplaced_cubes": self.unplaced_cubes,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class ShelfPacker:
    """Simple deterministic 2D packer used in each z-layer of a Sub-Cube."""

    def __init__(
        self,
        width: int,
        height: int,
        aspect_aware_weight: float = 0.45,
        fragmentation_penalty_weight: float = 0.20,
    ) -> None:
        self.width = width
        self.height = height
        self.rows: List[Dict[str, int]] = []
        self.used_height = 0
        self.aspect_aware_weight = max(0.0, float(aspect_aware_weight))
        self.fragmentation_penalty_weight = max(0.0, float(fragmentation_penalty_weight))

    def _options(self, w: int, h: int) -> List[Dict[str, float]]:
        options: List[Dict[str, float]] = []
        target_aspect = float(w) / max(float(h), 1.0)

        for row_idx, row in enumerate(self.rows):
            if h <= row["height"] and row["x_cursor"] + w <= self.width:
                x = row["x_cursor"]
                y = row["y"]
                remain_w = self.width - (x + w)
                remain_h = row["height"] - h
                waste = (self.width - (x + w)) + (row["height"] - h)
                slot_aspect = float(self.width - x) / max(float(row["height"]), 1.0)
                options.append(
                    {
                        "row_idx": float(row_idx),
                        "x": float(x),
                        "y": float(y),
                        "waste": float(waste),
                        "fit_score": float(waste),
                        "new_row": 0.0,
                        "aspect_penalty": abs(slot_aspect - target_aspect),
                        "fragment_penalty": float(remain_w) / max(float(self.width), 1.0)
                        + float(remain_h) / max(float(row["height"]), 1.0),
                    }
                )

        if self.used_height + h <= self.height:
            x = 0
            y = self.used_height
            remain_w = self.width - w
            remain_h = self.height - (y + h)
            waste = (self.width - w) + (self.height - (y + h))
            slot_aspect = float(self.width) / max(float(h), 1.0)
            options.append(
                {
                    "row_idx": -1.0,
                    "x": float(x),
                    "y": float(y),
                    "waste": float(waste),
                    "fit_score": float(waste),
                    "new_row": 1.0,
                    "aspect_penalty": abs(slot_aspect - target_aspect),
                    "fragment_penalty": float(remain_w) / max(float(self.width), 1.0)
                    + float(remain_h) / max(float(self.height), 1.0),
                }
            )

        for option in options:
            option["fit_score"] = (
                float(option["waste"])
                + self.aspect_aware_weight * float(option["aspect_penalty"])
                + self.fragmentation_penalty_weight * float(option["fragment_penalty"])
            )

        return options

    def try_place(self, w: int, h: int, policy: str) -> Optional[Tuple[int, int, float]]:
        options = self._options(w, h)
        if not options:
            return None

        if policy == "first_fit":
            option = options[0]
        elif policy == "aspect_aware":
            option = min(options, key=lambda item: (item["fit_score"], item["y"], item["x"]))
        else:
            option = min(options, key=lambda item: (item["waste"], item["y"], item["x"]))

        x, y = int(option["x"]), int(option["y"])
        if option["new_row"] == 1:
            self.rows.append({"y": y, "height": h, "x_cursor": w})
            self.used_height += h
        else:
            row = self.rows[int(option["row_idx"])]
            row["x_cursor"] += w

        return x, y, float(option["fit_score"])


class SubCubeAllocator:
    def __init__(
        self,
        num_subcubes: int,
        d: int,
        h: int,
        w: int,
        aspect_aware_weight: float = 0.45,
        fragmentation_penalty_weight: float = 0.20,
    ) -> None:
        self.num_subcubes = num_subcubes
        self.d = d
        self.h = h
        self.w = w
        self.aspect_aware_weight = max(0.0, float(aspect_aware_weight))
        self.fragmentation_penalty_weight = max(0.0, float(fragmentation_penalty_weight))
        self.layers: Dict[int, List[ShelfPacker]] = {
            sc: [
                ShelfPacker(
                    w,
                    h,
                    aspect_aware_weight=self.aspect_aware_weight,
                    fragmentation_penalty_weight=self.fragmentation_penalty_weight,
                )
                for _ in range(d)
            ]
            for sc in range(num_subcubes)
        }

    def try_place(
        self,
        cube: WeightCube,
        candidate_subcubes: Sequence[int],
        preferred_z: Optional[int],
        policy: str,
        subcube_bias: Optional[Dict[int, float]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        if cube.d != 1:
            raise ValueError("Current allocator supports d=1 cubes only")

        z_order = list(range(self.d))
        if preferred_z is not None and 0 <= preferred_z < self.d:
            z_order.remove(preferred_z)
            z_order.insert(0, preferred_z)

        candidates: List[Tuple[int, int, int, int, int]] = []

        if policy == "first_fit":
            for sc in candidate_subcubes:
                for z in z_order:
                    result = self.layers[sc][z].try_place(cube.w, cube.h, policy="first_fit")
                    if result is not None:
                        x, y, _ = result
                        return sc, z, x, y
            return None

        for sc in candidate_subcubes:
            for z in z_order:
                shadow = ShelfPacker(
                    self.w,
                    self.h,
                    aspect_aware_weight=self.aspect_aware_weight,
                    fragmentation_penalty_weight=self.fragmentation_penalty_weight,
                )
                shadow.rows = [dict(row) for row in self.layers[sc][z].rows]
                shadow.used_height = self.layers[sc][z].used_height
                result = shadow.try_place(cube.w, cube.h, policy=policy)
                if result is not None:
                    x, y, fit_score = result
                    bias = float(subcube_bias.get(sc, 0.0)) if subcube_bias else 0.0
                    score = fit_score + z * 4 + sc + bias
                    candidates.append((score, sc, z, x, y))

        if not candidates:
            return None

        _, sc, z, x, y = min(candidates, key=lambda item: item[0])
        commit = self.layers[sc][z].try_place(cube.w, cube.h, policy=policy)
        if commit is None:
            return None
        return sc, z, commit[0], commit[1]


def build_mutually_exclusive_groups(co_matrix: np.ndarray) -> List[List[int]]:
    """Greedy grouping: experts with pairwise zero co-occurrence share one group."""
    num_experts = int(co_matrix.shape[0])
    return _build_mutually_exclusive_groups_with_order(co_matrix, list(range(num_experts)))


def _build_mutually_exclusive_groups_with_order(
    co_matrix: np.ndarray,
    expert_order: Sequence[int],
) -> List[List[int]]:
    groups: List[List[int]] = []

    for expert in expert_order:
        placed = False
        for group in groups:
            if all(co_matrix[expert, g] == 0 for g in group):
                group.append(expert)
                placed = True
                break
        if not placed:
            groups.append([expert])

    groups = [sorted(group) for group in groups]
    groups.sort(key=lambda g: (g[0], len(g)))
    return groups


def _group_load_imbalance(groups: Sequence[Sequence[int]], freq: Dict[int, int]) -> float:
    if not groups:
        return 0.0
    loads = np.array([float(sum(freq.get(e, 0) for e in group)) for group in groups], dtype=np.float64)
    mean = float(np.mean(loads)) if len(loads) > 0 else 0.0
    if mean <= 0.0:
        return float(np.std(loads))
    return float(np.std(loads) / mean)


def _build_mutually_exclusive_groups_multistart(
    co_matrix: np.ndarray,
    freq: Dict[int, int],
    trials: int,
    seed_base: int,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    num_experts = int(co_matrix.shape[0])
    if num_experts <= 0:
        return [], {"enabled": False, "trials": 0}

    row_sum = co_matrix.sum(axis=1) if co_matrix.size > 0 else np.zeros((num_experts,), dtype=np.float64)
    experts = list(range(num_experts))
    orders: List[List[int]] = [
        experts,
        sorted(experts, key=lambda e: (freq.get(e, 0), row_sum[e], -e), reverse=True),
        sorted(experts, key=lambda e: (row_sum[e], freq.get(e, 0), -e), reverse=True),
        sorted(
            experts,
            key=lambda e: (float(freq.get(e, 0)) + float(row_sum[e]), freq.get(e, 0), -e),
            reverse=True,
        ),
    ]

    dedup_orders: List[List[int]] = []
    seen = set()
    for order in orders:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        dedup_orders.append(order)

    rng = np.random.default_rng(seed_base + 50000)
    target_trials = max(1, int(trials))
    while len(dedup_orders) < target_trials:
        order = list(experts)
        rng.shuffle(order)
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        dedup_orders.append(order)

    best_groups = _build_mutually_exclusive_groups_with_order(co_matrix, dedup_orders[0])
    best_score = (
        len(best_groups),
        _group_load_imbalance(best_groups, freq),
        -max((len(g) for g in best_groups), default=0),
    )

    for order in dedup_orders[1:]:
        groups = _build_mutually_exclusive_groups_with_order(co_matrix, order)
        score = (
            len(groups),
            _group_load_imbalance(groups, freq),
            -max((len(g) for g in groups), default=0),
        )
        if score < best_score:
            best_groups = groups
            best_score = score

    return best_groups, {
        "enabled": True,
        "trials": int(len(dedup_orders)),
        "best_group_count": int(len(best_groups)),
        "best_score": {
            "group_count": int(best_score[0]),
            "load_imbalance": float(best_score[1]),
            "largest_group_bonus": float(best_score[2]),
        },
    }


def _gini_coefficient(values: Sequence[float]) -> float:
    arr = np.array([max(0.0, float(v)) for v in values], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    arr = np.sort(arr)
    n = int(arr.size)
    cumsum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cumsum) / total) / n)


def _resolve_hot_subgraph_top_k(
    group_order: Sequence[int],
    group_demand: Dict[int, float],
    moe_cfg: MoEConfig,
) -> Tuple[int, Dict[str, Any]]:
    configured = max(0, int(moe_cfg.hot_subgraph_top_k))
    if configured <= 0 or not group_order:
        return 0, {
            "enabled": False,
            "configured_top_k": int(configured),
            "selected_top_k": 0,
        }

    if not moe_cfg.enable_dynamic_hot_subgraph_top_k:
        top_k = min(configured, len(group_order))
        return top_k, {
            "enabled": False,
            "configured_top_k": int(configured),
            "selected_top_k": int(top_k),
        }

    values = [float(group_demand.get(g, 0.0)) for g in group_order]
    gini = _gini_coefficient(values)
    ratio = float(moe_cfg.dynamic_hot_subgraph_min_ratio) + (
        float(moe_cfg.dynamic_hot_subgraph_max_ratio) - float(moe_cfg.dynamic_hot_subgraph_min_ratio)
    ) * (1.0 - gini)
    dynamic_k = max(1, int(round(len(group_order) * ratio)))
    top_k = min(configured, dynamic_k, len(group_order))
    return top_k, {
        "enabled": True,
        "configured_top_k": int(configured),
        "dynamic_candidate_top_k": int(dynamic_k),
        "selected_top_k": int(top_k),
        "group_demand_gini": float(gini),
        "ratio": float(ratio),
    }


def _select_hot_experts(
    freq: Dict[int, int],
    shared_experts: Sequence[int],
    quantile: float,
) -> set:
    if not freq:
        return set(shared_experts)

    values = np.array(list(freq.values()))
    threshold = np.quantile(values, 1.0 - quantile) if len(values) > 1 else values[0]
    hot = {expert for expert, count in freq.items() if count >= threshold and count > 0}
    hot.update(shared_experts)
    return hot


def _replica_count(
    cube: WeightCube,
    freq: Dict[int, int],
    max_freq: int,
    contention: Dict[int, float],
    max_contention: float,
    burstiness: Dict[int, float],
    max_burstiness: float,
    transition_influence: Dict[int, float],
    max_transition: float,
    moe_cfg: MoEConfig,
    enabled: bool,
) -> int:
    if not enabled:
        return 1

    area = cube.h * cube.w
    if area > moe_cfg.replication_area_limit:
        return 1

    if cube.expert_id is None:
        return 1

    if cube.is_shared_expert:
        return min(2, moe_cfg.max_replication)

    if max_freq <= 0:
        return 1

    freq_ratio = freq.get(cube.expert_id, 0) / max(max_freq, 1)
    cont_ratio = contention.get(cube.expert_id, 0.0) / max(max_contention, 1.0)
    burst_ratio = burstiness.get(cube.expert_id, 0.0) / max(max_burstiness, 1e-9)
    trans_ratio = transition_influence.get(cube.expert_id, 0.0) / max(max_transition, 1.0)

    pressure = (
        moe_cfg.replication_freq_weight * freq_ratio
        + moe_cfg.replication_contention_weight * cont_ratio
        + moe_cfg.replication_burst_weight * burst_ratio
        + moe_cfg.replication_transition_weight * trans_ratio
    )

    if pressure >= moe_cfg.replica_pressure_high_threshold:
        return min(3, moe_cfg.max_replication)
    if pressure >= moe_cfg.replica_pressure_low_threshold:
        return min(2, moe_cfg.max_replication)
    return 1


def _shared_replica_count(
    cube: WeightCube,
    parsed_model: ParsedModel,
    moe_cfg: MoEConfig,
    enabled: bool,
) -> int:
    if not enabled or not moe_cfg.enable_shared_operator_replication:
        return 1
    if cube.expert_id is not None:
        return 1
    op = parsed_model.operators.get(cube.operator_id, {})
    if op.get("type") not in set(moe_cfg.shared_replication_operator_types):
        return 1
    volume = cube.h * cube.w * cube.d
    if volume < moe_cfg.shared_replication_min_volume:
        return 1
    return min(int(moe_cfg.shared_replication_max_replicas), int(moe_cfg.max_replication))


def _compression_ratio(cube: WeightCube, hot_experts: set, moe_cfg: MoEConfig) -> float:
    quant_ratio = 1.0
    if moe_cfg.enable_blockwise_quantization:
        quant_bits = moe_cfg.hot_quant_bits if cube.expert_id in hot_experts else moe_cfg.cold_quant_bits
        quant_ratio = float(quant_bits) / 8.0

    sparse_ratio = 1.0
    if moe_cfg.enable_sparse_compression:
        sparsity = moe_cfg.hot_sparsity_ratio if cube.expert_id in hot_experts else moe_cfg.cold_sparsity_ratio
        sparse_ratio = max(0.0, 1.0 - float(sparsity))

    ratio = max(0.15, min(1.0, quant_ratio * sparse_ratio))
    return ratio


def _estimate_conflict_score(
    placements: Sequence[Placement],
    co_matrix: np.ndarray,
) -> float:
    expert_to_subcubes: Dict[int, set] = {}
    for p in placements:
        if p.expert_id is None:
            continue
        expert_to_subcubes.setdefault(p.expert_id, set()).add(p.subcube)

    experts = sorted(expert_to_subcubes.keys())
    if not experts:
        return 0.0

    conflict = 0.0
    for i, e1 in enumerate(experts):
        for e2 in experts[i + 1 :]:
            if co_matrix[e1, e2] <= 0:
                continue
            shared = expert_to_subcubes[e1] & expert_to_subcubes[e2]
            if shared:
                conflict += float(co_matrix[e1, e2]) / len(shared)
    return conflict


def _build_group_pair_cooccurrence(groups: Sequence[Sequence[int]], co_matrix: np.ndarray) -> np.ndarray:
    n_groups = len(groups)
    pair = np.zeros((n_groups, n_groups), dtype=np.float64)
    for i in range(n_groups):
        for j in range(i, n_groups):
            value = 0.0
            for e1 in groups[i]:
                for e2 in groups[j]:
                    value += float(co_matrix[e1, e2])
            pair[i, j] = value
            pair[j, i] = value
    return pair


def _assignment_from_order(order: Sequence[int], num_subcubes: int) -> Dict[int, int]:
    return {group_id: idx % num_subcubes for idx, group_id in enumerate(order)}


def _score_group_assignment(
    assignment: Dict[int, int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
) -> float:
    groups = sorted(assignment.keys())

    conflict = 0.0
    for i, g1 in enumerate(groups):
        sc1 = assignment[g1]
        d1 = float(group_demand.get(g1, 0.0))
        for g2 in groups[i + 1 :]:
            if sc1 == assignment[g2]:
                d2 = float(group_demand.get(g2, 0.0))
                denom = max(d1 + d2, 1e-9)
                imbalance_ratio = abs(d1 - d2) / denom
                propagation = 1.0 + float(moe_cfg.conflict_propagation_weight) * imbalance_ratio
                conflict += float(pair_co[g1, g2]) * propagation

    loads = np.zeros(num_subcubes, dtype=np.float64)
    for g, sc in assignment.items():
        loads[sc] += group_demand.get(g, 0.0)
    mean_load = float(np.mean(loads)) if len(loads) > 0 else 0.0
    if mean_load <= 0:
        imbalance = 0.0
        peak = 0.0
    else:
        imbalance = float(np.std(loads) / mean_load)
        peak = float(np.max(loads) / mean_load)

    return (
        moe_cfg.conflict_weight * conflict
        + moe_cfg.capacity_weight * (imbalance + float(moe_cfg.capacity_peak_weight) * peak)
    )


def _build_seed_order(
    group_order: Sequence[int],
    fixed_assignment: Dict[int, int],
    group_demand: Dict[int, float],
    num_subcubes: int,
) -> List[int]:
    buckets: Dict[int, List[int]] = {sc: [] for sc in range(num_subcubes)}
    assigned = set(fixed_assignment.keys())

    for g, sc in fixed_assignment.items():
        buckets[sc].append(g)

    for g in group_order:
        if g in assigned:
            continue
        target_sc = min(
            range(num_subcubes),
            key=lambda sc: (
                sum(group_demand.get(x, 0.0) for x in buckets[sc]),
                len(buckets[sc]),
                sc,
            ),
        )
        buckets[target_sc].append(g)

    seed_order: List[int] = []
    max_len = max((len(v) for v in buckets.values()), default=0)
    for row in range(max_len):
        for sc in range(num_subcubes):
            if row < len(buckets[sc]):
                seed_order.append(buckets[sc][row])
    return seed_order


def _build_initial_group_assignment(
    group_order: Sequence[int],
    fixed_assignment: Dict[int, int],
    group_demand: Dict[int, float],
    num_subcubes: int,
) -> Dict[int, int]:
    assignment: Dict[int, int] = dict(fixed_assignment)
    loads = np.zeros(num_subcubes, dtype=np.float64)
    for g, sc in assignment.items():
        loads[sc] += float(group_demand.get(g, 0.0))

    for g in group_order:
        if g in assignment:
            continue
        sc = int(np.argmin(loads))
        assignment[g] = sc
        loads[sc] += float(group_demand.get(g, 0.0))
    return assignment


def _single_seed_assignment_search(
    seed_assignment: Dict[int, int],
    fixed_assignment: Dict[int, int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
) -> Tuple[Dict[int, int], float, int]:
    current = dict(seed_assignment)
    fixed_groups = set(fixed_assignment.keys())
    movable_groups = [g for g in sorted(current.keys()) if g not in fixed_groups]
    current_score = _score_group_assignment(current, pair_co, group_demand, num_subcubes, moe_cfg)
    move_counter = 0

    if len(movable_groups) == 0:
        return current, current_score, move_counter

    for _ in range(moe_cfg.local_search_max_iters):
        best_assignment = dict(current)
        best_score = current_score
        improved = False

        # Move: relocate one group to another subcube.
        for g in movable_groups:
            sc_now = current[g]
            for sc in range(num_subcubes):
                if sc == sc_now:
                    continue
                cand = dict(current)
                cand[g] = sc
                cand_score = _score_group_assignment(cand, pair_co, group_demand, num_subcubes, moe_cfg)
                if cand_score + 1e-9 < best_score:
                    best_score = cand_score
                    best_assignment = cand
                    improved = True

        # Swap: exchange subcube assignments of two groups.
        n = len(movable_groups)
        for i in range(n - 1):
            g1 = movable_groups[i]
            for j in range(i + 1, n):
                g2 = movable_groups[j]
                if current[g1] == current[g2]:
                    continue
                cand = dict(current)
                cand[g1], cand[g2] = cand[g2], cand[g1]
                cand_score = _score_group_assignment(cand, pair_co, group_demand, num_subcubes, moe_cfg)
                if cand_score + 1e-9 < best_score:
                    best_score = cand_score
                    best_assignment = cand
                    improved = True

        if not improved:
            break

        current = best_assignment
        current_score = best_score
        move_counter += 1

    return current, current_score, move_counter


def _simulated_annealing_assignment(
    seed_assignment: Dict[int, int],
    fixed_assignment: Dict[int, int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
    rng: np.random.Generator,
) -> Tuple[Dict[int, int], float, int]:
    current = dict(seed_assignment)
    fixed_groups = set(fixed_assignment.keys())
    movable_groups = [g for g in sorted(current.keys()) if g not in fixed_groups]
    current_score = _score_group_assignment(current, pair_co, group_demand, num_subcubes, moe_cfg)
    best = dict(current)
    best_score = current_score
    accepted_worse = 0
    temp = moe_cfg.sa_init_temp

    if len(movable_groups) <= 0:
        return best, best_score, accepted_worse

    for _ in range(moe_cfg.sa_steps):
        cand = dict(current)
        if len(movable_groups) >= 2 and float(rng.random()) < 0.5:
            i, j = rng.integers(0, len(movable_groups), size=2)
            if i != j:
                g1 = movable_groups[int(i)]
                g2 = movable_groups[int(j)]
                cand[g1], cand[g2] = cand[g2], cand[g1]
        else:
            g = movable_groups[int(rng.integers(0, len(movable_groups)))]
            sc_now = cand[g]
            sc = int(rng.integers(0, num_subcubes))
            if sc == sc_now:
                sc = (sc + 1) % num_subcubes
            cand[g] = sc

        cand_score = _score_group_assignment(cand, pair_co, group_demand, num_subcubes, moe_cfg)
        delta = cand_score - current_score

        if delta <= 0 or float(rng.random()) < math.exp(-delta / max(temp, 1e-6)):
            if delta > 0:
                accepted_worse += 1
            current = cand
            current_score = cand_score
            if current_score + 1e-9 < best_score:
                best = dict(current)
                best_score = current_score

        temp *= moe_cfg.sa_cooling

    return best, best_score, accepted_worse


def _build_trial_assignment_seed(
    base_assignment: Dict[int, int],
    fixed_assignment: Dict[int, int],
    num_subcubes: int,
    trial_id: int,
    seed_base: int,
) -> Dict[int, int]:
    rng = np.random.default_rng(seed_base + trial_id)
    seed = dict(base_assignment)
    movable = [g for g in sorted(seed.keys()) if g not in set(fixed_assignment.keys())]
    if trial_id == 0 or not movable:
        return seed

    perturb = min(8, max(1, len(movable) // 3))
    for _ in range(perturb):
        g = movable[int(rng.integers(0, len(movable)))]
        sc_now = seed[g]
        sc_new = int(rng.integers(0, num_subcubes))
        if sc_new == sc_now:
            sc_new = (sc_new + 1) % num_subcubes
        seed[g] = sc_new
    return seed


def _search_single_assignment_seed(
    seed_assignment: Dict[int, int],
    fixed_assignment: Dict[int, int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
    trial_id: int,
    seed_base: int,
) -> Tuple[Dict[int, int], float, Dict[str, Any]]:
    local_assignment, local_score, moves = _single_seed_assignment_search(
        seed_assignment,
        fixed_assignment,
        pair_co,
        group_demand,
        num_subcubes,
        moe_cfg,
    )

    sa_meta = {"enabled": False, "accepted_worse": 0}
    final_assignment = local_assignment
    final_score = local_score
    if moe_cfg.enable_simulated_annealing:
        rng = np.random.default_rng(seed_base + 30000 + trial_id)
        sa_assignment, sa_score, accepted_worse = _simulated_annealing_assignment(
            local_assignment,
            fixed_assignment,
            pair_co,
            group_demand,
            num_subcubes,
            moe_cfg,
            rng,
        )
        sa_meta = {"enabled": True, "accepted_worse": accepted_worse}
        if sa_score + 1e-9 < final_score:
            final_assignment = sa_assignment
            final_score = sa_score

    return final_assignment, final_score, {
        "trial_id": trial_id,
        "local_moves": moves,
        "simulated_annealing": sa_meta,
        "score": final_score,
    }


def _optimize_group_assignment(
    base_assignment: Dict[int, int],
    fixed_assignment: Dict[int, int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    if not base_assignment:
        return {}, {"enabled": False, "moves": 0, "best_score": 0.0, "restarts": 0}

    total_trials = max(moe_cfg.local_search_restarts, moe_cfg.parallel_trials)
    seeds = [
        _build_trial_assignment_seed(
            base_assignment=base_assignment,
            fixed_assignment=fixed_assignment,
            num_subcubes=num_subcubes,
            trial_id=i,
            seed_base=moe_cfg.random_seed,
        )
        for i in range(total_trials)
    ]

    results: List[Tuple[Dict[int, int], float, Dict[str, Any]]] = []
    if moe_cfg.enable_parallel_search and total_trials > 1:
        workers = min(moe_cfg.parallel_workers, total_trials)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _search_single_assignment_seed,
                    seeds[i],
                    fixed_assignment,
                    pair_co,
                    group_demand,
                    num_subcubes,
                    moe_cfg,
                    i,
                    moe_cfg.random_seed,
                )
                for i in range(total_trials)
            ]
            for future in futures:
                results.append(future.result())
    else:
        for i in range(total_trials):
            results.append(
                _search_single_assignment_seed(
                    seeds[i],
                    fixed_assignment,
                    pair_co,
                    group_demand,
                    num_subcubes,
                    moe_cfg,
                    i,
                    moe_cfg.random_seed,
                )
            )

    best_assignment, best_score, _ = min(results, key=lambda item: item[1])
    move_counter = int(sum(item[2].get("local_moves", 0) for item in results))
    sa_accepts = int(sum(item[2].get("simulated_annealing", {}).get("accepted_worse", 0) for item in results))

    meta = {
        "enabled": True,
        "strategy": "direct_assignment_search",
        "moves": move_counter,
        "best_score": float(best_score),
        "restarts": moe_cfg.local_search_restarts,
        "max_iters": moe_cfg.local_search_max_iters,
        "parallel": {
            "enabled": bool(moe_cfg.enable_parallel_search and total_trials > 1),
            "workers": int(min(moe_cfg.parallel_workers, total_trials)),
            "trials": int(total_trials),
        },
        "simulated_annealing": {
            "enabled": bool(moe_cfg.enable_simulated_annealing),
            "steps": int(moe_cfg.sa_steps),
            "accepted_worse_total": sa_accepts,
        },
    }
    return best_assignment, meta


def _solve_hot_subgraph_exact(
    top_groups: Sequence[int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
) -> Tuple[Dict[int, int], float]:
    if not top_groups:
        return {}, 0.0

    best_score = float("inf")
    best_assignment: Dict[int, int] = {}
    current: Dict[int, int] = {}

    def dfs(idx: int) -> None:
        nonlocal best_score, best_assignment

        if idx == len(top_groups):
            score = _score_group_assignment(current, pair_co, group_demand, num_subcubes, moe_cfg)
            if score < best_score:
                best_score = score
                best_assignment = dict(current)
            return

        g = top_groups[idx]
        for sc in range(num_subcubes):
            current[g] = sc
            # Lightweight branch-and-bound on conflict lower bound.
            partial_conflict = 0.0
            for i in range(idx + 1):
                gi = top_groups[i]
                for j in range(i + 1, idx + 1):
                    gj = top_groups[j]
                    if current.get(gi) == current.get(gj):
                        partial_conflict += pair_co[gi, gj]
            if moe_cfg.conflict_weight * partial_conflict <= best_score:
                dfs(idx + 1)
            current.pop(g, None)

    dfs(0)
    return best_assignment, best_score


def _single_seed_local_search(
    seed_order: Sequence[int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
) -> Tuple[List[int], float, int]:
    current = list(seed_order)
    n = len(current)
    current_score = _score_group_assignment(
        _assignment_from_order(current, num_subcubes),
        pair_co,
        group_demand,
        num_subcubes,
        moe_cfg,
    )
    move_counter = 0

    for _ in range(moe_cfg.local_search_max_iters):
        best_neighbor = list(current)
        best_neighbor_score = current_score
        improved = False

        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = list(current)
                cand[i], cand[j] = cand[j], cand[i]
                cand_score = _score_group_assignment(
                    _assignment_from_order(cand, num_subcubes),
                    pair_co,
                    group_demand,
                    num_subcubes,
                    moe_cfg,
                )
                if cand_score + 1e-9 < best_neighbor_score:
                    best_neighbor = cand
                    best_neighbor_score = cand_score
                    improved = True

        if n >= 4:
            for i in range(n - 2):
                for j in range(i + 2, n):
                    cand = list(current)
                    cand[i : j + 1] = list(reversed(cand[i : j + 1]))
                    cand_score = _score_group_assignment(
                        _assignment_from_order(cand, num_subcubes),
                        pair_co,
                        group_demand,
                        num_subcubes,
                        moe_cfg,
                    )
                    if cand_score + 1e-9 < best_neighbor_score:
                        best_neighbor = cand
                        best_neighbor_score = cand_score
                        improved = True

        if not improved:
            break

        current = best_neighbor
        current_score = best_neighbor_score
        move_counter += 1

    return current, current_score, move_counter


def _simulated_annealing(
    seed_order: Sequence[int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
    rng: np.random.Generator,
) -> Tuple[List[int], float, int]:
    current = list(seed_order)
    n = len(current)
    current_score = _score_group_assignment(
        _assignment_from_order(current, num_subcubes),
        pair_co,
        group_demand,
        num_subcubes,
        moe_cfg,
    )
    best = list(current)
    best_score = current_score
    accepted_worse = 0
    temp = moe_cfg.sa_init_temp

    if n <= 1:
        return best, best_score, accepted_worse

    for _ in range(moe_cfg.sa_steps):
        cand = list(current)
        if n >= 4 and float(rng.random()) < 0.45:
            i = int(rng.integers(0, n - 2))
            j = int(rng.integers(i + 2, n))
            cand[i : j + 1] = list(reversed(cand[i : j + 1]))
        else:
            i, j = rng.integers(0, n, size=2)
            if i != j:
                cand[i], cand[j] = cand[j], cand[i]

        cand_score = _score_group_assignment(
            _assignment_from_order(cand, num_subcubes),
            pair_co,
            group_demand,
            num_subcubes,
            moe_cfg,
        )
        delta = cand_score - current_score

        if delta <= 0 or float(rng.random()) < math.exp(-delta / max(temp, 1e-6)):
            if delta > 0:
                accepted_worse += 1
            current = cand
            current_score = cand_score
            if current_score + 1e-9 < best_score:
                best = list(current)
                best_score = current_score

        temp *= moe_cfg.sa_cooling

    return best, best_score, accepted_worse


def _build_trial_seed(seed_order: Sequence[int], trial_id: int, n: int, seed_base: int) -> List[int]:
    rng = np.random.default_rng(seed_base + trial_id)
    seed = list(seed_order)
    if trial_id == 0:
        return seed

    swap_times = min(6, max(1, n // 3))
    for _ in range(swap_times):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            seed[i], seed[j] = seed[j], seed[i]

    if n >= 4:
        i = int(rng.integers(0, n - 2))
        j = int(rng.integers(i + 2, n))
        seed[i : j + 1] = list(reversed(seed[i : j + 1]))
    return seed


def _search_single_seed(
    seed: Sequence[int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
    trial_id: int,
    seed_base: int,
) -> Tuple[List[int], float, Dict[str, Any]]:
    local_order, local_score, moves = _single_seed_local_search(
        seed,
        pair_co,
        group_demand,
        num_subcubes,
        moe_cfg,
    )

    sa_meta = {"enabled": False, "accepted_worse": 0}
    final_order = local_order
    final_score = local_score
    if moe_cfg.enable_simulated_annealing:
        rng = np.random.default_rng(seed_base + 10000 + trial_id)
        sa_order, sa_score, accepted_worse = _simulated_annealing(
            local_order,
            pair_co,
            group_demand,
            num_subcubes,
            moe_cfg,
            rng,
        )
        sa_meta = {"enabled": True, "accepted_worse": accepted_worse}
        if sa_score + 1e-9 < final_score:
            final_order = sa_order
            final_score = sa_score

    return final_order, final_score, {
        "trial_id": trial_id,
        "local_moves": moves,
        "simulated_annealing": sa_meta,
        "score": final_score,
    }


def _optimize_group_order(
    seed_order: Sequence[int],
    pair_co: np.ndarray,
    group_demand: Dict[int, float],
    num_subcubes: int,
    moe_cfg: MoEConfig,
) -> Tuple[List[int], Dict[str, Any]]:
    if not seed_order:
        return [], {"enabled": False, "moves": 0, "best_score": 0.0, "restarts": 0}

    n = len(seed_order)
    total_trials = max(moe_cfg.local_search_restarts, moe_cfg.parallel_trials)
    seeds = [_build_trial_seed(seed_order, trial_id=i, n=n, seed_base=moe_cfg.random_seed) for i in range(total_trials)]

    results: List[Tuple[List[int], float, Dict[str, Any]]] = []
    if moe_cfg.enable_parallel_search and total_trials > 1:
        workers = min(moe_cfg.parallel_workers, total_trials)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _search_single_seed,
                    seeds[i],
                    pair_co,
                    group_demand,
                    num_subcubes,
                    moe_cfg,
                    i,
                    moe_cfg.random_seed,
                )
                for i in range(total_trials)
            ]
            for future in futures:
                results.append(future.result())
    else:
        for i in range(total_trials):
            results.append(
                _search_single_seed(
                    seeds[i],
                    pair_co,
                    group_demand,
                    num_subcubes,
                    moe_cfg,
                    i,
                    moe_cfg.random_seed,
                )
            )

    best_order, best_score, _ = min(results, key=lambda item: item[1])
    move_counter = int(sum(item[2].get("local_moves", 0) for item in results))
    sa_accepts = int(
        sum(item[2].get("simulated_annealing", {}).get("accepted_worse", 0) for item in results)
    )

    meta = {
        "enabled": True,
        "moves": move_counter,
        "best_score": float(best_score),
        "restarts": moe_cfg.local_search_restarts,
        "max_iters": moe_cfg.local_search_max_iters,
        "parallel": {
            "enabled": bool(moe_cfg.enable_parallel_search and total_trials > 1),
            "workers": int(min(moe_cfg.parallel_workers, total_trials)),
            "trials": int(total_trials),
        },
        "simulated_annealing": {
            "enabled": bool(moe_cfg.enable_simulated_annealing),
            "steps": int(moe_cfg.sa_steps),
            "accepted_worse_total": sa_accepts,
        },
    }
    return best_order, meta


def _estimate_group_conflict(
    assignment: Dict[int, int],
    pair_co: np.ndarray,
) -> float:
    groups = sorted(assignment.keys())
    conflict = 0.0
    for i, g1 in enumerate(groups):
        sc1 = assignment[g1]
        for g2 in groups[i + 1 :]:
            if sc1 == assignment[g2]:
                conflict += float(pair_co[g1, g2])
    return float(conflict)


def solve_mapping(
    parsed_model: ParsedModel,
    activation_trace: Dict[str, Any],
    cube_cfg: CubeConfig,
    moe_cfg: MoEConfig,
    packing_policy: str = "best_fit",
    enable_moe_optimization: bool = True,
    enable_adaptive_replication: bool = True,
) -> MappingResult:
    cube_cfg.validate()
    moe_cfg.validate()

    if packing_policy not in {"first_fit", "best_fit", "aspect_aware"}:
        raise ValueError("packing_policy must be one of 'first_fit'/'best_fit'/'aspect_aware'")

    effective_packing_policy = packing_policy
    if (
        packing_policy == "best_fit"
        and enable_moe_optimization
        and moe_cfg.enable_aspect_aware_packing
    ):
        effective_packing_policy = "aspect_aware"

    freq = activation_trace.get("expert_frequency", {})
    shared_experts = activation_trace.get("shared_experts", [])
    co_matrix = activation_trace.get("cooccurrence_matrix")
    if co_matrix is None:
        num_experts = activation_trace.get("num_experts", 0)
        co_matrix = np.zeros((num_experts, num_experts), dtype=np.int64)

    hot_experts = _select_hot_experts(freq, shared_experts, moe_cfg.hot_expert_quantile)
    if enable_moe_optimization:
        groups, grouping_meta = _build_mutually_exclusive_groups_multistart(
            co_matrix=co_matrix,
            freq=freq,
            trials=moe_cfg.grouping_multi_start_trials,
            seed_base=moe_cfg.random_seed,
        )
    else:
        groups = []
        grouping_meta = {"enabled": False, "trials": 0, "best_group_count": 0}

    cubes = list(parsed_model.weight_cubes.values())
    expert_area: Dict[int, int] = {}
    for cube in cubes:
        if cube.expert_id is None:
            continue
        expert_area[cube.expert_id] = expert_area.get(cube.expert_id, 0) + cube.h * cube.w * cube.d

    group_demand: Dict[int, float] = {
        gi: float(sum((freq.get(e, 0) + 1) * expert_area.get(e, 0) for e in group))
        for gi, group in enumerate(groups)
    }
    group_order = sorted(
        range(len(groups)),
        key=lambda i: (group_demand.get(i, 0.0), sum(freq.get(e, 0) for e in groups[i])),
        reverse=True,
    )

    ls_meta: Dict[str, Any] = {"enabled": False}
    exact_meta: Dict[str, Any] = {"enabled": False}
    hot_subgraph_topk_meta: Dict[str, Any] = {
        "enabled": False,
        "configured_top_k": int(moe_cfg.hot_subgraph_top_k),
        "selected_top_k": 0,
    }
    if enable_moe_optimization and groups:
        fixed_assignment: Dict[int, int] = {}
        pair_co = _build_group_pair_cooccurrence(groups, co_matrix)

        if moe_cfg.enable_exact_hot_subgraph and moe_cfg.hot_subgraph_top_k > 0:
            top_k, hot_subgraph_topk_meta = _resolve_hot_subgraph_top_k(
                group_order=group_order,
                group_demand=group_demand,
                moe_cfg=moe_cfg,
            )
            top_k = min(top_k, len(group_order))
            top_groups = group_order[:top_k]
            fixed_assignment, exact_score = _solve_hot_subgraph_exact(
                top_groups=top_groups,
                pair_co=pair_co,
                group_demand=group_demand,
                num_subcubes=cube_cfg.num_subcubes,
                moe_cfg=moe_cfg,
            )
            exact_meta = {
                "enabled": True,
                "top_k": top_k,
                "score": exact_score,
                "fixed_assignment": fixed_assignment,
                "dynamic_topk": hot_subgraph_topk_meta,
            }

        base_assignment = _build_initial_group_assignment(
            group_order=group_order,
            fixed_assignment=fixed_assignment,
            group_demand=group_demand,
            num_subcubes=cube_cfg.num_subcubes,
        )

        if moe_cfg.enable_group_local_search:
            group_to_subcube, ls_meta = _optimize_group_assignment(
                base_assignment=base_assignment,
                fixed_assignment=fixed_assignment,
                pair_co=pair_co,
                group_demand=group_demand,
                num_subcubes=cube_cfg.num_subcubes,
                moe_cfg=moe_cfg,
            )
        else:
            group_to_subcube = base_assignment
            ls_meta = {
                "enabled": False,
                "strategy": "direct_assignment_search",
                "best_score": float(
                    _score_group_assignment(
                        group_to_subcube,
                        pair_co,
                        group_demand,
                        cube_cfg.num_subcubes,
                        moe_cfg,
                    )
                ),
            }
    else:
        group_to_subcube = {}

    expert_to_group = {expert: gi for gi, group in enumerate(groups) for expert in group}

    allocator = SubCubeAllocator(
        cube_cfg.num_subcubes,
        cube_cfg.d,
        cube_cfg.h,
        cube_cfg.w,
        aspect_aware_weight=moe_cfg.aspect_aware_weight,
        fragmentation_penalty_weight=moe_cfg.fragmentation_penalty_weight,
    )

    expert_contention: Dict[int, float] = {}
    if co_matrix.size > 0:
        row_sum = co_matrix.sum(axis=1)
        for expert_id in range(len(row_sum)):
            expert_contention[expert_id] = float(row_sum[expert_id])
    max_contention = max(expert_contention.values()) if expert_contention else 0.0

    transition_matrix = activation_trace.get("transition_matrix")
    if transition_matrix is None:
        transition_matrix = np.zeros_like(co_matrix, dtype=np.float64)
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    expert_transition_mass: Dict[int, float] = {}
    if transition_matrix.size > 0:
        row_sum = transition_matrix.sum(axis=1)
        col_sum = transition_matrix.sum(axis=0)
        for expert_id in range(len(row_sum)):
            expert_transition_mass[expert_id] = float(row_sum[expert_id] + col_sum[expert_id])

    burstiness = activation_trace.get("expert_burstiness", {})
    max_burstiness = max(burstiness.values()) if burstiness else 0.0
    transition_influence = activation_trace.get("expert_transition_influence", {})
    max_transition = max(transition_influence.values()) if transition_influence else 0.0

    cubes.sort(
        key=lambda c: (
            0 if (c.expert_id in hot_experts or c.is_shared_expert) else 1,
            -(c.h * c.w * c.d),
            -freq.get(c.expert_id, 0) if c.expert_id is not None else 0,
        )
    )

    placements: List[Placement] = []
    cube_to_placements: Dict[str, List[str]] = {}
    placement_by_id: Dict[str, Placement] = {}
    unplaced: List[str] = []

    max_freq = max(freq.values()) if freq else 0
    logical_volume = float(sum(c.h * c.w * c.d for c in cubes))
    extra_replica_budget = logical_volume * moe_cfg.replication_volume_budget_ratio
    extra_replica_used = 0.0
    shared_replica_requested = 0
    shared_replica_mapped = 0
    subcube_used_volume: Dict[int, float] = {sc: 0.0 for sc in range(cube_cfg.num_subcubes)}
    subcube_expert_counter: Dict[int, Dict[int, int]] = {
        sc: {} for sc in range(cube_cfg.num_subcubes)
    }
    subcube_group_counter: Dict[int, Dict[int, int]] = {
        sc: {} for sc in range(cube_cfg.num_subcubes)
    }
    subcube_capacity = float(cube_cfg.subcube_volume)

    for cube in cubes:
        cube_comp_ratio = _compression_ratio(cube, hot_experts, moe_cfg)
        requested_replicas = _replica_count(
            cube,
            freq,
            max_freq=max_freq,
            contention=expert_contention,
            max_contention=max_contention,
            burstiness=burstiness,
            max_burstiness=max_burstiness,
            transition_influence=transition_influence,
            max_transition=max_transition,
            moe_cfg=moe_cfg,
            enabled=enable_adaptive_replication and moe_cfg.enable_adaptive_replication,
        )
        if requested_replicas == 1:
            requested_replicas = _shared_replica_count(
                cube,
                parsed_model,
                moe_cfg,
                enabled=enable_adaptive_replication,
            )
        cube_volume = float(cube.h * cube.w * cube.d)
        replicas = 1
        for _ in range(1, requested_replicas):
            if extra_replica_used + cube_volume <= extra_replica_budget + 1e-9:
                extra_replica_used += cube_volume
                replicas += 1
            else:
                break
        is_shared_operator_replica = cube.expert_id is None and replicas > 1
        if is_shared_operator_replica:
            shared_replica_requested += 1

        candidate_subcubes = list(range(cube_cfg.num_subcubes))
        if enable_moe_optimization and cube.expert_id is not None and cube.expert_id in expert_to_group:
            preferred_sc = group_to_subcube[expert_to_group[cube.expert_id]]
            candidate_subcubes = [preferred_sc] + [sc for sc in candidate_subcubes if sc != preferred_sc]

        preferred_z = 0 if cube.expert_id in hot_experts or cube.is_shared_expert else None

        mapped_replicas = 0
        used_subcubes_for_cube: set = set()

        for replica_id in range(replicas):
            replica_candidates = list(candidate_subcubes)
            if replicas > 1 and used_subcubes_for_cube:
                replica_candidates.sort(key=lambda sc: (sc in used_subcubes_for_cube, sc))

            dynamic_bias: Dict[int, float] = {}
            cube_group = expert_to_group.get(cube.expert_id) if cube.expert_id is not None else None
            for sc in replica_candidates:
                bias = 0.0

                if cube.expert_id is not None and co_matrix.size > 0:
                    expert_counts = subcube_expert_counter.get(sc, {})
                    conflict_mass = 0.0
                    transition_mass = 0.0
                    for other_e, cnt in expert_counts.items():
                        if other_e == cube.expert_id:
                            continue
                        conflict_mass += float(co_matrix[cube.expert_id, other_e]) * float(cnt)
                        if (
                            transition_matrix.size > 0
                            and cube.expert_id < transition_matrix.shape[0]
                            and other_e < transition_matrix.shape[1]
                        ):
                            pair_transition = (
                                float(transition_matrix[cube.expert_id, other_e])
                                + float(transition_matrix[other_e, cube.expert_id])
                            )
                            transition_mass += pair_transition * float(cnt)
                    expert_norm = max(expert_contention.get(cube.expert_id, 0.0), 1.0)
                    bias += moe_cfg.placement_conflict_weight * (conflict_mass / expert_norm)
                    transition_norm = max(expert_transition_mass.get(cube.expert_id, 0.0), 1.0)
                    bias += moe_cfg.transition_conflict_weight * (transition_mass / transition_norm)

                util_ratio = float(subcube_used_volume[sc]) / max(subcube_capacity, 1.0)
                bias += moe_cfg.placement_load_weight * util_ratio

                if enable_moe_optimization and cube_group is not None:
                    group_counts = subcube_group_counter.get(sc, {})
                    if group_counts and cube_group not in group_counts:
                        bias += moe_cfg.placement_group_penalty

                if replica_id > 0 and sc in used_subcubes_for_cube:
                    bias += moe_cfg.replica_diversity_penalty

                dynamic_bias[sc] = float(bias)

            result = allocator.try_place(
                cube=cube,
                candidate_subcubes=replica_candidates,
                preferred_z=preferred_z,
                policy=effective_packing_policy,
                subcube_bias=dynamic_bias,
            )
            if result is None:
                break

            sc, z, x, y = result
            physical_cube_id = f"{cube.cube_id}__r{replica_id}"
            strategy = "replicated" if replicas > 1 else "single"
            placement = Placement(
                logical_cube_id=cube.cube_id,
                physical_cube_id=physical_cube_id,
                operator_id=cube.operator_id,
                expert_id=cube.expert_id,
                replica_id=replica_id,
                subcube=sc,
                z=z,
                x=x,
                y=y,
                h=cube.h,
                w=cube.w,
                d=cube.d,
                strategy=strategy,
                compression_ratio=cube_comp_ratio,
            )
            placements.append(placement)
            placement_by_id[physical_cube_id] = placement
            cube_to_placements.setdefault(cube.cube_id, []).append(physical_cube_id)
            mapped_replicas += 1
            used_subcubes_for_cube.add(sc)
            if is_shared_operator_replica and replica_id > 0:
                shared_replica_mapped += 1

            subcube_used_volume[sc] += cube_volume
            if cube.expert_id is not None:
                e_counter = subcube_expert_counter[sc]
                e_counter[cube.expert_id] = int(e_counter.get(cube.expert_id, 0)) + 1
                if cube.expert_id in expert_to_group:
                    group_id = expert_to_group[cube.expert_id]
                    g_counter = subcube_group_counter[sc]
                    g_counter[group_id] = int(g_counter.get(group_id, 0)) + 1

        if mapped_replicas == 0:
            unplaced.append(cube.cube_id)

    placed_volume = float(sum(p.h * p.w * p.d for p in placements))
    uncompressed_physical_bytes = float(
        sum(parsed_model.weight_cubes[p.logical_cube_id].bytes_size for p in placements)
    )
    compressed_physical_bytes = float(
        sum(parsed_model.weight_cubes[p.logical_cube_id].bytes_size * p.compression_ratio for p in placements)
    )
    compression_ratio = compressed_physical_bytes / max(uncompressed_physical_bytes, 1.0)
    space_util = placed_volume / max(float(cube_cfg.total_volume), 1.0)
    mapping_rate = (len(cubes) - len(unplaced)) / max(len(cubes), 1)
    conflict_score = _estimate_conflict_score(placements, co_matrix)
    physical_ids = [p.physical_cube_id for p in placements]
    weight_stationary_valid = len(physical_ids) == len(set(physical_ids))

    result = MappingResult(
        placements=placements,
        cube_to_placements=cube_to_placements,
        placement_by_id=placement_by_id,
        unplaced_cubes=sorted(set(unplaced)),
        metrics={
            "space_utilization": space_util,
            "mapping_rate": mapping_rate,
            "placed_volume": placed_volume,
            "logical_volume": logical_volume,
            "conflict_score": conflict_score,
            "compression_ratio": compression_ratio,
            "compressed_physical_bytes": compressed_physical_bytes,
            "uncompressed_physical_bytes": uncompressed_physical_bytes,
        },
        metadata={
            "packing_policy": packing_policy,
            "effective_packing_policy": effective_packing_policy,
            "enable_moe_optimization": enable_moe_optimization,
            "enable_adaptive_replication": enable_adaptive_replication,
            "hot_experts": sorted(hot_experts),
            "mutually_exclusive_groups": groups,
            "grouping_strategy": grouping_meta,
            "group_to_subcube": group_to_subcube,
            "group_local_search": ls_meta,
            "exact_hot_subgraph": exact_meta,
            "hot_subgraph_topk": hot_subgraph_topk_meta,
            "group_assignment_estimated_conflict": float(
                _estimate_group_conflict(group_to_subcube, _build_group_pair_cooccurrence(groups, co_matrix))
            )
            if group_to_subcube and len(groups) > 0
            else 0.0,
            "extra_replica_budget": extra_replica_budget,
            "extra_replica_used": extra_replica_used,
            "shared_operator_replication": {
                "enabled": bool(moe_cfg.enable_shared_operator_replication and enable_adaptive_replication),
                "operator_types": list(moe_cfg.shared_replication_operator_types),
                "min_volume": int(moe_cfg.shared_replication_min_volume),
                "max_replicas": int(moe_cfg.shared_replication_max_replicas),
                "requested_logical_cubes": int(shared_replica_requested),
                "extra_physical_replicas": int(shared_replica_mapped),
                "replicated_logical_cubes": int(
                    sum(
                        1
                        for cube_id, pids in cube_to_placements.items()
                        if len(pids) > 1 and parsed_model.weight_cubes[cube_id].expert_id is None
                    )
                ),
            },
            "replica_spread_subcubes": int(len({(p.logical_cube_id, p.subcube) for p in placements})),
            "compression": {
                "enable_blockwise_quantization": moe_cfg.enable_blockwise_quantization,
                "enable_sparse_compression": moe_cfg.enable_sparse_compression,
                "hot_quant_bits": moe_cfg.hot_quant_bits,
                "cold_quant_bits": moe_cfg.cold_quant_bits,
                "hot_sparsity_ratio": moe_cfg.hot_sparsity_ratio,
                "cold_sparsity_ratio": moe_cfg.cold_sparsity_ratio,
                "compressed_physical_bytes": compressed_physical_bytes,
                "uncompressed_physical_bytes": uncompressed_physical_bytes,
                "compression_ratio": compression_ratio,
            },
            "replication_pressure_weights": {
                "freq": moe_cfg.replication_freq_weight,
                "contention": moe_cfg.replication_contention_weight,
                "burst": moe_cfg.replication_burst_weight,
                "transition": moe_cfg.replication_transition_weight,
            },
            "replication_pressure_thresholds": {
                "low": moe_cfg.replica_pressure_low_threshold,
                "high": moe_cfg.replica_pressure_high_threshold,
            },
            "placement_dynamic_weights": {
                "conflict": moe_cfg.placement_conflict_weight,
                "transition_conflict": moe_cfg.transition_conflict_weight,
                "load": moe_cfg.placement_load_weight,
                "group_penalty": moe_cfg.placement_group_penalty,
                "replica_diversity": moe_cfg.replica_diversity_penalty,
            },
            "packing_dynamic_weights": {
                "enable_aspect_aware": bool(moe_cfg.enable_aspect_aware_packing),
                "aspect_aware_weight": float(moe_cfg.aspect_aware_weight),
                "fragmentation_penalty_weight": float(moe_cfg.fragmentation_penalty_weight),
                "conflict_propagation_weight": float(moe_cfg.conflict_propagation_weight),
                "capacity_peak_weight": float(moe_cfg.capacity_peak_weight),
            },
            "constraints": {
                "weight_stationary": bool(weight_stationary_valid),
                "duplicate_physical_cube_ids": int(len(physical_ids) - len(set(physical_ids))),
            },
        },
    )
    return result


def save_mapping(path: Path, mapping: MappingResult) -> None:
    save_json(path=path, data=mapping.to_dict())
