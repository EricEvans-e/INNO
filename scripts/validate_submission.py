from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_json, save_json


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def _get_weight_cubes(solution: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = solution.get("weight_cubes")
    if isinstance(top, list):
        return top
    nested = solution.get("spatial_mapping", {}).get("weight_cubes", [])
    return nested if isinstance(nested, list) else []


def _get_schedule(solution: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = solution.get("schedule")
    if isinstance(top, list):
        return top
    nested = solution.get("static_schedule", [])
    return nested if isinstance(nested, list) else []


def _check_solution_shape(solution: Dict[str, Any], errors: List[str]) -> None:
    for key in ["hardware", "spatial_mapping", "static_schedule", "metrics", "weight_cubes", "schedule"]:
        if key not in solution:
            errors.append(f"missing solution key: {key}")
    if not _get_weight_cubes(solution):
        errors.append("solution has no weight_cubes")
    if not _get_schedule(solution):
        errors.append("solution has no schedule")


def _check_capacity(
    output_dir: Path,
    strict_capacity: bool,
    capacity_max_ratio: float,
    errors: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        warnings.append("run_manifest.json missing; capacity ratio can only be checked from solution metadata if present")
        return {}

    manifest = load_json(manifest_path)
    capacity = manifest.get("configs", {}).get("capacity", {})
    if not capacity:
        warnings.append("run_manifest.json has no configs.capacity")
        return {}

    ratio = float(capacity.get("capacity_ratio", 0.0))
    max_ratio = float(capacity.get("capacity_max_ratio", capacity_max_ratio))
    if strict_capacity and ratio > max_ratio + 1e-9:
        errors.append(f"capacity ratio {ratio:.6f} exceeds max {max_ratio:.6f}")
    return capacity


def _check_mapping(solution: Dict[str, Any], errors: List[str]) -> None:
    hw = solution.get("hardware", {})
    h = int(hw.get("subcube_size", [0, 0])[0])
    w = int(hw.get("subcube_size", [0, 0])[1])
    d = int(hw.get("D", 0))
    num_subcubes = int(hw.get("num_subcubes", 0))
    placed_by_subcube: Dict[int, List[Tuple[int, int, int, int, int, int, str]]] = {}

    for idx, cube in enumerate(_get_weight_cubes(solution)):
        coords = cube.get("coords", [])
        shape = cube.get("shape", [])
        if len(coords) != 3 or len(shape) != 3:
            errors.append(f"weight cube {idx} has invalid coords/shape")
            continue
        x, y, z = [int(v) for v in coords]
        ch, cw, cd = [int(v) for v in shape]
        sc = int(cube.get("subcube", -1))
        cid = str(cube.get("physical_cube_id", cube.get("id", idx)))

        if sc < 0 or sc >= num_subcubes:
            errors.append(f"{cid}: subcube {sc} out of range [0,{num_subcubes})")
        if x < 0 or y < 0 or z < 0 or ch <= 0 or cw <= 0 or cd <= 0:
            errors.append(f"{cid}: invalid non-positive coords/shape")
        if x + cw > w or y + ch > h or z + cd > d:
            errors.append(f"{cid}: placement exceeds subcube bounds")

        placed_by_subcube.setdefault(sc, []).append((x, y, z, cw, ch, cd, cid))

    for sc, cubes in placed_by_subcube.items():
        for i, a in enumerate(cubes):
            ax, ay, az, aw, ah, ad, aid = a
            for b in cubes[i + 1 :]:
                bx, by, bz, bw, bh, bd, bid = b
                if (
                    _intervals_overlap(ax, ax + aw, bx, bx + bw)
                    and _intervals_overlap(ay, ay + ah, by, by + bh)
                    and _intervals_overlap(az, az + ad, bz, bz + bd)
                ):
                    errors.append(f"spatial overlap in subcube {sc}: {aid} vs {bid}")

    spatial = solution.get("spatial_mapping", {})
    if spatial.get("unplaced_cubes"):
        errors.append(f"unplaced cubes present: {spatial.get('unplaced_cubes')}")
    if float(spatial.get("mapping_rate", solution.get("metrics", {}).get("mapping_rate", 0.0))) < 1.0:
        errors.append("mapping_rate is below 1.0")


def _check_schedule(solution: Dict[str, Any], errors: List[str]) -> None:
    by_subcube: Dict[int, List[Tuple[int, int, str]]] = {}
    for idx, rec in enumerate(_get_schedule(solution)):
        sc = int(rec.get("subcube", -1))
        if sc < 0:
            continue
        start = int(rec.get("start_cycle", rec.get("start", 0)))
        end = int(rec.get("end_cycle", rec.get("end", start)))
        if end < start:
            errors.append(f"schedule record {idx} has end before start")
            continue
        by_subcube.setdefault(sc, []).append((start, end, str(rec.get("task_id", idx))))

    for sc, items in by_subcube.items():
        items.sort(key=lambda x: (x[0], x[1]))
        prev_end = -1
        prev_task = ""
        for start, end, task_id in items:
            if start < prev_end:
                errors.append(f"subcube {sc} schedule overlap: {prev_task} vs {task_id}")
            if end > prev_end:
                prev_end = end
                prev_task = task_id

    constraints = solution.get("metadata", {}).get("simulation", {}).get("constraints", {})
    for key in ["subcube_exclusivity_violations", "dependency_violations", "parallel_limit_violations"]:
        if int(constraints.get(key, 0)) != 0:
            errors.append(f"simulation constraint violation: {key}={constraints.get(key)}")
    if constraints and not bool(constraints.get("valid", False)):
        errors.append("simulation constraints.valid is false")


def validate_output_dir(
    output_dir: Path,
    strict_capacity: bool = False,
    capacity_max_ratio: float = 2.0,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    solution_path = output_dir / "solution.json"
    if not solution_path.exists():
        errors.append(f"missing {solution_path}")
        solution: Dict[str, Any] = {}
    else:
        solution = load_json(solution_path)

    if solution:
        _check_solution_shape(solution, errors)
        _check_mapping(solution, errors)
        _check_schedule(solution, errors)

    capacity = _check_capacity(output_dir, strict_capacity, capacity_max_ratio, errors, warnings)
    report = {
        "output_dir": str(output_dir),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "capacity": capacity,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CIM scheduler submission artifacts")
    parser.add_argument("--output", type=Path, required=True, help="Output directory containing solution.json")
    parser.add_argument("--strict-capacity", action="store_true")
    parser.add_argument("--capacity-max-ratio", type=float, default=2.0)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = validate_output_dir(
        output_dir=args.output,
        strict_capacity=bool(args.strict_capacity),
        capacity_max_ratio=max(1e-9, float(args.capacity_max_ratio)),
    )
    if args.report:
        save_json(args.report, report)
    else:
        print(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
