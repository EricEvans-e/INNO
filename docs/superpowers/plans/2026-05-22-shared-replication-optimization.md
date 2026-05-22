# Shared Replication Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve strict-capacity latency by allowing selected large shared/non-expert weight cubes to use the existing replica-aware scheduler, and make the ablation labels reflect what each variant actually disables.

**Architecture:** Extend the existing mapping path in `src/mapping_solver.py` instead of creating a new scheduler. Shared operator replication is opt-in through `MoEConfig`, uses the same `cube_to_placements` interface as expert replication, and remains bounded by the existing extra replica volume budget. The pipeline keeps `optimized` as the submission variant and introduces clearer ablation metadata/names so output metrics are interpretable.

**Tech Stack:** Python 3.12, dataclasses, pytest, existing project virtual environment at `E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe`.

---

## File Structure

- Modify `config/moe_config.py`: add validated shared-operator replication knobs.
- Modify `src/mapping_solver.py`: decide replica count for non-expert shared cubes and report metadata.
- Modify `main.py`: rename/clarify ablation variants and add CLI flags for shared replication.
- Modify `tests/test_mapping.py`: add focused tests for shared cube replication and capacity budget.
- Modify `tests/test_pipeline_outputs.py`: assert new ablation keys/metadata are exported.
- Do not change `src/simulator.py` unless a test proves it is necessary; it already consumes multiple placements through `cube_to_placements`.

---

### Task 1: Add Shared-Operator Replication Config And Mapping Behavior

**Files:**
- Modify: `config/moe_config.py`
- Modify: `src/mapping_solver.py`
- Test: `tests/test_mapping.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_mapping.py`:

```python
from dataclasses import replace

from config.cube_config import CubeConfig
from config.moe_config import DEFAULT_MOE_CONFIG
from src.mapping_solver import solve_mapping
from src.model_parser import parse_activation_trace, parse_model


def test_shared_operator_replication_uses_extra_budget() -> None:
    cube_cfg = CubeConfig(n=3, d=4, h=4096, w=4096)
    model = parse_model(ROOT / "data" / "sample_model.json", cube_cfg)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")
    moe_cfg = replace(
        DEFAULT_MOE_CONFIG,
        enable_shared_operator_replication=True,
        shared_replication_operator_types=("linear",),
        shared_replication_min_volume=4096 * 4096,
        shared_replication_max_replicas=2,
        replication_volume_budget_ratio=0.5,
    )

    mapping = solve_mapping(
        model,
        trace,
        cube_cfg,
        moe_cfg,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    replicated = {
        cube_id: placements
        for cube_id, placements in mapping.cube_to_placements.items()
        if cube_id.startswith(("embed_proj", "shared_attn", "shared_mlp", "lm_head"))
        and len(placements) > 1
    }
    assert replicated
    assert mapping.metadata["shared_operator_replication"]["enabled"] is True
    assert mapping.metadata["shared_operator_replication"]["replicated_logical_cubes"] >= 1
    assert mapping.metadata["extra_replica_used"] <= mapping.metadata["extra_replica_budget"]


def test_shared_operator_replication_can_be_disabled() -> None:
    cube_cfg = CubeConfig(n=3, d=4, h=4096, w=4096)
    model = parse_model(ROOT / "data" / "sample_model.json", cube_cfg)
    trace = parse_activation_trace(ROOT / "data" / "sample_trace.json")
    moe_cfg = replace(
        DEFAULT_MOE_CONFIG,
        enable_shared_operator_replication=False,
        shared_replication_operator_types=("linear",),
        shared_replication_min_volume=4096 * 4096,
        shared_replication_max_replicas=2,
        replication_volume_budget_ratio=0.5,
    )

    mapping = solve_mapping(
        model,
        trace,
        cube_cfg,
        moe_cfg,
        packing_policy="best_fit",
        enable_moe_optimization=True,
        enable_adaptive_replication=True,
    )

    assert len(mapping.cube_to_placements["embed_proj__s0"]) == 1
    assert mapping.metadata["shared_operator_replication"]["enabled"] is False
    assert mapping.metadata["shared_operator_replication"]["replicated_logical_cubes"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY -m pytest tests/test_mapping.py::test_shared_operator_replication_uses_extra_budget tests/test_mapping.py::test_shared_operator_replication_can_be_disabled -q
```

Expected: FAIL because `MoEConfig` has no `enable_shared_operator_replication` field.

- [ ] **Step 3: Add config fields and validation**

In `config/moe_config.py`, add fields to `MoEConfig`:

```python
enable_shared_operator_replication: bool = True
shared_replication_operator_types: tuple[str, ...] = ("linear",)
shared_replication_min_volume: int = 16_777_216
shared_replication_max_replicas: int = 2
```

Extend `validate()`:

```python
if self.shared_replication_min_volume <= 0:
    raise ValueError("shared_replication_min_volume must be positive")
if self.shared_replication_max_replicas < 1:
    raise ValueError("shared_replication_max_replicas must be >= 1")
if not self.shared_replication_operator_types:
    raise ValueError("shared_replication_operator_types must not be empty")
```

- [ ] **Step 4: Implement shared replica decision**

In `src/mapping_solver.py`, add a helper near `_replica_count`:

```python
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
```

Inside `solve_mapping()`, after `requested_replicas = _replica_count(...)`, add:

```python
        if requested_replicas == 1:
            requested_replicas = _shared_replica_count(
                cube,
                parsed_model,
                moe_cfg,
                enabled=enable_adaptive_replication,
            )
```

Track metadata before the cube loop:

```python
    shared_replica_requested = 0
    shared_replica_mapped = 0
```

Inside the cube loop, after final `replicas` is known:

```python
        is_shared_operator_replica = cube.expert_id is None and replicas > 1
        if is_shared_operator_replica:
            shared_replica_requested += 1
```

Inside the successful placement block:

```python
            if is_shared_operator_replica and replica_id > 0:
                shared_replica_mapped += 1
```

Add metadata:

```python
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
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY -m pytest tests/test_mapping.py::test_shared_operator_replication_uses_extra_budget tests/test_mapping.py::test_shared_operator_replication_can_be_disabled -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add config/moe_config.py src/mapping_solver.py tests/test_mapping.py
git commit -m "feat: replicate shared operators under budget"
```

---

### Task 2: Expose CLI Knobs And Clarify Ablation Variants

**Files:**
- Modify: `main.py`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Write failing pipeline metadata test**

Add to `tests/test_pipeline_outputs.py`:

```python
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
    mapping = load_json(output_dir / "optimized_mapping.json")
    assert "shared_operator_replication" in mapping["metadata"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY -m pytest tests/test_pipeline_outputs.py::test_pipeline_exports_clear_ablation_variants -q
```

Expected: FAIL because the old keys are still exported.

- [ ] **Step 3: Rename variants**

In `main.py`, change the `variants` dict:

```python
    variants = {
        "baseline": {
            "packing_policy": "first_fit",
            "enable_moe_optimization": False,
            "enable_adaptive_replication": False,
        },
        "ablation_best_fit_replication_only": {
            "packing_policy": "best_fit",
            "enable_moe_optimization": False,
            "enable_adaptive_replication": True,
        },
        "ablation_moe_without_replication": {
            "packing_policy": "best_fit",
            "enable_moe_optimization": True,
            "enable_adaptive_replication": False,
        },
        "optimized": {
            "packing_policy": "best_fit",
            "enable_moe_optimization": True,
            "enable_adaptive_replication": True,
        },
    }
```

Update the CLI ablation print loop near the bottom:

```python
    for key in ["baseline", "ablation_best_fit_replication_only", "ablation_moe_without_replication", "optimized"]:
        print(f"  - {key}: {comparison['ablation'][key]['metrics']['latency']:.2f}")
```

- [ ] **Step 4: Add CLI arguments for shared replication**

Add arguments near the existing replication flags:

```python
    parser.add_argument("--disable-shared-replication", action="store_true", help="Disable adaptive replication for large non-expert shared operators")
    parser.add_argument("--shared-replication-min-volume", type=int, default=DEFAULT_MOE_CONFIG.shared_replication_min_volume, help="Minimum logical volume for non-expert shared-operator replication")
    parser.add_argument("--shared-replication-max-replicas", type=int, default=DEFAULT_MOE_CONFIG.shared_replication_max_replicas, help="Maximum replicas for eligible non-expert shared operators")
```

Pass these into `replace(DEFAULT_MOE_CONFIG, ...)`:

```python
        enable_shared_operator_replication=(not args.disable_shared_replication),
        shared_replication_min_volume=max(1, int(args.shared_replication_min_volume)),
        shared_replication_max_replicas=max(1, int(args.shared_replication_max_replicas)),
```

- [ ] **Step 5: Run focused test**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY -m pytest tests/test_pipeline_outputs.py::test_pipeline_exports_clear_ablation_variants -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add main.py tests/test_pipeline_outputs.py
git commit -m "chore: clarify ablation variants"
```

---

### Task 3: Verify Strict-Capacity Performance And Submission Compatibility

**Files:**
- Modify only if verification exposes a concrete bug.

- [ ] **Step 1: Run full tests**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run strict smoke with shared replication enabled**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/v6_shared_replication_smoke --profile --deterministic --search-trials 1 --parallel-workers 1 --local-restarts 1 --local-iters 5 --disable-sa --disable-parallel-search --cube-d 2 --strict-capacity --capacity-max-ratio 2.0 --replication-volume-budget-ratio 0.45 --shared-replication-max-replicas 2 --shared-replication-min-volume 16777216 --overlap-transfer-compute --overlap-alpha 0.7872434125774002 --overlap-model-mode nonlinear_bandwidth_aware --overlap-bw-power-law-alpha 1.190361785167963 --overlap-z-depth-penalty 0.028146737699334498
```

Expected: command exits 0 and prints optimized latency.

- [ ] **Step 3: Validate output**

Run:

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
& $PY scripts/validate_submission.py --output outputs/v6_shared_replication_smoke --strict-capacity --capacity-max-ratio 2.0 --report outputs/v6_shared_replication_smoke/validation_report.json
```

Expected: validator exits 0 and report has `"ok": true`.

- [ ] **Step 4: Inspect performance evidence**

Read:

```powershell
Get-Content outputs/v6_shared_replication_smoke/comparison_metrics.json
Get-Content outputs/v6_shared_replication_smoke/optimized_mapping.json
```

Acceptance:
- `comparison_metrics.json` exists.
- `optimized_mapping.json` metadata includes `shared_operator_replication`.
- `solution.json` validates.
- If latency does not improve under the smoke parameters, record the measured result in the final report and do not claim a speedup.

- [ ] **Step 5: Commit any verification fixes**

Only if files changed during Task 3:

```powershell
git add <changed-files>
git commit -m "fix: preserve strict shared replication validation"
```

---

## Self-Review Notes

- Spec coverage: Task 1 implements shared/non-expert replication; Task 2 clarifies ablations and CLI; Task 3 verifies strict-capacity compatibility and performance evidence.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: config fields are referenced consistently in tests, CLI, and mapping metadata.
