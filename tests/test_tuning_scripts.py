import json
import sys
from pathlib import Path

from config.cube_config import DEFAULT_CUBE_CONFIG
from scripts import tune_optuna, tune_optuna_multi


class StaticTrial:
    number = 0

    def __init__(self):
        self.user_attrs = {}

    def suggest_float(self, _name, low, _high):
        return low

    def suggest_int(self, _name, low, _high):
        return low

    def suggest_categorical(self, _name, choices):
        return choices[0]

    def report(self, _value, step=None):
        return None

    def should_prune(self):
        return False

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


def _fake_comparison():
    return {
        "optimized": {
            "latency": 100.0,
            "conflict_score": 0.0,
            "pipeline_bubble_cycles": 0.0,
            "space_utilization": 0.5,
            "temporal_utilization": 0.25,
        }
    }


def test_tuning_scripts_accept_project_venv_and_pin_multiprocessing(monkeypatch):
    captured = []

    monkeypatch.setattr(tune_optuna.sys, "executable", str(tune_optuna.EXPECTED_VENV_PYTHON))
    monkeypatch.setattr(tune_optuna.multiprocessing, "set_executable", captured.append)

    tune_optuna._ensure_project_venv_python()

    assert captured == [str(tune_optuna.EXPECTED_VENV_PYTHON)]


def test_tuning_scripts_reject_system_python(monkeypatch):
    monkeypatch.setattr(tune_optuna_multi.sys, "executable", r"C:\Python312\python.exe")

    try:
        tune_optuna_multi._ensure_project_venv_python()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError for non-project Python")

    assert "Project venv Python required" in message
    assert str(tune_optuna_multi.EXPECTED_VENV_PYTHON) in message


def test_single_objective_factory_can_disable_shared_replication(monkeypatch, tmp_path):
    captured_flags = []

    def fake_run_pipeline(**kwargs):
        captured_flags.append(kwargs["moe_cfg_override"].enable_shared_operator_replication)
        return _fake_comparison()

    monkeypatch.setattr(tune_optuna, "run_pipeline", fake_run_pipeline)

    objective = tune_optuna._objective_factory(
        model_path=Path("data/sample_model.json"),
        trace_paths=[Path("data/sample_trace.json")],
        trace_weights=[1.0],
        output_root=tmp_path,
        cube_cfg=DEFAULT_CUBE_CONFIG,
        capacity_max_ratio=2.0,
        strict_capacity=True,
        robust_worst_weight=0.0,
        tail_p95_weight=0.0,
        tail_p99_weight=0.0,
        enable_shared_operator_replication=False,
    )

    objective(StaticTrial())

    assert captured_flags == [False]


def test_multi_objective_factory_can_disable_shared_replication(monkeypatch, tmp_path):
    captured_flags = []

    def fake_run_pipeline(**kwargs):
        captured_flags.append(kwargs["moe_cfg_override"].enable_shared_operator_replication)
        return _fake_comparison()

    monkeypatch.setattr(tune_optuna_multi, "run_pipeline", fake_run_pipeline)

    objective = tune_optuna_multi._objective_factory(
        model_path=Path("data/sample_model.json"),
        trace_paths=[Path("data/sample_trace.json")],
        trace_weights=[1.0],
        output_root=tmp_path,
        cube_cfg=DEFAULT_CUBE_CONFIG,
        capacity_max_ratio=2.0,
        strict_capacity=True,
        overlap_transfer_compute=True,
        robust_worst_weight=0.0,
        tail_p95_weight=0.0,
        tail_p99_weight=0.0,
        enable_shared_operator_replication=False,
    )

    objective(StaticTrial())

    assert captured_flags == [False]


def test_single_tuning_payload_records_disabled_shared_replication(monkeypatch, tmp_path):
    captured_flags = []

    def fake_run_pipeline(**kwargs):
        captured_flags.append(kwargs["moe_cfg_override"].enable_shared_operator_replication)
        return _fake_comparison()

    monkeypatch.setattr(tune_optuna, "run_pipeline", fake_run_pipeline)

    result = tune_optuna.run_tuning(
        model_path=Path("data/sample_model.json"),
        trace_paths=[Path("data/sample_trace.json")],
        output_root=tmp_path,
        n_trials=1,
        trace_weight_map={},
        auto_trace_weight=False,
        robust_worst_weight=0.0,
        tail_p95_weight=0.0,
        tail_p99_weight=0.0,
        enable_two_stage_tuning=False,
        stage1_ratio=0.45,
        stage2_warmstart_topk=1,
        n_jobs=1,
        seed=2026,
        cube_cfg=DEFAULT_CUBE_CONFIG,
        capacity_max_ratio=2.0,
        strict_capacity=True,
        enable_shared_operator_replication=False,
    )

    assert captured_flags == [False]
    assert result["best_params"]["enable_shared_operator_replication"] is False


def test_multi_tuning_payload_and_holdout_record_disabled_shared_replication(monkeypatch, tmp_path):
    captured_flags = []

    def fake_run_pipeline(**kwargs):
        captured_flags.append(kwargs["moe_cfg_override"].enable_shared_operator_replication)
        return _fake_comparison()

    monkeypatch.setattr(tune_optuna_multi, "run_pipeline", fake_run_pipeline)

    tune_optuna_multi.run_multi_tuning(
        model_path=Path("data/sample_model.json"),
        trace_paths=[Path("data/sample_trace.json")],
        output_root=tmp_path,
        n_trials=1,
        trace_weight_map={},
        auto_trace_weight=False,
        overlap_transfer_compute=True,
        robust_worst_weight=0.0,
        tail_p95_weight=0.0,
        tail_p99_weight=0.0,
        enable_two_stage_tuning=False,
        stage1_ratio=0.45,
        stage2_warmstart_topk=1,
        n_jobs=1,
        seed=2026,
        holdout_paths=[Path("data/sample_trace.json")],
        holdout_topk=1,
        cube_cfg=DEFAULT_CUBE_CONFIG,
        capacity_max_ratio=2.0,
        strict_capacity=True,
        enable_shared_operator_replication=False,
    )

    payload = json.loads((tmp_path / "multiobjective_pareto.json").read_text(encoding="utf-8"))

    assert captured_flags == [False, False]
    assert payload["pareto"][0]["params"]["enable_shared_operator_replication"] is False
    assert payload["holdout_eval"][0]["params"]["enable_shared_operator_replication"] is False
