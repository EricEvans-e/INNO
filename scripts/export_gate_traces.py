#!/usr/bin/env python3
"""Export gate/top-k expert traces from models (mock + PyTorch support).

Usage examples:
  # mock mode (no torch required)
  python scripts/export_gate_traces.py --mode mock --output outputs/trace_exporter_mock.json \
      --num-experts 128 --n-inferences 1000 --top-k 2

  # torch mode: requires a module that exposes a factory: module_path:factory_fn
  # factory_fn(n_inferences) -> (model, inputs_iterable)
  python scripts/export_gate_traces.py --mode torch --torch-factory mymod:build_model_and_inputs \
      --gate-attrs "encoder.layers.0.moe.gate" --output outputs/trace_from_model.json --top-k 2

Note: torch mode expects the provided factory to return (model, inputs_iterable).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

try:
    import numpy as np
except Exception:
    np = None


def generate_mock_traces(num_experts: int, n_inferences: int, top_k: int, hot_ratio: float, alpha: float, seed: int):
    if np is None:
        raise RuntimeError("numpy is required for mock mode")
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, num_experts + 1)
    weights = 1.0 / (ranks ** alpha)
    n_hot = max(1, int(num_experts * hot_ratio))
    weights[:n_hot] *= (1.0 + 5.0)
    probs = weights / weights.sum()

    traces: List[List[int]] = []
    for _ in range(n_inferences):
        k = min(top_k, num_experts)
        raw = list(rng.choice(num_experts, size=k, replace=False, p=probs))
        traces.append([int(x) for x in raw])
    return traces


def save_trace(output: Path, num_experts: int, top_k: int, traces: List[List[int]], shared_experts: List[int] | None = None):
    payload: Dict[str, Any] = {
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "n_inferences": int(len(traces)),
        "traces": traces,
        "shared_experts": list(shared_experts or []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved traces to {output}")


def import_factory(factory_path: str) -> Callable:
    # factory_path examples: mypkg.mymod:build or /path/to/file.py:build
    if ":" not in factory_path:
        raise ValueError("factory must be MODULE_OR_PATH:CALLABLE_NAME")
    module_part, func_name = factory_path.split(":", 1)
    if module_part.endswith(".py") or module_part.startswith("./") or module_part.startswith("/"):
        spec = importlib.util.spec_from_file_location("custom_mod", module_part)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {module_part}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name or "custom_mod"] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_part)

    if not hasattr(mod, func_name):
        raise AttributeError(f"Module {module_part} has no attribute {func_name}")
    return getattr(mod, func_name)


def collect_traces_from_torch(factory: Callable, gate_attrs: List[str], n_inferences: int, top_k: int, device: str = "cpu") -> Tuple[int, List[List[int]]]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for torch mode") from e

    # factory should produce (model, inputs_iterable) when called with n_inferences
    res = factory(n_inferences)
    if isinstance(res, tuple) and len(res) >= 2:
        model, inputs = res[0], res[1]
    else:
        raise RuntimeError("Factory must return (model, inputs_iterable)")

    model.to(device)
    model.eval()

    traces: List[List[int]] = []

    hooks = []

    def make_hook(buf: List[List[int]]):
        def hook(module, inp, out):
            # assume out is logits tensor
            try:
                vals = out.detach().cpu()
                if vals.dim() == 1:
                    idx = torch.topk(vals, k=min(top_k, vals.shape[-1]))[1].tolist()
                    buf.append([int(x) for x in idx])
                else:
                    top = torch.topk(vals, k=min(top_k, vals.shape[-1]), dim=-1)[1]
                    for row in top.tolist():
                        buf.append([int(x) for x in row])
            except Exception:
                pass

        return hook

    # register hooks
    for attr in gate_attrs:
        parts = attr.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p)
        buf: List[List[int]] = []
        h = mod.register_forward_hook(make_hook(buf))
        hooks.append((h, buf))

    # run through inputs
    with torch.no_grad():
        count = 0
        for inp in inputs:
            # inputs can be tensor or tuple/list
            if isinstance(inp, (list, tuple)):
                model(*[x.to(device) if hasattr(x, 'to') else x for x in inp])
            else:
                if hasattr(inp, 'to'):
                    model(inp.to(device))
                else:
                    model(inp)
            count += 1
            if n_inferences and count >= n_inferences:
                break

    # gather buffers (flatten)
    for h, buf in hooks:
        h.remove()
        traces.extend(buf)

    # naive num_experts: max id + 1
    num_experts = 0
    for t in traces:
        if t:
            num_experts = max(num_experts, max(t) + 1)

    return num_experts, traces


def main():
    parser = argparse.ArgumentParser(description="Export gate/top-k expert traces")
    parser.add_argument("--mode", choices=("mock", "torch"), default="mock")
    parser.add_argument("--output", type=Path, default=Path("outputs/trace_exporter.json"))

    # mock args
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--n-inferences", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--hot-ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)

    # torch args
    parser.add_argument("--torch-factory", type=str, help="MODULE_OR_PATH:factory_fn returning (model, inputs_iterable)")
    parser.add_argument("--gate-attrs", type=str, help="semicolon-separated attribute paths on model for gate modules, e.g. 'encoder.layers.0.moe.gate;decoder.layers.1.moe.gate'")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.mode == "mock":
        traces = generate_mock_traces(args.num_experts, args.n_inferences, args.top_k, args.hot_ratio, args.alpha, args.seed)
        save_trace(args.output, args.num_experts, args.top_k, traces)
        return

    # torch mode
    if args.mode == "torch":
        if not args.torch_factory:
            raise SystemExit("--torch-factory is required in torch mode")
        factory = import_factory(args.torch_factory)
        gate_attrs = [s.strip() for s in (args.gate_attrs or "").split(";") if s.strip()]
        num_experts, traces = collect_traces_from_torch(factory, gate_attrs, args.n_inferences, args.top_k, device=args.device)
        save_trace(args.output, num_experts or args.num_experts, args.top_k, traces)


if __name__ == "__main__":
    main()
