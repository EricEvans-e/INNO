import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_cooccurrence_matrix(traces: Sequence[Sequence[int]], num_experts: int) -> np.ndarray:
    matrix = np.zeros((num_experts, num_experts), dtype=np.int64)
    for active in traces:
        unique = sorted(set(active))
        for i in unique:
            matrix[i, i] += 1
        for idx, i in enumerate(unique):
            for j in unique[idx + 1 :]:
                matrix[i, j] += 1
                matrix[j, i] += 1
    return matrix


def compute_expert_frequency(traces: Sequence[Sequence[int]], num_experts: int) -> Dict[int, int]:
    freq = {i: 0 for i in range(num_experts)}
    for active in traces:
        for expert in set(active):
            freq[expert] += 1
    return freq


def build_transition_matrix(traces: Sequence[Sequence[int]], num_experts: int) -> np.ndarray:
    matrix = np.zeros((num_experts, num_experts), dtype=np.int64)
    if len(traces) <= 1:
        return matrix
    for t in range(len(traces) - 1):
        prev_active = sorted(set(traces[t]))
        next_active = sorted(set(traces[t + 1]))
        for i in prev_active:
            for j in next_active:
                matrix[i, j] += 1
    return matrix


def compute_expert_burstiness(traces: Sequence[Sequence[int]], num_experts: int) -> Dict[int, float]:
    if not traces:
        return {i: 0.0 for i in range(num_experts)}

    longest = {i: 0 for i in range(num_experts)}
    current = {i: 0 for i in range(num_experts)}
    n = len(traces)

    for active in traces:
        active_set = set(active)
        for e in range(num_experts):
            if e in active_set:
                current[e] += 1
                if current[e] > longest[e]:
                    longest[e] = current[e]
            else:
                current[e] = 0

    return {e: float(longest[e]) / float(max(n, 1)) for e in range(num_experts)}


def compute_transition_influence(transition_matrix: np.ndarray) -> Dict[int, float]:
    if transition_matrix.size == 0:
        return {}
    in_flow = transition_matrix.sum(axis=0)
    out_flow = transition_matrix.sum(axis=1)
    return {int(i): float(in_flow[i] + out_flow[i]) for i in range(transition_matrix.shape[0])}


def normalize_frequency(freq: Dict[int, int]) -> Dict[int, float]:
    if not freq:
        return {}
    max_value = max(freq.values()) if freq else 1
    max_value = max(max_value, 1)
    return {k: v / max_value for k, v in freq.items()}


def check_2d_non_overlap(rects: Iterable[Tuple[int, int, int, int]]) -> bool:
    rect_list = list(rects)
    for i in range(len(rect_list)):
        x1, y1, w1, h1 = rect_list[i]
        for j in range(i + 1, len(rect_list)):
            x2, y2, w2, h2 = rect_list[j]
            overlap_x = not (x1 + w1 <= x2 or x2 + w2 <= x1)
            overlap_y = not (y1 + h1 <= y2 or y2 + h2 <= y1)
            if overlap_x and overlap_y:
                return False
    return True


def plot_cooccurrence_heatmap(matrix: np.ndarray, output_path: Path, title: str = "MoE Expert Co-occurrence") -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_title(title)
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Expert ID")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mapping_slices(
    placements: Sequence[Dict[str, Any]],
    n_subcubes: int,
    d: int,
    h: int,
    w: int,
    output_path: Path,
    max_layers: int = 3,
) -> None:
    ensure_dir(output_path.parent)
    layers = min(d, max_layers)
    fig, axes = plt.subplots(layers, 1, figsize=(12, 3.2 * layers), squeeze=False)

    for z in range(layers):
        ax = axes[z][0]
        ax.set_title(f"z={z} mapping slices")
        ax.set_xlim(0, n_subcubes * w)
        ax.set_ylim(0, h)
        ax.set_xlabel("Sub-Cube aligned width")
        ax.set_ylabel("Height")
        ax.invert_yaxis()

        for sub_idx in range(n_subcubes):
            offset_x = sub_idx * w
            ax.axvline(offset_x, color="lightgray", linewidth=0.8)
            ax.text(offset_x + 20, 120, f"SC{sub_idx}", fontsize=8)

        for p in placements:
            if p["z"] != z:
                continue
            sub = p["subcube"]
            offset_x = sub * w
            rect = plt.Rectangle(
                (offset_x + p["x"], p["y"]),
                p["w"],
                p["h"],
                fill=True,
                alpha=0.45,
                edgecolor="black",
                linewidth=0.7,
            )
            ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_schedule_gantt(
    schedule_records: Sequence[Dict[str, Any]],
    output_path: Path,
    max_records: int = 400,
) -> None:
    ensure_dir(output_path.parent)
    records = sorted(schedule_records, key=lambda x: (x["subcube"], x["start"]))[:max_records]
    if not records:
        return

    subcubes = sorted({r["subcube"] for r in records if r["subcube"] >= 0})
    y_map = {sc: i for i, sc in enumerate(subcubes)}

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab20")

    for idx, rec in enumerate(records):
        if rec["subcube"] < 0:
            continue
        y = y_map[rec["subcube"]]
        start = rec["start"]
        width = rec["end"] - rec["start"]
        ax.barh(y, width, left=start, color=cmap(idx % 20), alpha=0.85, edgecolor="black", linewidth=0.3)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([f"SC{s}" for s in subcubes])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sub-Cube")
    ax.set_title("Scheduling Timeline (Gantt)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latency_distribution(latency_by_inference: Sequence[int], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not latency_by_inference:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    arr = np.array(latency_by_inference, dtype=np.float64)
    bins = min(20, max(5, int(np.sqrt(len(arr)))))
    ax.hist(arr, bins=bins, color="#5B8FF9", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_title("Per-Inference Latency Distribution")
    ax.set_xlabel("Latency (cycles)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_subcube_contention(contention_records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not contention_records:
        return

    subcubes = [int(r["subcube"]) for r in contention_records]
    waits = [float(r.get("wait_cycles", 0.0)) for r in contention_records]
    bw_util = [float(r.get("bandwidth_utilization", 0.0)) for r in contention_records]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax2 = ax1.twinx()

    ax1.bar(subcubes, waits, color="#F6BD16", alpha=0.85, label="Wait cycles")
    ax2.plot(subcubes, bw_util, color="#3AA1FF", marker="o", linewidth=1.8, label="Bandwidth util")

    ax1.set_xlabel("Sub-Cube")
    ax1.set_ylabel("Wait cycles")
    ax2.set_ylabel("Bandwidth utilization")
    ax1.set_title("Sub-Cube Contention and Bandwidth Utilization")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
