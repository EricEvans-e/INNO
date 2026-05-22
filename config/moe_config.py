from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class MoEConfig:
    """MoE-oriented optimization knobs for mapping and scheduling."""

    enable_moe_optimization: bool = True
    enable_adaptive_replication: bool = True
    hot_expert_quantile: float = 0.2
    max_replication: int = 3
    replication_area_limit: int = 4_194_304
    conflict_weight: float = 1.0
    capacity_weight: float = 0.2
    latency_weight: float = 1.0
    enable_group_local_search: bool = True
    local_search_restarts: int = 6
    local_search_max_iters: int = 50
    enable_simulated_annealing: bool = True
    sa_steps: int = 120
    sa_init_temp: float = 1.5
    sa_cooling: float = 0.96
    enable_parallel_search: bool = True
    parallel_workers: int = 4
    parallel_trials: int = 10
    random_seed: int = 2026
    grouping_multi_start_trials: int = 6
    enable_exact_hot_subgraph: bool = True
    hot_subgraph_top_k: int = 6
    enable_dynamic_hot_subgraph_top_k: bool = True
    dynamic_hot_subgraph_min_ratio: float = 0.15
    dynamic_hot_subgraph_max_ratio: float = 0.30
    replication_volume_budget_ratio: float = 0.25
    enable_shared_operator_replication: bool = True
    shared_replication_operator_types: tuple[str, ...] = ("linear",)
    shared_replication_min_volume: int = 16_777_216
    shared_replication_max_replicas: int = 2
    replication_freq_weight: float = 0.55
    replication_contention_weight: float = 0.2
    replication_burst_weight: float = 0.15
    replication_transition_weight: float = 0.1
    replica_pressure_low_threshold: float = 0.52
    replica_pressure_high_threshold: float = 0.82
    placement_conflict_weight: float = 0.8
    transition_conflict_weight: float = 0.25
    placement_load_weight: float = 0.25
    placement_group_penalty: float = 0.15
    replica_diversity_penalty: float = 0.2
    enable_aspect_aware_packing: bool = True
    aspect_aware_weight: float = 0.45
    fragmentation_penalty_weight: float = 0.20
    conflict_propagation_weight: float = 0.35
    capacity_peak_weight: float = 0.20
    enable_blockwise_quantization: bool = True
    enable_sparse_compression: bool = True
    hot_quant_bits: int = 8
    cold_quant_bits: int = 6
    hot_sparsity_ratio: float = 0.15
    cold_sparsity_ratio: float = 0.30

    def validate(self) -> None:
        if not (0.0 <= self.hot_expert_quantile <= 1.0):
            raise ValueError("hot_expert_quantile must be in [0, 1]")
        if self.max_replication < 1:
            raise ValueError("max_replication must be >= 1")
        if self.replication_area_limit <= 0:
            raise ValueError("replication_area_limit must be positive")
        if self.local_search_restarts < 1:
            raise ValueError("local_search_restarts must be >= 1")
        if self.local_search_max_iters < 1:
            raise ValueError("local_search_max_iters must be >= 1")
        if self.sa_steps < 1:
            raise ValueError("sa_steps must be >= 1")
        if self.sa_init_temp <= 0:
            raise ValueError("sa_init_temp must be > 0")
        if not (0.0 < self.sa_cooling < 1.0):
            raise ValueError("sa_cooling must be in (0, 1)")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be >= 1")
        if self.parallel_trials < 1:
            raise ValueError("parallel_trials must be >= 1")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0")
        if self.grouping_multi_start_trials < 1:
            raise ValueError("grouping_multi_start_trials must be >= 1")
        if self.hot_subgraph_top_k < 0:
            raise ValueError("hot_subgraph_top_k must be >= 0")
        if not (0.0 < self.dynamic_hot_subgraph_min_ratio <= 1.0):
            raise ValueError("dynamic_hot_subgraph_min_ratio must be in (0, 1]")
        if not (0.0 < self.dynamic_hot_subgraph_max_ratio <= 1.0):
            raise ValueError("dynamic_hot_subgraph_max_ratio must be in (0, 1]")
        if self.dynamic_hot_subgraph_min_ratio > self.dynamic_hot_subgraph_max_ratio:
            raise ValueError("dynamic_hot_subgraph_min_ratio must be <= dynamic_hot_subgraph_max_ratio")
        if not (0.0 <= self.replication_volume_budget_ratio <= 1.0):
            raise ValueError("replication_volume_budget_ratio must be in [0, 1]")
        if self.shared_replication_min_volume <= 0:
            raise ValueError("shared_replication_min_volume must be positive")
        if self.shared_replication_max_replicas < 1:
            raise ValueError("shared_replication_max_replicas must be >= 1")
        if not self.shared_replication_operator_types:
            raise ValueError("shared_replication_operator_types must not be empty")
        if self.replica_pressure_low_threshold < 0:
            raise ValueError("replica_pressure_low_threshold must be >= 0")
        if self.replica_pressure_high_threshold < 0:
            raise ValueError("replica_pressure_high_threshold must be >= 0")
        if self.replica_pressure_low_threshold > self.replica_pressure_high_threshold:
            raise ValueError("replica_pressure_low_threshold must be <= replica_pressure_high_threshold")
        for name, v in [
            ("replication_freq_weight", self.replication_freq_weight),
            ("replication_contention_weight", self.replication_contention_weight),
            ("replication_burst_weight", self.replication_burst_weight),
            ("replication_transition_weight", self.replication_transition_weight),
            ("placement_conflict_weight", self.placement_conflict_weight),
            ("transition_conflict_weight", self.transition_conflict_weight),
            ("placement_load_weight", self.placement_load_weight),
            ("placement_group_penalty", self.placement_group_penalty),
            ("replica_diversity_penalty", self.replica_diversity_penalty),
            ("aspect_aware_weight", self.aspect_aware_weight),
            ("fragmentation_penalty_weight", self.fragmentation_penalty_weight),
            ("conflict_propagation_weight", self.conflict_propagation_weight),
            ("capacity_peak_weight", self.capacity_peak_weight),
        ]:
            if v < 0:
                raise ValueError(f"{name} must be >= 0")
        if (
            self.replication_freq_weight
            + self.replication_contention_weight
            + self.replication_burst_weight
            + self.replication_transition_weight
            <= 0
        ):
            raise ValueError("replication pressure weights sum must be > 0")
        if self.hot_quant_bits < 2 or self.hot_quant_bits > 8:
            raise ValueError("hot_quant_bits must be in [2, 8]")
        if self.cold_quant_bits < 2 or self.cold_quant_bits > 8:
            raise ValueError("cold_quant_bits must be in [2, 8]")
        if not (0.0 <= self.hot_sparsity_ratio < 1.0):
            raise ValueError("hot_sparsity_ratio must be in [0, 1)")
        if not (0.0 <= self.cold_sparsity_ratio < 1.0):
            raise ValueError("cold_sparsity_ratio must be in [0, 1)")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_MOE_CONFIG = MoEConfig()
