from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class CubeConfig:
    """Static 3D CIM resource definition used by mapping and simulation."""

    n: int = 3
    d: int = 8
    h: int = 4096
    w: int = 4096
    max_parallel_subcubes: int = 9
    switching_penalty: int = 3
    z_access_penalty: int = 1

    def validate(self) -> None:
        if not (2 <= self.n <= 4):
            raise ValueError(f"N must be in [2, 4], got {self.n}")
        if not (4096 <= self.h <= 16384):
            raise ValueError(f"H must be in [4096, 16384], got {self.h}")
        if not (4096 <= self.w <= 16384):
            raise ValueError(f"W must be in [4096, 16384], got {self.w}")
        if self.d <= 0:
            raise ValueError("D must be positive")
        if self.max_parallel_subcubes <= 0:
            raise ValueError("max_parallel_subcubes must be positive")

    @property
    def num_subcubes(self) -> int:
        return self.n * self.n

    @property
    def subcube_volume(self) -> int:
        return self.d * self.h * self.w

    @property
    def total_volume(self) -> int:
        return self.num_subcubes * self.subcube_volume

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


DEFAULT_CUBE_CONFIG = CubeConfig()
