from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Config:
    """Parameters for generating sample weights."""

    mean: Final[float] = 119.6
    std: Final[float] = 16.50
    lower_bound: Final[float] = 80.0
    upper_bound: Final[float] = 160.0
    target_portions: Final[float] = 19.0

    """Parameters defining the optimization problem."""
    quantization_interval: Final[int] = 3
    package_lower_bound: Final[int] = 640
    package_upper_bound: Final[int] = 680
    max_selections_per_package_class: Final[int] = 10

    """Parameters for the Wilson confidence interval estimation."""
    confidence_level: Final[float] = 0.05
    target_value: Final[float] = 19.0
    number_of_samples: Final[int] = 1
