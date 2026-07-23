"""Run the bin-covering solver on a fixed set of measured weights."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.solver import find_feasible_packages, find_optimal_selections


DEFAULT_DATA_PATH = Path(__file__).with_name("sample_data.json")


@dataclass(frozen=True)
class DemoConfig:
    """Configuration stored alongside the real-world measurements."""

    weights: list[int]
    interval: int
    upper_bound: int
    lower_bound: int
    max_selections: int

    @property
    def max_selections_per_package_class(self) -> int:
        """Adapt the demo configuration to the current solver interface."""
        return self.max_selections

    @classmethod
    def from_json(cls, path: Path) -> "DemoConfig":
        with path.open(encoding="utf-8") as data_file:
            config = cls(**json.load(data_file))
        config.validate()
        return config

    def validate(self) -> None:
        if not self.weights:
            raise ValueError("weights must not be empty")
        if any(type(weight) is not int or weight <= 0 for weight in self.weights):
            raise ValueError("weights must contain only positive integers")
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        if self.max_selections <= 0:
            raise ValueError("max_selections must be positive")


def quantize_weights(weights: list[int], interval: int) -> list[int]:
    """Round measurements down to their equivalence-class boundaries."""
    return sorted(weight // interval * interval for weight in weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the fixed real-world measurement dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"path to the dataset (default: {DEFAULT_DATA_PATH.name})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = DemoConfig.from_json(args.data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(f"error: could not load {args.data}: {error}") from error

    weight_values = quantize_weights(config.weights, config.interval)
    print(
        f"Loaded {len(weight_values)} measured weights from {args.data} "
        f"({len(set(weight_values))} equivalence classes)."
    )
    print(
        "Enumerating package classes with total weight in "
        f"[{config.lower_bound}, {config.upper_bound}]..."
    )
    feasible_package_classes = find_feasible_packages(
        weight_values, config.lower_bound, config.upper_bound
    )
    print(f"Found {len(feasible_package_classes)} feasible package classes.")

    result = find_optimal_selections(weight_values, feasible_package_classes, config)
    if result is None:
        raise SystemExit("error: the dataset did not produce an optimal solution")


if __name__ == "__main__":
    main()
