import argparse
import random
from dataclasses import replace

import numpy as np

from utils import (
    uniform_random_quantize,
    calculate_wilson_ci,
    percentage_of_value,
    generate_samples_until_sum,
)
from solver import find_feasible_packages, find_optimal_selections
from config import Config


def sample_from_num_solutions(config: Config) -> int:
    samples = generate_samples_until_sum(
        mean=config.mean,
        std=config.std,
        lower_bound=config.lower_bound,
        upper_bound=config.upper_bound,
        target_sum=config.target_portions * config.package_lower_bound,
    )
    weight_values = np.round(samples).astype(int).tolist()
    weight_values = uniform_random_quantize(weight_values, config.quantization_interval)
    feasible_package_classes = find_feasible_packages(
        weight_values, config.package_lower_bound, config.package_upper_bound
    )
    return find_optimal_selections(weight_values, feasible_package_classes, config)


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Estimate bin-covering outcomes with Monte Carlo trials."
    )
    parser.add_argument("--trials", type=int, default=defaults.number_of_samples)
    parser.add_argument("--seed", type=int, help="Seed both Python and NumPy RNGs")
    parser.add_argument("--mean", type=float, default=defaults.mean)
    parser.add_argument("--std", type=float, default=defaults.std)
    parser.add_argument(
        "--sample-lower-bound", type=float, default=defaults.lower_bound
    )
    parser.add_argument(
        "--sample-upper-bound", type=float, default=defaults.upper_bound
    )
    parser.add_argument(
        "--target-portions", type=float, default=defaults.target_portions
    )
    parser.add_argument(
        "--quantization-interval",
        type=int,
        default=defaults.quantization_interval,
    )
    parser.add_argument(
        "--package-lower-bound",
        type=int,
        default=defaults.package_lower_bound,
    )
    parser.add_argument(
        "--package-upper-bound",
        type=int,
        default=defaults.package_upper_bound,
    )
    parser.add_argument(
        "--max-selections",
        type=int,
        default=defaults.max_selections_per_package_class,
    )
    parser.add_argument("--target-value", type=float, default=defaults.target_value)
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.confidence_level,
        help="Significance level; 0.05 produces a 95%% confidence interval",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.std <= 0:
        raise ValueError("--std must be positive")
    if args.sample_lower_bound >= args.sample_upper_bound:
        raise ValueError("--sample-lower-bound must be less than --sample-upper-bound")
    if args.quantization_interval <= 0:
        raise ValueError("--quantization-interval must be positive")
    if args.package_lower_bound > args.package_upper_bound:
        raise ValueError("--package-lower-bound must not exceed --package-upper-bound")
    if args.max_selections <= 0:
        raise ValueError("--max-selections must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")

    return replace(
        Config(),
        mean=args.mean,
        std=args.std,
        lower_bound=args.sample_lower_bound,
        upper_bound=args.sample_upper_bound,
        target_portions=args.target_portions,
        quantization_interval=args.quantization_interval,
        package_lower_bound=args.package_lower_bound,
        package_upper_bound=args.package_upper_bound,
        max_selections_per_package_class=args.max_selections,
        confidence_level=args.alpha,
        target_value=args.target_value,
        number_of_samples=args.trials,
    )


def main() -> None:
    args = parse_args()
    try:
        config = config_from_args(args)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Generate samples by repeatedly solving the problem
    samples = []
    for _ in range(config.number_of_samples):
        num_solutions = sample_from_num_solutions(config)
        samples.append(num_solutions)

    # Calculate and display results
    matching_count = sum(1 for x in samples if x == config.target_value)

    print(
        f"Percentage of {config.target_value}: {percentage_of_value(samples, config.target_value)}%"
    )

    # Calculate Wilson confidence interval for the proportion
    lower, upper = calculate_wilson_ci(
        matching_count, config.number_of_samples, alpha=config.confidence_level
    )
    confidence = (1 - config.confidence_level) * 100
    print(f"Wilson {confidence:g}% CI: ({lower:.3f}, {upper:.3f})")


if __name__ == "__main__":
    main()
