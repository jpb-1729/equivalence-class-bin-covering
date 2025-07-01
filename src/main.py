
import numpy as np

from utils import uniform_random_quantize, calculate_wilson_ci, percentage_of_value, generate_samples_until_sum
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


def main() -> None:
    config = Config()

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
    print(f"Wilson 95% CI: ({lower:.3f}, {upper:.3f})")


if __name__ == "__main__":
    main()
