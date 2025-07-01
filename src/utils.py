import random
import numpy as np
from scipy.stats import truncnorm
from typing import Union, Tuple, List
from binomial import wilson_ci


def generate_samples_until_sum(
    mean: float,
    std: float,
    lower_bound: float,
    upper_bound: float,
    target_sum: float,
    max_samples: int = 10000,
    return_sum: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Generate samples from a truncated normal distribution until their sum reaches
    or exceeds a specified target sum.

    Args:
        mean: Mean of the underlying normal distribution
        std: Standard deviation of the underlying normal distribution
        lower_bound: Lower bound for truncation of individual samples
        upper_bound: Upper bound for truncation of individual samples
        target_sum: The minimum sum threshold to reach
        max_samples: Maximum number of samples to generate (prevents infinite loops)
        return_sum: If True, returns the final sum along with the samples

    Returns:
        If return_sum is False:
            NumPy array containing the generated samples
        If return_sum is True:
            Tuple containing (samples array, final sum)

    Raises:
        ValueError: If parameter configuration makes reaching the target sum impossible
        RuntimeError: If max_samples is reached without meeting the target sum
    """
    # Validate inputs
    if std <= 0:
        raise ValueError("Standard deviation must be positive")

    if lower_bound >= upper_bound:
        raise ValueError("Lower bound must be less than upper bound")

    # Check if it's mathematically possible to reach the target sum
    if lower_bound <= 0 and target_sum > 0:
        # With negative values possible, we might never reach the target
        print(
            "Warning: Lower bound <= 0 means negative samples are possible, "
            "which could make reaching the target sum difficult"
        )

    # Initialize
    samples = []
    current_sum = 0
    count = 0

    # Generate samples until reaching the target sum
    while current_sum < target_sum and count < max_samples:
        # Calculate standardized bounds
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std

        # Generate a single sample
        new_sample = float(truncnorm.rvs(a, b, loc=mean, scale=std, size=1))

        samples.append(new_sample)
        current_sum += new_sample
        count += 1

    # Check if we hit the maximum without reaching the target
    if count >= max_samples and current_sum < target_sum:
        raise RuntimeError(
            f"Failed to reach the target sum of {target_sum} after generating "
            f"{max_samples} samples. Current sum: {current_sum:.2f}"
        )

    # Convert to numpy array
    samples_array = np.array(samples)

    if return_sum:
        return samples_array, current_sum
    else:
        return samples_array


def uniform_random_quantize(numbers: List[float], interval_size: float) -> List[float]:
    """Randomly rounds to upper or lower interval with equal probability."""
    if interval_size == 0:
        raise ValueError("interval_size cannot be zero")

    quantized = []
    for num in numbers:
        lower = (num // interval_size) * interval_size
        upper = lower + interval_size

        # 50% chance of rounding up or down
        if random.random() < 0.5:
            quantized.append(upper)
        else:
            quantized.append(lower)

    return quantized


def sample_with_replacement(population: List, sample_size: int) -> List:
    """Simple random sampling with replacement."""
    return [random.choice(population) for _ in range(sample_size)]


def percentage_of_value(lst: List[float], target_value: float = 19.0) -> float:
    """Calculate what percentage of the list equals the target value."""
    if not lst:
        return 0.0

    count = sum(1 for x in lst if x == target_value)
    percentage = (count / len(lst)) * 100
    return percentage


def calculate_wilson_ci(matching_count: int, total_samples: int, alpha: float = 0.05):
    """Calculate Wilson confidence interval for proportion."""
    return wilson_ci(matching_count, total_samples, alpha)
