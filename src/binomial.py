import numpy as np
from scipy.stats import norm


def wilson_ci(x, n, alpha=0.05):
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    x : int
        Number of observed successes.
    n : int
        Number of trials.
    alpha : float, optional
        Significance level for the confidence interval (default is 0.05 for 95% CI).

    Returns
    -------
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.

    Notes
    -----
    - The Wilson interval has better coverage than the standard normal (Wald) interval, especially for small n or proportions near 0 or 1.
    - The returned interval is always within [0, 1].
    """
    # Input validation
    if n <= 0:
        raise ValueError("Number of trials 'n' must be positive.")
    if not (0 <= x <= n):
        raise ValueError("Number of successes 'x' must be between 0 and n.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")

    # Sample proportion
    p_hat = x / n
    # Z-score for two-tailed interval
    z = norm.ppf(1 - alpha / 2)
    # Wilson denominator
    denom = 1 + z**2 / n
    # Center of the interval (adjusted proportion)
    center = (p_hat + z**2 / (2 * n)) / denom
    # Half-width of the interval
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    # Ensure bounds are within [0, 1]
    lower = max(0, center - half)
    upper = min(1, center + half)
    return lower, upper


# Example usage
if __name__ == "__main__":
    # Simulate binomial data: n=20 trials, p=0.95 probability
    n = 100
    p = 0.95
    x = np.random.binomial(n, p)
    lower, upper = wilson_ci(x, n)
    print(f"Observed {x} successes out of {n} trials")
    print(f"Wilson 95% CI: ({lower:.3f}, {upper:.3f})")
