import json
import random
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy.stats import truncnorm
from typing import Union, Tuple, List, Optional

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


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    A custom solution callback for collecting intermediate solutions from the constraint solver.

    Attributes:
        variables (list[cp_model.IntVar]): A list of integer variables to track in the solutions.
        solution_count (int): The total count of solutions found.
        solutions (list): A list to store each solution found during the solving process.
    """

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solutions = []

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        solution = {str(v): self.Value(v) for v in self.__variables}
        self.__solutions.append(solution)
        if self.__solution_count % 10000 == 0:
            print(f"Appending {self.__solution_count}-th solution")

    @property
    def solution_count(self) -> int:
        return self.__solution_count

    @property
    def solutions(self) -> list:
        return self.__solutions


class CSPSolver:
    """
    A Constraint Satisfaction Problem (CSP) solver using Google OR-Tools.

    This class encapsulates the setup and solving of a CSP for weight selection
    problems. It uses the CP-SAT solver from OR-Tools to find all feasible
    solutions within given constraints.

    Attributes:
        weight_classes (List[str]): List of unique weight classes.
        model (cp_model.CpModel): The CP-SAT model.
        solver (cp_model.CpSolver): The CP-SAT solver.
        variables (Dict[str, cp_model.IntVar]): Dictionary of decision variables.
        solution_collector (SolutionCollector): Collector for all feasible solutions.

    Example:
        >>> weights = [1, 2, 2, 3, 3, 3]
        >>> solver = CSPSolver(weights)
        >>> solver.add_range_constraint(5, 10)
        >>> solutions = solver.find_all_solutions()
    """

    def __init__(self, weights: list[int]):
        """
        Initialize the CSP solver with given weights.

        Args:
            weights (List[int]): List of integer weights to be considered in the problem.

        This method sets up the CP-SAT model, creates decision variables for each
        unique weight class, and initializes a solution collector.
        """
        classes = Counter(weights)
        self.weight_classes = [str(item) for item in classes.keys()]

        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        self.variables: dict[str, cp_model.IntVar] = {
            weight_class: self.model.NewIntVar(0, max_count, str(weight_class))
            for weight_class, max_count in classes.items()
        }

        self.solution_collector = SolutionCollector(list(self.variables.values()))
        self.solver.parameters.enumerate_all_solutions = True

    def add_range_constraint(self, lower_bound: int, upper_bound: int) -> None:
        """
        Add a range constraint to the model.

        This method adds constraints to ensure that the sum of (weight * count)
        for all weight classes falls within the specified range.

        Args:
            lower_bound (int): The minimum allowed sum.
            upper_bound (int): The maximum allowed sum.

        Raises:
            ValueError: If lower_bound is greater than upper_bound.
        """
        if lower_bound > upper_bound:
            raise ValueError("Lower bound must be less than or equal to upper bound")

        weight_sum = sum(
            variable * int(weight)
            for variable, weight in zip(self.variables.values(), self.weight_classes)
        )

        self.model.Add(weight_sum >= lower_bound)
        self.model.Add(weight_sum <= upper_bound)

    def find_all_solutions(self) -> list[dict[str, int]]:
        """
        Solve the model and return all feasible solutions.

        This method invokes the CP-SAT solver to find all solutions that satisfy
        the model's constraints. It uses the SolutionCollector to gather all
        feasible solutions.

        Returns:
            List[Dict[str, int]]: A list of dictionaries, where each dictionary
            represents a feasible solution. The keys are weight classes (as strings)
            and the values are the counts for each weight class.

        Raises:
            RuntimeError: If the solver fails or returns an unexpected status.
        """
        status = self.solver.Solve(self.model, self.solution_collector)
        match status:
            case cp_model.OPTIMAL | cp_model.FEASIBLE:
                return self.solution_collector.solutions
            case cp_model.INFEASIBLE:
                raise RuntimeError("The problem is infeasible.")
            case cp_model.MODEL_INVALID:
                raise RuntimeError("The model is invalid.")
            case _:
                raise RuntimeError(f"Solver failed with status: {status}")


def uniform_random_quantize(numbers, interval_size):
    """
    Randomly rounds to upper or lower interval with equal probability.
    """
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


def sample_with_replacement(population, sample_size):
    return [random.choice(population) for _ in range(sample_size)]


@dataclass
class Config:
    weights: list[int]
    interval: int
    upper_bound: int
    lower_bound: int
    max_selections: int

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


def load_weights(config):
    # quantize and sort weights to improve performance
    weight_values = quantize(config.weights, config.interval)
    weight_values.sort()
    return weight_values


def find_feasible_packages(weight_values, lower, upper):
    # create constraint solver to find all possible solutions
    solver = CSPSolver(weight_values)
    solver.add_range_constraint(lower, upper)
    return solver.find_all_solutions()


def find_optimal_selections(weight_values, feasible_package_classes, config):
    # create lp solver with SCIP backend.
    lp_solver = pywraplp.Solver.CreateSolver("SCIP")
    lp_solver.set_time_limit(100000)
    lp_solver.EnableOutput()

    # compute total number of times each weight value occurs.
    weight_occurrences = Counter([str(weight) for weight in weight_values])

    # Define variables.
    # Each variable represents the number of times a package class is selected.
    selections = []
    for package_class in feasible_package_classes:
        selections.append(
            (
                package_class,
                lp_solver.IntVar(0, config.max_selections, str(package_class)),
            )
        )

    # Define the constraints.
    # Observe that sum of all selections of each weight value must be bounded above by total occurences the dataset.
    # Package classes from CSP adhere to this but only for individual packages.
    # So we have to iterate over all package class selection variables and compute how many times each weight class is selected.
    # To ensure the constraint holds, we add a linear inequality for each weight class.
    for weight_class, total_occurrences in weight_occurrences.items():
        num_selections = 0
        for package_class, times_selected in selections:
            num_selections += package_class[weight_class] * times_selected
        lp_solver.Add(num_selections <= total_occurrences)

    # define the objective => maximize the number of packages selected
    lp_solver.Maximize(lp_solver.Sum([count for _, count in selections]))
    status = lp_solver.Solve()

    # print the solution.
    if status == pywraplp.Solver.OPTIMAL:
        num_bins = 0
        for package_class, count_var in selections:
            if count_var.solution_value() > 0:
                num_bins += count_var.solution_value()
                print(
                    {key: value for key, value in package_class.items() if value != 0}
                )
                print(sum(int(key) * value for key, value in package_class.items()))

        print(f"Number of bins used: {num_bins}")
        return num_bins
    else:
        print("The problem does not have an optimal solution.")



def sample_from_num_solutions(config) -> int:
    # constants are found from running MLE on sample data. target sum constant lets use get a shot at 19 portions.
    samples = generate_samples_until_sum(
        mean=119.6,
        std=16.50,
        lower_bound=80.0,
        upper_bound=160.0,
        target_sum=19 * 640 + 1.0,
    )
    weight_values = np.round(samples).astype(int).tolist()
    weight_values = uniform_random_quantize(weight_values, 3)
    feasible_package_classes = find_feasible_packages(
        weight_values, config.lower_bound, config.upper_bound
    )
    return find_optimal_selections(weight_values, feasible_package_classes, config)


def percentage_of_value(lst, target_value=19.0):
    """
    Calculate what percentage of the list equals the target value.

    Args:
        lst: List of floats
        target_value: The value to check for (default 19.0)

    Returns:
        Percentage as a float
    """
    if not lst:  # Handle empty list
        return 0.0

    count = sum(1 for x in lst if x == target_value)
    percentage = (count / len(lst)) * 100
    return percentage


def main() -> None:
    # Configuration file containing problem parameters
    filename = "sample_data.json"
    config = Config.from_json(filename)

    # Number of Monte Carlo samples to generate
    num_samples = 100
    samples = []

    # Generate samples by repeatedly solving the problem
    for _ in range(num_samples):
        num_solutions = sample_from_num_solutions(config)
        samples.append(num_solutions)

    # Calculate and display results
    target_value = 19.0  # Expected number of solutions
    matching_count = sum(1 for x in samples if x == target_value)

    print(f"Number of solutions: {num_solutions}")
    print(f"Percentage of {target_value}: {percentage_of_value(samples, target_value)}%")

    # Calculate Wilson confidence interval for the proportion
    lower, upper = wilson_ci(matching_count, num_samples, alpha=0.05)
    print(f"Wilson 95% CI: ({lower:.3f}, {upper:.3f})")


if __name__ == "__main__":
    main()
