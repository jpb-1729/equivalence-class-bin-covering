import json
import random
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from collections import Counter
from dataclasses import dataclass


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


def quantize(numbers, interval_size):
    # Quantizing each number to the nearest multiple of interval_size
    quantized = [num // interval_size * interval_size for num in numbers]
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


def main() -> None:
    # load config from file.
    filename = "sample_data.json"
    config = Config.from_json(filename)

    weight_values = load_weights(config)
    feasible_package_classes = find_feasible_packages(
        weight_values, config.lower_bound, config.upper_bound
    )
    find_optimal_selections(weight_values, feasible_package_classes, config)


if __name__ == "__main__":
    main()
