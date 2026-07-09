#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "csp_solver.h"
#include "ortools/linear_solver/linear_solver.h"

namespace mp = operations_research;

// Rounds each number down to the nearest multiple of interval_size.
std::vector<int64_t> Quantize(const std::vector<int64_t>& numbers,
                              int64_t interval_size) {
  std::vector<int64_t> quantized;
  quantized.reserve(numbers.size());
  for (const int64_t number : numbers) {
    quantized.push_back(number / interval_size * interval_size);
  }
  return quantized;
}

// Quantizes and sorts the weights, which improves solver performance.
std::vector<int64_t> LoadWeights(const Config& config) {
  std::vector<int64_t> weight_values = Quantize(config.weights, config.interval);
  std::sort(weight_values.begin(), weight_values.end());
  return weight_values;
}

// Selects package classes to maximize the number of packages built, without
// using more items of any weight than the dataset actually contains. Each
// package class from the CSP respects the weight counts on its own, but a class
// may be selected many times, so the limit is enforced across all selections.
void FindOptimalSelections(const CSPSolver& csp_solver,
                           const std::vector<PackageClass>& package_classes,
                           const Config& config) {
  std::unique_ptr<mp::MPSolver> solver(mp::MPSolver::CreateSolver("SCIP"));
  if (solver == nullptr) {
    throw std::runtime_error("Could not create SCIP solver.");
  }
  solver->EnableOutput();

  // Each variable is the number of times a package class is selected.
  std::vector<mp::MPVariable*> selections;
  selections.reserve(package_classes.size());
  for (size_t i = 0; i < package_classes.size(); ++i) {
    selections.push_back(solver->MakeIntVar(
        0, static_cast<double>(config.max_selections), "class_" + std::to_string(i)));
  }

  const std::vector<int64_t>& distinct_weights = csp_solver.distinct_weights();
  const std::vector<int64_t>& occurrences = csp_solver.occurrences();

  for (size_t weight_index = 0; weight_index < distinct_weights.size();
       ++weight_index) {
    mp::MPConstraint* const constraint = solver->MakeRowConstraint(
        0, static_cast<double>(occurrences[weight_index]));
    for (size_t class_index = 0; class_index < package_classes.size();
         ++class_index) {
      constraint->SetCoefficient(
          selections[class_index],
          static_cast<double>(package_classes[class_index][weight_index]));
    }
  }

  // Maximize the number of packages selected.
  mp::MPObjective* const objective = solver->MutableObjective();
  for (mp::MPVariable* const selection : selections) {
    objective->SetCoefficient(selection, 1);
  }
  objective->SetMaximization();

  if (solver->Solve() != mp::MPSolver::OPTIMAL) {
    return;
  }

  for (size_t class_index = 0; class_index < package_classes.size();
       ++class_index) {
    if (selections[class_index]->solution_value() <= 0) {
      continue;
    }

    const PackageClass& package_class = package_classes[class_index];
    std::cout << "{";
    bool first = true;
    for (size_t i = 0; i < distinct_weights.size(); ++i) {
      if (package_class[i] == 0) {
        continue;
      }
      if (!first) {
        std::cout << ", ";
      }
      std::cout << "'" << distinct_weights[i] << "': " << package_class[i];
      first = false;
    }
    std::cout << "}\n";
  }
}

int main() {
  try {
    const Config config = Config::FromJson("sample_data.json");

    const std::vector<int64_t> weight_values = LoadWeights(config);

    // Create constraint solver to find all possible packages.
    CSPSolver csp_solver(weight_values);
    csp_solver.AddRangeConstraint(config.lower_bound, config.upper_bound);
    const std::vector<PackageClass> feasible_package_classes =
        csp_solver.FindAllSolutions();

    FindOptimalSelections(csp_solver, feasible_package_classes, config);
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}
