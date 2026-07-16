#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "csp_solver.h"
#include "monte_carlo.h"
#include "ortools/linear_solver/linear_solver.h"

namespace mp = operations_research;

int64_t FindOptimalSelections(const CSPSolver& csp_solver,
                              const std::vector<PackageClass>& package_classes,
                              const Config& config) {
  std::unique_ptr<mp::MPSolver> solver(mp::MPSolver::CreateSolver("SCIP"));
  if (solver == nullptr) {
    throw std::runtime_error("Could not create SCIP solver");
  }
  solver->set_time_limit(100000);

  std::vector<mp::MPVariable*> selections;
  selections.reserve(package_classes.size());
  for (size_t i = 0; i < package_classes.size(); ++i) {
    selections.push_back(solver->MakeIntVar(
        0, static_cast<double>(config.max_selections_per_package_class),
        "class_" + std::to_string(i)));
  }

  const std::vector<int64_t>& occurrences = csp_solver.occurrences();
  for (size_t weight = 0; weight < occurrences.size(); ++weight) {
    mp::MPConstraint* constraint = solver->MakeRowConstraint(
        0, static_cast<double>(occurrences[weight]));
    for (size_t package = 0; package < package_classes.size(); ++package) {
      constraint->SetCoefficient(
          selections[package],
          static_cast<double>(package_classes[package][weight]));
    }
  }

  mp::MPObjective* objective = solver->MutableObjective();
  for (mp::MPVariable* selection : selections) {
    objective->SetCoefficient(selection, 1.0);
  }
  objective->SetMaximization();

  if (solver->Solve() != mp::MPSolver::OPTIMAL) {
    throw std::runtime_error("The optimization problem has no optimal solution");
  }
  return static_cast<int64_t>(std::llround(objective->Value()));
}

struct TrialResult {
  int64_t number_of_bins;
  double solve_time_seconds;
};

TrialResult RunTrial(const Config& config, std::mt19937_64& generator) {
  const std::vector<double> samples = GenerateSamplesUntilSum(
      config.mean, config.standard_deviation, config.sample_lower_bound,
      config.sample_upper_bound,
      config.target_portions * config.package_lower_bound, generator);

  std::vector<int64_t> rounded;
  rounded.reserve(samples.size());
  for (const double sample : samples) {
    rounded.push_back(static_cast<int64_t>(std::llround(sample)));
  }
  std::vector<int64_t> weights = UniformRandomQuantize(
      rounded, config.quantization_interval, generator);
  std::sort(weights.begin(), weights.end());

  const auto solve_start = std::chrono::steady_clock::now();
  CSPSolver csp_solver(weights);
  csp_solver.AddRangeConstraint(config.package_lower_bound,
                                config.package_upper_bound);
  const std::vector<PackageClass> packages = csp_solver.FindAllSolutions();
  const int64_t number_of_bins =
      FindOptimalSelections(csp_solver, packages, config);
  const auto solve_end = std::chrono::steady_clock::now();
  const double solve_time_seconds =
      std::chrono::duration<double>(solve_end - solve_start).count();
  return {number_of_bins, solve_time_seconds};
}

int main(int argc, char* argv[]) {
  try {
    Config config;
    if (argc > 2) {
      throw std::invalid_argument("Usage: bin_covering [number_of_trials]");
    }
    if (argc == 2) {
      config.number_of_samples = std::stoll(argv[1]);
    }
    if (config.number_of_samples <= 0) {
      throw std::invalid_argument("Number of samples must be positive");
    }

    std::random_device random_device;
    std::mt19937_64 generator(random_device());
    std::vector<int64_t> results;
    std::vector<double> solve_times;
    results.reserve(static_cast<size_t>(config.number_of_samples));
    solve_times.reserve(static_cast<size_t>(config.number_of_samples));
    for (int64_t i = 0; i < config.number_of_samples; ++i) {
      const TrialResult trial = RunTrial(config, generator);
      results.push_back(trial.number_of_bins);
      solve_times.push_back(trial.solve_time_seconds);
      std::cout << "Completed trial " << i + 1 << "/"
                << config.number_of_samples << " in " << std::fixed
                << std::setprecision(3) << trial.solve_time_seconds << " s\n";
    }

    const int64_t matches = static_cast<int64_t>(
        std::count(results.begin(), results.end(), config.target_value));
    std::cout << "Percentage of " << config.target_value << ": "
              << PercentageOfValue(results, config.target_value) << "%\n";

    const auto [lower, upper] = WilsonConfidenceInterval(
        matches, config.number_of_samples, config.confidence_level);
    std::cout << std::fixed << std::setprecision(3) << "Wilson "
              << (1.0 - config.confidence_level) * 100.0 << "% CI: (" << lower
              << ", " << upper << ")\n";

    const auto [minimum, maximum] =
        std::minmax_element(solve_times.begin(), solve_times.end());
    const double total =
        std::accumulate(solve_times.begin(), solve_times.end(), 0.0);
    std::cout << "Solve time (seconds): average="
              << total / static_cast<double>(solve_times.size())
              << ", min=" << *minimum << ", max=" << *maximum << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}
