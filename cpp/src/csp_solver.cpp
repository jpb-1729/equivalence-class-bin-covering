#include "csp_solver.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/sorted_interval_list.h"

namespace sat = operations_research::sat;

CSPSolver::CSPSolver(const std::vector<int64_t>& weights) {
  // Sorted input means equal weights are adjacent, so counting is a linear scan.
  for (const int64_t weight : weights) {
    if (distinct_weights_.empty() || distinct_weights_.back() != weight) {
      distinct_weights_.push_back(weight);
      occurrences_.push_back(1);
    } else {
      ++occurrences_.back();
    }
  }

  variables_.reserve(distinct_weights_.size());
  for (size_t i = 0; i < distinct_weights_.size(); ++i) {
    variables_.push_back(
        model_.NewIntVar(operations_research::Domain(0, occurrences_[i]))
            .WithName(std::to_string(distinct_weights_[i])));
  }
}

void CSPSolver::AddRangeConstraint(int64_t lower_bound, int64_t upper_bound) {
  if (lower_bound > upper_bound) {
    throw std::invalid_argument(
        "Lower bound must be less than or equal to upper bound");
  }

  const sat::LinearExpr weight_sum =
      sat::LinearExpr::WeightedSum(variables_, distinct_weights_);

  model_.AddGreaterOrEqual(weight_sum, lower_bound);
  model_.AddLessOrEqual(weight_sum, upper_bound);
}

std::vector<PackageClass> CSPSolver::FindAllSolutions() {
  sat::Model model;

  sat::SatParameters parameters;
  parameters.set_enumerate_all_solutions(true);
  model.Add(sat::NewSatParameters(parameters));

  std::vector<PackageClass> solutions;
  model.Add(sat::NewFeasibleSolutionObserver(
      [&](const sat::CpSolverResponse& response) {
        PackageClass solution;
        solution.reserve(variables_.size());
        for (const sat::IntVar& variable : variables_) {
          solution.push_back(sat::SolutionIntegerValue(response, variable));
        }
        solutions.push_back(std::move(solution));

        if (solutions.size() % 10000 == 0) {
          std::cout << "Appending " << solutions.size() << "-th solution\n";
        }
      }));

  const sat::CpSolverResponse response = sat::SolveCpModel(model_.Build(), &model);
  switch (response.status()) {
    case sat::CpSolverStatus::OPTIMAL:
    case sat::CpSolverStatus::FEASIBLE:
      return solutions;
    case sat::CpSolverStatus::INFEASIBLE:
      throw std::runtime_error("The problem is infeasible.");
    case sat::CpSolverStatus::MODEL_INVALID:
      throw std::runtime_error("The model is invalid.");
    default:
      throw std::runtime_error("Solver failed with status: " +
                               sat::CpSolverStatus_Name(response.status()));
  }
}
