#ifndef CSP_SOLVER_H_
#define CSP_SOLVER_H_

#include <cstdint>
#include <vector>

#include "ortools/sat/cp_model.h"

// Counts of each distinct weight, index-aligned with
// CSPSolver::distinct_weights().
using PackageClass = std::vector<int64_t>;

// Enumerates every combination of weight counts whose weighted sum falls within
// a given range, using the CP-SAT solver.
class CSPSolver {
 public:
  // `weights` must be sorted ascending.
  explicit CSPSolver(const std::vector<int64_t>& weights);

  // Constrains the weighted sum of the selected counts to [lower, upper].
  void AddRangeConstraint(int64_t lower_bound, int64_t upper_bound);

  // Returns every combination satisfying the constraints.
  std::vector<PackageClass> FindAllSolutions();

  const std::vector<int64_t>& distinct_weights() const {
    return distinct_weights_;
  }

  // Number of times each distinct weight occurs in the input.
  const std::vector<int64_t>& occurrences() const { return occurrences_; }

 private:
  std::vector<int64_t> distinct_weights_;
  std::vector<int64_t> occurrences_;
  operations_research::sat::CpModelBuilder model_;
  std::vector<operations_research::sat::IntVar> variables_;
};

#endif  // CSP_SOLVER_H_
