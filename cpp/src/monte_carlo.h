#ifndef MONTE_CARLO_H_
#define MONTE_CARLO_H_

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

// Draws from a doubly truncated normal distribution until target_sum is met.
std::vector<double> GenerateSamplesUntilSum(
    double mean, double standard_deviation, double lower_bound,
    double upper_bound, double target_sum, std::mt19937_64& generator,
    int64_t max_samples = 10000);

// Randomly rounds each value to the interval immediately above or below it.
std::vector<int64_t> UniformRandomQuantize(
    const std::vector<int64_t>& numbers, int64_t interval_size,
    std::mt19937_64& generator);

double PercentageOfValue(const std::vector<int64_t>& values,
                         int64_t target_value);

// Returns a two-sided Wilson score interval for a binomial proportion.
std::pair<double, double> WilsonConfidenceInterval(int64_t successes,
                                                   int64_t trials,
                                                   double alpha = 0.05);

#endif  // MONTE_CARLO_H_
