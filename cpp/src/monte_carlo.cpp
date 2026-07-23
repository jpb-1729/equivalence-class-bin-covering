#include "monte_carlo.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// Peter J. Acklam's rational approximation of the inverse normal CDF.
double InverseNormalCdf(double probability) {
  if (probability <= 0.0 || probability >= 1.0) {
    throw std::invalid_argument("Probability must be between 0 and 1");
  }

  constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                          -2.759285104469687e+02, 1.383577518672690e+02,
                          -3.066479806614716e+01, 2.506628277459239e+00};
  constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                          -1.556989798598866e+02, 6.680131188771972e+01,
                          -1.328068155288572e+01};
  constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                          -2.400758277161838e+00, -2.549732539343734e+00,
                          4.374664141464968e+00, 2.938163982698783e+00};
  constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                          2.445134137142996e+00, 3.754408661907416e+00};
  constexpr double lower_tail = 0.02425;
  constexpr double upper_tail = 1.0 - lower_tail;

  if (probability < lower_tail) {
    const double q = std::sqrt(-2.0 * std::log(probability));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
            c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (probability > upper_tail) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
             c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }

  const double q = probability - 0.5;
  const double r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
          a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r +
          1.0);
}

}  // namespace

std::vector<double> GenerateSamplesUntilSum(
    double mean, double standard_deviation, double lower_bound,
    double upper_bound, double target_sum, std::mt19937_64& generator,
    int64_t max_samples) {
  if (standard_deviation <= 0.0) {
    throw std::invalid_argument("Standard deviation must be positive");
  }
  if (lower_bound >= upper_bound) {
    throw std::invalid_argument("Lower bound must be less than upper bound");
  }
  if (max_samples <= 0) {
    throw std::invalid_argument("Maximum sample count must be positive");
  }

  std::normal_distribution<double> distribution(mean, standard_deviation);
  std::vector<double> samples;
  double sum = 0.0;
  while (sum < target_sum &&
         samples.size() < static_cast<size_t>(max_samples)) {
    double sample;
    do {
      sample = distribution(generator);
    } while (sample < lower_bound || sample > upper_bound);
    samples.push_back(sample);
    sum += sample;
  }

  if (sum < target_sum) {
    throw std::runtime_error("Failed to reach target sum before sample limit");
  }
  return samples;
}

std::vector<int64_t> UniformRandomQuantize(
    const std::vector<int64_t>& numbers, int64_t interval_size,
    std::mt19937_64& generator) {
  if (interval_size <= 0) {
    throw std::invalid_argument("Quantization interval must be positive");
  }

  std::bernoulli_distribution round_up(0.5);
  std::vector<int64_t> quantized;
  quantized.reserve(numbers.size());
  for (const int64_t number : numbers) {
    const int64_t lower = number / interval_size * interval_size;
    quantized.push_back(lower + (round_up(generator) ? interval_size : 0));
  }
  return quantized;
}

double PercentageOfValue(const std::vector<int64_t>& values,
                         int64_t target_value) {
  if (values.empty()) {
    return 0.0;
  }
  const auto matches = std::count(values.begin(), values.end(), target_value);
  return 100.0 * static_cast<double>(matches) /
         static_cast<double>(values.size());
}

std::pair<double, double> WilsonConfidenceInterval(int64_t successes,
                                                   int64_t trials,
                                                   double alpha) {
  if (trials <= 0) {
    throw std::invalid_argument("Number of trials must be positive");
  }
  if (successes < 0 || successes > trials) {
    throw std::invalid_argument("Successes must be between zero and trials");
  }
  if (alpha <= 0.0 || alpha >= 1.0) {
    throw std::invalid_argument("Alpha must be between 0 and 1");
  }

  const double n = static_cast<double>(trials);
  const double p = static_cast<double>(successes) / n;
  const double z = InverseNormalCdf(1.0 - alpha / 2.0);
  const double z_squared = z * z;
  const double denominator = 1.0 + z_squared / n;
  const double center = (p + z_squared / (2.0 * n)) / denominator;
  const double half_width =
      z / denominator *
      std::sqrt(p * (1.0 - p) / n + z_squared / (4.0 * n * n));
  return {std::max(0.0, center - half_width),
          std::min(1.0, center + half_width)};
}
