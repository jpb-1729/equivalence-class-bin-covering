#ifndef CONFIG_H_
#define CONFIG_H_

#include <cstdint>

// Parameters for the Monte Carlo evaluation and the optimization problem.
// These defaults mirror origin/dev's Python Config.
struct Config {
  double mean = 119.6;
  double standard_deviation = 16.50;
  double sample_lower_bound = 80.0;
  double sample_upper_bound = 160.0;
  double target_portions = 19.0;

  int64_t quantization_interval = 3;
  int64_t package_lower_bound = 640;
  int64_t package_upper_bound = 680;
  int64_t max_selections_per_package_class = 10;

  double confidence_level = 0.05;
  int64_t target_value = 19;
  int64_t number_of_samples = 100;
};

#endif  // CONFIG_H_
