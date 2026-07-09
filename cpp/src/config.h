#ifndef CONFIG_H_
#define CONFIG_H_

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Input parameters for the bin covering problem, loaded from a JSON file.
struct Config {
  std::vector<int64_t> weights;
  int64_t interval;
  int64_t upper_bound;
  int64_t lower_bound;
  int64_t max_selections;

  static Config FromJson(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
      throw std::runtime_error("Could not open config file: " + path);
    }

    const nlohmann::json data = nlohmann::json::parse(file);
    return Config{
        data.at("weights").get<std::vector<int64_t>>(),
        data.at("interval").get<int64_t>(),
        data.at("upper_bound").get<int64_t>(),
        data.at("lower_bound").get<int64_t>(),
        data.at("max_selections").get<int64_t>(),
    };
  }
};

#endif  // CONFIG_H_
