
## Introduction

This project is a personal implementation of a bin covering solver based on the formulation presented in the paper: *[On the Use of Equivalence Classes for Optimal and Sub-Optimal Bin Packing and Bin Covering](https://doi.org/10.1109/TASE.2020.3022986)* by Roselli, Hagebring, Riazi, Fabian, and Åkesson.

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jpb-1729/equivalence-class-bin-covering.git
   cd equivalence-class-bin-covering
   ```

2. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install project dependencies:**
   ```bash
   poetry install
   ```

## Usage

After installation, you can run the program with the following command:

```bash
poetry run python src/main.py
```

This will execute the program and solve the bin covering problem using the data provided in `sample_data.json`.

## Configuration

The project uses a JSON configuration file (`sample_data.json`) to provide the necessary input data for solving the bin packing/covering problems. The following fields should be included in the file:

- **weights**: A list of integer weights representing the values of items.
- **interval**: The quantization interval used for grouping weights into equivalence classes.
- **upper_bound**: The maximum total value allowed in each bin for bin packing.
- **lower_bound**: The minimum total value required in each bin for bin covering.
- **max_selections**: The maximum number of times a package class can be selected in the optimization process.

### Example `sample_data.json`:

```json
{
    "weights": [10, 20, 30, 40, 50],
    "interval": 1,
    "upper_bound": 100,
    "lower_bound": 50,
    "max_selections": 10
}
```

### Citation:
Roselli, S., Hagebring, F., Riazi, S., Fabian, M., Åkesson, K. (2021). On the Use of Equivalence Classes for Optimal and Sub-Optimal Bin Packing and Bin Covering. *IEEE Transactions on Automation Science and Engineering*, 18(1), 369-381. [DOI: 10.1109/TASE.2020.3022986](https://doi.org/10.1109/TASE.2020.3022986).

## License

This project is licensed under the [Apache License 2.0](./LICENSE). Please see the `LICENSE` file for more details.
