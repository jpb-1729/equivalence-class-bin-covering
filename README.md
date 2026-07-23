# Equivalence-Class Bin Covering

This project estimates how often a bin-covering solver reaches a target number
of bins. Each Monte Carlo trial:

1. draws item weights from a truncated normal distribution until their total
   reaches the configured target;
2. randomly quantizes those weights into equivalence classes;
3. enumerates feasible package classes with OR-Tools CP-SAT;
4. uses SCIP to maximize the number of packages without reusing items; and
5. records whether the optimum matches the target value.

After all trials, the program reports the matching percentage and a Wilson
confidence interval for that proportion.

The formulation is based on Roselli et al., [*On the Use of Equivalence Classes
for Optimal and Sub-Optimal Bin Packing and Bin
Covering*](https://doi.org/10.1109/TASE.2020.3022986).

## Requirements

- Python 3.12 or newer
- [Poetry](https://python-poetry.org/docs/#installation)

Install the locked dependencies from the repository root:

```bash
poetry install
```

If Poetry selects the wrong Python version, point it at a Python 3.12
interpreter first:

```bash
poetry env use /path/to/python3.12
poetry install
```

## CLI usage

Run one trial with the defaults:

```bash
poetry run python src/main.py
```

Run 10 reproducible trials:

```bash
poetry run python src/main.py --trials 10 --seed 42
```

Show every available option and its current default:

```bash
poetry run python src/main.py --help
```

Common options:

| Option | Meaning | Default |
| --- | --- | ---: |
| `--trials` | Number of independent Monte Carlo trials | `1` |
| `--seed` | Seed for reproducible sampling and quantization | random |
| `--quantization-interval` | Width of each weight equivalence class | `3` |
| `--package-lower-bound` | Minimum weight accepted for one package | `640` |
| `--package-upper-bound` | Maximum weight accepted for one package | `680` |
| `--max-selections` | Maximum uses of one feasible package class | `10` |
| `--target-value` | Optimal bin count considered a match | `19` |
| `--alpha` | Significance level for the Wilson interval | `0.05` |

Distribution options are also available: `--mean`, `--std`,
`--sample-lower-bound`, `--sample-upper-bound`, and `--target-portions`.

For example, this runs 100 trials with tighter package limits and a 99%
confidence interval:

```bash
poetry run python src/main.py \
  --trials 100 \
  --package-lower-bound 645 \
  --package-upper-bound 675 \
  --alpha 0.01
```

SCIP prints detailed optimization logs, so longer runs can produce substantial
terminal output and may take several minutes.

### C++ Monte Carlo evaluation

The C++ implementation mirrors the Monte Carlo evaluation on the `dev` branch.
It requires CMake, a C++17 compiler, and an OR-Tools installation that CMake can
discover.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
./cpp/build/bin_covering
```

The default evaluation runs 100 trials. Pass a trial count to the executable for
a shorter smoke test, such as `./cpp/build/bin_covering 1`.

Monte Carlo, optimization, and confidence-interval parameters are defined in
`cpp/src/config.h`. The executable samples weights from the configured truncated
normal distribution, solves each generated bin-covering problem, and reports the
percentage matching the target along with its Wilson confidence interval.
It also reports the average, minimum, and maximum solver time across all trials;
weight generation is excluded from these timings.

## Configuration

The Monte Carlo defaults live in [`src/config.py`](src/config.py). CLI options
override those defaults for a single run.

`sample_data.json` is retained for the original fixed-dataset experiment, but
the Monte Carlo entry point does **not** read it. Changing that file will not
change `src/main.py` results.

The key distinction between the two sets of bounds is:

- sample bounds constrain each randomly generated item weight;
- package bounds constrain the total weight placed in each solved package.

## Project layout

- `src/main.py` — CLI and Monte Carlo loop
- `src/config.py` — default experiment parameters
- `src/solver.py` — CP-SAT enumeration and SCIP optimization
- `src/utils.py` — sampling, quantization, and statistics
- `src/binomial.py` — Wilson confidence interval calculation

## License

Licensed under the [Apache License 2.0](LICENSE).
