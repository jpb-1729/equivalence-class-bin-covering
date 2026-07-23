# Equivalence-Class Bin Covering

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1109%2FTASE.2020.3022986-B31B1B)](https://doi.org/10.1109/TASE.2020.3022986)

Python and C++ experiments for solving bin-covering problems through weight
equivalence classes. The project combines OR-Tools CP-SAT package enumeration
with SCIP optimization, and includes both a preserved real-world problem
instance and reproducible Monte Carlo evaluation.

> **Preserved demo result:** 102 measurements become 33 equivalence classes and
> 83,484 feasible package classes, producing a proven optimum of **19 packages**.

## Highlights

- Reproduces a fixed problem instance from real-world weight measurements.
- Reduces combinatorial complexity by grouping weights into equivalence classes.
- Enumerates feasible package classes with OR-Tools CP-SAT.
- Maximizes non-overlapping package selections with SCIP.
- Runs seeded Monte Carlo experiments from a truncated normal distribution.
- Reports Wilson confidence intervals for the target-outcome proportion.
- Provides matching Python and C++ Monte Carlo implementations.

## Quick start

The Python implementation requires Python 3.12 or newer and
[Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone https://github.com/jpb-1729/equivalence-class-bin-covering.git
cd equivalence-class-bin-covering
poetry install
poetry run python demo.py
```

The demo is intentionally a full optimization run and can take tens of seconds,
depending on the machine and solver build. SCIP also prints detailed progress
logs.

## Workflows

### Fixed measurement demo

`demo.py` loads `sample_data.json`, deterministically maps each measurement to
the lower boundary of its configured equivalence class, enumerates all package
classes within the configured weight window, and reports the optimum:

```bash
poetry run python demo.py
```

Use another dataset with the same JSON schema:

```bash
poetry run python demo.py --data path/to/measurements.json
```

See [`SAMPLE_DATA.md`](SAMPLE_DATA.md) for the schema, exact transformation,
verified result, licensing, and currently known provenance gaps.

### Python Monte Carlo evaluation

Run one trial with the default configuration:

```bash
poetry run python src/main.py
```

Run ten reproducible trials:

```bash
poetry run python src/main.py --trials 10 --seed 42
```

Show every option:

```bash
poetry run python src/main.py --help
```

Common options:

| Option | Meaning | Default |
| --- | --- | ---: |
| `--trials` | Number of independent Monte Carlo trials | `1` |
| `--seed` | Seed for Python and NumPy random generators | random |
| `--quantization-interval` | Width of each weight equivalence class | `3` |
| `--package-lower-bound` | Lower package-weight enumeration bound | `640` |
| `--package-upper-bound` | Upper package-weight enumeration bound | `680` |
| `--max-selections` | Maximum uses of one feasible package class | `10` |
| `--target-value` | Optimal package count considered a match | `19` |
| `--alpha` | Significance level for the Wilson interval | `0.05` |

Distribution options include `--mean`, `--std`, `--sample-lower-bound`,
`--sample-upper-bound`, and `--target-portions`.

For example:

```bash
poetry run python src/main.py \
  --trials 100 \
  --package-lower-bound 645 \
  --package-upper-bound 675 \
  --alpha 0.01
```

### C++ Monte Carlo evaluation

The C++17 implementation mirrors the Python Monte Carlo workflow. It requires
CMake and an OR-Tools installation that CMake can discover.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
./cpp/build/bin_covering
```

The default C++ evaluation runs 100 trials. Pass a trial count for a shorter
smoke test:

```bash
./cpp/build/bin_covering 1
```

Parameters are defined in `cpp/src/config.h`. The program reports the target
match percentage, Wilson confidence interval, and average, minimum, and maximum
solver times. Sample generation is excluded from the timing measurements.

## Method

Each Monte Carlo trial:

1. draws weights from a truncated normal distribution until their total reaches
   the configured target;
2. quantizes the weights into equivalence classes;
3. enumerates feasible package classes with OR-Tools CP-SAT;
4. uses SCIP to maximize the number of selected packages without reusing items;
5. records whether the optimum matches the target package count.

Across trials, the program reports the matching percentage and a Wilson
confidence interval for that proportion.

The formulation is based on Roselli et al.,
[*On the Use of Equivalence Classes for Optimal and Suboptimal Bin Packing and
Bin Covering*](https://doi.org/10.1109/TASE.2020.3022986).

## Configuration and reproducibility

Python Monte Carlo defaults live in [`src/config.py`](src/config.py), and CLI
arguments override them for one run. C++ defaults live in
[`cpp/src/config.h`](cpp/src/config.h).

`sample_data.json` drives only the fixed demo. Modifying it does not affect
`src/main.py` or the C++ Monte Carlo implementation.

The two kinds of bounds have different roles:

- sample bounds truncate each generated item weight;
- package bounds define the package-total window enumerated by the solver.

Use `--seed` when comparing Python Monte Carlo configurations. Solver logs and
the exact selected package classes may vary between compatible solver versions
when multiple optimal solutions exist.

## Project layout

```text
.
├── demo.py                  Fixed real-world measurement workflow
├── sample_data.json         Preserved measurements and demo parameters
├── SAMPLE_DATA.md           Data schema, provenance, and expected result
├── src/
│   ├── main.py              Python Monte Carlo CLI
│   ├── config.py            Python experiment defaults
│   ├── solver.py            CP-SAT enumeration and SCIP optimization
│   ├── utils.py             Sampling, quantization, and statistics
│   └── binomial.py          Wilson confidence interval
└── cpp/
    ├── CMakeLists.txt
    └── src/                 C++ Monte Carlo implementation
```

## Citation

GitHub exposes citation metadata from [`CITATION.cff`](CITATION.cff). If this
software contributes to published work, cite both the repository and the
underlying equivalence-class formulation paper.

## License

This project and its included sample are licensed under the
[Apache License 2.0](LICENSE). Third-party dependencies retain their respective
licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
