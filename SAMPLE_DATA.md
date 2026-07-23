# Sample problem instance

`sample_data.json` defines the fixed real-world instance used by `demo.py`. It
contains the measured item weights and the parameters required to reproduce the
original bin-covering problem.

The measurements were collected in January 2023 with a digital scale whose
three-digit seven-segment display reports ounces to two decimal places. The JSON
stores each displayed reading as an integer number of hundredths of an ounce:
`123` represents `1.23 oz`. This preserves the scale readings exactly while
keeping all solver coefficients integral.

## Instance summary

| Property | Value |
| --- | ---: |
| Number of items | 102 |
| Collection period | January 2023 |
| Measurement resolution | 0.01 oz |
| Minimum observed weight | 0.84 oz (`84`) |
| Maximum observed weight | 1.56 oz (`156`) |
| Total observed weight | 122.11 oz (`12,211`) |
| Distinct observed weights | 36 |
| Quantization interval | 0.02 oz (`2`) |
| Package-weight window | 6.40–6.50 oz (`640`–`650`) |
| Maximum selections per package class | 10 |

## File schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "weights",
    "interval",
    "lower_bound",
    "upper_bound",
    "max_selections"
  ],
  "properties": {
    "weights": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "integer",
        "minimum": 1
      }
    },
    "interval": {
      "type": "integer",
      "minimum": 1
    },
    "lower_bound": {
      "type": "integer"
    },
    "upper_bound": {
      "type": "integer"
    },
    "max_selections": {
      "type": "integer",
      "minimum": 1
    }
  }
}
```

In addition to the schema above, `lower_bound` must not exceed `upper_bound`.

The fields mean:

- `weights`: the displayed measurements in hundredths of an ounce;
- `interval`: the equivalence-class width in hundredths of an ounce;
- `lower_bound` and `upper_bound`: the preferred package-total window in
  hundredths of an ounce;
- `max_selections`: the maximum number of times the optimizer may select one
  package class.

The bounds are soft in the problem interpretation. The current demo uses them as
a hard filter while enumerating candidate package classes.

## Quantization

Quantization operates on the integer hundredths-of-an-ounce representation.
Each encoded weight \(w\) is mapped deterministically to the lower multiple of
the interval:

\[
q(w) =
\left\lfloor \frac{w}{\text{interval}} \right\rfloor
\cdot \text{interval}
\]

For the committed interval of 2:

\[
q(w) = 2\left\lfloor \frac{w}{2} \right\rfloor
\]

For example, `119` (`1.19 oz`) becomes `118` (`1.18 oz`), while `120`
(`1.20 oz`) remains unchanged. This reduces the 36 exact observed weights to 33
equivalence classes.

## Optimization problem

The demo solves the instance in two stages.

### 1. Enumerate package classes

For every equivalence class \(j\):

- \(r_j\) is its quantized representative weight;
- \(n_j\) is the number of available items in that class;
- \(z_j\) is the number of items from that class placed in one package.

OR-Tools CP-SAT enumerates every integer vector \(z\) satisfying:

\[
0 \le z_j \le n_j
\]

and

\[
640 \le \sum_j r_j z_j \le 650.
\]

The weighted sum is expressed in hundredths of an ounce, so these limits
correspond to 6.40–6.50 oz. Each feasible vector is one package class.

### 2. Select packages

SCIP assigns an integer selection count \(x_p\) to every enumerated package
class \(p\). It maximizes:

\[
\sum_p x_p
\]

subject to:

\[
0 \le x_p \le 10
\]

and, for every equivalence class \(j\):

\[
\sum_p z_{p,j}x_p \le n_j.
\]

The final constraint prevents any measured item from being reused.

## Verified result

With the committed data and locked Python dependencies:

| Result | Value |
| --- | ---: |
| Equivalence classes | 33 |
| Feasible package classes | 83,484 |
| Optimal number of packages | 19 |

Run the instance from the repository root:

```bash
poetry install
poetry run python demo.py
```

SCIP may return a different set of selected package classes when multiple
optimal solutions exist, but the optimal package count should remain 19.

## Scope and licensing

This file preserves one problem instance; it is not intended as a representative
statistical dataset. The repository does not yet record the scale manufacturer
or model, its calibration status, the type of items measured, or the sampling
and measurement protocol. The instance is distributed with the project under
the [Apache License 2.0](LICENSE).
