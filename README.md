# SPADE.jl

SPADE is a solver for large, sparse nonlinear models with bounds. It solves square
systems of equations, and it minimizes an objective subject to equality and
inequality constraints and bounds.
SPADE.jl is its Julia interface: use it through JuMP, or give it your own residual
and Jacobian functions.

This is an evaluation version, version 0.1.0. It works until the end of 2026.

## What SPADE is good at

- **Large systems.** SPADE solves models with more than three million equations,
  such as dynamic economic models over long horizons.
- **Poor starts.** SPADE takes Newton steps in a trust region and keeps the
  variables within their bounds. When a trial point is outside the domain of a
  function, such as the log of a negative number, SPADE takes a shorter step instead
  of stopping.
- **Recursive equations.** SPADE finds the equations that it can solve one at a time,
  and solves them outside the coupled Newton system. Long recursive chains add little
  to the cost of a solve.
- **Clear failures.** When SPADE cannot solve a model, it says why and names the
  variables behind the stop: variables at a bound that blocks progress, or the
  variables in a singular part of the Jacobian.
- **Models with an objective.** A reduced-gradient method minimizes the objective
  subject to equality and inequality constraints. It suits models with many
  constraints and few degrees of freedom, such as calibration problems.

SPADE finds a local solution. It does not handle integer variables.

## Requirements

- 64-bit Windows.
- A CPU with AVX2: most Intel Core CPUs since 2013 and AMD CPUs since 2015, but not
  all Pentium and Celeron CPUs.
- Julia 1.11 or later.
- About 105 MB to download and 375 MB of disk space for the compiled solver.

A Linux version, and versions for other CPU types, can be made available on request.
To ask for one, write to martin@bonde.dk.

## Install

In a Julia project:

```julia
using Pkg
Pkg.add(url = "https://github.com/MartinBonde/SPADE.jl", rev = "v0.1.0")
```

Or add these lines to the `Project.toml` of your project, and run
`using Pkg; Pkg.instantiate()`:

```toml
[deps]
SPADE = "c3e1a8d2-6f4b-4d7a-8e15-2a9b7c0d4e63"

[sources]
SPADE = {url = "https://github.com/MartinBonde/SPADE.jl", rev = "v0.1.0"}
```

While Pkg prints `Building SPADE`, it downloads the compiled solver from GitHub.

The first solve in a Julia session starts the solver in a separate process, which
takes a few seconds. The functions of your model stay in your Julia process, so they
can be any Julia code.

## Use with JuMP

Without an objective, SPADE solves the constraints as a square system. Then all
constraints must be equations, and after you remove fixed variables, there must be as
many equations as variables:

```julia
using JuMP, SPADE
model = Model(SPADE.Optimizer)
@variable(model, x >= 0, start = 1)
@variable(model, y >= 0, start = 1)
@constraint(model, x^2 + y == 3)
@constraint(model, x + y^2 == 5)
optimize!(model)
termination_status(model), value(x), value(y)
```

With an objective, SPADE minimizes it subject to the constraints and bounds. The
constraints can be equations and inequalities:

```julia
model = Model(SPADE.Optimizer)
@variable(model, 0 <= x <= 2, start = 0.5)
@variable(model, y, start = 0.5)
@constraint(model, y == exp(x) - 1)
@constraint(model, x + y <= 1.5)
@objective(model, Min, (x - 1)^2 + (y - 1)^2)
optimize!(model)
termination_status(model), objective_value(model)
```

To find a feasible point of a model with inequalities, give it a constant objective,
such as `@objective(model, Min, 0)`.

SPADE takes linear, quadratic and nonlinear constraints with `==`, `<=` and `>=`,
user-defined operators, and bounds on variables. Set options with `set_attribute`
and the time limit with `set_time_limit_sec`:

```julia
set_attribute(model, "maxiters", 1000)
set_attribute(model, "verbose", true)
set_time_limit_sec(model, 60)
```

## Use without JuMP

Give the residuals, the Jacobian and its sparsity pattern:

```julia
using SPADE, SparseArrays
system = SquareSystem(
    (F, x) -> (F .= x .^ 2 .- 2),
    (J, x) -> (J.nzval .= 2 .* x),
    sparse([1], [1], [1.0]),
)
result = solve_square(system, [1.0]; maxiters = 100)
result.accepted, result.x
```

`minimize` takes an `OptimizationSystem` in the same way. Its rows are equations: for
an inequality, add a slack variable with the bounds of the row. The help of
`SquareSystem` and `OptimizationSystem` (type `?SquareSystem` in the REPL) lists the
optional arguments: bounds, fixed variables, row scales, a domain check, and
`evaluate_rows!`, which lets SPADE solve recursive equations one at a time.

`result.termination` names the reason for the stop, and `result.message` explains
it. After a stop at a bound or a singular Jacobian, `result.stop_variables` lists the
variables behind it, the most important first.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `abstol` | `1e-10` | Largest allowed residual of a scaled row |
| `maxiters` | `100` | Largest number of Newton trials, rejected trials included |
| `time_limit_seconds` | `600` | Time limit |
| `memory_limit_bytes` | `8_000_000_000` | Memory limit of the solver process |
| `optimality_tolerance` | `1e-7` | Largest scaled reduced gradient at an optimum, relative to `max(1, abs(objective))` |
| `verbose` | `false` | Print the progress of the solve |

In a square system, SPADE divides each row whose largest term `abs(J[i, j] * x[j])`
exceeds one by that term before it tests `abstol`. The solver checks the time and
memory limits between steps, so a solve can pass them by a little.

## Report a problem

Open an issue on this repository with the output of the solve, run with
`verbose = true`. For a failed solve without JuMP, also attach the file that
`write_failure_bundle(path, system, result; x0, options)` writes. It holds the point
and the residuals, but no code of your model. To keep a model private, send these
files to martin@bonde.dk instead.

## License

`LICENSE.md` gives the terms of the evaluation version. SPADE uses
[MUMPS](https://mumps-solver.org) under the CeCILL-C license. The `share/licenses`
folder of the downloaded solver holds the license of each third-party part. To find
that folder:

```julia
using Artifacts, SPADE
toml = joinpath(pkgdir(SPADE), "Artifacts.toml")
joinpath(artifact_path(artifact_hash("spade", toml)), "share", "licenses")
```

The release page of this repository holds the source of each part under the LGPL or
the CeCILL-C license.
