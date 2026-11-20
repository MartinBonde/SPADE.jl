# SPADE.jl

SPADE.jl is the Julia interface of SPADE, a solver for sparse nonlinear
models with bounds: square systems of equations (`solve_square`) and models with an
objective (`minimize`, or `SPADE.Optimizer` in JuMP).

This is an evaluation version for 64-bit Windows. It works until the end of 2026, on
a CPU with AVX2: most Intel Core CPUs since 2013 and AMD CPUs since 2015, but not all
Pentium and Celeron CPUs.

This package holds no solver code. When you install it, Pkg also downloads the
compiled solver from the releases of this repository: a download of 100 MB that
needs 390 MB of disk space. The first solve
of a Julia session starts the solver in its own process. The functions of your model
stay in your Julia process.

## Install

You need Julia 1.11 or later. Add these lines to the `Project.toml` of your project:

```toml
[deps]
SPADE = "c3e1a8d2-6f4b-4d7a-8e15-2a9b7c0d4e63"

[sources]
SPADE = {url = "https://github.com/MartinBonde/SPADE.jl", rev = "v0.1.0"}
```

Then run `using Pkg; Pkg.instantiate()` in that project. Pkg first looks for the
solver on the Julia package server and prints `Failure artifact: spade`. Then
it downloads the solver from GitHub. Or, in any project:

```julia
using Pkg
Pkg.add(url = "https://github.com/MartinBonde/SPADE.jl", rev = "v0.1.0")
```

## Use

With JuMP:

```julia
using JuMP, SPADE
model = Model(SPADE.Optimizer)
set_attribute(model, "maxiters", 1000)
```

Without JuMP, give the residuals, the Jacobian and its sparsity pattern:

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

`SolveOptions` lists the options. At its start, the solver prints its build date and
the last year in which it works.

## Report a problem

Send the output of the solve. For a failed solve without JuMP, also send the file
that `write_failure_bundle(path, system, result; x0, options)` writes. It holds the
point and the residuals, but no code of your model.

## License

`LICENSE.md` gives the terms. The solver uses MUMPS under the CeCILL-C license. The
`share/licenses` folder of the downloaded solver holds the license of each part.
