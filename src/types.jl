_unrestricted_domain(x) = true

"""
Rows over physical variables, with bounds, and an objective or `nothing`.
`SquareSystem` and `OptimizationSystem` name the two kinds; their constructors give
the meaning of each field. Each derived system, such as a triangular group or the
rows of a basis, is an `EquationSystem` too.
"""
struct EquationSystem{RF, JF, EF, D, O}
    residual!::RF
    jacobian!::JF
    evaluate_rows!::EF
    jac_prototype::SparseMatrixCSC{Float64, Int}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    free_variables::Vector{Int}
    active_rows::Vector{Int}
    constant_rows::Vector{Int}
    row_scale::Vector{Float64}
    domain::D
    objective::O
end

"The objective of an `OptimizationSystem`: its value, gradient, and Hessian product."
struct Objective{VF, GF, HF}
    value::VF
    gradient!::GF
    hessian_product!::HF
end

const SquareSystem{RF, JF, EF, D} = EquationSystem{RF, JF, EF, D, Nothing}
const OptimizationSystem{RF, JF, EF, D, O <: Objective} = EquationSystem{RF, JF, EF, D, O}

"""
    SquareSystem(residual!, jacobian!, jac_prototype; lower_bounds, upper_bounds,
        free_variables, row_scale, evaluate_rows!, domain)

Physical residuals and a fixed sparse CSC Jacobian pattern over the free variables.

`residual!(F, x)` and `jacobian!(J, x)` overwrite all rows and see every physical
variable. The Jacobian has the original rows and the free columns, in
`free_variables` order. Include every entry that can be nonzero away from the start.
Rows with no entry are constant. After you remove fixed variables and constant
rows, the system must be square.

Equal bounds fix a variable. The solver ignores its start value.

`evaluate_rows!(residual, jacobian, x, rows)` is optional. It writes only the
requested rows into full-size outputs, either of which can be `nothing`. With it,
the solver removes triangular equations from the coupled solve.

A false `domain(x)`, a `DomainError`, or a value that is not finite rejects the
evaluation. Other callback errors propagate.
"""
function SquareSystem(residual!, jacobian!, jac_prototype::SparseMatrixCSC; arguments...)
    system = _equation_system(residual!, jacobian!, jac_prototype, nothing; arguments...)
    length(system.active_rows) == length(system.free_variables) || throw(
        DimensionMismatch("the active rows and free variables must form a square system"),
    )
    system
end

"""
    OptimizationSystem(residual!, jacobian!, jac_prototype, objective, gradient!;
        lower_bounds, upper_bounds, free_variables, row_scale, evaluate_rows!, domain,
        hessian_product!)

Minimize `objective(x)` subject to the rows and the bounds. The other arguments are
as for `SquareSystem`, but the system need not be square. The free variables can
outnumber the rows with a free variable: the difference is the number of degrees of
freedom. The rows can outnumber the free variables when some rows depend on the
others, as Walras' law makes one row of a general-equilibrium model depend on the
others: the solver then checks that the dependent rows hold. `objective(x)`
returns a `Float64`, and `gradient!(g, x)` overwrites the derivative with respect
to each physical variable. For a maximum, negate both. An inequality row needs a
slack variable with the bounds of the row.

`hessian_product!(h, x, σ, y, v)` overwrites `h` with the product of `v` and the
Hessian of `σ objective(x) + y ⋅ residual(x)`, all in physical variables, with `y`
a multiplier for each row. With it, the search takes Newton steps in the reduced
space; without it (`nothing`), it takes limited-memory BFGS steps.

A `DomainError` or a value that is not finite rejects the objective.
"""
function OptimizationSystem(
    residual!,
    jacobian!,
    jac_prototype::SparseMatrixCSC,
    objective,
    gradient!;
    hessian_product! = nothing,
    arguments...,
)
    _equation_system(
        residual!,
        jacobian!,
        jac_prototype,
        Objective(objective, gradient!, hessian_product!);
        arguments...,
    )
end

"Check the arguments that both kinds of system share, and build the system."
function _equation_system(
    residual!,
    jacobian!,
    jac_prototype,
    objective;
    lower_bounds = fill(-Inf, size(jac_prototype, 2)),
    upper_bounds = fill(Inf, size(jac_prototype, 2)),
    free_variables = findall(lower_bounds .!= upper_bounds),
    row_scale = ones(size(jac_prototype, 1)),
    evaluate_rows! = nothing,
    domain = _unrestricted_domain,
)
    length(lower_bounds) == length(upper_bounds) ||
        throw(DimensionMismatch("lower and upper bounds have different lengths"))
    all(lower_bounds .<= upper_bounds) ||
        throw(ArgumentError("bounds must be ordered and cannot contain NaN"))
    all(isfinite, lower_bounds[lower_bounds .== upper_bounds]) ||
        throw(ArgumentError("fixed variable values must be finite"))
    sort(free_variables) == findall(lower_bounds .< upper_bounds) ||
        throw(ArgumentError("free_variables must list each nonfixed variable once"))
    length(free_variables) == size(jac_prototype, 2) ||
        throw(DimensionMismatch("Jacobian columns must follow free_variables"))
    length(row_scale) == size(jac_prototype, 1) ||
        throw(DimensionMismatch("row_scale must contain one multiplier per original row"))
    all(s -> isfinite(s) && s > 0, row_scale) ||
        throw(ArgumentError("row multipliers must be finite and positive"))
    pattern = SparseMatrixCSC{Float64, Int}(jac_prototype)
    active = sort!(unique(rowvals(pattern)))
    EquationSystem(
        residual!,
        jacobian!,
        evaluate_rows!,
        pattern,
        Float64.(lower_bounds),
        Float64.(upper_bounds),
        Int.(free_variables),
        active,
        setdiff(axes(pattern, 1), active),
        Float64.(row_scale),
        domain,
        objective,
    )
end

"""
Return the physical point of the start `x0`: the start of each free variable, and the
value of each fixed variable, whose start the solver ignores.
"""
function _physical_start(system, x0)
    length(x0) == length(system.lower_bounds) ||
        throw(DimensionMismatch("the start must contain every physical variable"))
    x = copy(system.lower_bounds)
    x[system.free_variables] .= x0[system.free_variables]
    x
end

const STOP_MESSAGES = (
    root = "The scaled rows meet the tolerance.",
    invalid_start = "The requested start has an invalid evaluation.",
    constant_residual = "A constant row is outside tolerance.",
    iteration_limit = "The trial-attempt limit was reached.",
    time_limit = "The soft time limit was reached.",
    memory_limit = "The soft process resident memory limit was reached.",
    stagnation = "The trust step cannot make further progress.",
    bound = "Bounds block the descent: the residual decreases only if the stop variables move past their bounds.",
    singular = "The residual cannot decrease near this point, where the Jacobian is singular or nearly singular.",
    evaluation_failure = "Invalid trial evaluations prevent further progress.",
    numerical_breakdown = "Solver arithmetic produced a non-finite value.",
    missing_multipliers = "The point meets tolerance, but it has no row multipliers.",
    optimum = "The reduced gradient meets the optimality tolerance.",
    infeasible = "The sum of infeasibilities has a positive local minimum.",
    unbounded = "A variable passed 1e10 while the objective decreased.",
)

"""
The fully evaluated valid point with the smallest scaled residual.

`accepted` is true exactly when `x` and `residual` are finite, `x` meets the
bounds and domain, and `maximum(abs, row_scale .* residual; init = 0.0) <= abstol`.
`termination` gives the stop reason, one of the keys of `STOP_MESSAGES`. A point
can meet tolerance and exceed a soft limit. `x` and `residual` are `nothing` when
no evaluation was finite.

`stop_variables` holds the physical indices of the variables behind a `:bound`
or `:singular` stop, the most important first. For `:bound`, they are at a bound
that blocks the descent. For `:singular`, they carry the Newton step, or the null
space when the Jacobian is singular. Other stops leave it empty. `linear_failure`
gives why the last Newton system that failed had no factor, or `nothing`; the search
can go on after such a failure.
"""
struct SquareResult
    x::Union{Nothing, Vector{Float64}}
    residual::Union{Nothing, Vector{Float64}}
    accepted::Bool
    termination::Symbol
    message::String
    iterations::Int
    elapsed_seconds::Float64
    observed_memory_bytes::Int
    linear_failure::Union{Nothing, String}
    stop_variables::Vector{Int}
end

"""
The result of `minimize`: the fields of `SquareResult`, the objective at `x`, the
row multipliers, and the reduced cost and status of each physical variable.

`objective` is `nothing` without a point, and when the objective is invalid at the
point, which the `:evaluation_failure` stop then reports. At a feasible point of the
search, `multipliers` hold one value per row, so that the objective gradient plus
the transposed Jacobian times the multipliers gives the reduced cost of each free
variable. A row without a free variable gets zero. `statuses` gives `:basic`,
`:superbasic`, `:lower` or `:upper` for a free variable and `:fixed` for a fixed
one. A nonbasic variable, at a bound, has its reduced cost; the others have zero,
and a fixed variable `NaN`. These three fields are `nothing` at other points, after
a time or memory limit, and when the Jacobian gives no multipliers, which a
`:missing_multipliers` stop reports.
"""
struct OptimizationResult
    x::Union{Nothing, Vector{Float64}}
    residual::Union{Nothing, Vector{Float64}}
    accepted::Bool
    termination::Symbol
    message::String
    iterations::Int
    elapsed_seconds::Float64
    observed_memory_bytes::Int
    linear_failure::Union{Nothing, String}
    stop_variables::Vector{Int}
    objective::Union{Nothing, Float64}
    multipliers::Union{Nothing, Vector{Float64}}
    reduced_costs::Union{Nothing, Vector{Float64}}
    statuses::Union{Nothing, Vector{Symbol}}
end

function OptimizationResult(
    result::SquareResult,
    objective,
    multipliers,
    reduced_costs,
    statuses,
)
    OptimizationResult(
        result.x,
        result.residual,
        result.accepted,
        result.termination,
        result.message,
        result.iterations,
        result.elapsed_seconds,
        result.observed_memory_bytes,
        result.linear_failure,
        result.stop_variables,
        objective,
        multipliers,
        reduced_costs,
        statuses,
    )
end
