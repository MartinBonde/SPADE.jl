function MOI.optimize!(opt::Optimizer)
    started_ns = time_ns()
    opt.result = nothing
    problem = Base.invokelatest(_transform_moi_model, opt.model)
    opt.variable_positions = problem.positions
    opt.constraint_rows = problem.rows
    opt.fixed = problem.fixed
    opt.objective_sign = problem.sign
    opt.result = _moi_solve(problem.system, problem.start, opt.options, started_ns)
    nothing
end

_moi_solve(system::SquareSystem, arguments...) = solve_square(system, arguments...)
_moi_solve(system::OptimizationSystem, arguments...) = minimize(system, arguments...)

const TERMINATION_STATUS = (
    root = MOI.LOCALLY_SOLVED,
    time_limit = MOI.TIME_LIMIT,
    memory_limit = MOI.MEMORY_LIMIT,
    iteration_limit = MOI.ITERATION_LIMIT,
    stagnation = MOI.SLOW_PROGRESS,
    bound = MOI.LOCALLY_INFEASIBLE,
    singular = MOI.LOCALLY_INFEASIBLE,
    invalid_start = MOI.INVALID_MODEL,
    evaluation_failure = MOI.INVALID_MODEL,
    numerical_breakdown = MOI.NUMERICAL_ERROR,
    constant_residual = MOI.OTHER_ERROR,
    missing_multipliers = MOI.OTHER_ERROR,
    optimum = MOI.LOCALLY_SOLVED,
    infeasible = MOI.LOCALLY_INFEASIBLE,
    unbounded = MOI.NORM_LIMIT,
)

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    opt.result === nothing ? MOI.OPTIMIZE_NOT_CALLED :
    TERMINATION_STATUS[opt.result.termination]
end

function MOI.get(opt::Optimizer, attribute::MOI.PrimalStatus)
    1 <= attribute.result_index <= MOI.get(opt, MOI.ResultCount()) || return MOI.NO_SOLUTION
    opt.result.accepted ? MOI.FEASIBLE_POINT : MOI.UNKNOWN_RESULT_STATUS
end
function MOI.get(opt::Optimizer, attribute::MOI.DualStatus)
    1 <= attribute.result_index <= MOI.get(opt, MOI.ResultCount()) || return MOI.NO_SOLUTION
    _multipliers(opt.result) === nothing ? MOI.NO_SOLUTION : MOI.FEASIBLE_POINT
end
function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    opt.result === nothing || opt.result.x === nothing ? 0 : 1
end
function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    opt.result === nothing ? 0.0 : opt.result.elapsed_seconds
end
function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
    opt.result === nothing ? "not called" :
    "$(opt.result.termination): $(opt.result.message)"
end
MOI.get(::Optimizer, ::MOI.SolverName) = "SPADE"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.1.0"

function MOI.get(opt::Optimizer, attribute::MOI.VariablePrimal, variable::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, attribute)
    opt.result.x[opt.variable_positions[variable]]
end

function MOI.get(opt::Optimizer, attribute::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attribute)
    _objective_value(opt.result, opt.objective_sign, attribute)
end

function _objective_value(result::OptimizationResult, sign, attribute)
    sign * something(result.objective, NaN)
end
function _objective_value(::SquareResult, sign, attribute)
    throw(
        MOI.GetAttributeNotAllowed(attribute, "a model without an objective has no value"),
    )
end

_multipliers(::SquareResult) = nothing
_multipliers(result::OptimizationResult) = result.multipliers

function _checked_multipliers(opt, attribute)
    MOI.check_result_index_bounds(opt, attribute)
    multipliers = _multipliers(opt.result)
    multipliers === nothing &&
        throw(MOI.GetAttributeNotAllowed(attribute, "the result has no row multipliers"))
    multipliers
end

# MOI duals make the objective gradient equal the sum of each dual times the row
# gradient, for a minimum and a maximum alike. The multipliers of the minimized
# objective make that sum zero, so the duals are the negative multipliers.
function MOI.get(opt::Optimizer, attribute::MOI.ConstraintDual, index::MOI.ConstraintIndex)
    -_checked_multipliers(opt, attribute)[opt.constraint_rows[index]]
end

# The dual of an active bound is the reduced cost of its variable, which only a
# nonbasic variable has. A bound that fixes a variable has a dual, but the solver
# does not have the Jacobian column of a fixed variable.
function MOI.get(
    opt::Optimizer,
    attribute::MOI.ConstraintDual,
    index::MOI.ConstraintIndex{MOI.VariableIndex},
)
    _checked_multipliers(opt, attribute)
    position = opt.variable_positions[MOI.VariableIndex(index.value)]
    opt.fixed[position] && throw(
        MOI.GetAttributeNotAllowed(attribute, "SPADE has no dual for a fixed variable"),
    )
    _bound_dual(
        MOI.get(opt.model, MOI.ConstraintSet(), index),
        opt.result.x[position],
        opt.result.reduced_costs[position],
    )
end

_bound_dual(set::MOI.GreaterThan, x, cost) = x == set.lower ? cost : 0.0
_bound_dual(set::MOI.LessThan, x, cost) = x == set.upper ? cost : 0.0
_bound_dual(set::MOI.Interval, x, cost) = x == set.lower || x == set.upper ? cost : 0.0
