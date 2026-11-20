function _moi_bound!(lower, upper, i, set::MOI.EqualTo{Float64})
    lower[i] <= set.value <= upper[i] ||
        throw(ArgumentError("fixed variable $i violates its bounds"))
    lower[i] = upper[i] = set.value
end

function _moi_bound!(lower, upper, i, set::MOI.GreaterThan{Float64})
    lower[i] = max(lower[i], set.lower)
end

function _moi_bound!(lower, upper, i, set::MOI.LessThan{Float64})
    upper[i] = min(upper[i], set.upper)
end

function _moi_bound!(lower, upper, i, set::MOI.Interval{Float64})
    lower[i] = max(lower[i], set.lower)
    upper[i] = min(upper[i], set.upper)
end

function _moi_bound!(lower, upper, i, set)
    throw(MOI.UnsupportedConstraint{MOI.VariableIndex, typeof(set)}())
end

_substitute_fixed(value::Real, fixed) = value
_substitute_fixed(variable::MOI.VariableIndex, fixed) = get(fixed, variable, variable)
function _substitute_fixed(f::MOI.ScalarNonlinearFunction, fixed)
    MOI.ScalarNonlinearFunction(
        f.head,
        Any[_substitute_fixed(arg, fixed) for arg in f.args],
    )
end
function _substitute_fixed(
    f::Union{MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}},
    fixed,
)
    _substitute_fixed(convert(MOI.ScalarNonlinearFunction, f), fixed)
end

const _MOI_FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

const _MOI_INEQUALITIES =
    Union{MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.Interval{Float64}}

function _moi_equation!(rows, bounds, f, set)
    throw(MOI.UnsupportedConstraint{typeof(f), typeof(set)}())
end

function _moi_equation!(rows, bounds, f::_MOI_FUNCTIONS, set::MOI.EqualTo{Float64})
    (convert(MOI.ScalarNonlinearFunction, f), set.value)
end

# An inequality row equates its function with a slack variable in the bounds of the row.
function _moi_equation!(rows, bounds, f::_MOI_FUNCTIONS, set::_MOI_INEQUALITIES)
    slack = MOI.VariableIndex(rows.first_slack + length(rows.slacks))
    push!(rows.slacks, slack)
    push!(rows.slack_rows, length(rows.targets) + 1)
    push!(bounds.lower, -Inf)
    push!(bounds.upper, Inf)
    _moi_bound!(bounds.lower, bounds.upper, length(bounds.lower), set)
    (
        MOI.ScalarNonlinearFunction(
            :-,
            Any[convert(MOI.ScalarNonlinearFunction, f), slack],
        ),
        0.0,
    )
end

function _moi_objective(f::Union{MOI.VariableIndex, _MOI_FUNCTIONS})
    convert(MOI.ScalarNonlinearFunction, f)
end
_moi_objective(f) = throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{typeof(f)}()))

function _transform_moi_model(model)
    attributes = MOI.get(model, MOI.ListOfModelAttributesSet())
    foreach(_check_moi_model_attribute, attributes)
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    sense = MOI.get(model, MOI.ObjectiveSense())
    bounds = (;
        positions = Dict(variable => i for (i, variable) in pairs(variables)),
        lower = fill(-Inf, length(variables)),
        upper = fill(Inf, length(variables)),
    )
    nonlinear = MOI.Nonlinear.Model()
    foreach(attribute -> _register_moi_operator!(nonlinear, model, attribute), attributes)
    # MOI parses each row onto one tape. A model with an objective also keeps the
    # rows for the Hessian products of MOI's evaluator. Slack variables follow the
    # variables of the model, with the next indices.
    rows = (;
        nonlinear,
        tape = MOI.Nonlinear.Expression(),
        starts = Int[],
        targets = Float64[],
        equations = sense == MOI.FEASIBILITY_SENSE ? nothing :
                    MOI.ScalarNonlinearFunction[],
        indices = Dict{MOI.ConstraintIndex, Int}(),
        slacks = MOI.VariableIndex[],
        slack_rows = Int[],
        first_slack = maximum(variable -> variable.value, variables; init = 0) + 1,
    )
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        foreach(MOI.get(model, MOI.ListOfConstraintIndices{F, S}())) do index
            f = MOI.get(model, MOI.ConstraintFunction(), index)
            set = MOI.get(model, MOI.ConstraintSet(), index)
            _collect_moi_row!(rows, bounds, index, f, set)
        end
    end
    (; lower, upper) = bounds
    targets = rows.targets
    physical = [variables; rows.slacks]
    free = findall(lower .!= upper)
    fixed = Dict(physical[i] => lower[i] for i in findall(lower .== upper))
    start = copy(lower)
    for i in filter(<=(length(variables)), free)
        value = MOI.get(model, MOI.VariablePrimalStart(), variables[i])
        value === nothing &&
            throw(ArgumentError("missing start for free variable $(variables[i])"))
        start[i] = value
    end
    slack_positions = length(variables) .+ eachindex(rows.slacks)
    start[slack_positions] .= 0.0

    tape = _RowTape(nonlinear.operators, rows.tape, rows.starts, physical, lower, upper)
    free_positions = zeros(Int, length(physical))
    free_positions[free] = eachindex(free)
    # The tape holds the sorted columns of each row, which is the transpose of the
    # Jacobian structure. Each entry of `order` holds its position on the tape.
    order = copy(
        transpose(
            SparseMatrixCSC(
                length(free),
                length(targets),
                tape.column_starts,
                free_positions[tape.columns],
                collect(eachindex(tape.columns)),
            ),
        ),
    )
    positions = invperm(nonzeros(order))
    residual! = function (output, x)
        _evaluate_tape!(output, nothing, tape, x, eachindex(targets), targets, positions)
        nothing
    end
    jacobian! = function (jacobian, x)
        _evaluate_tape!(nothing, jacobian, tape, x, eachindex(targets), targets, positions)
        nothing
    end
    evaluate_rows! = function (residual, jacobian, x, selected_rows)
        _evaluate_tape!(residual, jacobian, tape, x, selected_rows, targets, positions)
        nothing
    end
    start[slack_positions] .= _slack_start(
        residual!,
        length(targets),
        start,
        rows.slack_rows,
        lower[slack_positions],
        upper[slack_positions],
    )
    arguments = (;
        lower_bounds = lower,
        upper_bounds = upper,
        free_variables = free,
        evaluate_rows! = evaluate_rows!,
    )
    sign = sense == MOI.MAX_SENSE ? -1.0 : 1.0
    system =
        sense == MOI.FEASIBILITY_SENSE ?
        SquareSystem(residual!, jacobian!, order; arguments...) :
        _moi_optimization_system(
            model,
            attributes,
            physical,
            fixed,
            sign,
            residual!,
            jacobian!,
            order,
            rows,
            arguments,
        )
    (;
        system,
        start,
        positions = bounds.positions,
        rows = rows.indices,
        fixed = lower .== upper,
        sign,
    )
end

"""
Return the `OptimizationSystem` that minimizes `sign` times the objective of `model`.
The tape evaluates the rows. For the Hessian products of the Lagrangian, MOI's
evaluator takes the `equations` and `targets` of `rows`, and the objective has its
own evaluator: MOI evaluates the objective of an evaluator with its rows, so an
objective outside its domain would reject valid rows.
"""
function _moi_optimization_system(
    model,
    attributes,
    physical,
    fixed,
    sign,
    residual!,
    jacobian!,
    order,
    rows,
    arguments,
)
    row_model = MOI.Nonlinear.Model()
    foreach(attribute -> _register_moi_operator!(row_model, model, attribute), attributes)
    foreach(rows.equations, rows.targets) do expression, target
        MOI.Nonlinear.add_constraint(
            row_model,
            _substitute_fixed(expression, fixed),
            MOI.EqualTo(target),
        )
    end
    row_evaluator =
        MOI.Nonlinear.Evaluator(row_model, MOI.Nonlinear.SparseReverseMode(), physical)
    MOI.initialize(row_evaluator, [:HessVec])
    F = MOI.get(model, MOI.ObjectiveFunctionType())
    nonlinear = MOI.Nonlinear.Model()
    foreach(attribute -> _register_moi_operator!(nonlinear, model, attribute), attributes)
    MOI.Nonlinear.set_objective(
        nonlinear,
        _substitute_fixed(
            _moi_objective(MOI.get(model, MOI.ObjectiveFunction{F}())),
            fixed,
        ),
    )
    evaluator =
        MOI.Nonlinear.Evaluator(nonlinear, MOI.Nonlinear.SparseReverseMode(), physical)
    MOI.initialize(evaluator, [:Grad, :HessVec])
    work = zeros(length(physical))
    OptimizationSystem(
        residual!,
        jacobian!,
        order,
        x -> sign * MOI.eval_objective(evaluator, x),
        function (gradient, x)
            MOI.eval_objective_gradient(evaluator, gradient, x)
            gradient .*= sign
            nothing
        end;
        arguments...,
        hessian_product! = function (h, x, σ, y, v)
            MOI.eval_hessian_lagrangian_product(row_evaluator, h, x, v, 0.0, y)
            MOI.eval_hessian_lagrangian_product(evaluator, work, x, v, sign * σ, Float64[])
            h .+= work
            nothing
        end,
    )
end

_check_moi_model_attribute(::MOI.AbstractModelAttribute) = nothing
function _check_moi_model_attribute(attribute::MOI.NLPBlock)
    throw(MOI.UnsupportedAttribute(attribute, "use scalar nonlinear rows"))
end

_register_moi_operator!(nonlinear, model, ::MOI.AbstractModelAttribute) = nothing
function _register_moi_operator!(nonlinear, model, attribute::MOI.UserDefinedFunction)
    MOI.Nonlinear.register_operator(
        nonlinear,
        attribute.name,
        attribute.arity,
        MOI.get(model, attribute)...,
    )
end

function _collect_moi_row!(rows, bounds, index, variable::MOI.VariableIndex, set)
    _moi_bound!(bounds.lower, bounds.upper, bounds.positions[variable], set)
end
function _collect_moi_row!(rows, bounds, index, f, set)
    expression, target = _moi_equation!(rows, bounds, f, set)
    push!(rows.starts, length(rows.tape.nodes) + 1)
    MOI.Nonlinear.parse_expression(rows.nonlinear, rows.tape, expression, -1)
    push!(rows.targets, target)
    _keep_equation!(rows.equations, expression)
    rows.indices[index] = length(rows.targets)
end

_keep_equation!(::Nothing, expression) = nothing
_keep_equation!(equations, expression) = push!(equations, expression)

"""
Return the starts of the slack variables of `rows`, whose bounds are `lower` and
`upper`. `residual!` writes the residuals of all `row_count` rows. Each start is
the residual of its row at `point`, where the slacks are zero, projected on the
bounds. The projection chooses the start of a new variable; it hides no error.
When a row cannot be evaluated, the start is invalid anyway, and the slack starts
at its bound point nearest to zero. Without slacks, the rows need no evaluation.
"""
function _slack_start(residual!, row_count, point, rows, lower, upper)
    isempty(rows) && return Float64[]
    residual = fill(NaN, row_count)
    try
        residual!(residual, point)
    catch err
        err isa DomainError || rethrow()
        fill!(residual, NaN)
    end
    activity = residual[rows]
    clamp.(ifelse.(isfinite.(activity), activity, 0.0), lower, upper)
end
