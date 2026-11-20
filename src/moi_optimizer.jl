"""
MathOptInterface optimizer for scalar affine, quadratic, or nonlinear rows with
variable bounds. An inequality row gets a slack variable. Without an objective,
`solve_square` solves the rows; with one, `minimize` does. Raw options use the
`SolveOptions` names; `MOI.TimeLimitSec` sets the time limit. Only accepted results
get `MOI.FEASIBLE_POINT`, and only results with row multipliers get a dual point.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    model::MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}
    options::SolveOptions
    result::Union{Nothing, SquareResult, OptimizationResult}
    variable_positions::Dict{MOI.VariableIndex, Int}
    constraint_rows::Dict{MOI.ConstraintIndex, Int}
    fixed::BitVector
    objective_sign::Float64
end

function Optimizer()
    Optimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        SolveOptions(),
        nothing,
        Dict{MOI.VariableIndex, Int}(),
        Dict{MOI.ConstraintIndex, Int}(),
        falses(0),
        1.0,
    )
end

# The optimizer takes rows one at a time, so JuMP copies a model into it through
# the bridges, and MOI adds no second cache.
MOI.supports_incremental_interface(::Optimizer) = true
MOI.add_variable(opt::Optimizer) = MOI.add_variable(opt.model)
MOI.add_variables(opt::Optimizer, n::Int) = MOI.add_variables(opt.model, n)
function MOI.add_constraint(
    opt::Optimizer,
    f::Union{MOI.VariableIndex, _MOI_FUNCTIONS},
    set::Union{MOI.EqualTo{Float64}, _MOI_INEQUALITIES},
)
    MOI.add_constraint(opt.model, f, set)
end
MOI.is_valid(opt::Optimizer, index::MOI.Index) = MOI.is_valid(opt.model, index)
function MOI.set(
    opt::Optimizer,
    attribute::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction, MOI.UserDefinedFunction},
    value,
)
    MOI.set(opt.model, attribute, value)
end
function MOI.set(
    opt::Optimizer,
    attribute::MOI.VariablePrimalStart,
    variable::MOI.VariableIndex,
    value,
)
    MOI.set(opt.model, attribute, variable, value)
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex, _MOI_FUNCTIONS}},
)
    true
end
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true
MOI.supports(::Optimizer, ::MOI.UserDefinedFunction) = true
function MOI.supports(::Optimizer, attribute::MOI.RawOptimizerAttribute)
    Symbol(attribute.name) in fieldnames(SolveOptions)
end

function MOI.set(opt::Optimizer, attribute::MOI.RawOptimizerAttribute, value)
    MOI.supports(opt, attribute) || throw(MOI.UnsupportedAttribute(attribute))
    opt.options = _replace(opt.options; Symbol(attribute.name) => value)
    nothing
end

function MOI.get(opt::Optimizer, attribute::MOI.RawOptimizerAttribute)
    MOI.supports(opt, attribute) || throw(MOI.UnsupportedAttribute(attribute))
    getfield(opt.options, Symbol(attribute.name))
end

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.options = _replace(opt.options; verbose = !value)
end
MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.options.verbose
function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Real)
    opt.options = _replace(opt.options; time_limit_seconds = value)
end
function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    MOI.set(opt, MOI.TimeLimitSec(), SolveOptions().time_limit_seconds)
end
MOI.get(opt::Optimizer, ::MOI.TimeLimitSec) = opt.options.time_limit_seconds

MOI.is_empty(opt::Optimizer) = MOI.is_empty(opt.model)

function MOI.empty!(opt::Optimizer)
    MOI.empty!(opt.model)
    opt.result = nothing
    empty!(opt.variable_positions)
    empty!(opt.constraint_rows)
    opt.fixed = falses(0)
    opt.objective_sign = 1.0
    nothing
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    MOI.copy_to(dest.model, src)
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.AbstractFunction},
    ::Type{<:MOI.AbstractSet},
)
    false
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{
        <:Union{
            MOI.EqualTo{Float64},
            MOI.GreaterThan{Float64},
            MOI.LessThan{Float64},
            MOI.Interval{Float64},
        },
    },
)
    true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_MOI_FUNCTIONS},
    ::Type{<:Union{MOI.EqualTo{Float64}, _MOI_INEQUALITIES}},
)
    true
end

function MOI.get(::Optimizer, ::MOI.ListOfSupportedNonlinearOperators)
    MOI.get(MOI.Nonlinear.OperatorRegistry(), MOI.ListOfSupportedNonlinearOperators())
end
