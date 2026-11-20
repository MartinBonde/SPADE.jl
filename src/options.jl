"""
    SolveOptions(; abstol = 1e-10, maxiters = 100, verbose = false,
        time_limit_seconds = 600, memory_limit_bytes = 8_000_000_000,
        optimality_tolerance = 1e-7)

`abstol` holds on the rows after scaling. In `solve_square`, a row whose largest term
`|J[i, j] x[j]|` exceeds one is divided by that term.
`maxiters` counts all trials across equation groups, rejected trials included.
Time and memory limits are soft: the solver checks them between operations.
Time covers conversion, compilation, and numerical work after entry. Memory is the
resident memory of the process. At a local optimum of `minimize`, each scaled
reduced gradient is at most `optimality_tolerance` times `max(1, |objective|)`, as
CONOPT's `Tol_Optimality`.
"""
Base.@kwdef struct SolveOptions
    abstol::Float64 = 1e-10
    maxiters::Int = 100
    verbose::Bool = false
    time_limit_seconds::Float64 = 600.0
    memory_limit_bytes::Int = 8_000_000_000
    optimality_tolerance::Float64 = 1e-7

    function SolveOptions(
        abstol,
        maxiters,
        verbose,
        time_limit_seconds,
        memory_limit_bytes,
        optimality_tolerance,
    )
        isfinite(abstol) && abstol >= 0 ||
            throw(ArgumentError("abstol must be finite and nonnegative"))
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        isfinite(time_limit_seconds) && time_limit_seconds >= 0 ||
            throw(ArgumentError("time_limit_seconds must be finite and nonnegative"))
        memory_limit_bytes >= 0 ||
            throw(ArgumentError("memory_limit_bytes must be nonnegative"))
        isfinite(optimality_tolerance) && optimality_tolerance > 0 ||
            throw(ArgumentError("optimality_tolerance must be finite and positive"))
        new(
            abstol,
            maxiters,
            verbose,
            time_limit_seconds,
            memory_limit_bytes,
            optimality_tolerance,
        )
    end
end

"Return `options` with the given fields replaced."
function _replace(options::SolveOptions; changes...)
    SolveOptions(;
        (name => getfield(options, name) for name in fieldnames(SolveOptions))...,
        changes...,
    )
end
