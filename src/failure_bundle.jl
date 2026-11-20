"""
    write_failure_bundle(path, system, result; x0, options)

Write `path/failure.toml` and return its absolute path. Record stored evidence
without evaluating the model. The model source defines any domain callback;
this file records bounds but does not serialize executable code. For an
`OptimizationSystem`, the file also records the objective, `NaN` without a
value, the row multipliers, and the reduced cost and status of each variable.
"""
function write_failure_bundle(
    path::AbstractString,
    system::SquareSystem,
    result::SquareResult;
    x0,
    options::SolveOptions,
)
    _write_evidence(path, _evidence(system, result, x0, options))
end

function write_failure_bundle(
    path::AbstractString,
    system::OptimizationSystem,
    result::OptimizationResult;
    x0,
    options::SolveOptions,
)
    payload = _evidence(system, result, x0, options)
    payload["objective"] = something(result.objective, NaN)
    payload["has_multipliers"] = result.multipliers !== nothing
    payload["multipliers"] = something(result.multipliers, Float64[])
    payload["reduced_costs"] = something(result.reduced_costs, Float64[])
    payload["statuses"] = String.(something(result.statuses, Symbol[]))
    _write_evidence(path, payload)
end

function _evidence(system, result, x0, options)
    Dict{String, Any}(
        "start" => _physical_start(system, x0),
        "has_point" => result.x !== nothing,
        "x" => something(result.x, Float64[]),
        "residual" => something(result.residual, Float64[]),
        "row_scale" => system.row_scale,
        "free_variables" => system.free_variables,
        "options" => Dict(
            String(name) => getfield(options, name) for name in fieldnames(SolveOptions)
        ),
        "domain" => Dict(
            "lower_bounds" => system.lower_bounds,
            "upper_bounds" => system.upper_bounds,
        ),
        "diagnostics" => Dict(
            "accepted" => result.accepted,
            "termination" => String(result.termination),
            "message" => result.message,
            "iterations" => result.iterations,
            "elapsed_seconds" => result.elapsed_seconds,
            "observed_memory_bytes" => result.observed_memory_bytes,
            "linear_failure" => something(result.linear_failure, ""),
            "stop_variables" => result.stop_variables,
        ),
    )
end

function _write_evidence(path, payload)
    mkpath(path)
    filename = abspath(joinpath(path, "failure.toml"))
    open(filename, "w") do io
        TOML.print(io, payload)
    end
    filename
end
