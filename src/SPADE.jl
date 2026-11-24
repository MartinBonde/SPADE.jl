"""
The Julia interface of the SPADE solver. It has the types, options, results
and `Optimizer` of SPADE, and solves in a server process: the compiled
solver that Pkg installs with this package. The callbacks of a system run in this
process, and the server asks for each evaluation.
"""
module SPADE

using Artifacts: artifact_hash, artifact_path
using SparseArrays
using Sockets
using TOML
using MathOptInterface
import NaNMath
import PrecompileTools

const MOI = MathOptInterface

export SquareSystem,
    solve_square,
    SolveOptions,
    SquareResult,
    OptimizationSystem,
    minimize,
    OptimizationResult,
    Optimizer,
    write_failure_bundle

include("options.jl")
include("types.jl")
include("protocol.jl")
include("failure_bundle.jl")
include("moi_tape.jl")
include("moi_transform.jl")
include("moi_optimizer.jl")
include("moi_wrapper.jl")

"""
The command that starts the server; the client adds the path of the pipe. Without a
command, the first solve starts the server of the artifact `spade`, which the build
of this package downloads. The tests of the repository set another command.
"""
const SERVER_COMMAND = Ref{Cmd}()

"The command of the server in the artifact `spade`."
function _artifact_server()
    hash = artifact_hash("spade", joinpath(dirname(@__DIR__), "Artifacts.toml"))
    hash === nothing && error("The SPADE server runs only on 64-bit Windows.")
    server = joinpath(artifact_path(hash), "bin", "spade-server.exe")
    isfile(server) ||
        error("The SPADE solver is not installed. Run `import Pkg; Pkg.build(\"SPADE\")`.")
    `$server`
end

"The running server, which serves one solve at a time."
struct Server
    process::Base.Process
    connection::Connection
end

const SERVER = Ref{Union{Nothing, Server}}(nothing)
const SERVER_LOCK = ReentrantLock()

"Return the running server, or start one."
function _server()
    server = SERVER[]
    server !== nothing && process_running(server.process) && return server
    isassigned(SERVER_COMMAND) || (SERVER_COMMAND[] = _artifact_server())
    SERVER[] = _start_server(SERVER_COMMAND[])
end

function _start_server(command)
    path = Sys.iswindows() ? "\\\\.\\pipe\\spade-$(getpid())-$(time_ns())" : tempname()
    listener = Sockets.listen(path)
    process = run(pipeline(`$command $path`; stdout, stderr); wait = false)
    # A server that stops before it connects must not leave `accept` waiting.
    errormonitor(@async (wait(process); close(listener)))
    socket = try
        accept(listener)
    catch
        process_exited(process) || rethrow()
        error(
            "The SPADE server stopped with code $(process.exitcode) before it connected; see its output.",
        )
    end
    close(listener)
    Server(process, Connection(socket))
end

"Stop the server; the next solve starts a new one."
function _stop_server()
    kill(SERVER[].process)
    SERVER[] = nothing
end

"""
    solve_square(system, x0; options...)
    solve_square(system, x0, options::SolveOptions, started_ns = time_ns())

Find a root from the physical start `x0` in the server. The time limit counts from
`started_ns`, and the memory limit holds for the server. Return a `SquareResult`.
"""
function solve_square(system::SquareSystem, x0; kwargs...)
    solve_square(system, x0, SolveOptions(; kwargs...))
end

function solve_square(
    system::SquareSystem,
    x0,
    options::SolveOptions,
    started_ns = time_ns(),
)
    _remote_solve(system, x0, options, started_ns)
end

"""
    minimize(system, x0; options...)
    minimize(system, x0, options::SolveOptions, started_ns = time_ns())

Minimize the objective of `system` from the physical start `x0` in the server.
Return an `OptimizationResult`.
"""
function minimize(system::OptimizationSystem, x0; kwargs...)
    minimize(system, x0, SolveOptions(; kwargs...))
end

function minimize(
    system::OptimizationSystem,
    x0,
    options::SolveOptions,
    started_ns = time_ns(),
)
    _remote_solve(system, x0, options, started_ns)
end

"""
Solve `system` in the server. After an error during the exchange, the state of the
server is unknown, so the client stops it. When a callback fails, the server stops
the solve, and the client throws the error of the callback.
"""
function _remote_solve(system, x0, options, started_ns)
    start = _physical_start(system, x0)
    outcome = lock(SERVER_LOCK) do
        connection = _server().connection
        try
            _send_problem(connection.message, system, start, options, started_ns)
            _flush(connection)
            _serve(connection, _Evaluation(system, start))
        catch
            _stop_server()
            rethrow()
        end
    end
    _result(outcome)
end

_kind(::SquareSystem) = SOLVE_SQUARE
_kind(::OptimizationSystem) = SOLVE_OPTIMIZATION

_has_hessian(::Nothing) = false
_has_hessian(objective::Objective) = objective.hessian_product! !== nothing

function _send_problem(io, system, start, options, started_ns)
    pattern = system.jac_prototype
    write(io, _kind(system))
    write(io, system.evaluate_rows! !== nothing)
    write(io, system.domain !== _unrestricted_domain)
    write(io, _has_hessian(system.objective))
    write(io, size(pattern, 1))
    _send(io, pattern.colptr)
    _send(io, pattern.rowval)
    _send(io, system.lower_bounds)
    _send(io, system.upper_bounds)
    _send(io, system.free_variables)
    _send(io, system.row_scale)
    _send(io, start)
    _send_fields(io, options)
    write(io, (time_ns() - started_ns) / 1e9)
end

"The system, the point that the server and the client share, and the outputs."
mutable struct _Evaluation{S}
    const system::S
    const x::Vector{Float64}
    const residual::Vector{Float64}
    const jacobian::SparseMatrixCSC{Float64, Int}
    const gradient::Vector{Float64}
    const product::Vector{Float64}
    const row_positions::SparseMatrixCSC{Int, Int}
    # The first error of a callback other than a `DomainError`.
    error::Union{Nothing, Exception}
end

function _Evaluation(system, start)
    pattern = system.jac_prototype
    _Evaluation(
        system,
        copy(start),
        zeros(size(pattern, 1)),
        copy(pattern),
        zeros(length(start)),
        zeros(length(start)),
        _row_positions(pattern),
        nothing,
    )
end

"A solve that the server stopped, with its message and the error of a callback."
struct _Failure
    message::String
    error::Union{Nothing, Exception}
end

_result(result) = result
function _result(failure::_Failure)
    throw(something(failure.error, ErrorException("SPADE server: $(failure.message)")))
end

"Answer the requests of the server until it returns the result or a failure."
function _serve(connection, evaluation)
    socket = connection.socket
    while true
        code = read(socket, UInt8)
        code == RESULT && return _receive_fields(socket, _result_type(evaluation.system))
        code == FAILURE && return _Failure(_receive(socket, String), evaluation.error)
        _evaluate(Val(code), socket, connection.message, evaluation)
        _flush(connection)
    end
end

_result_type(::SquareSystem) = SquareResult
_result_type(::OptimizationSystem) = OptimizationResult

"""
Call `evaluate()`, and write its status and, when valid, the values that `send(io)`
writes. A `DomainError` makes the evaluation invalid; the client keeps another
error, and the server stops the solve.
"""
function _reply(send, io, evaluation, evaluate)
    try
        evaluate()
    catch error
        error isa DomainError || return _failed(io, evaluation, error)
        write(io, INVALID)
        return _send(io, sprint(showerror, error))
    end
    write(io, VALID)
    send(io)
end

function _failed(io, evaluation, error)
    evaluation.error = something(evaluation.error, error)
    write(io, FAILED)
end

function _evaluate(::Val{RESIDUAL}, socket, io, e)
    _receive_point!(socket, e.x)
    _reply(io -> write(io, e.residual), io, e, () -> e.system.residual!(e.residual, e.x))
end

function _evaluate(::Val{JACOBIAN}, socket, io, e)
    _receive_point!(socket, e.x)
    _reply(io, e, () -> e.system.jacobian!(e.jacobian, e.x)) do io
        write(io, nonzeros(e.jacobian))
    end
end

function _evaluate(::Val{ROWS}, socket, io, e)
    _receive_point!(socket, e.x)
    rows = _receive(socket, Vector{Int})
    residual = read(socket, Bool) ? e.residual : nothing
    jacobian = read(socket, Bool) ? e.jacobian : nothing
    _reply(io, e, () -> e.system.evaluate_rows!(residual, jacobian, e.x, rows)) do io
        _send_rows(io, residual, rows)
        _send_rows(io, jacobian, _positions(e.row_positions, rows))
    end
end

_send_rows(io, ::Nothing, positions) = nothing
_send_rows(io, residual::Vector{Float64}, rows) = _send(io, residual[rows])
function _send_rows(io, jacobian::SparseMatrixCSC, positions)
    _send(io, nonzeros(jacobian)[positions])
end

function _evaluate(::Val{DOMAIN}, socket, io, e)
    _receive_point!(socket, e.x)
    inside = Ref(false)
    _reply(io -> write(io, inside[]), io, e, () -> inside[] = e.system.domain(e.x)::Bool)
end

function _evaluate(::Val{OBJECTIVE}, socket, io, e)
    _receive_point!(socket, e.x)
    value = Ref(NaN)
    _reply(io, e, () -> value[] = e.system.objective.value(e.x)::Float64) do io
        write(io, value[])
    end
end

function _evaluate(::Val{GRADIENT}, socket, io, e)
    _receive_point!(socket, e.x)
    _reply(io, e, () -> e.system.objective.gradient!(e.gradient, e.x)) do io
        write(io, e.gradient)
    end
end

function _evaluate(::Val{HESSIAN}, socket, io, e)
    _receive_point!(socket, e.x)
    σ = read(socket, Float64)
    y = _receive(socket, Vector{Float64})
    v = _receive(socket, Vector{Float64})
    _reply(io, e, () -> e.system.objective.hessian_product!(e.product, e.x, σ, y, v)) do io
        write(io, e.product)
    end
end

include("precompile.jl")

end
