# The messages between the SPADE client and server, which both include.
#
# The client starts the server and sends a problem: the pattern, bounds and start of
# the system, the options, and the seconds that passed before the request. The server
# solves it and asks the client for each evaluation. A request carries the entries of
# the point that changed since the last request. The client replies with a status
# and, when valid, the values. The server ends the solve with the result, or with a
# failure and its message.

# Requests of the client.
const SOLVE_SQUARE = 0x01
const SOLVE_OPTIMIZATION = 0x02
# Requests of the server during a solve, and the two messages that end it.
const RESIDUAL = 0x11
const JACOBIAN = 0x12
const ROWS = 0x13
const DOMAIN = 0x14
const OBJECTIVE = 0x15
const GRADIENT = 0x16
const HESSIAN = 0x17
const RESULT = 0x18
const FAILURE = 0x19
# The status of a reply: the values follow; a `DomainError` message follows; or
# another error stopped the callback, and the client keeps it.
const VALID = 0x00
const INVALID = 0x01
const FAILED = 0x02

"A socket, and the buffer of the message that this side sends next."
struct Connection
    socket::Base.PipeEndpoint
    message::IOBuffer
end

Connection(socket) = Connection(socket, IOBuffer())

"Send the buffered message as one write."
_flush(connection) = write(connection.socket, take!(connection.message))

_send(io, value::Union{Bool, Int, Float64}) = write(io, value)
function _send(io, values::Vector{<:Union{Int, Float64}})
    (write(io, length(values)); write(io, values))
end
_send(io, text::String) = (write(io, ncodeunits(text)); write(io, text))
_send(io, name::Symbol) = _send(io, String(name))
function _send(io, names::Vector{Symbol})
    (write(io, length(names)); foreach(n -> _send(io, n), names))
end

_receive(io, T::Type{<:Union{Bool, Int, Float64}}) = read(io, T)
_receive(io, ::Type{Vector{T}}) where {T} = read!(io, Vector{T}(undef, read(io, Int)))
_receive(io, ::Type{String}) = String(read(io, read(io, Int)))
_receive(io, ::Type{Symbol}) = Symbol(_receive(io, String))
_receive(io, ::Type{Vector{Symbol}}) = [_receive(io, Symbol) for _ in 1:read(io, Int)]

_send_optional(io, ::Nothing) = write(io, false)
_send_optional(io, value) = (write(io, true); _send(io, value))

_send_field(io, value, ::Type) = _send(io, value)
_send_field(io, value, ::Type{Union{Nothing, T}}) where {T} = _send_optional(io, value)
_receive_field(io, T::Type) = _receive(io, T)
function _receive_field(io, ::Type{Union{Nothing, T}}) where {T}
    read(io, Bool) ? _receive(io, T) : nothing
end

"Send each field of `value`: `SolveOptions`, a `SquareResult` or an `OptimizationResult`."
function _send_fields(io, value::T) where {T}
    foreach(fieldnames(T)) do name
        _send_field(io, getfield(value, name), fieldtype(T, name))
    end
end

function _receive_fields(io, T)
    T((_receive_field(io, fieldtype(T, name)) for name in fieldnames(T))...)
end

"Send the entries of `x` that differ from `sent`, the point of the client, and update `sent`."
function _send_point(io, x, sent)
    changed = findall(.!isequal.(x, sent))
    sent[changed] = x[changed]
    _send(io, changed)
    _send(io, sent[changed])
end

"Receive the entries of the point `x` that changed."
function _receive_point!(io, x)
    changed = _receive(io, Vector{Int})
    x[changed] = _receive(io, Vector{Float64})
end

"""
The positions in `nonzeros(pattern)` of the entries of each row: column `i` of the
result holds those of row `i`.
"""
function _row_positions(pattern)
    numbered = SparseMatrixCSC(
        size(pattern)...,
        pattern.colptr,
        pattern.rowval,
        collect(eachindex(pattern.rowval)),
    )
    copy(transpose(numbered))
end

"The positions in `nonzeros(pattern)` of the entries of `rows`."
function _positions(row_positions, rows)
    [p for row in rows for p in view(nonzeros(row_positions), nzrange(row_positions, row))]
end
