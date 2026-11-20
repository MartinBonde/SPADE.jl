const _Node = MOI.Nonlinear.Node
const _NODE_VARIABLE = MOI.Nonlinear.NODE_VARIABLE
const _NODE_MOI_VARIABLE = MOI.Nonlinear.NODE_MOI_VARIABLE
const _NODE_VALUE = MOI.Nonlinear.NODE_VALUE
const _NODE_CALL_MULTIVARIATE = MOI.Nonlinear.NODE_CALL_MULTIVARIATE
const _NODE_CALL_UNIVARIATE = MOI.Nonlinear.NODE_CALL_UNIVARIATE
const _NODE_COMPARISON = MOI.Nonlinear.NODE_COMPARISON
const _NODE_LOGIC = MOI.Nonlinear.NODE_LOGIC

"""
All rows of a model on one tape for reverse-mode derivatives. MOI's evaluator
keeps about 15 arrays for each row, which for millions of rows take many
gigabytes and much time to collect. The tape keeps one set of arrays, and does
the arithmetic of `MOI.Nonlinear.ReverseAD` in the same order, so values and
derivatives are the same to the last bit.

`nodes` holds the nodes of each row as MOI parses them: a parent comes before its
children, and each parent and value index counts from the start of the tape. The
nodes of row `i` are `starts[i]:starts[i + 1] - 1`. `children` holds the children
of each node in argument order, from `child_starts`, and `columns` the sorted
variables of each row, from `column_starts`. The other vectors hold the work of
one row.
"""
struct _RowTape
    operators::MOI.Nonlinear.OperatorRegistry
    nodes::Vector{_Node}
    values::Vector{Float64}
    starts::Vector{Int}
    child_starts::Vector{Int}
    children::Vector{Int}
    column_starts::Vector{Int}
    columns::Vector{Int}
    forward::Vector{Float64}
    partials::Vector{Float64}
    reverse::Vector{Float64}
    gradient::Vector{Float64}
    arguments::Vector{Float64}
    argument_gradient::Vector{Float64}
end

"""
Return the tape of `expression`, where MOI parsed row `i` from node `starts[i]`.
Variable `j` of `physical` becomes column `j`; a fixed variable, with
`lower[j] == upper[j]`, becomes its value. The tape takes the arrays of `expression`.
"""
function _RowTape(operators, expression, starts, physical, lower, upper)
    positions = Dict(variable => j for (j, variable) in pairs(physical))
    values = expression.values
    nodes = map!(expression.nodes, expression.nodes) do node
        node.type == _NODE_MOI_VARIABLE || return node
        j = positions[MOI.VariableIndex(node.index)]
        lower[j] == upper[j] || return _Node(_NODE_VARIABLE, j, node.parent)
        push!(values, lower[j])
        _Node(_NODE_VALUE, length(values), node.parent)
    end
    row_starts = [starts; length(nodes) + 1]
    child_counts = zeros(Int, length(nodes))
    foreach(node -> node.parent > 0 && (child_counts[node.parent] += 1), nodes)
    child_starts = cumsum([1; child_counts])
    children = zeros(Int, length(nodes) - length(starts))
    next = child_starts[1:(end - 1)]
    foreach(pairs(nodes)) do (k, node)
        node.parent > 0 || return
        children[next[node.parent]] = k
        next[node.parent] += 1
    end
    columns, seen = Int[], zeros(Int, length(physical))
    column_starts = [
        1
        [
            _append_columns!(columns, seen, nodes, row_starts, row) for
            row in eachindex(starts)
        ]
    ]
    longest = maximum(diff(row_starts); init = 0)
    widest = maximum(child_counts; init = 0)
    _RowTape(
        operators,
        nodes,
        values,
        row_starts,
        child_starts,
        children,
        column_starts,
        columns,
        zeros(longest),
        zeros(longest),
        zeros(longest),
        zeros(length(physical)),
        zeros(widest),
        zeros(widest),
    )
end

"Append the sorted variables of `row` to `columns`, and return the next start."
function _append_columns!(columns, seen, nodes, row_starts, row)
    first = length(columns) + 1
    for node in view(nodes, row_starts[row]:(row_starts[row + 1] - 1))
        node.type == _NODE_VARIABLE && seen[node.index] != row || continue
        seen[node.index] = row
        push!(columns, node.index)
    end
    sort!(view(columns, first:length(columns)))
    length(columns) + 1
end

"""
Evaluate `rows` of `tape` at `x`. Write each value minus its target to
`residual`, and each derivative to its entry in `jacobian`: entry `k` of the
columns of the tape is `nonzeros(jacobian)[positions[k]]`. `nothing` skips an
output.
"""
function _evaluate_tape!(residual, jacobian, tape, x, rows, targets, positions)
    # A `DomainError` names the row where it arises.
    current = 0
    try
        for row in rows
            current = row
            value = _forward!(tape, x, row)
            _store_value!(residual, row, value - targets[row])
            _store_gradient!(jacobian, tape, row, positions)
        end
    catch err
        err isa DomainError || rethrow()
        throw(DomainError(err.val, "Row $current: $(err.msg)"))
    end
end

_store_value!(::Nothing, row, value) = nothing
_store_value!(residual, row, value) = (residual[row] = value)

_store_gradient!(::Nothing, tape, row, positions) = nothing
function _store_gradient!(jacobian, tape, row, positions)
    _reverse!(tape, row)
    entries = tape.column_starts[row]:(tape.column_starts[row + 1] - 1)
    gradient = tape.gradient
    foreach(k -> gradient[tape.columns[k]] = 0.0, entries)
    offset = tape.starts[row] - 1
    for k in tape.starts[row]:(tape.starts[row + 1] - 1)
        node = tape.nodes[k]
        node.type == _NODE_VARIABLE && (gradient[node.index] += tape.reverse[k - offset])
    end
    foreach(k -> nonzeros(jacobian)[positions[k]] = gradient[tape.columns[k]], entries)
end

"""
Return the value of `row` at `x`, and store the derivative of each parent by each
child, as `MOI.Nonlinear.ReverseAD._forward_eval` does. The node type is a value,
so the loop branches on it instead of dispatching.
"""
function _forward!(tape, x, row)
    offset = tape.starts[row] - 1
    for k in (tape.starts[row + 1] - 1):-1:tape.starts[row]
        node = tape.nodes[k]
        children = tape.child_starts[k]:(tape.child_starts[k + 1] - 1)
        tape.partials[k - offset] = 0.0
        tape.forward[k - offset] = if node.type == _NODE_VARIABLE
            x[node.index]
        elseif node.type == _NODE_VALUE
            tape.values[node.index]
        elseif node.type == _NODE_CALL_MULTIVARIATE
            _multivariate!(tape, node.index, children, offset)
        elseif node.type == _NODE_CALL_UNIVARIATE
            _univariate!(tape, node.index, tape.children[first(children)] - offset)
        elseif node.type == _NODE_COMPARISON
            _comparison!(tape, node.index, children, offset)
        else
            @assert node.type == _NODE_LOGIC
            _logic!(tape, node.index, children, offset)
        end
    end
    tape.forward[1]
end

function _univariate!(tape, id, i)
    value, derivative = MOI.Nonlinear.eval_univariate_function_and_gradient(
        tape.operators,
        id,
        tape.forward[i],
    )
    tape.partials[i] = derivative
    convert(Float64, value)
end

"The operator `id` on `children`, whose nodes count from `offset`."
function _multivariate!(tape, id, children, offset)
    f, p = tape.forward, tape.partials
    child(r) = tape.children[r] - offset
    if id == 1 # +
        total = 0.0
        for r in children
            p[child(r)] = 1.0
            total += f[child(r)]
        end
        total
    elseif id == 2 # -
        @assert length(children) == 2
        i, j = child(first(children)), child(last(children))
        p[i], p[j] = 1.0, -1.0
        f[i] - f[j]
    elseif id == 3 # *
        product = 1.0
        for r in children
            product *= f[child(r)]
        end
        # As in MOI, the product divided by a factor gives the other factors only
        # for three or more nonzero factors. A closure here would box `product`.
        multiply = product == 0.0 || length(children) <= 2
        for r in children
            p[child(r)] =
                multiply ? _other_factors(f, children, r, child) : product / f[child(r)]
        end
        product
    elseif id == 4 # ^
        @assert length(children) == 2
        i, j = child(first(children)), child(last(children))
        base, exponent = f[i], f[j]
        value, p[i] =
            exponent == 2 ? (base * base, 2 * base) :
            exponent == 1 ? (base, 1.0) :
            (NaNMath.pow(base, exponent), exponent * NaNMath.pow(base, exponent - 1))
        p[j] = value * NaNMath.log(base)
        value
    elseif id == 5 # /
        @assert length(children) == 2
        i, j = child(first(children)), child(last(children))
        numerator, inverse = f[i], 1 / f[j]
        p[i], p[j] = inverse, -numerator * inverse * inverse
        numerator * inverse
    elseif id == 6 # ifelse
        @assert length(children) == 3
        c, i, j = child.(first(children) .+ (0, 1, 2))
        condition = f[c] == 1
        p[i], p[j] = condition, !condition
        ifelse(condition, f[i], f[j])
    else
        _registered_multivariate!(tape, id, child.(children))
    end
end

function _other_factors(f, children, r, child)
    others = 1.0
    for s in children
        s == r || (others *= f[child(s)])
    end
    others
end

function _registered_multivariate!(tape, id, nodes)
    arguments = view(tape.arguments, eachindex(nodes))
    gradient = view(tape.argument_gradient, eachindex(nodes))
    arguments .= view(tape.forward, nodes)
    gradient .= 0.0
    operator = tape.operators.multivariate_operators[id]
    value = MOI.Nonlinear.eval_multivariate_function(tape.operators, operator, arguments)
    MOI.Nonlinear.eval_multivariate_gradient(tape.operators, operator, gradient, arguments)
    tape.partials[nodes] = gradient
    convert(Float64, value)
end

function _comparison!(tape, id, children, offset)
    nodes = tape.children[children] .- offset
    operator = tape.operators.comparison_operators[id]
    tape.partials[nodes] .= 0.0
    f = tape.forward
    comparisons = (
        MOI.Nonlinear.eval_comparison_function(tape.operators, operator, f[i], f[j]) for
        (i, j) in zip(nodes[1:(end - 1)], nodes[2:end])
    )
    Float64(all(comparisons))
end

function _logic!(tape, id, children, offset)
    i, j = tape.children[children[1:2]] .- offset
    tape.partials[i] = tape.partials[j] = 0.0
    operator = tape.operators.logic_operators[id]
    f = tape.forward
    Float64(
        MOI.Nonlinear.eval_logic_function(tape.operators, operator, f[i] == 1, f[j] == 1),
    )
end

"""
Store the derivative of `row` by each of its nodes, as
`MOI.Nonlinear.ReverseAD._reverse_eval` does. A value, comparison or logic node
gets zero, as MOI's evaluator keeps from its start.
"""
function _reverse!(tape, row)
    offset = tape.starts[row] - 1
    tape.reverse[1] = 1.0
    for k in (tape.starts[row] + 1):(tape.starts[row + 1] - 1)
        node = tape.nodes[k]
        parent, partial = tape.reverse[node.parent - offset], tape.partials[k - offset]
        tape.reverse[k - offset] =
            node.type == _NODE_VALUE ||
            node.type == _NODE_COMPARISON ||
            node.type == _NODE_LOGIC ? 0.0 :
            ifelse(parent == 0.0 && !isfinite(partial), parent, parent * partial)
    end
end
