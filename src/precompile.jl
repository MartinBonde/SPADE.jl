# Compile the calls of a JuMP model before the solve into the package image, so the
# first solve runs native code, as the server does. JuMP copies its cache through
# the bridges into the optimizer, and `MOI.optimize!` transforms the rows.
PrecompileTools.@compile_workload begin
    cache = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    x, y, z = MOI.add_variables(cache, 3)
    MOI.set(cache, MOI.VariablePrimalStart(), [x, y], [1.0, 2.0])
    MOI.add_constraint(cache, y, MOI.GreaterThan(0.0))
    MOI.add_constraint(cache, z, MOI.EqualTo(3.0))
    MOI.add_constraint(
        cache,
        MOI.ScalarNonlinearFunction(
            :*,
            Any[1.0x + 2.0, MOI.ScalarNonlinearFunction(:log, Any[y]), 1.0y * z],
        ),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint(cache, 1.0x + 1.0y, MOI.EqualTo(3.0))
    backend = MOI.Utilities.CachingOptimizer(
        cache,
        MOI.instantiate(Optimizer; with_bridge_type = Float64),
    )
    MOI.Utilities.attach_optimizer(backend)
    (; system, start) = _transform_moi_model(backend.optimizer.model.model)
    jacobian = copy(system.jac_prototype)
    system.residual!(zeros(2), start)
    system.jacobian!(jacobian, start)
    system.evaluate_rows!(zeros(2), jacobian, start, [2])
end
