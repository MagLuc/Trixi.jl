module TimeDisc

include("pairedrk.jl")

using ..Trixi
using ..Solvers: AbstractSolver, rhs!, update_level_info!
using ..Auxiliary: timer, parameter
using ..Mesh: TreeMesh
using ..Mesh.Trees: minimum_level, maximum_level
using .PairedRk: calc_coefficients, calc_c, calc_a_multilevel
using TimerOutputs: @timeit

export timestep!


# Second-order paired Runge-Kutta method (multilevel version)
function timestep!(solver::AbstractSolver, mesh::TreeMesh,
                   ::Val{:paired_rk_2_multi}, t::Float64, dt::Float64)
  # Get parameters
  n_stages = parameter("n_stages", valid=(2, 4, 8, 16))
  derivative_evaluations = parameter("derivative_evaluations", valid=(2, 4, 8, 16))

  # Determine Runge-Kutta coefficients "c"
  c = calc_c(n_stages)

  # Store for convenience
  u   = solver.elements.u
  k   = solver.elements.u_t
  k1 = solver.elements.u_rungekutta
  un  = similar(k)

  # Implement general Runge-Kutta method (not storage-optimized) for paired RK schemes, where
  # aᵢⱼ= 0 except for j = 1 or j = i - 1
  # bₛ = 1, bᵢ = 0  for i ≠ s
  # c₁ = 0
  #
  #                 s
  # uⁿ⁺¹ = uⁿ + Δt  ∑ bᵢkᵢ = uⁿ + Δt kₛ
  #                i=1
  # k₁ = rhs(tⁿ, uⁿ)
  # k₂ = rhs(tⁿ + c₂Δt, uⁿ + Δt(a₂₁ k₁))
  # k₃ = rhs(tⁿ + c₃Δt, uⁿ + Δt(a₃₁ k₁ + a₃₂ k₂))
  # k₄ = rhs(tⁿ + c₄Δt, uⁿ + Δt(a₄₁ k₁ + a₄₃ k₃))
  # ...
  # kₛ = rhs(tⁿ + cₛΔt, uⁿ + Δt(aₛ₁ k₁ + aₛ,ₛ₋₁ kₛ₋₁))

  # Update level info for each element
  @timeit timer() "update level info" update_level_info!(solver, mesh)

  # Stage 1
  stage = 1
  t_stage = t + dt * c[stage]
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Store permanently
  @timeit timer() "Runge-Kutta step" begin
    @. un = u
    @. k1 = k
  end

  # Stage 2
  stage = 2
  t_stage = t + dt * c[stage]
  @timeit timer() "calc_a_multilevel" a_1, _ = calc_a_multilevel(n_stages,
                                                                 stage,
                                                                 derivative_evaluations,
                                                                 solver.n_elements,
                                                                 solver.level_info_elements)
  a_1_rs = reshape(a_1, 1, 1, 1, :)
  @timeit timer() "Runge-Kutta step" @. u = un + dt * a_1_rs * k1
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Stages 3-n_stages
  for stage in 3:n_stages
    t_stage = t + dt * c[stage]
    @timeit timer() "calc_a_multilevel" a_1, a_2 = calc_a_multilevel(n_stages,
                                                                     stage,
                                                                     derivative_evaluations,
                                                                     solver.n_elements,
                                                                     solver.level_info_elements)
    a_1_rs = reshape(a_1, 1, 1, 1, :)
    a_2_rs = reshape(a_2, 1, 1, 1, :)
    @timeit timer() "Runge-Kutta step" @. u = un + dt * (a_1_rs * k1 + a_2_rs * k)
    @timeit timer() "rhs" rhs!(solver, t_stage, stage)
  end

  # Final update to u
  @timeit timer() "Runge-Kutta step" @. u = un + dt * k
end


# Second-order paired Runge-Kutta method
function timestep!(solver::AbstractSolver, mesh::TreeMesh,
                   ::Val{:paired_rk_2_s}, t::Float64, dt::Float64)
  # Get parameters
  n_stages = parameter("n_stages", valid=(2, 4, 8, 16))
  derivative_evaluations = parameter("derivative_evaluations", valid=(2, 4, 8, 16))

  # Determine Runge-Kutta coefficients
  a, c = calc_coefficients(n_stages, derivative_evaluations)

  # Store for convenience
  u   = solver.elements.u
  k   = solver.elements.u_t
  k1 = solver.elements.u_rungekutta
  un  = similar(k)

  # Implement general Runge-Kutta method (not storage-optimized) for paired RK schemes, where
  # aᵢⱼ= 0 except for j = 1 or j = i - 1
  # bₛ = 1, bᵢ = 0  for i ≠ s
  # c₁ = 0
  #
  #                 s
  # uⁿ⁺¹ = uⁿ + Δt  ∑ bᵢkᵢ = uⁿ + Δt kₛ
  #                i=1
  # k₁ = rhs(tⁿ, uⁿ)
  # k₂ = rhs(tⁿ + c₂Δt, uⁿ + Δt(a₂₁ k₁))
  # k₃ = rhs(tⁿ + c₃Δt, uⁿ + Δt(a₃₁ k₁ + a₃₂ k₂))
  # k₄ = rhs(tⁿ + c₄Δt, uⁿ + Δt(a₄₁ k₁ + a₄₃ k₃))
  # ...
  # kₛ = rhs(tⁿ + cₛΔt, uⁿ + Δt(aₛ₁ k₁ + aₛ,ₛ₋₁ kₛ₋₁))

  # Stage 1
  stage = 1
  t_stage = t + dt * c[stage]
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Store permanently
  @timeit timer() "Runge-Kutta step" begin
    @. un = u
    @. k1 = k
  end

  # Stage 2
  stage = 2
  t_stage = t + dt * c[stage]
  @timeit timer() "Runge-Kutta step" @. u = un + dt * a[ 2, 1] * k1
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Stages 3-n_stages
  for stage in 3:n_stages
    t_stage = t + dt * c[stage]
    @timeit timer() "Runge-Kutta step" @. u = un + dt * (a[stage, 1] * k1 + a[stage, stage-1] * k)
    @timeit timer() "rhs" rhs!(solver, t_stage, stage)
  end

  # Final update to u
  @timeit timer() "Runge-Kutta step" @. u = un + dt * k
end


# Carpenter's 4th-order 5-stage low-storage Runge-Kutta method
function timestep!(solver::AbstractSolver, mesh::TreeMesh,
                   ::Val{:carpenter_4_5}, t::Float64, dt::Float64)
  a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  for stage = 1:5
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end


end
