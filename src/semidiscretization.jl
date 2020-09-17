
# TODO: Taal Mesh<:AbstractMesh{NDIMS}, Equations<:AbstractEquations{NDIMS} ?
"""
    Semidiscretization

A struct containing everything needed to describe a spatial Semidiscretization
of a hyperbolic conservation law.
"""
struct Semidiscretization{Mesh, Equations, InitialConditions, BoundaryConditions, SourceTerms, Solver, Cache}
  mesh::Mesh

  equations::Equations

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_conditions::InitialConditions

  # TODO: Taal BCs
  boundary_conditions::BoundaryConditions

  source_terms::SourceTerms

  solver::Solver

  cache::Cache

  function Semidiscretization{Mesh, Equations, InitialConditions, BoundaryConditions, SourceTerms, Solver, Cache}(
      mesh::Mesh, equations::Equations,
      initial_conditions::InitialConditions, boundary_conditions::BoundaryConditions,
      source_terms::SourceTerms,
      solver::Solver, cache::Cache) where {Mesh, Equations, InitialConditions, BoundaryConditions, SourceTerms, Solver, Cache}
    @assert ndims(mesh) == ndims(equations)

    new(mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
  end
end

@inline source_terms_nothing(args...) = nothing

function Semidiscretization(mesh, equations, initial_conditions, solver;
                            source_terms=source_terms_nothing,
                            boundary_conditions=nothing, RealT=real(solver))

  cache = create_cache(mesh, equations, boundary_conditions, solver, RealT)

  Semidiscretization{typeof(mesh), typeof(equations), typeof(initial_conditions), typeof(boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
end

# TODO: Taal bikeshedding, implement a method with reduced information and the signature
# function Base.show(io::IO, semi::Semidiscretization)
function Base.show(io::IO, ::MIME"text/plain", semi::Semidiscretization)
  println(io, "Semidiscretization using")
  println(io, "- ", semi.mesh)
  println(io, "- ", semi.equations)
  println(io, "- ", semi.initial_conditions)
  println(io, "- ", semi.boundary_conditions)
  println(io, "- ", semi.source_terms)
  print(io, "- cache with fields:")
  for key in keys(semi.cache)
    print(io, " ", key)
  end
end


@inline Base.ndims(semi::Semidiscretization) = ndims(semi.mesh)

@inline nvariables(semi::Semidiscretization) = nvariables(semi.equations)

@inline nnodes(semi::Semidiscretization) = nnodes(semi.solver)

@inline ndofs(semi::Semidiscretization) = ndofs(semi.mesh, semi.solver, semi.cache)



@inline function get_node_coords(x, semi::Semidiscretization, indices...)
  @unpack equations, solver = semi

  get_node_coords(x, equations, solver, indices...)
end

@inline function get_node_vars(u, semi::Semidiscretization, indices...)
  @unpack equations, solver = semi

  get_node_vars(u, equations, solver, indices...)
end

@inline function get_surface_node_vars(u, semi::Semidiscretization, indices...)
  @unpack equations, solver = semi

  get_surface_node_vars(u, equations, solver, indices...)
end

@inline function set_node_vars!(u, u_node, semi::Semidiscretization, indices...)
  @unpack equations, solver = semi

  set_node_vars!(u, u_node, equations, solver, indices...)
  return nothing
end

@inline function add_to_node_vars!(u, u_node, semi::Semidiscretization, indices...)
  @unpack equations, solver = semi

  add_to_node_vars!(u, u_node, equations, solver, indices...)
  return nothing
end


function integrate(func, u, semi::Semidiscretization; normalize=true)
  @unpack mesh, equations, solver, cache = semi

  integrate(func, u, mesh, equations, solver, cache)
end


function calc_error_norms(func, u, t, analyzer, semi::Semidiscretization)
  @unpack mesh, equations, initial_conditions, solver, cache = semi

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_conditions, solver, cache)
end
calc_error_norms(u, t, analyzer, semi::Semidiscretization) = calc_error_norms(cons2cons, u, t, analyzer, semi)


function compute_coefficients(t, semi::Semidiscretization)
  compute_coefficients(semi.initial_conditions, t, semi)
end

function compute_coefficients(func, t, semi::Semidiscretization)
  @unpack mesh, equations, solver, cache = semi

  u = allocate_coefficients(mesh, equations, solver, cache)
  compute_coefficients!(u, func, t, semi)
  u
end

function compute_coefficients!(u, func, t, semi::Semidiscretization)
  @unpack mesh, equations, solver, cache = semi

  compute_coefficients!(u, func, t, mesh, equations, solver, cache)
end


function semidiscretize(semi::Semidiscretization, tspan)
  u0 = compute_coefficients(first(tspan), semi)
  ODEProblem(rhs!, u0, tspan, semi)
end


# TODO: Taal better name
function rhs!(du, u, semi::Semidiscretization, t)
  @unpack mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache = semi

  # TODO: Taal decide, do we need to pass the mesh?
  rhs!(du, u, t, mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
end


# TODO: Taal implement/interface
# struct CacheVariable{X}
#   cache::X
#   visualize::Bool
# end



# TODO: Taal interface
# New mesh/solver combinations have to implement
# - ndims(mesh)
# - nnodes(solver)
# - real(solver)
# - ndofs(mesh, solver, cache)
# - create_cache(mesh, equations, boundary_conditions, solver)
# - integrate(func, u, mesh, equations, solver, cache)
# - calc_error_norms(func, u, t, analyzer, mesh, equations, initial_conditions, solver, cache)
# - allocate_coefficients(mesh, equations, solver, cache)
# - compute_coefficients!(u, func, mesh, equations, solver, cache)
# - rhs!(du, u, t, mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
