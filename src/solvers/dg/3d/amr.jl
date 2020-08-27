# This file contains functions that are related to the AMR capabilities of the DG solver

# Refine elements in the DG solver based on a list of cell_ids that should be refined
function refine!(dg::Dg3D{Eqn, NVARS, N}, mesh::TreeMesh,
                 cells_to_refine::AbstractArray{Int}) where {Eqn, NVARS, N}
  # Return early if there is nothing to do
  if isempty(cells_to_refine)
    return
  end

  # Determine for each existing element whether it needs to be refined
  needs_refinement = falses(nelements(dg.elements))
  tree = mesh.tree
  # The "Ref(...)" is such that we can vectorize the search but not the array that is searched
  elements_to_refine = searchsortedfirst.(Ref(dg.elements.cell_ids[1:nelements(dg.elements)]),
                                          cells_to_refine)
  needs_refinement[elements_to_refine] .= true

  # Retain current solution data
  old_n_elements = nelements(dg.elements)
  old_u = dg.elements.u

  # Get new list of leaf cells
  leaf_cell_ids = leaf_cells(tree)

  # Initialize new elements container
  elements = init_elements(leaf_cell_ids, mesh, Val(NVARS), Val(N))
  n_elements = nelements(elements)

  # Loop over all elements in old container and either copy them or refine them
  element_id = 1
  for old_element_id in 1:old_n_elements
    if needs_refinement[old_element_id]
      # Refine element and store solution directly in new data structure
      refine_element!(elements.u, element_id, old_u, old_element_id,
                      dg.mortar_forward_upper, dg.mortar_forward_lower, dg)
      element_id += 2^ndims(dg)
    else
      # Copy old element data to new element container
      @views elements.u[:, :, :, :, element_id] .= old_u[:, :, :, :, old_element_id]
      element_id += 1
    end
  end

  # Initialize new interfaces container
  interfaces = init_interfaces(leaf_cell_ids, mesh, Val(NVARS), Val(N), elements)
  n_interfaces = ninterfaces(interfaces)

  # Initialize boundaries
  boundaries = init_boundaries(leaf_cell_ids, mesh, Val(NVARS), Val(N), elements)
  n_boundaries = nboundaries(boundaries)

  # Initialize new mortar containers
  l2mortars = init_mortars(leaf_cell_ids, mesh, Val(NVARS), Val(N), elements, dg.mortar_type)
  n_l2mortars = nmortars(l2mortars)

  # Sanity check
  if isperiodic(mesh.tree) && n_l2mortars == 0
    @assert n_interfaces == 3*n_elements ("For 3D and periodic domains and conforming elements, "
                                        * "n_surf must be the same as 3*n_elem")
  end

  # Update DG instance with new data
  dg.elements = elements
  dg.n_elements = n_elements
  dg.interfaces = interfaces
  dg.n_interfaces = n_interfaces
  dg.boundaries = boundaries
  dg.n_boundaries = n_boundaries
  dg.l2mortars = l2mortars
  dg.n_l2mortars = n_l2mortars
end


# Refine solution data u for an element, using L2 projection (interpolation)
function refine_element!(u, element_id, old_u, old_element_id,
                         forward_upper, forward_lower, dg::Dg3D)
  # Store new element ids
  bottom_lower_left_id  = element_id
  bottom_lower_right_id = element_id + 1
  bottom_upper_left_id  = element_id + 2
  bottom_upper_right_id = element_id + 3
  top_lower_left_id     = element_id + 4
  top_lower_right_id    = element_id + 5
  top_upper_left_id     = element_id + 6
  top_upper_right_id    = element_id + 7

  # Interpolate to bottom lower left element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_lower[i, ii]
              * forward_lower[j, jj]
              * forward_lower[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, bottom_lower_left_id)
  end

  # Interpolate to bottom lower right element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_upper[i, ii]
              * forward_lower[j, jj]
              * forward_lower[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, bottom_lower_right_id)
  end

  # Interpolate to bottom upper left element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_lower[i, ii]
              * forward_upper[j, jj]
              * forward_lower[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, bottom_upper_left_id)
  end

  # Interpolate to bottom upper right element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_upper[i, ii]
              * forward_upper[j, jj]
              * forward_lower[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, bottom_upper_right_id)
  end

  # Interpolate to top lower left element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_lower[i, ii]
              * forward_lower[j, jj]
              * forward_upper[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, top_lower_left_id)
  end

  # Interpolate to top lower right element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_upper[i, ii]
              * forward_lower[j, jj]
              * forward_upper[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, top_lower_right_id)
  end

  # Interpolate to top upper left element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_lower[i, ii]
              * forward_upper[j, jj]
              * forward_upper[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, top_upper_left_id)
  end

  # Interpolate to top upper right element
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, old_element_id)
              * forward_upper[i, ii]
              * forward_upper[j, jj]
              * forward_upper[k, kk])
    end
    set_node_vars!(u, acc, dg, i, j, k, top_upper_right_id)
  end
end


# Coarsen elements in the DG solver based on a list of cell_ids that should be removed
function coarsen!(dg::Dg3D{Eqn, NVARS, N}, mesh::TreeMesh,
                  child_cells_to_coarsen::AbstractArray{Int}) where {Eqn, NVARS, N}
  # Return early if there is nothing to do
  if isempty(child_cells_to_coarsen)
    return
  end

  # Determine for each old element whether it needs to be removed
  to_be_removed = falses(nelements(dg.elements))
  # The "Ref(...)" is such that we can vectorize the search but not the array that is searched
  elements_to_remove = searchsortedfirst.(Ref(dg.elements.cell_ids[1:nelements(dg.elements)]),
                                          child_cells_to_coarsen)
  to_be_removed[elements_to_remove] .= true

  # Retain current solution data
  old_n_elements = nelements(dg.elements)
  old_u = dg.elements.u

  # Get new list of leaf cells
  leaf_cell_ids = leaf_cells(mesh.tree)

  # Initialize new elements container
  elements = init_elements(leaf_cell_ids, mesh, Val(NVARS), Val(N))
  n_elements = nelements(elements)

  # Loop over all elements in old container and either copy them or coarsen them
  skip = 0
  element_id = 1
  for old_element_id in 1:old_n_elements
    # If skip is non-zero, we just coarsened 2^ndims elements and need to omit the following elements
    if skip > 0
      skip -= 1
      continue
    end

    if to_be_removed[old_element_id]
      # If an element is to be removed, sanity check if the following elements
      # are also marked - otherwise there would be an error in the way the
      # cells/elements are sorted
      @assert all(to_be_removed[old_element_id:(old_element_id+2^ndims(dg)-1)]) "bad cell/element order"

      # Coarsen elements and store solution directly in new data structure
      coarsen_elements!(elements.u, element_id, old_u, old_element_id,
                        dg.l2mortar_reverse_upper, dg.l2mortar_reverse_lower, dg)
      element_id += 1
      skip = 2^ndims(dg) - 1
    else
      # Copy old element data to new element container
      @views elements.u[:, :, :, :, element_id] .= old_u[:, :, :, :, old_element_id]
      element_id += 1
    end
  end

  # Initialize new interfaces container
  interfaces = init_interfaces(leaf_cell_ids, mesh, Val(NVARS), Val(N), elements)
  n_interfaces = ninterfaces(interfaces)

  # Initialize boundaries
  boundaries = init_boundaries(leaf_cell_ids, mesh, Val(NVARS), Val(N), elements)
  n_boundaries = nboundaries(boundaries)

  # Initialize new mortar containers
  l2mortars = init_mortars(leaf_cell_ids, mesh, Val(NVARS), Val(N), elements, dg.mortar_type)
  n_l2mortars = nmortars(l2mortars)

  # Sanity check
  if isperiodic(mesh.tree) && n_l2mortars == 0
    @assert n_interfaces == 3*n_elements ("For 3D and periodic domains and conforming elements, "
                                        * "n_surf must be the same as 3*n_elem")
  end

  # Update DG instance with new data
  dg.elements = elements
  dg.n_elements = n_elements
  dg.interfaces = interfaces
  dg.n_interfaces = n_interfaces
  dg.boundaries = boundaries
  dg.n_boundaries = n_boundaries
  dg.l2mortars = l2mortars
  dg.n_l2mortars = n_l2mortars
end


# Coarsen solution data u for four elements, using L2 projection
function coarsen_elements!(u, element_id, old_u, old_element_id,
                           reverse_upper, reverse_lower, dg::Dg3D)
  # Store old element ids
  bottom_lower_left_id  = old_element_id
  bottom_lower_right_id = old_element_id + 1
  bottom_upper_left_id  = old_element_id + 2
  bottom_upper_right_id = old_element_id + 3
  top_lower_left_id     = old_element_id + 4
  top_lower_right_id    = old_element_id + 5
  top_upper_left_id     = old_element_id + 6
  top_upper_right_id    = old_element_id + 7

  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, j, k, element_id))

    # Project from bottom lower left element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, bottom_lower_left_id)
              * reverse_lower[i, ii]
              * reverse_lower[j, jj]
              * reverse_lower[k, kk])
    end

    # Project from bottom lower right element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, bottom_lower_right_id)
              * reverse_upper[i, ii]
              * reverse_lower[j, jj]
              * reverse_lower[k, kk])
    end

    # Project from bottom upper left element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, bottom_upper_left_id)
              * reverse_lower[i, ii]
              * reverse_upper[j, jj]
              * reverse_lower[k, kk])
    end

    # Project from bottom upper right element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, bottom_upper_right_id)
              * reverse_upper[i, ii]
              * reverse_upper[j, jj]
              * reverse_lower[k, kk])
    end

    # Project from top lower left element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, top_lower_left_id)
              * reverse_lower[i, ii]
              * reverse_lower[j, jj]
              * reverse_upper[k, kk])
    end

    # Project from top lower right element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, top_lower_right_id)
              * reverse_upper[i, ii]
              * reverse_lower[j, jj]
              * reverse_upper[k, kk])
    end

    # Project from top upper left element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, top_upper_left_id)
              * reverse_lower[i, ii]
              * reverse_upper[j, jj]
              * reverse_upper[k, kk])
    end

    # Project from top upper right element
    for kk in 1:nnodes(dg), jj in 1:nnodes(dg), ii in 1:nnodes(dg)
      acc += (get_node_vars(old_u, dg, ii, jj, kk, top_upper_right_id)
              * reverse_upper[i, ii]
              * reverse_upper[j, jj]
              * reverse_upper[k, kk])
    end

    # Update value
    set_node_vars!(u, acc, dg, i, j, k, element_id)
  end
end


# Calculate an AMR indicator value for each element/leaf cell
#
# The indicator value λ ∈ [-1,1] is ≈ -1 for cells that should be coarsened, ≈
# 0 for cells that should remain as-is, and ≈ 1 for cells that should be
# refined.
#
# Note: The implementation here implicitly assumes that we have an element for
# each leaf cell and that they are in the same order.
#
# FIXME: This is currently implemented for each test case - we need something
# appropriate that is both equation and test case independent
function calc_amr_indicator(dg::Dg3D, mesh::TreeMesh, time::Float64)
  lambda = zeros(dg.n_elements)

  if dg.amr_indicator === :gauss
    base_level = 4
    max_level = 6
    threshold_high = 0.6
    threshold_low = 0.1

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      # Determine target level from peak value
      peak = maximum(dg.elements.u[:, :, :, :, element_id])
      if peak > threshold_high
        target_level = max_level
      elseif peak > threshold_low
        target_level = max_level - 1
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :density_pulse
    # Works with initial_conditions_density_pulse for compressible Euler equations
    base_level = 4
    max_level = 6
    threshold_high = 1.3
    threshold_low = 1.05

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      # Determine target level from peak value
      peak = maximum(dg.elements.u[1, :, :, :, element_id])
      if peak > threshold_high
        target_level = max_level
      elseif peak > threshold_low
        target_level = max_level - 1
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :isentropic_vortex
    base_level = 3
    max_level = 5
    radius_high = 2
    radius_low = 3

    # Domain size needed to handle periodicity
    domain_length = mesh.tree.length_level_0

    # Get analytical vortex center (based on assumption that center=[0.0,0.0]
    # at t=0.0 and that we stop after one period)
    if time < domain_length/2
      center = Float64[time, time]
    else
      center = Float64[time-domain_length, time-domain_length]
    end

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      cell_id = dg.elements.cell_ids[element_id]
      r = periodic_distance_3d(mesh.tree.coordinates[:, cell_id], center, domain_length)
      if r < radius_high
        target_level = max_level
      elseif r < radius_low
        target_level = max_level - 1
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :khi
    base_level = 4
    max_level = 6
    # to make the simulation smaller and quicker wall clock time, choose super_max_level = 6
    super_max_level = 7
    blending_factor_threshold0 = 0.3
    blending_factor_threshold1 = 0.003
    blending_factor_threshold2 = 0.0003

    # (Re-)initialize element variable storage for blending factor
    if (!haskey(dg.element_variables, :amr_indicator_values) ||
        length(dg.element_variables[:amr_indicator_values]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values] = Vector{Float64}(undef, dg.n_elements)
    end
    if (!haskey(dg.element_variables, :amr_indicator_values_tmp) ||
        length(dg.element_variables[:amr_indicator_values_tmp]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values_tmp] = Vector{Float64}(undef, dg.n_elements)
    end

    alpha     = dg.element_variables[:amr_indicator_values]
    alpha_tmp = dg.element_variables[:amr_indicator_values_tmp]
    calc_blending_factors!(alpha, alpha_tmp, dg.elements.u, dg.amr_alpha_max, dg.amr_alpha_min, false,
                           Val(:density), dg)

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      target_level = actual_level
      if alpha[element_id] >= blending_factor_threshold0
        target_level = super_max_level
      elseif alpha[element_id] >= blending_factor_threshold1
        target_level = max_level
      elseif alpha[element_id] <= blending_factor_threshold2
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :blob
    base_level = 4
    max_level = 7
    # to make the simulation smaller and quicker wall clock time, choose super_max_level = 6
    super_max_level = 7
    blending_factor_threshold0 = 0.3
    blending_factor_threshold1 = 0.003
    blending_factor_threshold2 = 0.0003

    # (Re-)initialize element variable storage for blending factor
    if (!haskey(dg.element_variables, :amr_indicator_values) ||
        length(dg.element_variables[:amr_indicator_values]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values] = Vector{Float64}(undef, dg.n_elements)
    end
    if (!haskey(dg.element_variables, :amr_indicator_values_tmp) ||
        length(dg.element_variables[:amr_indicator_values_tmp]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values_tmp] = Vector{Float64}(undef, dg.n_elements)
    end

    alpha     = dg.element_variables[:amr_indicator_values]
    alpha_tmp = dg.element_variables[:amr_indicator_values_tmp]
    calc_blending_factors!(alpha, alpha_tmp, dg.elements.u, dg.amr_alpha_max, dg.amr_alpha_min, false,
                           Val(:density), dg)

    # (Re-)initialize element variable storage for blending factor
    if (!haskey(dg.element_variables, :blending_factor) ||
        length(dg.element_variables[:blending_factor]) != dg.n_elements)
      dg.element_variables[:blending_factor] = Vector{Float64}(undef, dg.n_elements)
    end
    if (!haskey(dg.element_variables, :blending_factor_tmp) ||
        length(dg.element_variables[:blending_factor_tmp]) != dg.n_elements)
      dg.element_variables[:blending_factor_tmp] = Vector{Float64}(undef, dg.n_elements)
    end

    alpha1     = dg.element_variables[:blending_factor]
    alpha1_tmp = dg.element_variables[:blending_factor_tmp]
    calc_blending_factors!(alpha1, alpha1_tmp, dg.elements.u, dg.shock_alpha_max, dg.shock_alpha_min, true,
                           Val(dg.shock_indicator_variable), dg)

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      target_level = actual_level
      # adapt for the amr indicator
      if alpha[element_id] >= blending_factor_threshold0
        target_level = super_max_level
      elseif alpha[element_id] >= blending_factor_threshold1
        target_level = max_level
      elseif alpha[element_id] <= blending_factor_threshold2
        target_level = base_level
      end
      # make sure that a highly troubled shock cell is not coarsened
      if isapprox.(dg.shock_alpha_max, alpha1[element_id], atol=1e-12)
        target_level = max_level
      end
      # Compare target level with actual level to set indicator
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :blast_wave
    base_level = 4
    max_level = 6
    blending_factor_threshold = 0.01

    # (Re-)initialize element variable storage for blending factor
    if (!haskey(dg.element_variables, :amr_indicator_values) ||
        length(dg.element_variables[:amr_indicator_values]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values] = Vector{Float64}(undef, dg.n_elements)
    end
    if (!haskey(dg.element_variables, :amr_indicator_values_tmp) ||
        length(dg.element_variables[:amr_indicator_values_tmp]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values_tmp] = Vector{Float64}(undef, dg.n_elements)
    end

    alpha     = dg.element_variables[:amr_indicator_values]
    alpha_tmp = dg.element_variables[:amr_indicator_values_tmp]
    calc_blending_factors!(alpha, alpha_tmp, dg.elements.u, dg.amr_alpha_max, dg.amr_alpha_min, dg.amr_alpha_smooth,
                           Val(:density_pressure), dg)

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      if alpha[element_id] > blending_factor_threshold
        target_level = max_level
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :sedov_self_gravity
    base_level = 2
    max_level = 7
    blending_factor_threshold = 0.0003

    # (Re-)initialize element variable storage for blending factor
    if (!haskey(dg.element_variables, :amr_indicator_values) ||
        length(dg.element_variables[:amr_indicator_values]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values] = Vector{Float64}(undef, dg.n_elements)
    end
    if (!haskey(dg.element_variables, :amr_indicator_values_tmp) ||
        length(dg.element_variables[:amr_indicator_values_tmp]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values_tmp] = Vector{Float64}(undef, dg.n_elements)
    end

    alpha     = dg.element_variables[:amr_indicator_values]
    alpha_tmp = dg.element_variables[:amr_indicator_values_tmp]
    calc_blending_factors!(alpha, alpha_tmp, dg.elements.u, dg.amr_alpha_max, dg.amr_alpha_min, dg.amr_alpha_smooth,
                           Val(:density_pressure), dg)

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      if alpha[element_id] > blending_factor_threshold
        target_level = max_level
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  else
    error("unknown AMR indicator '$(dg.amr_indicator)'")
  end

  return lambda
end


# For periodic domains, distance between two points must take into account
# periodic extensions of the domain
function periodic_distance_3d(coordinates, center, domain_length)
  dx = abs.(coordinates - center)
  dx_periodic = min.(dx, domain_length .- dx)
  return sqrt(sum(dx_periodic.^2))
end
