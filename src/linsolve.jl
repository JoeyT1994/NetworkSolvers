import ITensorNetworks as itn
import ITensors as it
import NamedGraphs.PartitionedGraphs as npg
using NamedGraphs: NamedEdge, is_tree
using Printf
using ITensorNetworks: AbstractITensorNetwork

@kwdef mutable struct LinsolveProblem{State, QFNetwork, LFNetwork}
  state::State
  xAxnetwork::QFNetwork
  xbnetwork::LFNetwork
  update_region
end

state(L::LinsolveProblem) = L.state
xAxnetwork(L::LinsolveProblem) = L.xAxnetwork
xbnetwork(L::LinsolveProblem) = L.xbnetwork
update_region(L::LinsolveProblem) = L.update_region

bra_region_to_ket_region(bra_region) = [(first(v), "ket") for v in bra_region]
bra_tensor_to_ket_tensor(L::LinsolveProblem, t::ITensor) = dag(itn.dual_inv_index_map(itn.tensornetwork(xAxnetwork(L)))(t))
function bra_to_ket(L::LinsolveProblem, bra::AbstractITensorNetwork)
  ket = copy(bra)
  for v in vertices(bra)
    ket[v] = bra_tensor_to_ket_tensor(L, bra[v])
  end
  return ket
end

function extracter!(problem::LinsolveProblem, region; kws...)

  prev_region = update_region(problem)
  tn = state(problem)
  xAx = xAxnetwork(problem)
  xb = xbnetwork(problem)
  prev_ket_region = bra_region_to_ket_region(prev_region)
  xAx = itn.update_factors(
    xAx, Dict(zip(prev_region, [tn[v] for v in prev_region]))
  )
  xAx = itn.update_factors(
    xAx, Dict(zip(prev_ket_region, [bra_tensor_to_ket_tensor(problem, tn[v]) for v in prev_region]))
  )
  xb = itn.update_factors(
    xb, Dict(zip(prev_region, [tn[v] for v in prev_region]))
  )

  #Update the cache for xAx
  path = itn.edge_sequence_between_regions(itn.tensornetwork(xAx), prev_region, region)
  tn = itn.gauge_walk(itn.Algorithm("orthogonalize"), tn, path)
  pe_path = npg.partitionedges(itn.partitioned_tensornetwork(xAx), path)
  xAx = itn.update(
    itn.Algorithm("bp"), xAx, pe_path; message_update_function_kwargs=(; normalize=false)
  )

  #Update the cache for xb
  path = itn.edge_sequence_between_regions(itn.tensornetwork(xb), prev_region, region)
  pe_path = npg.partitionedges(itn.partitioned_tensornetwork(xb), path)
  xb = itn.update(
    itn.Algorithm("bp"), xb, pe_path; message_update_function_kwargs=(; normalize=false)
  )
  
  problem.xbnetwork = xb
  problem.xAxnetwork = xAx
  problem.update_region = region

  localxAx_tensors = itn.environment(xAx, vcat(region, bra_region_to_ket_region(region))); 
  localxb_tensor = itn.environment(xb, region);
  seq = itn.contraction_sequence(localxb_tensor; alg = "optimal")
  localxb_tensor = contract(localxb_tensor; sequence = seq)
  init = itn.factors(xb, region)
  seq = itn.contraction_sequence(init; alg = "optimal")
  init = contract(init; sequence = seq)

  return [localxAx_tensors, localxb_tensor, init]
end

function prepare_subspace!(problem::LinsolveProblem, local_tensor, region; sweep, kws...)
  local_tensor = subspace_expand!(problem, local_tensor, region; sweep, kws...)
  return local_tensor
end

function updater!(L::LinsolveProblem, local_tensors, region; solver = linsolve_solver, outputlevel, kws...)
  @assert length(local_tensors) == 3
  local_tensor, _ = solver(L, local_tensors[1], local_tensors[2], local_tensors[3]; kws...)
  if outputlevel >= 2
    @printf("  Region %s: ", region)
  end
  return local_tensor
end

function KrylovKit.linsolve(
  A::AbstractITensorNetwork,
  b::AbstractITensorNetwork,
  x0::AbstractITensorNetwork,
  args...;
  a₀::Number=0,
  a₁::Number=1,
  nsweeps=25,
  nsites=1,
  outputlevel=0,
  extracter_kwargs=(;),
  updater_kwargs=(;),
  truncation_kwargs=(;),
  normalize=true,
  kws...,
)
  xAx_network = itn.QuadraticFormNetwork(A, x0)
  xAx = itn.BeliefPropagationCache(xAx_network)
  xb_network = itn.BilinearFormNetwork(x0, b)
  xb = itn.BeliefPropagationCache(xb_network)
  bra_vs = itn.bra_vertex_map(xb_network).(collect(vertices(x0)))
  bra_tn = first(itn.induced_subgraph(xAx_network, bra_vs))
  init_prob = LinsolveProblem(;
    state=bra_tn, xAxnetwork = xAx, xbnetwork = xb, update_region=bra_vs
  )

  updater_kwargs = (; a₀, a₁, updater_kwargs...)
  truncation_kwargs = (; truncation_kwargs..., normalize, set_orthogonal_region=false)
  common_sweep_kwargs = (; nsites, outputlevel, updater_kwargs, truncation_kwargs)
  kwargs_array = [(; common_sweep_kwargs..., sweep=s) for s in 1:nsweeps]
  sweep_iter = sweep_iterator(init_prob, kwargs_array)
  converged_prob = alternating_update(sweep_iter; outputlevel, kws...)
  bra = state(converged_prob)
  ket = bra_to_ket(converged_prob, bra)
  return itn.rename_vertices(itn.inv_vertex_map(xAx_network), ket)
end