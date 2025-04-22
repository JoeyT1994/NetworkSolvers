import ITensors as it
import ITensorNetworks as itn
using Printf
import NetworkSolvers as ns
using Random
using Graphs: edges, dst, src
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using TensorOperations: TensorOperations
using StableRNGs: StableRNG
using ITensorNetworks: ITensorNetwork, apply, inner, siteinds, random_tensornetwork, ttn
using ITensorNetworks.ModelHamiltonians: heisenberg

it.disable_warn_order()

rng = StableRNG(1234)

cutoff = 1E-11
maxdim = 4
nsweeps = 1

#g = named_comb_tree((3, 2))
g = named_grid((3,1))
s = siteinds("S=1/2", g)

b = random_tensornetwork(rng, s; link_space=2)
H = ITensorNetwork(ttn(heisenberg(g), s))

x0 = copy(b)

x = ns.linsolve(H, b, x0; nsweeps, nsites = 1, truncation_kwargs = (; cutoff, maxdim), normalize = false)

#b_alt = apply(H, x; maxdim=itn.maxlinkdim(H)*itn.maxlinkdim(x), nsites=1, normalize=false)

#f = inner(b, b_alt; alg = "exact") / sqrt(inner(b,b; alg = "exact") * inner(b_alt,b_alt; alg = "exact"))

#@show f