using KrylovKit: linsolve

function internal_f(L::LinsolveProblem, xbra::ITensor, operator::Vector{<:ITensor})
  xket = bra_tensor_to_ket_tensor(L, xbra)
  ts = [operator; xket]
  seq = itn.contraction_sequence(ts; alg = "optimal")
  return contract(ts; sequence = seq)
end

function linsolve_solver(
  L::LinsolveProblem,
  operator::Vector{<:ITensor},
  b::ITensor,
  init::ITensor;
  ishermitian=false,
  tol=1E-11,
  krylovdim=30,
  maxiter=10,
  verbosity=0,
  a₀,
  a₁,
  kws...
)
  f = xbra -> internal_f(L, xbra, operator)

  lhs = contract(ITensor[init; f(init)]; sequence = "automatic")
  rhs = contract(ITensor[init; b]; sequence = "automatic")
  println("Beginning LinSolve")
  @show lhs[], rhs[]

  x, info = linsolve(
   f, b, init, a₀, a₁; ishermitian, tol, krylovdim, maxiter, verbosity
  )
  lhs = contract(ITensor[x; f(x)]; sequence = "automatic")
  rhs = contract(ITensor[x; b]; sequence = "automatic")
  @show lhs[], rhs[]
  #x /= norm(x)
  return x, (;)
end
