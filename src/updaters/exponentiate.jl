using KrylovKit: KrylovKit

function exponentiate_updater(
  operator,
  init;
  time,
  krylovdim=30,
  maxiter=100,
  verbosity=0,
  tol=1E-12,
  ishermitian=true,
  issymmetric=true,
  eager=true,
)
  result, exp_info = KrylovKit.exponentiate(
    operator,
    time,
    init;
    eager,
    krylovdim,
    maxiter,
    verbosity,
    tol,
    ishermitian,
    issymmetric,
  )
  return result, exp_info
end
