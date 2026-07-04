#pragma once
// Internal helpers for scipp::sparse: int64 / double array <-> std::vector.

#include <cstdint>
#include <memory>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::sparse::detail {

namespace ld = scipp::linalg::detail;

inline std::vector<int64_t> iv(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kInt64).ascontiguousarray();
  const int64_t* p = c.typed_data<int64_t>();
  return std::vector<int64_t>(p, p + c.size());
}
inline numpp::ndarray from_iv(const std::vector<int64_t>& v) {
  numpp::ndarray a(numpp::Shape{static_cast<int64_t>(v.size())}, numpp::kInt64);
  int64_t* p = a.typed_data<int64_t>();
  for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
  return a;
}
inline std::vector<double> dv(const numpp::ndarray& a) { return ld::to_vec(a); }
inline numpp::ndarray from_dv(const std::vector<double>& v) { return ld::from_vec(v); }

// ---- preconditioners (see precond.cpp) ----
// kind: 0 None, 1 Jacobi, 2 IC0 (incomplete Cholesky), 3 ILU0 (incomplete LU).
// A is given as a canonical CSR (row-sorted columns). IC0/ILU0 that fail to build
// (non-SPD / zero pivot) silently degrade to Jacobi so `apply` stays usable.
struct Precond {
  int kind = 0;
  int64_t n = 0;
  std::vector<double> invdiag;                                  // Jacobi
  std::vector<int64_t> Lp, Li;                                  // IC0 lower (incl diag)
  std::vector<double> Lx, Ldiag;                                //   M = L Lᵀ
  std::vector<int64_t> Up, Ui, dpos;                            // ILU0 combined LU (CSR of A)
  std::vector<double> Ux;                                       //   L unit-lower, U upper

  std::vector<double> apply(const std::vector<double>& r) const;
};

Precond make_precond(int kind, int64_t n, const std::vector<int64_t>& Ap,
                     const std::vector<int64_t>& Ai, const std::vector<double>& Ax);

// ---- sparse direct factorization (see factor.cpp) ----
// A is passed as canonical CSR. ordering: 0 Natural, 1 RCM, 2 AMD (→ RCM for now).
struct DirectResult {
  std::vector<double> x;      // solution (empty if !ok)
  int64_t factor_nnz = 0;     // nonzeros in L
  bool used_cholesky = false; // true if the SPD Cholesky path was taken
  bool ok = false;            // false ⇒ factorization failed (singular)
};

// Factor A and solve A x = b sparsely (SPD → Cholesky, else LU).
DirectResult sparse_direct_solve(int64_t n, const std::vector<int64_t>& Ap,
                                 const std::vector<int64_t>& Ai, const std::vector<double>& Ax,
                                 const std::vector<double>& b, int ordering);

// Factor A only and report nnz(L) (Cholesky when SPD, else LU). ok=false ⇒ singular.
DirectResult sparse_direct_factor_nnz(int64_t n, const std::vector<int64_t>& Ap,
                                      const std::vector<int64_t>& Ai,
                                      const std::vector<double>& Ax, int ordering);

// Reusable factorization (factor once, solve many) — the shift-invert operator
// for the sparse eigensolver. `Factorization` is opaque (defined in factor.cpp);
// callers hold it via shared_ptr and never need its layout.
struct Factorization;
std::shared_ptr<Factorization> sparse_direct_factorize(int64_t n, const std::vector<int64_t>& Ap,
                                                       const std::vector<int64_t>& Ai,
                                                       const std::vector<double>& Ax, int ordering);
bool factorization_ok(const std::shared_ptr<Factorization>& F);
// True iff the factored matrix was SPD (the Cholesky path was taken) — a free
// Sturm/definiteness bit for the buckling shift walk.
bool factorization_definite(const std::shared_ptr<Factorization>& F);
std::vector<double> factorization_solve(const std::shared_ptr<Factorization>& F,
                                        const std::vector<double>& b);

}  // namespace scipp::sparse::detail
