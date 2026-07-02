// Oracle tests for scipp::sparse against frozen SciPy golden data.
#include <cmath>
#include <vector>

#include "golden.hpp"
#include "numpp/core/dtype.hpp"
#include "numpp/core/ndarray.hpp"
#include "scipp/sparse/sparse.hpp"
#include "scipp_test.hpp"

namespace sp = scipp::sparse;
namespace {
numpp::ndarray vec(const double* d, int n) {
  numpp::ndarray a(numpp::Shape{n}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
numpp::ndarray mat(const double* d, int r, int c) {
  numpp::ndarray a(numpp::Shape{r, c}, numpp::kFloat64);
  double* p = a.typed_data<double>();
  for (int i = 0; i < r * c; ++i) p[i] = d[i];
  return a;
}
std::vector<double> tov(const numpp::ndarray& a) {
  numpp::ndarray c = a.astype(numpp::kFloat64).ascontiguousarray();
  const double* p = c.typed_data<double>();
  return std::vector<double>(p, p + c.size());
}
numpp::ndarray ivec(const long long* d, int n) {  // int64 array from golden longs
  numpp::ndarray a(numpp::Shape{n}, numpp::kInt64);
  int64_t* p = a.typed_data<int64_t>();
  for (int i = 0; i < n; ++i) p[i] = d[i];
  return a;
}
// Build a CsrMatrix from golden CSR arrays emitted by emit_csr(name, ...).
#define CSR(name)                                                                      \
  sp::CsrMatrix(vec(golden::name##_data, golden::name##_data_n),                        \
                ivec(golden::name##_indices, golden::name##_indices_n),                 \
                ivec(golden::name##_indptr, golden::name##_indptr_n),                   \
                static_cast<int64_t>(golden::name##_rows),                              \
                static_cast<int64_t>(golden::name##_cols))
double relL2(const std::vector<double>& a, const double* b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return std::sqrt(num / den);
}
void cv(const numpp::ndarray& got, const double* exp, int n, double rtol = 1e-9, double atol = 1e-11) {
  auto g = tov(got);
  for (int i = 0; i < n && i < (int)g.size(); ++i) CHECK_CLOSE(g[i], exp[i], rtol, atol);
}
}  // namespace
#define M(name) mat(golden::name##_d, golden::name##_r, golden::name##_c)
#define G(name) golden::name, golden::name##_n

TEST_CASE("formats and constructors") {
  auto A = sp::CsrMatrix::from_dense(M(sp_A));
  cv(A.toarray(), golden::sp_A_toarray_d, golden::sp_A_toarray_r * golden::sp_A_toarray_c);
  CHECK(A.nnz() == 8);
  cv(A.diagonal(), G(sp_diag));
  cv(sp::eye(4).toarray(), golden::sp_eye_d, 16);
  double d0[] = {1, 2, 3, 4}, d1[] = {5, 6, 7};
  std::vector<numpp::ndarray> diags{vec(d0, 4), vec(d1, 3)};
  cv(sp::diags(diags, {0, 1}, 4).toarray(), golden::sp_diags_d, 16);
  cv(A.add(sp::eye(4)).toarray(), golden::sp_add_d, 16);
  // CSC round-trip via transpose
  auto At = A.transpose();
  CHECK(At.nnz() == 8);
}

TEST_CASE("products and backend dispatch") {
  auto A = sp::CsrMatrix::from_dense(M(sp_A));
  cv(A.spmv(vec(G(sp_x))), G(sp_spmv));
  cv(A.spmm(M(sp_X)), golden::sp_spmm_d, golden::sp_spmm_r * golden::sp_spmm_c);
  // Dispatch now delegates to numpp::csr_spmv, which owns device offload + the
  // CPU fallback. The dispatched SpMV still matches the SciPy oracle, and
  // last_backend() reflects NumPP's actual choice (CPU locally as NumPP is
  // CPU-only here; Device where a NumPP GPU variant is present).
  cv(sp::spmv(A, vec(G(sp_x))), G(sp_spmv));
  auto ycpu = tov(sp::spmv(A, vec(G(sp_x)), sp::Backend::Cpu));
  CHECK(sp::last_backend() == sp::Backend::Cpu);
  auto ydev = tov(sp::spmv(A, vec(G(sp_x)), sp::Backend::Device));
  auto bk = sp::last_backend();
  CHECK(bk == sp::Backend::Cpu || bk == sp::Backend::Device);
  for (size_t i = 0; i < ycpu.size(); ++i) CHECK_CLOSE(ycpu[i], ydev[i], 1e-12, 1e-12);
}

TEST_CASE("sparse solvers") {
  auto A = sp::CsrMatrix::from_dense(M(sp_A));
  auto b = vec(G(sp_b));
  cv(sp::spsolve(A, b), G(sp_spsolve), 1e-9, 1e-11);
  // iterative solvers drive the residual to zero (A is SPD here)
  auto xcg = sp::cg(A, b);
  cv(A.spmv(xcg), G(sp_b), 1e-5, 1e-7);
  auto xg = sp::gmres(A, b);
  cv(A.spmv(xg), G(sp_b), 1e-5, 1e-7);
  CHECK_CLOSE(sp::norm(A), golden::sp_norm_fro, 1e-9, 1e-11);
}

TEST_CASE("preconditioned iterative solvers + reporting") {
  auto K = CSR(sp_spd);  // stiff, ill-conditioned SPD system (80 DOF)
  auto b = vec(golden::sp_spd_b, golden::sp_spd_b_n);
  const double* xref = golden::sp_spd_x;

  // IC0-preconditioned CG converges to the sparse-direct solution.
  auto ic = sp::cg_report(K, b, sp::Preconditioner::IC0, 1e-9, 5000);
  CHECK(ic.converged);
  CHECK(ic.final_residual < 1e-9);
  CHECK(relL2(tov(ic.x), xref) < 1e-6);

  // Jacobi-preconditioned CG likewise.
  auto jac = sp::cg_report(K, b, sp::Preconditioner::Jacobi, 1e-9, 5000);
  CHECK(jac.converged);
  CHECK(relL2(tov(jac.x), xref) < 1e-6);

  // Preconditioning accelerates: no more iterations than unpreconditioned CG.
  auto none = sp::cg_report(K, b, sp::Preconditioner::None, 1e-9, 5000);
  CHECK(ic.iterations <= none.iterations);
  CHECK(jac.iterations <= none.iterations);

  // Non-convergence is signaled, not hidden by a silently-wrong vector.
  auto stalled = sp::cg_report(K, b, sp::Preconditioner::None, 1e-12, 3);
  CHECK(!stalled.converged);
  CHECK(stalled.iterations == 3);
  CHECK(stalled.final_residual > 1e-12);
  // Reported residual matches the true residual ‖b − A x‖ / ‖b‖ of the returned x.
  auto Ax = tov(K.spmv(stalled.x));
  auto bb = tov(b);
  double num = 0, den = 0;
  for (size_t i = 0; i < bb.size(); ++i) { double d = bb[i] - Ax[i]; num += d * d; den += bb[i] * bb[i]; }
  CHECK_CLOSE(stalled.final_residual, std::sqrt(num / den), 1e-7, 1e-12);

  // GMRES with ILU0 preconditioning solves the same system.
  auto gi = sp::gmres_report(K, b, sp::Preconditioner::ILU0, 1e-9, 2000);
  CHECK(gi.converged);
  CHECK(relL2(tov(gi.x), xref) < 1e-6);
}

TEST_CASE("sparse direct factorization") {
  // SPD Cholesky matches SciPy's spsolve (n=80 → sparse path by default).
  auto K = CSR(sp_spd);
  auto b = vec(golden::sp_spd_b, golden::sp_spd_b_n);
  CHECK(relL2(tov(sp::spsolve(K, b)), golden::sp_spd_x) < 1e-10);
  CHECK(relL2(tov(sp::spsolve(K, b, sp::OrderingMethod::Natural)), golden::sp_spd_x) < 1e-10);
  cv(K.spmv(sp::spsolve(K, b)), golden::sp_spd_b, golden::sp_spd_b_n, 1e-8, 1e-10);

  // General (non-symmetric) LU matches SciPy; the ordering overload forces the
  // sparse path even below the small-N dense threshold.
  auto Ag = CSR(sp_gen);
  auto bg = vec(golden::sp_gen_b, golden::sp_gen_b_n);
  CHECK(relL2(tov(sp::spsolve(Ag, bg, sp::OrderingMethod::Rcm)), golden::sp_gen_x) < 1e-10);
  CHECK(relL2(tov(sp::spsolve(Ag, bg, sp::OrderingMethod::Natural)), golden::sp_gen_x) < 1e-10);

  // RCM reduces fill on the arrow matrix (hub ordered last) and still solves.
  auto Ka = CSR(sp_arrow);
  auto ba = vec(golden::sp_arrow_b, golden::sp_arrow_b_n);
  CHECK(relL2(tov(sp::spsolve(Ka, ba, sp::OrderingMethod::Rcm)), golden::sp_arrow_x) < 1e-10);
  CHECK(sp::factor_nnz(Ka, sp::OrderingMethod::Rcm) < sp::factor_nnz(Ka, sp::OrderingMethod::Natural));

  // Symmetric indefinite: Cholesky rejects the non-positive pivot and the solver
  // falls back to LU, still matching SciPy.
  auto Ki = CSR(sp_indef);
  auto bi = vec(golden::sp_indef_b, golden::sp_indef_b_n);
  CHECK(relL2(tov(sp::spsolve(Ki, bi, sp::OrderingMethod::Rcm)), golden::sp_indef_x) < 1e-10);
}

TEST_CASE("csgraph") {
  auto G = sp::CsrMatrix::from_dense(M(sp_G));
  cv(sp::csgraph::dijkstra(G, true), golden::sp_dijkstra_d, golden::sp_dijkstra_r * golden::sp_dijkstra_c);
  cv(sp::csgraph::floyd_warshall(G, true), golden::sp_floyd_d, golden::sp_floyd_r * golden::sp_floyd_c);
  cv(sp::csgraph::bellman_ford(G, true), golden::sp_dijkstra_d, golden::sp_dijkstra_r * golden::sp_dijkstra_c);
  auto cc = sp::csgraph::connected_components(sp::CsrMatrix::from_dense(M(sp_Gu)), false);
  CHECK(cc.n_components == static_cast<int>(golden::sp_ncomp));
  auto mst = sp::csgraph::minimum_spanning_tree(sp::CsrMatrix::from_dense(M(sp_Gm)));
  auto t = tov(mst.toarray());
  double w = 0; for (double v : t) w += v;
  CHECK_CLOSE(w, golden::sp_mst_weight, 1e-9, 1e-11);
}
