// Sparse linear algebra: spsolve (direct), cg / gmres (matrix-free), norm.
#include "scipp/sparse/sparse.hpp"

#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scipp/sparse/detail.hpp"

namespace scipp::sparse {
namespace d = detail;
namespace {
double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}
int precond_kind(Preconditioner p) { return static_cast<int>(p); }  // enum order matches
d::Precond build_precond(const CsrMatrix& A, Preconditioner p) {
  return d::make_precond(precond_kind(p), A.rows(), d::iv(A.indptr()), d::iv(A.indices()),
                         d::dv(A.data()));
}
}  // namespace

namespace {
// Below this size the dense LU is cheaper than building a sparse factorization,
// and it keeps the tiny golden systems on the exact same numeric path.
constexpr int64_t kDenseThreshold = 64;

int ordering_code(OrderingMethod o) { return static_cast<int>(o); }

ndarray sparse_solve(const CsrMatrix& A, const ndarray& b, OrderingMethod ordering) {
  d::DirectResult r = d::sparse_direct_solve(A.rows(), d::iv(A.indptr()), d::iv(A.indices()),
                                             d::dv(A.data()), d::dv(b), ordering_code(ordering));
  if (!r.ok) return numpp::linalg::solve(A.toarray(), b);  // singular pattern → dense fallback
  return d::from_dv(r.x);
}
}  // namespace

ndarray spsolve(const CsrMatrix& A, const ndarray& b) {
  // Small systems: dense LU. Larger systems: genuine sparse factorization
  // (SPD → Cholesky, else LU) with a fill-reducing RCM ordering.
  if (A.rows() != A.cols() || A.rows() <= kDenseThreshold)
    return numpp::linalg::solve(A.toarray(), b);
  return sparse_solve(A, b, OrderingMethod::Rcm);
}

ndarray spsolve(const CsrMatrix& A, const ndarray& b, OrderingMethod ordering) {
  return sparse_solve(A, b, ordering);
}

int64_t factor_nnz(const CsrMatrix& A, OrderingMethod ordering) {
  d::DirectResult r = d::sparse_direct_factor_nnz(A.rows(), d::iv(A.indptr()), d::iv(A.indices()),
                                                  d::dv(A.data()), ordering_code(ordering));
  return r.ok ? r.factor_nnz : -1;
}

// Preconditioned conjugate gradient. Solves M⁻¹ A x = M⁻¹ b for SPD A; reports the
// iteration count, final relative residual, and whether tol was met within maxiter.
IterationResult cg_report(const CsrMatrix& A, const ndarray& b, Preconditioner precond,
                          double tol, int maxiter) {
  std::vector<double> bv = d::dv(b);
  int n = static_cast<int>(bv.size());
  IterationResult out;
  out.x = d::from_dv(std::vector<double>(n, 0.0));
  double bnorm = std::sqrt(dot(bv, bv));
  if (bnorm == 0) { out.converged = true; return out; }

  d::Precond M = build_precond(A, precond);
  std::vector<double> x(n, 0.0), r = bv, z = M.apply(r), p = z;
  double rz = dot(r, z);
  double res = std::sqrt(dot(r, r)) / bnorm;
  for (int it = 0; it < maxiter; ++it) {
    std::vector<double> Ap = d::dv(A.spmv(d::from_dv(p)));
    double pAp = dot(p, Ap);
    double alpha = rz / pAp;
    for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
    res = std::sqrt(dot(r, r)) / bnorm;
    out.iterations = it + 1;
    if (res < tol) { out.converged = true; break; }
    z = M.apply(r);
    double rz_new = dot(r, z);
    double beta = rz_new / rz;
    for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    rz = rz_new;
  }
  out.x = d::from_dv(x);
  out.final_residual = res;
  return out;
}

ndarray cg(const CsrMatrix& A, const ndarray& b, double tol, int maxiter) {
  return cg_report(A, b, Preconditioner::None, tol, maxiter).x;
}

// Restarted GMRES(50) with optional right preconditioning (solves A M⁻¹ u = b,
// x = M⁻¹ u, so the Krylov residual equals the true residual ‖b − A x‖). Reports
// the iteration count, final relative residual, and convergence flag.
IterationResult gmres_report(const CsrMatrix& A, const ndarray& b, Preconditioner precond,
                             double tol, int maxiter) {
  std::vector<double> bv = d::dv(b);
  int n = static_cast<int>(bv.size());
  IterationResult out;
  double bnorm = std::sqrt(dot(bv, bv));
  if (bnorm == 0) { out.x = d::from_dv(std::vector<double>(n, 0.0)); out.converged = true; return out; }

  d::Precond M = build_precond(A, precond);
  std::vector<double> x(n, 0.0);
  int restart = std::min(n, 50);
  double beta = bnorm;
  int total_iters = 0;
  for (int outer = 0; outer < maxiter; ++outer) {
    std::vector<double> r = bv, Ax = d::dv(A.spmv(d::from_dv(x)));
    for (int i = 0; i < n; ++i) r[i] -= Ax[i];
    beta = std::sqrt(dot(r, r));
    if (beta / bnorm < tol) break;
    std::vector<std::vector<double>> V;
    V.push_back(r); for (double& v : V[0]) v /= beta;
    std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart, 0.0));
    std::vector<double> cs(restart, 0), sn(restart, 0), g(restart + 1, 0.0);
    g[0] = beta;
    int k = 0;
    for (; k < restart && total_iters < maxiter; ++k, ++total_iters) {
      std::vector<double> w = d::dv(A.spmv(d::from_dv(M.apply(V[k]))));  // A M⁻¹ v
      for (int j = 0; j <= k; ++j) { H[j][k] = dot(w, V[j]); for (int i = 0; i < n; ++i) w[i] -= H[j][k] * V[j][i]; }
      H[k + 1][k] = std::sqrt(dot(w, w));
      if (H[k + 1][k] > 1e-14) { for (double& v : w) v /= H[k + 1][k]; V.push_back(w); }
      for (int j = 0; j < k; ++j) {  // apply previous Givens rotations
        double t = cs[j] * H[j][k] + sn[j] * H[j + 1][k];
        H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k];
        H[j][k] = t;
      }
      double rho = std::hypot(H[k][k], H[k + 1][k]);
      cs[k] = H[k][k] / rho; sn[k] = H[k + 1][k] / rho;
      H[k][k] = rho; H[k + 1][k] = 0.0;
      g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
      if (std::fabs(g[k + 1]) / bnorm < tol) { ++k; ++total_iters; break; }
    }
    std::vector<double> yk(k, 0.0);  // back-substitution
    for (int i = k - 1; i >= 0; --i) {
      double s = g[i];
      for (int j = i + 1; j < k; ++j) s -= H[i][j] * yk[j];
      yk[i] = s / H[i][i];
    }
    std::vector<double> u(n, 0.0);  // u = Σ y_j v_j, then x += M⁻¹ u
    for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) u[i] += yk[j] * V[j][i];
    std::vector<double> zu = M.apply(u);
    for (int i = 0; i < n; ++i) x[i] += zu[i];
  }
  std::vector<double> r = bv, Ax = d::dv(A.spmv(d::from_dv(x)));
  for (int i = 0; i < n; ++i) r[i] -= Ax[i];
  out.x = d::from_dv(x);
  out.iterations = total_iters;
  out.final_residual = std::sqrt(dot(r, r)) / bnorm;
  out.converged = out.final_residual < tol;
  return out;
}

ndarray gmres(const CsrMatrix& A, const ndarray& b, double tol, int maxiter) {
  return gmres_report(A, b, Preconditioner::None, tol, maxiter).x;
}

double norm(const CsrMatrix& A, const std::string& ord) {
  auto da = d::dv(A.data());
  if (ord == "fro") { double s = 0; for (double v : da) s += v * v; return std::sqrt(s); }
  auto ip = d::iv(A.indptr()), id = d::iv(A.indices());
  std::vector<double> colsum(A.cols(), 0.0), rowsum(A.rows(), 0.0);
  for (int64_t i = 0; i < A.rows(); ++i)
    for (int64_t k = ip[i]; k < ip[i + 1]; ++k) { rowsum[i] += std::fabs(da[k]); colsum[id[k]] += std::fabs(da[k]); }
  double best = 0;
  if (ord == "1") for (double c : colsum) best = std::max(best, c);
  else if (ord == "inf") for (double r : rowsum) best = std::max(best, r);
  return best;
}

}  // namespace scipp::sparse
