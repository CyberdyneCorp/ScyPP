// Sparse generalized symmetric eigensolver: shift-invert Lanczos for
// K x = λ M x (K symmetric, M SPD). The operator C = (K − σ M)⁻¹ M is
// self-adjoint in the M-inner product ⟨a,b⟩_M = aᵀ M b; running Lanczos in that
// inner product yields a small symmetric tridiagonal T whose eigenpairs (θ, s)
// give Ritz values θ (largest |θ| ↔ λ nearest σ) and M-orthonormal Ritz vectors
// V s. Eigenvalues are shifted back as λ = σ + 1/θ. (K − σ M) is factored once
// with the sparse direct factorization and reused across iterations. The small
// projected problem is solved with NumPP's dense symmetric eigensolver.
#include "scipp/sparse/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scipp/sparse/detail.hpp"

namespace scipp::sparse {
namespace d = detail;
namespace {

using Vec = std::vector<double>;

double dot(const Vec& a, const Vec& b) {
  double s = 0.0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}

Vec spmv(const CsrMatrix& A, const Vec& x) { return d::dv(A.spmv(d::from_dv(x))); }

// Deterministic, reproducible start vector that avoids accidental M-orthogonality
// to a wanted eigenvector (a hash-style spread, no RNG so the result is stable).
Vec start_vector(int64_t n) {
  Vec v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    double s = std::sin(static_cast<double>(i + 1) * 12.9898) * 43758.5453;
    v[i] = (s - std::floor(s)) - 0.5;   // in [-0.5, 0.5)
  }
  return v;
}

// One M-orthonormal Lanczos run of dimension up to `m` on C = (K − σ M)⁻¹ M.
// Fills alpha (diagonal) and beta (off-diagonal) of the tridiagonal T and stores
// the M-orthonormal basis in V. Full reorthogonalization keeps V M-orthonormal.
struct Lanczos { std::vector<Vec> V; Vec alpha, beta; };

Lanczos lanczos(const CsrMatrix& K, const CsrMatrix& M,
                const std::shared_ptr<d::Factorization>& fac, int m, const Vec& v0) {
  int64_t n = K.rows();
  Lanczos L;
  // M-normalize the start vector.
  Vec v = v0, Mv = spmv(M, v);
  double nrm = std::sqrt(dot(v, Mv));
  for (int64_t i = 0; i < n; ++i) { v[i] /= nrm; Mv[i] /= nrm; }
  L.V.push_back(v);
  std::vector<Vec> MV{Mv};                       // M v_j, reused for inner products

  Vec w = d::factorization_solve(fac, Mv);       // C v_0
  double a = dot(w, Mv);
  L.alpha.push_back(a);
  for (int64_t i = 0; i < n; ++i) w[i] -= a * v[i];

  for (int j = 1; j < m; ++j) {
    // Full reorthogonalization in the M-inner product (twice for stability).
    for (int pass = 0; pass < 2; ++pass)
      for (size_t t = 0; t < L.V.size(); ++t) {
        double c = dot(MV[t], w);
        for (int64_t i = 0; i < n; ++i) w[i] -= c * L.V[t][i];
      }
    Vec Mw = spmv(M, w);
    double beta = std::sqrt(dot(w, Mw));
    if (!(beta > 1e-12)) break;                  // invariant subspace reached
    L.beta.push_back(beta);
    v.assign(w.begin(), w.end());
    for (int64_t i = 0; i < n; ++i) { v[i] /= beta; Mw[i] /= beta; }
    L.V.push_back(v);
    MV.push_back(Mw);

    w = d::factorization_solve(fac, MV.back());  // C v_j
    for (int64_t i = 0; i < n; ++i) w[i] -= beta * L.V[j - 1][i];
    a = dot(w, MV.back());
    L.alpha.push_back(a);
    for (int64_t i = 0; i < n; ++i) w[i] -= a * v[i];
  }
  return L;
}

// Solve the projected symmetric tridiagonal eigenproblem T s = θ s via NumPP's
// dense symmetric eigensolver. Returns (θ ascending, S columns).
numpp::linalg::EighResult solve_tridiagonal(const Vec& alpha, const Vec& beta) {
  int m = static_cast<int>(alpha.size());
  numpp::ndarray T(numpp::Shape{m, m}, numpp::kFloat64);
  double* t = T.typed_data<double>();
  std::fill(t, t + static_cast<size_t>(m) * m, 0.0);
  for (int i = 0; i < m; ++i) t[i * m + i] = alpha[i];
  for (int i = 0; i + 1 < m; ++i) { t[i * m + (i + 1)] = beta[i]; t[(i + 1) * m + i] = beta[i]; }
  return numpp::linalg::eigh(T);
}

// Reconstruct the Ritz vector x = Σ_t S[t, idx] v_t (M-orthonormal ⇒ xᵀ M x = 1).
Vec ritz_vector(const std::vector<Vec>& V, const double* S, int built, int idx) {
  Vec x(V.front().size(), 0.0);
  for (int t = 0; t < built; ++t) {
    double sti = S[t * built + idx];
    const Vec& vt = V[t];
    for (size_t i = 0; i < x.size(); ++i) x[i] += sti * vt[i];
  }
  return x;
}

// The k Ritz indices nearest σ ⇔ largest |θ| (θ = 1/(λ − σ)), then reordered so
// their eigenvalues λ = σ + 1/θ come out ascending.
std::vector<int> select_nearest(const double* theta, int built, int k, double sigma) {
  std::vector<int> order(built);
  for (int i = 0; i < built; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return std::fabs(theta[a]) > std::fabs(theta[b]); });
  std::vector<int> pick(order.begin(), order.begin() + std::min(k, built));
  std::sort(pick.begin(), pick.end(),
            [&](int a, int b) { return 1.0 / theta[a] < 1.0 / theta[b]; });
  return pick;
}

// True generalized residual ‖K x − λ M x‖ / ‖K x‖ of a Ritz pair.
double gen_residual(const CsrMatrix& K, const CsrMatrix& M, const Vec& x, double lambda) {
  Vec Kx = spmv(K, x), Mx = spmv(M, x);
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double r = Kx[i] - lambda * Mx[i];
    num += r * r; den += Kx[i] * Kx[i];
  }
  return std::sqrt(num / (den > 0.0 ? den : 1.0));
}

}  // namespace

EigshResult eigsh(const CsrMatrix& K, const CsrMatrix& M, int k,
                  double sigma, double tol, int maxiter) {
  int64_t n = K.rows();
  EigshResult out;
  if (k <= 0 || n <= 0) return out;
  if (k > n) k = static_cast<int>(n);

  // Shift-invert operator A = K − σ M, factored once and reused every iteration.
  CsrMatrix A = (sigma == 0.0) ? K : K.add(M.scaled(-sigma));
  auto fac = d::sparse_direct_factorize(A.rows(), d::iv(A.indptr()), d::iv(A.indices()),
                                        d::dv(A.data()), /*RCM*/ 1);
  if (!d::factorization_ok(fac)) return out;     // (K − σ M) singular: pick another sigma

  // Grow the Krylov dimension up to the subspace cap until the k modes converge.
  int cap = (maxiter > 0) ? std::min<int>(static_cast<int>(n), maxiter)
                          : std::min<int>(static_cast<int>(n), std::max(2 * k + 1, 20));
  Vec v0 = start_vector(n);
  int m = std::min<int>(cap, std::max(k + 1, std::min<int>(static_cast<int>(n), 2 * k + 1)));

  Lanczos L;
  numpp::linalg::EighResult tri;
  std::vector<int> pick;
  bool converged = false;
  while (true) {
    L = lanczos(K, M, fac, m, v0);
    int built = static_cast<int>(L.V.size());
    tri = solve_tridiagonal(L.alpha, L.beta);
    const double* theta = tri.eigenvalues.typed_data<double>();
    const double* S = tri.eigenvectors.typed_data<double>();  // (built × built), columns
    pick = select_nearest(theta, built, k, sigma);

    converged = true;
    for (int idx : pick) {
      Vec x = ritz_vector(L.V, S, built, idx);
      if (gen_residual(K, M, x, sigma + 1.0 / theta[idx]) > tol) { converged = false; break; }
    }
    if (converged || m >= cap || built < m) break;   // done, capped, or invariant subspace
    m = std::min(cap, m + std::max(k, 8));
  }

  // Materialize eigenvalues and (n × k) mass-normalized eigenvectors.
  int kk = static_cast<int>(pick.size());
  int built = static_cast<int>(L.V.size());
  const double* theta = tri.eigenvalues.typed_data<double>();
  const double* S = tri.eigenvectors.typed_data<double>();
  numpp::ndarray evals(numpp::Shape{kk}, numpp::kFloat64);
  numpp::ndarray evecs(numpp::Shape{n, kk}, numpp::kFloat64);
  double* wv = evals.typed_data<double>();
  double* vv = evecs.typed_data<double>();
  for (int c = 0; c < kk; ++c) {
    wv[c] = sigma + 1.0 / theta[pick[c]];
    Vec x = ritz_vector(L.V, S, built, pick[c]);
    for (int64_t i = 0; i < n; ++i) vv[i * kk + c] = x[i];
  }
  out.eigenvalues = evals;
  out.eigenvectors = evecs;
  out.iterations = built;
  out.converged = converged;
  return out;
}

}  // namespace scipp::sparse
