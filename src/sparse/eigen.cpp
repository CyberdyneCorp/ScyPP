// Sparse generalized symmetric eigensolver: thick-restart shift-invert Lanczos
// for K x = λ M x (K symmetric, M SPD). The operator C = (K − σ M)⁻¹ M is
// self-adjoint in the M-inner product ⟨a,b⟩_M = aᵀ M b. To keep the recurrence
// well-scaled on stiff pencils (whose eigenvalues can be ~1e13, so C's are
// ~1e-13), the operator is rescaled by s = trace(K)/trace(M): Ĉ = s·C has O(1)
// eigenvalues θ = s/(λ − σ), and eigenvalues shift back as λ = σ + s/θ.
//
// A single M-orthonormal Lanczos cycle builds a small projected matrix solved by
// NumPP's dense eigh. When the k wanted Ritz pairs have not converged, the
// solver thick-restarts (Wu & Simon): it retains the Ritz vectors nearest σ —
// their Ritz values on the diagonal, an arrowhead spike β_last·s_i[last] to the
// residual vector — and continues, deflating converged pairs so clustered
// spectra converge. The invariant-subspace test is relative to the operator
// scale, never an absolute β threshold. (K − σ M) is factored once and reused.
#include "scipp/sparse/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "numpp/linalg/linalg.hpp"
#include "scipp/sparse/detail.hpp"

namespace scipp::sparse {
namespace d = detail;
namespace {

using Vec = std::vector<double>;

// Declare an invariant subspace only when β is negligible relative to the
// operator scale — a genuine invariant subspace gives β/scale ~ machine-eps,
// whereas a stiff operator's legitimately-small β stays orders above this.
constexpr double kBreakRel = 1e-12;

double dot(const Vec& a, const Vec& b) {
  double s = 0.0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}
void axpy(Vec& y, double a, const Vec& x) {
  for (size_t i = 0; i < y.size(); ++i) y[i] += a * x[i];
}
Vec spmv(const CsrMatrix& A, const Vec& x) { return d::dv(A.spmv(d::from_dv(x))); }

double trace_sum(const CsrMatrix& A) {
  Vec diag = d::dv(A.diagonal());
  double s = 0.0; for (double v : diag) s += v; return s;
}

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

// The rescaled shift-invert operator Ĉ v = s · (K − σ M)⁻¹ (M v).
struct Operator {
  const CsrMatrix* M;
  std::shared_ptr<d::Factorization> fac;
  double s;
  Vec apply(const Vec& v) const {
    Vec w = d::factorization_solve(fac, spmv(*M, v));
    for (double& x : w) x *= s;
    return w;
  }
};

// Solve the small projected symmetric eigenproblem (dim×dim, stored in the
// leading block of the ncv×ncv H) via NumPP's dense eigensolver. Returns
// (θ ascending, S columns).
numpp::linalg::EighResult solve_projected(const std::vector<double>& H, int dim, int ncv) {
  numpp::ndarray T(numpp::Shape{dim, dim}, numpp::kFloat64);
  double* t = T.typed_data<double>();
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) t[i * dim + j] = H[static_cast<size_t>(i) * ncv + j];
  return numpp::linalg::eigh(T);
}

// Ritz-value indices ordered by |θ| descending (θ = s/(λ − σ), so largest |θ| ↔
// nearest σ).
std::vector<int> order_by_abs_desc(const double* theta, int dim) {
  std::vector<int> o(dim);
  for (int i = 0; i < dim; ++i) o[i] = i;
  std::sort(o.begin(), o.end(), [&](int a, int b) { return std::fabs(theta[a]) > std::fabs(theta[b]); });
  return o;
}

// The k Ritz indices nearest σ, reordered so their eigenvalues λ = σ + s/θ come
// out ascending (1/θ is monotone in λ for s > 0).
std::vector<int> select_nearest(const double* theta, int dim, int k) {
  std::vector<int> o = order_by_abs_desc(theta, dim);
  std::vector<int> pick(o.begin(), o.begin() + std::min(k, dim));
  std::sort(pick.begin(), pick.end(), [&](int a, int b) { return 1.0 / theta[a] < 1.0 / theta[b]; });
  return pick;
}

// Reconstruct the Ritz vector x = Σ_t S[t, idx] v_t (M-orthonormal ⇒ xᵀ M x = 1).
Vec ritz_vector(const std::vector<Vec>& V, const double* S, int dim, int idx) {
  Vec x(V.front().size(), 0.0);
  for (int t = 0; t < dim; ++t) axpy(x, S[t * dim + idx], V[t]);
  return x;
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

// Thick-restart M-orthonormal Lanczos on Ĉ. Holds the M-orthonormal basis V (and
// M·V), the dense projected matrix H, and the trailing residual so a cycle can
// restart from the retained Ritz vectors.
class TRLanczos {
 public:
  TRLanczos(const CsrMatrix& M, Operator op, int ncv)
      : M_(M), op_(std::move(op)), ncv_(ncv), H_(static_cast<size_t>(ncv) * ncv, 0.0) {}

  void seed(const Vec& v0) {
    Vec v = v0, Mv = spmv(M_, v0);
    double nrm = std::sqrt(dot(v, Mv));
    for (size_t i = 0; i < v.size(); ++i) { v[i] /= nrm; Mv[i] /= nrm; }
    V_ = {std::move(v)}; MV_ = {std::move(Mv)};
    dim_ = 0; invariant_ = false;
  }

  // Extend the basis from index `start` (V_[start] must exist) up to ncv,
  // filling H. The first step after a restart couples to the retained Ritz
  // vectors through the arrowhead spikes already stored in column `start` of H.
  // Returns the number of Lanczos steps performed.
  int extend(int start) {
    for (int j = start; j < ncv_; ++j) {
      Vec w = op_.apply(V_[j]);
      if (j == start)
        for (int i = 0; i < start; ++i) axpy(w, -H_[at(i, start)], V_[i]);   // arrowhead spikes
      else
        axpy(w, -H_[at(j - 1, j)], V_[j - 1]);                              // β_{j-1} v_{j-1}
      double alpha = dot(MV_[j], w);
      axpy(w, -alpha, V_[j]);
      H_[at(j, j)] = alpha;
      reorthogonalize(w);
      Vec Mw = spmv(M_, w);
      double beta = std::sqrt(std::max(0.0, dot(w, Mw)));
      opscale_ = std::max({opscale_, std::fabs(alpha), beta});
      if (beta <= kBreakRel * opscale_) {                 // invariant subspace reached
        invariant_ = true; beta_last_ = beta; resid_.clear(); Mresid_.clear();
        dim_ = j + 1; return j - start + 1;
      }
      for (double& x : w) x /= beta;
      for (double& x : Mw) x /= beta;
      if (j + 1 < ncv_) {                                 // v_{j+1} joins the basis
        H_[at(j, j + 1)] = beta; H_[at(j + 1, j)] = beta;
        V_.push_back(std::move(w)); MV_.push_back(std::move(Mw));
      } else {                                            // last vector is the residual
        resid_ = std::move(w); Mresid_ = std::move(Mw); beta_last_ = beta;
      }
    }
    dim_ = ncv_; invariant_ = false;
    return ncv_ - start;
  }

  // Restart: keep the `nkeep` Ritz vectors nearest σ (Ritz values on the
  // diagonal, spike β_last·s_i[last] to the residual) plus the residual vector.
  // Returns the new start index (nkeep).
  int restart(const double* theta, const double* S, int k) {
    int nkeep = std::min(ncv_ - 1, k + (ncv_ - k) / 2);
    nkeep = std::min(std::max(nkeep, k), dim_ - 1);
    std::vector<int> ord = order_by_abs_desc(theta, dim_);
    std::vector<Vec> nV, nMV;
    std::vector<double> nH(static_cast<size_t>(ncv_) * ncv_, 0.0);
    for (int jk = 0; jk < nkeep; ++jk) {
      int idx = ord[jk];
      Vec y(V_.front().size(), 0.0), My(V_.front().size(), 0.0);
      for (int t = 0; t < dim_; ++t) { axpy(y, S[t * dim_ + idx], V_[t]); axpy(My, S[t * dim_ + idx], MV_[t]); }
      nV.push_back(std::move(y)); nMV.push_back(std::move(My));
      nH[static_cast<size_t>(jk) * ncv_ + jk] = theta[idx];
      double spike = beta_last_ * S[(dim_ - 1) * dim_ + idx];
      nH[static_cast<size_t>(jk) * ncv_ + nkeep] = spike;
      nH[static_cast<size_t>(nkeep) * ncv_ + jk] = spike;
    }
    nV.push_back(resid_); nMV.push_back(Mresid_);
    V_ = std::move(nV); MV_ = std::move(nMV); H_ = std::move(nH);
    return nkeep;
  }

  int dim() const { return dim_; }
  int ncv() const { return ncv_; }
  bool invariant() const { return invariant_; }
  const std::vector<Vec>& V() const { return V_; }
  const std::vector<double>& H() const { return H_; }

 private:
  size_t at(int i, int j) const { return static_cast<size_t>(i) * ncv_ + j; }
  void reorthogonalize(Vec& w) {                          // full, twice, in the M-inner product
    for (int pass = 0; pass < 2; ++pass)
      for (size_t t = 0; t < V_.size(); ++t) axpy(w, -dot(MV_[t], w), V_[t]);
  }

  const CsrMatrix& M_;
  Operator op_;
  int ncv_;
  std::vector<double> H_;
  std::vector<Vec> V_, MV_;
  Vec resid_, Mresid_;
  double beta_last_ = 0.0, opscale_ = 0.0;
  int dim_ = 0;
  bool invariant_ = false;
};

// ---- buckling shift-walk helpers (see eigsh_buckling) ----

// A trial-shift seed magnitude in μ-space from the matrix scales, so the walk
// starts near the spectrum without any knowledge of the load factors.
double frob_scale(const CsrMatrix& K, const CsrMatrix& Kg) {
  double a = norm(Kg, "fro"), b = norm(K, "fro");
  double r = (b > 0.0) ? a / b : 1.0;
  return (std::isfinite(r) && r > 1e-300) ? r : 1.0;
}

// Factor-only definiteness probe: (K_geo − σ K) is SPD iff σ lies below the whole
// generalized spectrum of K_geo φ = μ (K φ). Counts one factorization in `shifts`.
bool probe_definite(const CsrMatrix& K, const CsrMatrix& Kg, double sigma, int& shifts) {
  ++shifts;
  CsrMatrix S = Kg.add(K.scaled(-sigma));
  auto f = d::sparse_direct_factorize(S.rows(), d::iv(S.indptr()), d::iv(S.indices()),
                                      d::dv(S.data()), /*RCM*/ 1);
  return d::factorization_definite(f);
}

// A bracket in the (negative) σ axis: `lo` is definite (below the spectrum), `hi`
// is indefinite (above μ_min). Both shifts are negative; |lo| > |hi|.
struct Bracket { double lo; double hi; };

// Geometrically expand from `seed` until one definite and one indefinite σ bound
// the spectrum boundary μ_min. Terminates because (K_geo + |σ| K) → SPD as σ → −∞.
Bracket bracket_below_spectrum(const CsrMatrix& K, const CsrMatrix& Kg, double seed,
                               int& shifts, int cap) {
  double s = (seed < 0.0) ? seed : -std::max(seed, 1e-300);
  if (probe_definite(K, Kg, s, shifts)) {                 // already below: shrink |σ| toward 0
    double lo = s, hi = s * 0.5;
    for (int i = 0; i < cap && probe_definite(K, Kg, hi, shifts); ++i) { lo = hi; hi *= 0.5; }
    return {lo, hi};
  }
  double hi = s, lo = s * 2.0;                             // still above: grow |σ| toward −∞
  for (int i = 0; i < cap && !probe_definite(K, Kg, lo, shifts); ++i) { hi = lo; lo *= 2.0; }
  return {lo, hi};
}

// Bisect the bracket a few times; stop loosely once σ* = lo sits ~20–50% below
// μ_min (magnitude ratio < 2) — close enough for strong shift-invert amplification,
// far enough for a well-conditioned SPD factor in the production solve.
double refine_shift(const CsrMatrix& K, const CsrMatrix& Kg, Bracket br, int& shifts, int cap) {
  for (int i = 0; i < cap && std::fabs(br.lo) > 2.0 * std::fabs(br.hi); ++i) {
    double mid = 0.5 * (br.lo + br.hi);
    if (probe_definite(K, Kg, mid, shifts)) br.lo = mid; else br.hi = mid;
  }
  return br.lo;
}

// From ascending pencil eigenvalues μ (of K_geo φ = μ K φ), keep the negative ones
// as load factors λ = −1/μ (already ascending) with their modes, up to k.
void collect_positive(const double* mu, const std::vector<Vec>& modes, int m, int k,
                      std::vector<double>& lam, std::vector<Vec>& out) {
  for (int i = 0; i < m && static_cast<int>(lam.size()) < k; ++i) {
    if (mu[i] < 0.0) { lam.push_back(-1.0 / mu[i]); out.push_back(modes[i]); }
  }
}

}  // namespace

// Shared thick-restart shift-invert Lanczos for the symmetric pencil A x = θ B x
// (B SPD; A may be indefinite). eigsh and eigsh_gen are thin forwarders; the only
// difference between the two public entry points is which operand is named the
// stiffness — the recurrence only ever requires the inner-product matrix B to be
// SPD, and A's indefiniteness is absorbed by the one-time (A − σ B) factorization.
EigshResult eigsh_core(const CsrMatrix& A, const CsrMatrix& B, int k,
                       double sigma, double tol, int maxiter) {
  int64_t n = A.rows();
  EigshResult out;
  if (k <= 0 || n <= 0) return out;
  if (k > n) k = static_cast<int>(n);

  // Rescale the operator so its eigenvalues θ = s/(λ − σ) are O(1) (stiff pencils
  // otherwise put the whole recurrence at magnitude ~1/λ). λ shifts back as σ + s/θ.
  // s only conditions the projected problem — it cancels exactly in λ = σ + s/θ —
  // so |trace(A)| keeps s well-scaled even when A is indefinite (trace ≤ 0). For
  // every SPD pencil (trace(A) ≥ trace(B) > 0) this equals the old trace(A)/trace(B).
  double trA = trace_sum(A), trB = trace_sum(B);
  double s = (std::isfinite(trA) && std::isfinite(trB) && trB > 0.0)
                 ? std::max(std::fabs(trA), trB) / trB
                 : 1.0;

  // Shift-invert operator (A − σ B), factored once and reused every iteration.
  CsrMatrix Ash = (sigma == 0.0) ? A : A.add(B.scaled(-sigma));
  auto fac = d::sparse_direct_factorize(Ash.rows(), d::iv(Ash.indptr()), d::iv(Ash.indices()),
                                        d::dv(Ash.data()), /*RCM*/ 1);
  if (!d::factorization_ok(fac)) return out;     // (A − σ B) singular: pick another sigma

  int ncv = std::min<int>(static_cast<int>(n), std::max(2 * k + 1, 20));
  int maxcycles = (maxiter > 0) ? maxiter : 200;
  Operator op{&B, fac, s};
  TRLanczos lz(B, op, ncv);
  lz.seed(start_vector(n));

  int start = 0, steps = 0;
  bool converged = false;
  numpp::linalg::EighResult tri;
  std::vector<int> pick;
  for (int cyc = 0; cyc < maxcycles; ++cyc) {
    steps += lz.extend(start);
    tri = solve_projected(lz.H(), lz.dim(), lz.ncv());
    const double* theta = tri.eigenvalues.typed_data<double>();
    const double* S = tri.eigenvectors.typed_data<double>();
    pick = select_nearest(theta, lz.dim(), k);

    converged = true;
    for (int idx : pick) {
      Vec x = ritz_vector(lz.V(), S, lz.dim(), idx);
      if (gen_residual(A, B, x, sigma + s / theta[idx]) > tol) { converged = false; break; }
    }
    if (converged || lz.invariant() || cyc == maxcycles - 1) break;
    start = lz.restart(theta, S, k);
  }

  // Materialize eigenvalues and (n × k) B-normalized eigenvectors.
  int kk = static_cast<int>(pick.size()), dim = lz.dim();
  const double* theta = tri.eigenvalues.typed_data<double>();
  const double* S = tri.eigenvectors.typed_data<double>();
  numpp::ndarray evals(numpp::Shape{kk}, numpp::kFloat64);
  numpp::ndarray evecs(numpp::Shape{n, kk}, numpp::kFloat64);
  double* wv = evals.typed_data<double>();
  double* vv = evecs.typed_data<double>();
  for (int c = 0; c < kk; ++c) {
    wv[c] = sigma + s / theta[pick[c]];
    Vec x = ritz_vector(lz.V(), S, dim, pick[c]);
    for (int64_t i = 0; i < n; ++i) vv[i * kk + c] = x[i];
  }
  out.eigenvalues = evals;
  out.eigenvectors = evecs;
  out.iterations = steps;
  out.converged = converged;
  return out;
}

EigshResult eigsh(const CsrMatrix& K, const CsrMatrix& M, int k,
                  double sigma, double tol, int maxiter) {
  return eigsh_core(K, M, k, sigma, tol, maxiter);
}

EigshResult eigsh_gen(const CsrMatrix& A, const CsrMatrix& B, int k,
                      double sigma, double tol, int maxiter) {
  return eigsh_core(A, B, k, sigma, tol, maxiter);
}

BucklingResult eigsh_buckling(const CsrMatrix& K, const CsrMatrix& K_geo, int k,
                              double sigma0, double tol, int maxiter) {
  int64_t n = K.rows();
  BucklingResult out;
  if (k <= 0 || n <= 0) return out;
  if (k > n) k = static_cast<int>(n);

  // Locate a shift strictly below the whole μ-spectrum of K_geo φ = μ (K φ) using
  // cheap factor-only definiteness probes, so the k eigenvalues nearest σ* are the
  // k most-negative μ = k smallest positive load factors λ = −1/μ.
  double seed = (sigma0 > 0.0) ? -1.0 / sigma0 : -frob_scale(K, K_geo);
  int shifts = 0;
  Bracket br = bracket_below_spectrum(K, K_geo, seed, shifts, /*cap*/ 60);
  double star = refine_shift(K, K_geo, br, shifts, /*cap*/ 8);

  // One generalized shift-invert solve; probe a few extra modes so ≥ k negative-μ
  // survivors remain after filtering. Nudge σ* further below on a singular ridge.
  int kp = std::min<int>(static_cast<int>(n), k + std::max(k, 4));
  EigshResult r = eigsh_gen(K_geo, K, kp, star, tol, maxiter);
  if (r.eigenvalues.size() == 0) {
    star *= 1.1;
    r = eigsh_gen(K_geo, K, kp, star, tol, maxiter);
  }

  int m = static_cast<int>(r.eigenvalues.size());
  if (m == 0) { out.shifts = shifts; return out; }
  const double* mu = r.eigenvalues.typed_data<double>();
  const double* ev = r.eigenvectors.typed_data<double>();
  std::vector<Vec> modes(m, Vec(static_cast<size_t>(n)));
  for (int c = 0; c < m; ++c)
    for (int64_t i = 0; i < n; ++i) modes[c][static_cast<size_t>(i)] = ev[i * m + c];

  std::vector<double> lam;
  std::vector<Vec> keep;
  collect_positive(mu, modes, m, k, lam, keep);

  int kk = static_cast<int>(lam.size());
  numpp::ndarray lf(numpp::Shape{kk}, numpp::kFloat64);
  numpp::ndarray md(numpp::Shape{n, kk}, numpp::kFloat64);
  double* lp = lf.typed_data<double>();
  double* mp = md.typed_data<double>();
  for (int c = 0; c < kk; ++c) {
    lp[c] = lam[static_cast<size_t>(c)];
    for (int64_t i = 0; i < n; ++i) mp[i * kk + c] = keep[static_cast<size_t>(c)][static_cast<size_t>(i)];
  }
  out.load_factors = lf;
  out.modes = md;
  out.iterations = r.iterations;
  out.shifts = shifts;
  out.converged = r.converged && kk >= k;
  return out;
}

}  // namespace scipp::sparse
