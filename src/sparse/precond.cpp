// Preconditioners for the sparse iterative solvers: Jacobi (inverse diagonal),
// IC0 (zero-fill incomplete Cholesky, M = L Lᵀ), and ILU0 (zero-fill incomplete
// LU, M = L U). Each builds from a canonical CSR matrix and exposes `apply`,
// computing M⁻¹ r via triangular solves. A failed incomplete factorization
// (non-SPD or zero pivot) degrades to Jacobi so callers always get a usable M.
#include <cmath>
#include <vector>

#include "scipp/sparse/detail.hpp"

namespace scipp::sparse::detail {
namespace {

constexpr double kTiny = 1e-300;

// Jacobi: invdiag[i] = 1 / A_ii (0 where the diagonal is missing/zero).
Precond build_jacobi(int64_t n, const std::vector<int64_t>& Ap,
                     const std::vector<int64_t>& Ai, const std::vector<double>& Ax) {
  Precond m; m.kind = 1; m.n = n; m.invdiag.assign(n, 0.0);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = Ap[i]; k < Ap[i + 1]; ++k)
      if (Ai[k] == i) m.invdiag[i] = (Ax[k] != 0.0) ? 1.0 / Ax[k] : 0.0;
  return m;
}

// IC0: lower triangle (col ≤ row) keeps A's pattern; values overwritten in place
// with the incomplete Cholesky factor. Returns false on a non-positive pivot.
bool build_ic0(int64_t n, const std::vector<int64_t>& Ap, const std::vector<int64_t>& Ai,
               const std::vector<double>& Ax, Precond& m) {
  m.Lp.assign(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t k = Ap[i]; k < Ap[i + 1]; ++k)
      if (Ai[k] <= i) { m.Li.push_back(Ai[k]); m.Lx.push_back(Ax[k]); }
    m.Lp[i + 1] = static_cast<int64_t>(m.Li.size());
  }
  m.Ldiag.assign(n, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t p = m.Lp[i]; p < m.Lp[i + 1]; ++p) {
      int64_t j = m.Li[p];
      double s = m.Lx[p];
      // s -= Σ_{c<j} L_ic L_jc  over columns shared by rows i and j
      int64_t pi = m.Lp[i], pj = m.Lp[j];
      while (pi < p && pj < m.Lp[j + 1]) {
        int64_t ci = m.Li[pi], cj = m.Li[pj];
        if (cj >= j) break;
        if (ci < cj) ++pi;
        else if (ci > cj) ++pj;
        else { s -= m.Lx[pi] * m.Lx[pj]; ++pi; ++pj; }
      }
      if (j < i) {
        m.Lx[p] = s / m.Ldiag[j];
      } else {  // diagonal
        if (s <= 0.0) return false;
        m.Lx[p] = std::sqrt(s);
        m.Ldiag[i] = m.Lx[p];
      }
    }
  }
  return true;
}

// ILU0: keep A's full pattern; overwrite with the combined L\U factor (unit L).
// Returns false when a diagonal entry is missing or a pivot is zero.
bool build_ilu0(int64_t n, const std::vector<int64_t>& Ap, const std::vector<int64_t>& Ai,
                const std::vector<double>& Ax, Precond& m) {
  m.Up = Ap; m.Ui = Ai; m.Ux = Ax; m.dpos.assign(n, -1);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = m.Up[i]; k < m.Up[i + 1]; ++k)
      if (m.Ui[k] == i) m.dpos[i] = k;
  for (int64_t i = 0; i < n; ++i) {
    if (m.dpos[i] < 0) return false;
    for (int64_t p = m.Up[i]; p < m.Up[i + 1]; ++p) {
      int64_t k = m.Ui[p];
      if (k >= i) break;                       // L part only (cols < i)
      double akk = m.Ux[m.dpos[k]];
      if (akk == 0.0) return false;
      double piv = m.Ux[p] / akk;
      m.Ux[p] = piv;
      // row_i[cols>k] -= piv * row_k[cols>k], restricted to the shared pattern
      int64_t pj = p + 1, pk = m.dpos[k] + 1;
      while (pj < m.Up[i + 1] && pk < m.Up[k + 1]) {
        int64_t cj = m.Ui[pj], ck = m.Ui[pk];
        if (cj < ck) ++pj;
        else if (cj > ck) ++pk;
        else { m.Ux[pj] -= piv * m.Ux[pk]; ++pj; ++pk; }
      }
    }
    if (m.Ux[m.dpos[i]] == 0.0) return false;
  }
  return true;
}

// IC0 apply: solve (L Lᵀ) z = r via forward then backward substitution.
std::vector<double> apply_ic0(const Precond& m, const std::vector<double>& r) {
  std::vector<double> y = r;
  for (int64_t i = 0; i < m.n; ++i) {                              // forward: L y = r
    double s = y[i];
    for (int64_t p = m.Lp[i]; p < m.Lp[i + 1] && m.Li[p] < i; ++p) s -= m.Lx[p] * y[m.Li[p]];
    y[i] = s / (m.Ldiag[i] != 0.0 ? m.Ldiag[i] : kTiny);
  }
  for (int64_t j = m.n - 1; j >= 0; --j) {                         // backward: Lᵀ z = y
    y[j] /= (m.Ldiag[j] != 0.0 ? m.Ldiag[j] : kTiny);
    for (int64_t p = m.Lp[j]; p < m.Lp[j + 1] && m.Li[p] < j; ++p) y[m.Li[p]] -= m.Lx[p] * y[j];
  }
  return y;
}

// ILU0 apply: solve (L U) z = r, L unit-lower and U upper.
std::vector<double> apply_ilu0(const Precond& m, const std::vector<double>& r) {
  std::vector<double> y = r;
  for (int64_t i = 0; i < m.n; ++i) {                              // forward: L y = r
    double s = y[i];
    for (int64_t p = m.Up[i]; p < m.Up[i + 1] && m.Ui[p] < i; ++p) s -= m.Ux[p] * y[m.Ui[p]];
    y[i] = s;
  }
  for (int64_t i = m.n - 1; i >= 0; --i) {                         // backward: U z = y
    double s = y[i];
    for (int64_t p = m.dpos[i] + 1; p < m.Up[i + 1]; ++p) s -= m.Ux[p] * y[m.Ui[p]];
    double d = m.Ux[m.dpos[i]];
    y[i] = s / (d != 0.0 ? d : kTiny);
  }
  return y;
}

}  // namespace

Precond make_precond(int kind, int64_t n, const std::vector<int64_t>& Ap,
                     const std::vector<int64_t>& Ai, const std::vector<double>& Ax) {
  if (kind == 0) { Precond m; m.kind = 0; m.n = n; return m; }
  if (kind == 1) return build_jacobi(n, Ap, Ai, Ax);
  if (kind == 2) {
    Precond m; m.kind = 2; m.n = n;
    if (build_ic0(n, Ap, Ai, Ax, m)) return m;
    return build_jacobi(n, Ap, Ai, Ax);       // degrade: matrix not SPD
  }
  // kind == 3 (ILU0)
  Precond m; m.kind = 3; m.n = n;
  if (build_ilu0(n, Ap, Ai, Ax, m)) return m;
  return build_jacobi(n, Ap, Ai, Ax);         // degrade: zero pivot
}

std::vector<double> Precond::apply(const std::vector<double>& r) const {
  if (kind == 1) {                                                  // Jacobi
    std::vector<double> z(n);
    for (int64_t i = 0; i < n; ++i) z[i] = r[i] * invdiag[i];
    return z;
  }
  if (kind == 2) return apply_ic0(*this, r);                        // IC0
  if (kind == 3) return apply_ilu0(*this, r);                       // ILU0
  return r;                                                         // None: identity
}

}  // namespace scipp::sparse::detail
