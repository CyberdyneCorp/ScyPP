// Sparse direct factorization for scipp::sparse: a fill-reducing ordering (RCM),
// an up-looking sparse Cholesky (SPD), and a Gilbert–Peierls left-looking sparse
// LU with partial pivoting (general). Algorithms follow Tim Davis' CSparse
// formulations (etree/ereach for Cholesky; reach/spsolve for LU), implemented
// from the published algorithm. Everything operates on compressed arrays — the
// matrix is never densified.
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "scipp/sparse/detail.hpp"

namespace scipp::sparse::detail {

using Idx = std::vector<int64_t>;
using Val = std::vector<double>;

// Cholesky factor (up-looking): A(perm,perm) = L Lᵀ with L in CSC (Lp/Li/Lx).
struct Chol { int64_t n = 0; Idx Lp, Li; Val Lx; Idx perm; bool ok = false; };

// LU factor (Gilbert–Peierls, partial pivoting): P A Q = L U (P = pinv, Q = q).
struct Lu { int64_t n = 0; Idx Lp, Li; Val Lx; Idx Up, Ui; Val Ux; Idx pinv, q; bool ok = false; };

// A reusable sparse direct factor (factor once, solve many): SPD → Cholesky,
// otherwise LU. Only the branch selected by `used_cholesky` is populated. Held
// opaquely by callers through a shared_ptr — see sparse_direct_factorize.
struct Factorization {
  int64_t n = 0;
  bool ok = false;
  bool used_cholesky = false;
  int64_t factor_nnz = 0;
  Chol chol;
  Lu lu;
};

namespace {

// Column-compressed matrix (CSC).
struct Csc {
  int64_t n = 0;   // columns
  int64_t m = 0;   // rows
  Idx p, i;
  Val x;
};

// ---- conversions ----------------------------------------------------------

// CSR (rows Ap/Ai/Ax over n×n) → CSC. Because CSC of A is the CSR of Aᵀ, this
// doubles as the transpose. Row indices within each column come out ascending.
Csc csr_to_csc(int64_t n, const Idx& Ap, const Idx& Ai, const Val& Ax) {
  int64_t nnz = static_cast<int64_t>(Ai.size());
  Csc C; C.n = n; C.m = n; C.p.assign(n + 1, 0); C.i.resize(nnz); C.x.resize(nnz);
  for (int64_t k = 0; k < nnz; ++k) C.p[Ai[k] + 1]++;
  for (int64_t j = 0; j < n; ++j) C.p[j + 1] += C.p[j];
  Idx next(C.p.begin(), C.p.end());
  for (int64_t r = 0; r < n; ++r)
    for (int64_t k = Ap[r]; k < Ap[r + 1]; ++k) {
      int64_t c = Ai[k], dst = next[c]++;
      C.i[dst] = r; C.x[dst] = Ax[k];
    }
  return C;
}

// True if A (canonical CSR) is numerically symmetric within tol.
bool is_symmetric(int64_t n, const Idx& Ap, const Idx& Ai, const Val& Ax) {
  Csc T = csr_to_csc(n, Ap, Ai, Ax);   // arrays of Aᵀ in CSR form
  if (T.p.size() != Ap.size() || T.i.size() != Ai.size()) return false;
  for (int64_t k = 0; k <= n; ++k) if (T.p[k] != Ap[k]) return false;
  for (size_t k = 0; k < Ai.size(); ++k) {
    if (T.i[k] != Ai[k]) return false;
    if (std::fabs(T.x[k] - Ax[k]) > 1e-12 * (1.0 + std::fabs(Ax[k]))) return false;
  }
  return true;
}

// ---- RCM ordering ----------------------------------------------------------

// Symmetric adjacency (pattern of A ∪ Aᵀ, no self-loops) as a CSR-like list.
void build_adjacency(int64_t n, const Idx& Ap, const Idx& Ai, Idx& adjp, Idx& adji) {
  std::vector<std::vector<int64_t>> nb(n);
  for (int64_t r = 0; r < n; ++r)
    for (int64_t k = Ap[r]; k < Ap[r + 1]; ++k) {
      int64_t c = Ai[k];
      if (c != r) { nb[r].push_back(c); nb[c].push_back(r); }
    }
  adjp.assign(n + 1, 0);
  for (int64_t v = 0; v < n; ++v) {
    auto& l = nb[v];
    std::sort(l.begin(), l.end());
    l.erase(std::unique(l.begin(), l.end()), l.end());
    adjp[v + 1] = adjp[v] + static_cast<int64_t>(l.size());
  }
  adji.resize(adjp[n]);
  for (int64_t v = 0; v < n; ++v)
    std::copy(nb[v].begin(), nb[v].end(), adji.begin() + adjp[v]);
}

// Reverse Cuthill–McKee. Returns perm with perm[new] = old node. Each connected
// component is started from an unvisited minimum-degree node; within each BFS
// level neighbours are appended in ascending degree.
Idx rcm(int64_t n, const Idx& Ap, const Idx& Ai) {
  Idx adjp, adji; build_adjacency(n, Ap, Ai, adjp, adji);
  Idx deg(n);
  for (int64_t v = 0; v < n; ++v) deg[v] = adjp[v + 1] - adjp[v];
  std::vector<char> seen(n, 0);
  Idx order; order.reserve(n);
  for (int64_t start = 0; start < n; ++start) {
    // pick the next unvisited node of minimum degree as a component root
    if (seen[start]) continue;
    int64_t root = start;
    for (int64_t v = start; v < n; ++v)
      if (!seen[v] && deg[v] < deg[root]) root = v;
    seen[root] = 1;
    size_t head = order.size();
    order.push_back(root);
    while (head < order.size()) {
      int64_t c = order[head++];
      Idx nbrs;
      for (int64_t k = adjp[c]; k < adjp[c + 1]; ++k) {
        int64_t w = adji[k];
        if (!seen[w]) { seen[w] = 1; nbrs.push_back(w); }
      }
      std::sort(nbrs.begin(), nbrs.end(), [&](int64_t a, int64_t b) { return deg[a] < deg[b]; });
      for (int64_t w : nbrs) order.push_back(w);
    }
  }
  std::reverse(order.begin(), order.end());
  return order;
}

// Permuted CSC of B = A(p,p) given iperm (iperm[old] = new). Natural ordering
// (iperm = identity) yields the plain CSC of A. Columns are row-sorted.
Csc permuted_csc(int64_t n, const Idx& Ap, const Idx& Ai, const Val& Ax, const Idx& iperm) {
  Csc C; C.n = n; C.m = n; C.p.assign(n + 1, 0);
  int64_t nnz = static_cast<int64_t>(Ai.size());
  C.i.resize(nnz); C.x.resize(nnz);
  for (int64_t r = 0; r < n; ++r)
    for (int64_t k = Ap[r]; k < Ap[r + 1]; ++k) C.p[iperm[Ai[k]] + 1]++;
  for (int64_t j = 0; j < n; ++j) C.p[j + 1] += C.p[j];
  Idx next(C.p.begin(), C.p.end());
  for (int64_t r = 0; r < n; ++r)
    for (int64_t k = Ap[r]; k < Ap[r + 1]; ++k) {
      int64_t jn = iperm[Ai[k]], in = iperm[r], dst = next[jn]++;
      C.i[dst] = in; C.x[dst] = Ax[k];
    }
  // sort each column by row index
  for (int64_t j = 0; j < n; ++j) {
    int64_t s = C.p[j], e = C.p[j + 1];
    Idx ord(e - s); for (int64_t t = 0; t < e - s; ++t) ord[t] = s + t;
    std::sort(ord.begin(), ord.end(), [&](int64_t a, int64_t b) { return C.i[a] < C.i[b]; });
    Idx ci(e - s); Val cx(e - s);
    for (int64_t t = 0; t < e - s; ++t) { ci[t] = C.i[ord[t]]; cx[t] = C.x[ord[t]]; }
    for (int64_t t = 0; t < e - s; ++t) { C.i[s + t] = ci[t]; C.x[s + t] = cx[t]; }
  }
  return C;
}

// ---- Cholesky (up-looking) -------------------------------------------------

// Elimination tree of a symmetric matrix from its upper triangle (CSC).
Idx etree(const Csc& C) {
  int64_t n = C.n;
  Idx parent(n, -1), ancestor(n, -1);
  for (int64_t k = 0; k < n; ++k)
    for (int64_t p = C.p[k]; p < C.p[k + 1]; ++p) {
      int64_t i = C.i[p];
      while (i != -1 && i < k) {
        int64_t inext = ancestor[i];
        ancestor[i] = k;
        if (inext == -1) parent[i] = k;
        i = inext;
      }
    }
  return parent;
}

// Nonzero pattern of row k of L (columns i<k). Returns top; s[top..n-1] holds the
// pattern in topological order. `mark[i]==k` flags nodes visited this step.
int64_t ereach(const Csc& C, int64_t k, const Idx& parent, Idx& s, Idx& mark) {
  int64_t n = C.n, top = n;
  mark[k] = k;
  for (int64_t p = C.p[k]; p < C.p[k + 1]; ++p) {
    int64_t i = C.i[p];
    if (i > k) continue;
    int64_t len = 0;
    while (mark[i] != k) { s[len++] = i; mark[i] = k; i = parent[i]; }
    while (len > 0) s[--top] = s[--len];
  }
  return top;
}

// Up-looking numeric Cholesky of C (SPD, symmetric CSC). Returns ok=false on a
// non-positive pivot (matrix not positive definite).
Chol chol_factor(const Csc& C, Idx perm) {
  int64_t n = C.n;
  Chol F; F.n = n; F.perm = std::move(perm);
  Idx parent = etree(C);
  Idx s(n), mark(n, -1), colcount(n, 1);          // 1 per diagonal
  for (int64_t k = 0; k < n; ++k) {
    int64_t top = ereach(C, k, parent, s, mark);
    for (int64_t t = top; t < n; ++t) colcount[s[t]]++;
  }
  F.Lp.assign(n + 1, 0);
  for (int64_t i = 0; i < n; ++i) F.Lp[i + 1] = F.Lp[i] + colcount[i];
  F.Li.assign(F.Lp[n], 0); F.Lx.assign(F.Lp[n], 0.0);
  Idx cnext(n);
  for (int64_t i = 0; i < n; ++i) { F.Li[F.Lp[i]] = i; cnext[i] = F.Lp[i] + 1; }  // reserve diagonal
  std::fill(mark.begin(), mark.end(), -1);
  Val x(n, 0.0);
  for (int64_t k = 0; k < n; ++k) {
    int64_t top = ereach(C, k, parent, s, mark);
    double d = 0.0;
    for (int64_t p = C.p[k]; p < C.p[k + 1]; ++p) {
      int64_t i = C.i[p];
      if (i < k) x[i] = C.x[p]; else if (i == k) d = C.x[p];
    }
    for (int64_t t = top; t < n; ++t) {
      int64_t i = s[t];
      double lki = x[i] / F.Lx[F.Lp[i]];
      x[i] = 0.0;
      for (int64_t p = F.Lp[i] + 1; p < cnext[i]; ++p) x[F.Li[p]] -= F.Lx[p] * lki;
      d -= lki * lki;
      F.Li[cnext[i]] = k; F.Lx[cnext[i]] = lki; cnext[i]++;
    }
    if (d <= 0.0) { F.ok = false; return F; }
    F.Lx[F.Lp[k]] = std::sqrt(d);
  }
  F.ok = true;
  return F;
}

// Solve A x = b where A(perm,perm) = L Lᵀ.
Val chol_solve(const Chol& F, const Val& b) {
  int64_t n = F.n;
  Val y(n);
  for (int64_t i = 0; i < n; ++i) y[i] = b[F.perm[i]];           // y = P b
  for (int64_t j = 0; j < n; ++j) {                             // L z = y
    y[j] /= F.Lx[F.Lp[j]];
    for (int64_t p = F.Lp[j] + 1; p < F.Lp[j + 1]; ++p) y[F.Li[p]] -= F.Lx[p] * y[j];
  }
  for (int64_t j = n - 1; j >= 0; --j) {                        // Lᵀ w = z
    for (int64_t p = F.Lp[j] + 1; p < F.Lp[j + 1]; ++p) y[j] -= F.Lx[p] * y[F.Li[p]];
    y[j] /= F.Lx[F.Lp[j]];
  }
  Val x(n);
  for (int64_t i = 0; i < n; ++i) x[F.perm[i]] = y[i];          // x = Pᵀ w
  return x;
}

// ---- LU (Gilbert–Peierls left-looking, partial pivoting) -------------------

// Depth-first search from node j over the columns of the partially-built L,
// pushing finished nodes onto xi[top..]. `mark` flags visited nodes.
int64_t lu_dfs(int64_t j0, const Idx& Lp, const Idx& Li, const Idx& pinv,
               int64_t top, Idx& xi, Idx& pstack, std::vector<char>& mark) {
  int64_t head = 0; xi[0] = j0;
  while (head >= 0) {
    int64_t j = xi[head];
    int64_t jnew = pinv[j];
    if (!mark[j]) { mark[j] = 1; pstack[head] = (jnew < 0) ? 0 : Lp[jnew]; }
    bool done = true;
    if (jnew >= 0) {
      int64_t pend = Lp[jnew + 1];
      for (int64_t p = pstack[head]; p < pend; ++p) {
        int64_t i = Li[p];
        if (mark[i]) continue;
        pstack[head] = p; xi[++head] = i; done = false; break;
      }
    }
    if (done) { --head; xi[--top] = j; }
  }
  return top;
}

// Reachable set (in topological order) of the RHS A(:,col) through L; xi[top..]
// holds the pattern, which is then unmarked.
int64_t lu_reach(const Idx& Lp, const Idx& Li, const Csc& A, int64_t col,
                 const Idx& pinv, Idx& xi, Idx& pstack, std::vector<char>& mark) {
  int64_t n = A.n, top = n;
  for (int64_t p = A.p[col]; p < A.p[col + 1]; ++p) {
    int64_t i = A.i[p];
    if (!mark[i]) top = lu_dfs(i, Lp, Li, pinv, top, xi, pstack, mark);
  }
  for (int64_t p = top; p < n; ++p) mark[xi[p]] = 0;
  return top;
}

// x = L \ A(:,col) using the existing L columns; returns top with xi[top..]=pattern.
int64_t lu_spsolve(const Idx& Lp, const Idx& Li, const Val& Lx, const Csc& A, int64_t col,
                   const Idx& pinv, Idx& xi, Idx& pstack, Val& x, std::vector<char>& mark) {
  int64_t n = A.n;
  int64_t top = lu_reach(Lp, Li, A, col, pinv, xi, pstack, mark);
  for (int64_t p = top; p < n; ++p) x[xi[p]] = 0.0;
  for (int64_t p = A.p[col]; p < A.p[col + 1]; ++p) x[A.i[p]] = A.x[p];
  for (int64_t px = top; px < n; ++px) {
    int64_t j = xi[px], jnew = pinv[j];
    if (jnew < 0) continue;                          // j not yet pivotal
    for (int64_t p = Lp[jnew] + 1; p < Lp[jnew + 1]; ++p) x[Li[p]] -= Lx[p] * x[j];
  }
  return top;
}

Lu lu_factor(const Csc& A, Idx q) {
  int64_t n = A.n;
  Lu F; F.n = n; F.q = std::move(q);
  F.Lp.assign(n + 1, 0); F.Up.assign(n + 1, 0); F.pinv.assign(n, -1);
  Val x(n, 0.0);
  Idx xi(n), pstack(n);
  std::vector<char> mark(n, 0);
  const double tol = 1.0;                            // partial pivoting
  for (int64_t k = 0; k < n; ++k) {
    F.Lp[k] = static_cast<int64_t>(F.Li.size());
    F.Up[k] = static_cast<int64_t>(F.Ui.size());
    int64_t col = F.q[k];
    int64_t top = lu_spsolve(F.Lp, F.Li, F.Lx, A, col, F.pinv, xi, pstack, x, mark);
    int64_t ipiv = -1; double a = -1.0;
    for (int64_t p = top; p < n; ++p) {
      int64_t i = xi[p];
      if (F.pinv[i] < 0) { double t = std::fabs(x[i]); if (t > a) { a = t; ipiv = i; } }
      else { F.Ui.push_back(F.pinv[i]); F.Ux.push_back(x[i]); }   // entry of U
    }
    if (ipiv == -1 || a <= 0.0) { F.ok = false; return F; }       // singular
    if (F.pinv[col] < 0 && std::fabs(x[col]) >= a * tol) ipiv = col;  // prefer diagonal
    double pivot = x[ipiv];
    F.Ui.push_back(k); F.Ux.push_back(pivot);                     // U diagonal (last)
    F.pinv[ipiv] = k;
    F.Li.push_back(ipiv); F.Lx.push_back(1.0);                    // L unit diagonal (first)
    for (int64_t p = top; p < n; ++p) {
      int64_t i = xi[p];
      if (F.pinv[i] < 0) { F.Li.push_back(i); F.Lx.push_back(x[i] / pivot); }
      x[i] = 0.0;
    }
  }
  F.Lp[n] = static_cast<int64_t>(F.Li.size());
  F.Up[n] = static_cast<int64_t>(F.Ui.size());
  for (size_t p = 0; p < F.Li.size(); ++p) F.Li[p] = F.pinv[F.Li[p]];  // remap L rows
  F.ok = true;
  return F;
}

// Solve A x = b with P A Q = L U (P = pinv, Q from q).
Val lu_solve(const Lu& F, const Val& b) {
  int64_t n = F.n;
  Val xw(n, 0.0);
  for (int64_t i = 0; i < n; ++i) xw[F.pinv[i]] = b[i];          // xw = P b
  for (int64_t j = 0; j < n; ++j)                                // L xw = P b (unit L)
    for (int64_t p = F.Lp[j] + 1; p < F.Lp[j + 1]; ++p) xw[F.Li[p]] -= F.Lx[p] * xw[j];
  for (int64_t j = n - 1; j >= 0; --j) {                         // U xw = ...
    double d = F.Ux[F.Up[j + 1] - 1];
    xw[j] /= d;
    for (int64_t p = F.Up[j]; p < F.Up[j + 1] - 1; ++p) xw[F.Ui[p]] -= F.Ux[p] * xw[j];
  }
  Val x(n);
  for (int64_t i = 0; i < n; ++i) x[F.q[i]] = xw[i];             // x = Q xw
  return x;
}

// ---- ordering selection ----------------------------------------------------

// Returns perm (new→old) and fills iperm (old→new). ordering 0=Natural, else RCM.
Idx make_ordering(int64_t n, const Idx& Ap, const Idx& Ai, int ordering, Idx& iperm) {
  Idx perm;
  if (ordering == 0) { perm.resize(n); for (int64_t i = 0; i < n; ++i) perm[i] = i; }
  else perm = rcm(n, Ap, Ai);                                    // RCM (also for AMD, for now)
  iperm.assign(n, 0);
  for (int64_t i = 0; i < n; ++i) iperm[perm[i]] = i;
  return perm;
}

// Factor A (canonical CSR) with the given ordering into a reusable Factorization.
// SPD → up-looking Cholesky, otherwise Gilbert–Peierls LU. ok=false ⇒ singular.
Factorization build_factorization(int64_t n, const Idx& Ap, const Idx& Ai, const Val& Ax,
                                  int ordering) {
  Factorization F; F.n = n;
  Idx iperm;
  Idx perm = make_ordering(n, Ap, Ai, ordering, iperm);
  if (is_symmetric(n, Ap, Ai, Ax)) {
    Csc C = permuted_csc(n, Ap, Ai, Ax, iperm);
    Chol c = chol_factor(C, perm);
    if (c.ok) {
      F.ok = true; F.used_cholesky = true; F.factor_nnz = c.Lp[n]; F.chol = std::move(c);
      return F;
    }
  }
  Csc Acsc = csr_to_csc(n, Ap, Ai, Ax);        // plain CSC of A
  Lu lu = lu_factor(Acsc, perm);               // RCM order as the column ordering
  if (!lu.ok) { F.ok = false; return F; }
  F.ok = true; F.used_cholesky = false;
  F.factor_nnz = static_cast<int64_t>(lu.Li.size()); F.lu = std::move(lu);
  return F;
}

// Solve A x = b reusing an existing factor.
Val apply_factorization(const Factorization& F, const Val& b) {
  return F.used_cholesky ? chol_solve(F.chol, b) : lu_solve(F.lu, b);
}

DirectResult to_result(const Factorization& F) {
  DirectResult R;
  R.ok = F.ok; R.used_cholesky = F.used_cholesky; R.factor_nnz = F.factor_nnz;
  return R;
}

}  // namespace

DirectResult sparse_direct_solve(int64_t n, const Idx& Ap, const Idx& Ai, const Val& Ax,
                                 const Val& b, int ordering) {
  Factorization F = build_factorization(n, Ap, Ai, Ax, ordering);
  DirectResult R = to_result(F);
  if (F.ok) R.x = apply_factorization(F, b);
  return R;
}

DirectResult sparse_direct_factor_nnz(int64_t n, const Idx& Ap, const Idx& Ai, const Val& Ax,
                                      int ordering) {
  return to_result(build_factorization(n, Ap, Ai, Ax, ordering));
}

std::shared_ptr<Factorization> sparse_direct_factorize(int64_t n, const Idx& Ap, const Idx& Ai,
                                                       const Val& Ax, int ordering) {
  return std::make_shared<Factorization>(build_factorization(n, Ap, Ai, Ax, ordering));
}

bool factorization_ok(const std::shared_ptr<Factorization>& F) { return F && F->ok; }

// True when the factored matrix is symmetric-positive-definite: the SPD Cholesky
// path was taken (Cholesky is attempted only for symmetric input and succeeds
// iff every pivot is positive). For a symmetric pencil (A − σ B) with B SPD this
// is a free Sturm bit — it holds iff σ lies below the whole generalized spectrum.
bool factorization_definite(const std::shared_ptr<Factorization>& F) {
  return F && F->ok && F->used_cholesky;
}

Val factorization_solve(const std::shared_ptr<Factorization>& F, const Val& b) {
  return apply_factorization(*F, b);
}

}  // namespace scipp::sparse::detail
