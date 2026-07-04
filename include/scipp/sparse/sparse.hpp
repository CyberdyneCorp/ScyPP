#pragma once
// scipp::sparse — port of scipy.sparse (Phase 9): COO/CSR/CSC formats, sparse
// products with backend dispatch, sparse linear solvers, and graph algorithms.

#include <cstdint>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scipp::sparse {

using numpp::ndarray;
class CsrMatrix;
class CscMatrix;

struct CooMatrix {
  ndarray data, row, col;
  int64_t rows = 0, cols = 0;
};

class CsrMatrix {
 public:
  CsrMatrix() = default;
  CsrMatrix(ndarray data, ndarray indices, ndarray indptr, int64_t rows, int64_t cols);
  static CsrMatrix from_coo(const CooMatrix& coo);
  static CsrMatrix from_dense(const ndarray& a);

  ndarray toarray() const;
  int64_t nnz() const;
  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }
  const ndarray& data() const { return data_; }
  const ndarray& indices() const { return indices_; }
  const ndarray& indptr() const { return indptr_; }

  CscMatrix transpose() const;
  ndarray diagonal() const;
  CsrMatrix add(const CsrMatrix& b) const;
  CsrMatrix scaled(double s) const;
  ndarray spmv(const ndarray& x) const;
  ndarray spmm(const ndarray& X) const;

 private:
  ndarray data_, indices_, indptr_;
  int64_t rows_ = 0, cols_ = 0;
};

class CscMatrix {
 public:
  CscMatrix(ndarray data, ndarray indices, ndarray indptr, int64_t rows, int64_t cols);
  ndarray toarray() const;
  int64_t nnz() const;
  CsrMatrix tocsr() const;
  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }

 private:
  ndarray data_, indices_, indptr_;
  int64_t rows_ = 0, cols_ = 0;
};

// ---- constructors ----
CsrMatrix eye(int64_t n);
CsrMatrix identity(int64_t n);
CsrMatrix diags(const std::vector<ndarray>& diagonals, const std::vector<int>& offsets, int64_t n);

// ---- product dispatch ----
enum class Backend { Cpu, Device };
Backend last_backend();
ndarray spmv(const CsrMatrix& A, const ndarray& x, Backend forced = Backend::Cpu);

// ---- sparse.linalg ----

// Preconditioner for the iterative solvers. Jacobi is the inverse diagonal;
// IC0 is a zero-fill incomplete Cholesky (SPD, use with cg); ILU0 is a zero-fill
// incomplete LU (general, use with gmres).
enum class Preconditioner { None, Jacobi, IC0, ILU0 };

// Result of a reporting iterative solve. `converged` is false when the solver hit
// `maxiter` without driving the relative residual below `tol` — callers MUST check
// it rather than trusting `x` blindly. `final_residual` is ‖b − A x‖ / ‖b‖.
struct IterationResult {
  ndarray x;
  int iterations = 0;
  double final_residual = 0.0;
  bool converged = false;
};

// Fill-reducing ordering for the sparse direct solver. RCM (reverse
// Cuthill-McKee) is the default; Natural disables reordering. Amd is reserved
// for a later PR and currently resolves to RCM.
enum class OrderingMethod { Natural, Rcm, Amd };

// Direct solve of A x = b. The no-ordering overload keeps a dense fallback for
// very small N and otherwise factors sparsely (SPD → Cholesky, else LU). The
// ordering overload always factors sparsely with the requested reordering.
ndarray spsolve(const CsrMatrix& A, const ndarray& b);
ndarray spsolve(const CsrMatrix& A, const ndarray& b, OrderingMethod ordering);

// Nonzeros in the sparse direct factor (L of the Cholesky/LU factorization) for
// the given ordering — a diagnostic for comparing fill across orderings.
int64_t factor_nnz(const CsrMatrix& A, OrderingMethod ordering = OrderingMethod::Rcm);

ndarray cg(const CsrMatrix& A, const ndarray& b, double tol = 1e-5, int maxiter = 1000);
ndarray gmres(const CsrMatrix& A, const ndarray& b, double tol = 1e-5, int maxiter = 1000);

// Reporting, preconditioner-aware variants. Unlike cg/gmres they never return a
// silently non-converged vector: inspect `IterationResult::converged`.
IterationResult cg_report(const CsrMatrix& A, const ndarray& b,
                          Preconditioner precond = Preconditioner::None,
                          double tol = 1e-5, int maxiter = 1000);
IterationResult gmres_report(const CsrMatrix& A, const ndarray& b,
                             Preconditioner precond = Preconditioner::None,
                             double tol = 1e-5, int maxiter = 1000);

double norm(const CsrMatrix& A, const std::string& ord = "fro");

// ---- sparse generalized symmetric eigensolver ----

// Result of eigsh. `eigenvalues` are the k requested eigenvalues in ascending
// order; `eigenvectors` is (n, k) with column i the mass-normalized eigenvector
// (xᵀ M x = 1) for eigenvalues[i]. `iterations` is the Lanczos subspace dimension
// built; `converged` is false when the k modes did not meet `tol` within the
// subspace cap — callers MUST check it, like IterationResult::converged.
struct EigshResult {
  ndarray eigenvalues;
  ndarray eigenvectors;
  int iterations = 0;
  bool converged = false;
};

// Lowest-k (or nearest-`sigma`) eigenpairs of the sparse generalized symmetric
// problem K x = λ M x, with K symmetric and M symmetric-positive-definite, via
// shift-invert Lanczos. The operator (K − σ M)⁻¹ is factored once (reusing the
// sparse direct factorization) and applied across the M-orthonormal Lanczos
// iterations; the small projected tridiagonal problem is solved densely and the
// eigenvalues are shifted back as λ = σ + 1/θ. `sigma` selects the target
// (0 for the lowest modes of an SPD pencil; a small negative shift moves off a
// rigid-body/near-zero mode). `maxiter <= 0` picks a default subspace cap.
EigshResult eigsh(const CsrMatrix& K, const CsrMatrix& M, int k,
                  double sigma = 0.0, double tol = 1e-8, int maxiter = 0);

// ---- sparse.csgraph ----
namespace csgraph {
ndarray dijkstra(const CsrMatrix& graph, bool directed = true);          // all-pairs distances
ndarray bellman_ford(const CsrMatrix& graph, bool directed = true);
ndarray floyd_warshall(const CsrMatrix& graph, bool directed = true);
struct ComponentsResult { int n_components; ndarray labels; };
ComponentsResult connected_components(const CsrMatrix& graph, bool directed = true,
                                      const std::string& connection = "weak");
CsrMatrix minimum_spanning_tree(const CsrMatrix& graph);

// Traversal: node_array is the visitation order; predecessors[i] is the parent
// of node i in the traversal tree (-9999 for the start node and unreachable
// nodes). Both arrays are returned as float64 (read with typed_data<double>()).
struct TraversalResult { ndarray node_array; ndarray predecessors; };
TraversalResult breadth_first_order(const CsrMatrix& graph, int64_t i_start,
                                    bool directed = true);
TraversalResult depth_first_order(const CsrMatrix& graph, int64_t i_start,
                                  bool directed = true);

// Johnson's all-pairs shortest paths (handles negative edge weights).
ndarray johnson(const CsrMatrix& graph, bool directed = true);

// Maximum flow on an integer-capacity directed graph. flow holds the per-edge
// flow on the original edges (float64 entries).
struct MaximumFlowResult { int64_t flow_value; CsrMatrix flow; };
MaximumFlowResult maximum_flow(const CsrMatrix& graph, int64_t source, int64_t sink);

// Maximum-cardinality bipartite matching. perm_type "row" returns a length-cols
// array whose j-th entry is the row matched to column j; "column" returns a
// length-rows array whose i-th entry is the column matched to row i. Unmatched
// vertices are -1. Returned as float64 (read with typed_data<double>()).
ndarray maximum_bipartite_matching(const CsrMatrix& graph,
                                   const std::string& perm_type = "row");
}  // namespace csgraph

}  // namespace scipp::sparse
