# sparse Specification

## Purpose
TBD - created by archiving change add-sparse. Update Purpose after archive.
## Requirements
### Requirement: Sparse matrix formats and conversion

`scipp::sparse` SHALL provide `CooMatrix`, `CsrMatrix` and `CscMatrix` with
construction from triplets and from dense, conversion between formats, `toarray`,
`nnz`, `transpose` and `diagonal`, matching SciPy. (oracle: scipy/sparse)

#### Scenario: Format round-trip
- GIVEN a dense matrix `A`
- WHEN it is converted to CSR/CSC/COO and back with `toarray`
- THEN the result equals `A`, `nnz` counts the nonzeros, and the canonical CSR
  matches SciPy's `csr_array(A)`

#### Scenario: Transpose and diagonal
- GIVEN a sparse matrix
- WHEN `transpose` and `diagonal` are computed
- THEN they match SciPy's

### Requirement: Sparse constructors and arithmetic

`scipp::sparse` SHALL provide `eye`, `identity` and `diags`, sparse+sparse
addition, and scalar multiplication, matching SciPy. (oracle: scipy/sparse/_construct.py)

#### Scenario: Constructors match SciPy
- WHEN `eye(n)` and `diags(diagonals, offsets)` are built and densified
- THEN they equal SciPy's

#### Scenario: Sparse addition
- GIVEN two sparse matrices of the same shape
- WHEN they are added
- THEN the dense result equals the dense sum and matches SciPy

### Requirement: Sparse matrix products with backend dispatch

`scipp::sparse` SHALL provide `spmv` (CSR · dense vector) and `spmm` (CSR · dense
matrix). `spmv` SHALL select an implementation via NumPP's capability registry —
a portable CPU kernel is always present and a device kernel is used when available
and the problem is large enough — with the chosen backend observable and the
device result equal to the CPU result within tolerance. (oracle: scipy/sparse)

#### Scenario: SpMV/SpMM match the dense product
- GIVEN a sparse matrix and a dense vector/matrix
- WHEN `spmv`/`spmm` are computed
- THEN the results equal the dense `A @ x` and match SciPy

#### Scenario: Backend dispatch and equivalence
- GIVEN an SpMV run once on the CPU path and once on the device-reference path
- WHEN the selected backend is queried
- THEN the two results are equal within tolerance and the backend is reported,
  with the CPU kernel always available as fallback

### Requirement: Sparse linear solvers

`scipp::sparse` SHALL provide `spsolve` (direct), `cg` and `gmres` (matrix-free
iterative), and `norm`, matching SciPy within documented tolerance. (oracle: scipy/sparse/linalg)

#### Scenario: Direct and iterative solves
- GIVEN a sparse system `A x = b`
- WHEN `spsolve(A, b)`, `cg(A, b)` (SPD `A`) and `gmres(A, b)` are computed
- THEN each returns `x` with `A @ x` ≈ `b`, matching SciPy

### Requirement: Compressed-sparse graph algorithms

`scipp::sparse::csgraph` SHALL provide `dijkstra`, `bellman_ford`,
`floyd_warshall`, `connected_components` and `minimum_spanning_tree` over a CSR
graph, matching SciPy. (oracle: scipy/sparse/csgraph)

#### Scenario: Shortest paths
- GIVEN a weighted graph as a CSR matrix
- WHEN `dijkstra` (and `bellman_ford`, `floyd_warshall`) compute shortest paths
- THEN the distance matrices match SciPy's

#### Scenario: Components and spanning tree
- GIVEN a graph
- WHEN `connected_components` and `minimum_spanning_tree` are computed
- THEN the component labeling matches SciPy (up to relabeling) and the MST total
  weight matches SciPy's

### Requirement: csgraph traversal and flow algorithms
The system SHALL provide `breadth_first_order`, `depth_first_order`, `johnson`,
`maximum_flow`, and `maximum_bipartite_matching` in `scipp::sparse::csgraph`,
operating on CSR graphs and matching `scipy.sparse.csgraph` to integer exactness
for orders/matchings/flow values and to `allclose` tolerance (~1e-12) for
distances.

#### Scenario: Breadth- and depth-first order
- GIVEN a directed CSR graph and a start node
- WHEN `breadth_first_order` (resp. `depth_first_order`) is called
- THEN the returned node array SHALL list the reachable nodes in the same order
  as SciPy (neighbours visited in CSR column order), and the predecessor array
  SHALL give each node's parent in the traversal tree with `-9999` for the start
  node and unreachable nodes.

#### Scenario: Undirected traversal symmetrisation
- GIVEN an asymmetric CSR graph traversed with `directed=false`
- WHEN `breadth_first_order` or `depth_first_order` is called
- THEN each node's neighbours SHALL be its out-neighbours (CSR order) followed by
  its in-neighbours (ascending source order), reproducing SciPy's symmetrised
  traversal order and predecessors.

#### Scenario: Johnson all-pairs shortest paths with negative edges
- GIVEN a directed CSR graph containing negative edge weights but no negative
  cycle
- WHEN `johnson` is called
- THEN the returned distance matrix SHALL equal `scipy.sparse.csgraph.johnson`
  to `allclose` tolerance, with `inf` for unreachable pairs.

#### Scenario: Maximum flow value
- GIVEN a directed CSR graph with integer capacities, a source and a sink
- WHEN `maximum_flow` is called
- THEN the returned `flow_value` SHALL equal SciPy's `.flow_value` exactly, and
  the net flow leaving the source in the returned flow matrix SHALL equal that
  value.

#### Scenario: Maximum bipartite matching
- GIVEN a CSR biadjacency matrix
- WHEN `maximum_bipartite_matching` is called with `perm_type="row"` (length
  cols) or `perm_type="column"` (length rows)
- THEN the returned matching array SHALL equal SciPy's, with `-1` for unmatched
  vertices, for graphs whose maximum matching is unique.

### Requirement: Device-accelerated CSR sparse matrix-vector product
`scipp::sparse` SHALL compute CSR `spmv` and `spmm` through NumPP's `csr_spmv`
device kernel, auto-selecting a GPU backend above the size threshold, always
falling back to the portable CPU path, with the selection observable via
`last_backend()`. (oracle: scipy.sparse CSR matvec)

#### Scenario: SpMV is backend-independent and matches SciPy
- GIVEN a CSR matrix and a dense vector
- WHEN `spmv` runs (CPU, and a device backend when one is present)
- THEN every backend's result is `allclose` to the other and to `scipy.sparse`
- AND `last_backend()` reports the backend actually used

### Requirement: Sparse generalized symmetric eigensolver

`scipp::sparse` SHALL provide `eigsh(K, M, k, sigma, tol, maxiter)`, a
shift-invert Lanczos solver for the generalized symmetric eigenproblem
`K x = λ M x` with `K` symmetric and `M` symmetric-positive-definite. It SHALL
return the `k` eigenpairs whose eigenvalues are nearest `sigma` (the lowest `k`
modes for an SPD pencil at `sigma = 0`), as an `EigshResult` carrying
eigenvalues in **ascending** order, **mass-normalized** eigenvectors
(`xᵀ M x = 1`, one per column), the total Lanczos steps performed, and a
`converged` flag. The shift-invert operator `(K − σ M)⁻¹` SHALL be produced by a
single sparse direct factorization reused across the Lanczos iterations (it
SHALL NOT refactor per iteration and SHALL NOT densify `K` or `M`).

The solver SHALL be robust to operator scale and to clustered spectra: it SHALL
declare an invariant subspace only **relative** to the operator scale (never on
an absolute `beta` threshold), and SHALL **restart** (thick/implicit restart,
deflating converged Ritz pairs) rather than stop when the wanted modes have not
converged within one Krylov cycle. `maxiter` bounds the number of restart cycles
(`≤ 0` selects a default). Convergence SHALL be gated on the true generalized
residual `‖K x − λ M x‖ / ‖K x‖`. Results SHALL match SciPy's
`scipy.sparse.linalg.eigsh` within documented tolerance. (oracle:
scipy/sparse/linalg)

#### Scenario: Lowest modes of an SPD pencil
- GIVEN a sparse SPD pencil `K x = λ M x` above the small-N threshold
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN it returns `k` ascending eigenvalues matching SciPy, with each
  eigenvector satisfying `‖K x − λ M x‖ / ‖K x‖ < tol` and `xᵀ M x ≈ 1`, and
  `converged = true`

#### Scenario: Mass-orthonormal eigenvectors
- GIVEN the eigenpairs returned by `eigsh`
- WHEN two distinct modes `x_i`, `x_j` are compared
- THEN `x_iᵀ M x_j ≈ 0` and `x_iᵀ M x_i ≈ 1` (M-orthonormal, mass-normalized)

#### Scenario: Stiff pencil with large-magnitude eigenvalues
- GIVEN an SPD pencil whose eigenvalues are large (e.g. `λ ~ 1e10`), so the
  shift-invert operator's eigenvalues (and the Lanczos `beta`) are `~1e-11`
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN the small `beta` is NOT mistaken for an invariant subspace, and the `k`
  modes converge to `‖K x − λ M x‖ / ‖K x‖ < tol` matching SciPy

#### Scenario: Clustered spectrum converges via restart
- GIVEN an SPD pencil with tightly-clustered / near-degenerate low modes that a
  single Krylov cycle cannot resolve to `tol`
- WHEN `eigsh(K, M, k, sigma = 0)` is computed
- THEN it restarts across cycles and converges the `k` modes to `tol`, matching
  SciPy, with `converged = true`

#### Scenario: Non-convergence is signaled, not hidden
- GIVEN a restart-cycle budget too small to resolve the `k` requested modes to
  `tol`
- WHEN `eigsh` exhausts `maxiter` cycles without meeting `tol`
- THEN the result reports `converged = false` (with the Lanczos steps
  performed), rather than returning silently-wrong eigenpairs
</content>

