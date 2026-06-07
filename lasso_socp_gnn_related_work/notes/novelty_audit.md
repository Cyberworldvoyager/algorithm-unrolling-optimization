# Novelty audit: LASSO+GNN and SOCP+GNN

Date: 2026-06-07

This audit is based on arXiv, OpenReview, Optimization Online, publisher/author pages, the PDFs already present in the workspace, and an additional targeted search for `SOCP + PDHG + GNN + unrolling + channel expansion + cone projection`. It is not a mathematical proof that no paper exists, but it is enough to set a defensible initial research claim.

## Short conclusion

Do not claim that "LASSO+GNN" is new. LISTA, HyperLISTA, graph unrolling networks, and graph unrolling sparse coding are close enough that the broad claim would be weak.

Do not claim that "SOCP+GNN" is new. Li, Liang, and Chen already have SOCP-GNN expressivity/universality work, including a 2026 conference version.

The stronger claim is narrower:

> I found SOCP-GNN representation theory and PDHG-based conic solvers, but no exact prior work on a PDHG-unrolled, cone-aware, dimension-agnostic GNN optimizer for SOCPs that learns primal-dual update parameters and warm-starts a classical SOCP first-order method.

For LASSO, the safer role is:

> LASSO-GNN is a smaller testbed for dimension-agnostic graph-parametric proximal optimization, not the main novelty by itself.

## Closest works

| Direction | Closest paper | What it already does | Why it does not fully cover our target |
|---|---|---|---|
| LASSO unrolling | Gregor and LeCun, 2010 | Unrolls sparse coding/LASSO-style inference into LISTA | Fixed dictionary setting, not a general graph-parametric solver for variable-size instances |
| LASSO theory | Chen et al., 2018; Chen et al., 2021 | Gives convergence theory and adaptive parameter tuning for unfolded ISTA | Still not a GNN over arbitrary problem graphs |
| Graph plus unrolling | Chen, Eldar, and Zhao, 2021 | Graph unrolling networks, including graph unrolling sparse coding | Graph-signal restoration, not generic LASSO instance optimization |
| LP plus GNN | Chen et al., 2022 | Proves GNNs can represent LP properties | Linear constraints only |
| LP plus PDHG unrolling | Li et al., 2024 | PDHG-Net with channel expansion for LPs | No cone geometry |
| QP plus GNN/unrolling | Chen et al., 2024; Yang et al., 2024 | QP expressivity and PDQP-Net | Quadratic programs, not SOCPs |
| QCQP plus GNN | Wu et al., 2024 | GNN representation for convex QCQPs | Representation theory, not PDHG unrolling |
| SOCP plus GNN | Li, Liang, and Chen, 2025/2026 | SOCP graph representation, universality, sample complexity | Does not appear to be an unrolled PDHG solver |
| SOCP plus PDHG | Lin, Xiong, Ge, and Ye, 2026 | PDHG-based matrix-free conic solver with GPU-enhanced cone projections | Classical solver, not a learned GNN/channel-expanded unrolled architecture |
| General constrained unrolled GNN | Hadou and Ribeiro, 2025 | Unrolls dual ascent into coupled GNNs | General framework, not cone-specific PDHG for SOCP |
| SDP GNN limits | Qian and Morris, 2026 | Shows architectural expressivity matters for SDP | Motivates careful SOCP graph architecture |

## Recommended project title framing

Strong:

- Dimension-Agnostic Primal-Dual Graph Unrolling for Second-Order Cone Programs
- Cone-Aware PDHG Networks for Scalable SOCP Solving
- Learning to Warm-Start SOCP First-Order Solvers with Unrolled Graph Neural Networks

Risky:

- LASSO+GNN
- SOCP+GNN
- A GNN Solver for SOCP

The risky titles are too broad because closely related papers already exist.

## Suggested contribution bullets

1. A cone-aware graph representation for SOCP primal-dual iterates that keeps variable nodes, linear constraint nodes, and cone component nodes explicit.
2. A PDHG-unrolled architecture whose layers perform learned primal-dual updates with SOCP cone projections.
3. A dimension-agnostic parameterization using shared message functions or channel expansion, allowing training on small SOCPs and testing on larger ones.
4. An unsupervised primal-dual/KKT residual loss, so labels from high-accuracy solvers are optional.
5. A two-stage inference scheme: neural warm start followed by standard PDHG or another first-order conic solver.

## Main risk

The 2026 SOCP-GNN paper is very close at the topic level, and the 2026 PDCS/cuPDCS paper is close at the solver level. Our paper must clearly distinguish algorithm-unrolled solving from WL-based supervised representation/prediction, and distinguish learned graph/channel parameterization from classical hand-designed conic PDHG. The method section should emphasize the PDHG update equations, cone projection operator, learned step sizes/sampling probabilities/channel mixing, and solver warm-start objective.
