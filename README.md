# Julia Package GISolver
| Status | Coverage |
| :----: | :----: |
| ![Build Status](https://github.com/fabienlefloch/GISolver.jl/actions/workflows/julia-runtests.yml/badge.svg) | [![codecov.io](http://codecov.io/github/fabienlefloch/GISolver.jl/coverage.svg?branch=main)](http://codecov.io/github/fabienlefloch/GISolver.jl?branch=main) |

Goldfarb and Idnani quadratic programming solver in native Julia.

This is a port of Berwin A. Turlach [quadprog](https://github.com/cran/quadprog) to the Julia language.

## Motivation

Julia has access to several high quality convex optimizer, which are able to solve quadratic programming problems, such as, for example, COSMO. But those tend to be not optimized for pure basic quadratic problems.

This solver can be *100 times* faster on some problems (and without any specific code level optimization).

## Usage

The solveQPcompact function implements the dual method of Goldfarb and Idnani (1982, 1983) for solving quadratic programming problems of the form `min(-d^T b + 1/2 b^T D b)` with the constraints `A^T b >= b_0`.

- Create a sparse matrix with A, then transform it to the quadprog sparse format via

`aind, amat = convertSparse(A)`

- and solve your problem

 `sol, lagr, crval, iact, nact, iter = solveQPcompact(D, d, amat, aind, b, 0, false)`

 The solution is in sol. It is possible through `factorized=true` to skip the factorization step for further speed-up. In this case, the matrix D should contain its inverse square root.