# Julia Package GoldfarbIdnaniSolver
| Status | Coverage |
| :----: | :----: |
| ![Build Status](https://github.com/fabienlefloch/GoldfarbIdnaniSolver.jl/actions/workflows/julia-runtests.yml/badge.svg) | [![codecov.io](http://codecov.io/github/fabienlefloch/GoldfarbIdnaniSolver.jl/coverage.svg?branch=main)](http://codecov.io/github/fabienlefloch/GoldfarbIdnaniSolver.jl?branch=main) |

Goldfarb and Idnani quadratic programming solver in native Julia.

This is a port of Berwin A. Turlach [quadprog](https://github.com/cran/quadprog) to the Julia language.

## Motivation

Julia has access to several high quality convex optimizer, which are able to solve quadratic programming problems, such as, for example, COSMO. But those tend to be not optimized for pure basic quadratic problems. It supports dense as well as sparse matrices.

This solver can be *100 times* faster on some problems (and without any specific code level optimization). Being native Julia, it also supports multiple precision big floats.

## Usage

The solveQPcompact function implements the dual method of Goldfarb and Idnani (1982, 1983) for solving quadratic programming problems of the form `min(-d^T b + 1/2 b^T D b)` with the constraints `A^T b >= b_0`.

- Create a sparse matrix with A, then transform it to the quadprog sparse format via

`amat, aind = convertSparse(A)`

- and solve your problem

 `sol, lagr, crval, iact, nact, iter = solveQPcompact(D, d, amat, aind, b_0)`

 The solution is in sol. It is possible through `factorized=true` to skip the factorization step for further speed-up. In this case, the matrix D should contain its inverse square root. The parameter `meq=0` allows to specify the number of equalities (as in quadprog R package).


 The solveQP function is the same algorithm for dense matrices:

- solve your problem using

 `sol, lagr, crval, iact, nact, iter = solveQP(D, d, A, b_0)`


 ## Example
A good example that shows the performance of the solver is to find the convex set closest to a set of financial option prices. This ensures the so-called arbitrage-free property of the option prices, see [An arbitrage-free interpolation of class C2 for option prices](https://arxiv.org/abs/2004.08650).

In terms of code, we fill up the constraints in the sparse matrix G, and associated vector h. The D matrix is simply a diagonal matrix of squared weights, here we input the inverse square root directly. And then we call the `GoldfarbIrdaniSolver`

 ```julia
using GoldfarbIdnaniSolver, SparseArrays, LinearAlgebra
function filterConvexCallPrices(
    strikes::Vector{T},
    callPrices::Vector{T}, #undiscounted!
    weights::Vector{T},
    forward::T;
    tol = 1e-8
)::Tuple{Vector{T},Vector{T},Vector{T}} where {T}
    n = length(callPrices)
    G = spzeros(T, 2 * n, n)
    h = zeros(T, 2 * n)
    for i = 2:n-1
        dym = (strikes[i] - strikes[i-1])
        dy = (strikes[i+1] - strikes[i])
        G[i, i-1] = -1 / dym
        G[i, i] = 1 / dym + 1 / dy
        G[i, i+1] = -1 / dy
    end
    G[1, 1] = 1 / (strikes[2] - strikes[1])
    G[1, 2] = -G[1, 1]
    G[n, n] = 1 / (strikes[n] - strikes[n-1])
    G[n, n-1] = -G[n, n]
    for i = 1:n
        h[i] = -tol
        G[n+i, i] = -1
        h[n+i] = -max(forward - strikes[i], 0) - tol
    end
    h[1] = 1 - tol
    strikesf = strikes
    #call solver
    amat, aind = convertSparse(copy(-G'))
    factorized = true
    dmat = diagm(1.0 ./ weights)
    dvec = @. callPrices * weights^2
    nEqualities = (strikes[1] == 0) ? 1 : 0
    pricesf, lagr, crval, iact, nact, iter = solveQPcompact(dmat, dvec, amat, aind, -h, meq=nEqualities, factorized = true)
    return strikesf, pricesf, weights
end
```

In terms of Black-Scholes implied volatility, the convex filtering is very visible, and the implied volatility look smoother (linear interpolation is used in the figure):
![Implied volatilities](/resources/images/tsla_convex_iv.png)


For the COSMO solver, the code below the comment would read
```julia
    W = spdiagm(weights)
    z = Variable(n)
    problem = minimize(square(norm(W * (z - callPrices))), G * z <= h)
    #solve!(problem, () -> SCS.Optimizer(verbose = 0))
    Convex.solve!(problem, () -> COSMO.Optimizer(verbose = false, eps_rel = 1e-8, eps_abs = 1e-8))
    pricesf = Convex.evaluate(z)
    return strikesf, pricesf, weights
```

We can benchmark the solvers as follows
```julia
using BenchmarkTools, StatsBase
strikes = Float64.([20, 25, 50, 55, 75, 100, 120, 125, 140, 150, 160, 175, 180, 195, 200, 210, 230, 240, 250, 255, 260, 270, 275, 280, 285, 290, 300, 310, 315, 320, 325, 330, 335, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 550, 580, 590, 600, 650, 670, 680, 690, 700])
prices = [337.9478782712897, 333.2264008151271, 310.49584376491333, 306.5789869909477, 288.3326474521462, 266.1436620149496, 249.44466994191146, 244.47045906634085, 232.02358238485036, 223.57092985061374, 215.47932505838847, 203.49665261662705, 199.45085022051444, 187.64870164975878, 183.73184487579312, 176.72338330960227, 161.59752782675685, 154.84695750485992, 147.27113520122242, 144.51478902657934, 141.11371474120148, 134.64682478802777, 131.65837649352014, 128.3346695814304, 125.60411253121669, 122.28040561912707, 115.9424612881003, 110.99712967626056, 108.21499437718802, 105.35549170482741, 102.75388027676055, 100.15226884869381, 97.60223566948576, 95.07799161470709, 90.23581650058495, 85.61989310557237, 80.97785070828226, 76.85159079958001, 72.72533089087777, 67.85118637372317, 64.96280443763155, 61.352327017517055, 58.025529966125816, 54.95662415902851, 51.810350978643, 49.07670378912775, 46.34305659961247, 43.11941604593882, 41.05628609158771, 38.735264892942695, 36.285298072150695, 34.8668962285343, 28.935397609774668, 24.370722585772782, 23.003898991015205, 21.27602765424615, 16.324515763803383, 14.545066178175517, 13.719814196435081, 13.049296961270962, 12.22404497953049]
weights = [0.18348804852853492, 0.14552754066163864, 0.06087460548064749, 0.05109612140267557, 0.03629649739986478, 0.025613847407177506, 0.020011216481149133, 0.019345036201403803, 0.01663883576809264, 0.015257762783202483, 0.014011545380885547, 0.012468184742920784, 0.012034729348791627, 0.010873040937928326, 0.010535157435308888, 0.009879102038500473, 0.008855540796164947, 0.00841084879707799, 0.008032374522333945, 0.007845822148042107, 0.00767850819280654, 0.00737017036267451, 0.007228761921893696, 0.007096391696183667, 0.00697143229125802, 0.006853569353455795, 0.006637272585167783, 0.0064509208672822085, 0.006365169495070308, 0.006284168069410054, 0.006209646991746033, 0.0061399098141406445, 0.0060750536990634395, 0.006014709985843373, 0.0059077251648834515, 0.0058176064495057645, 0.005741621638057057, 0.005682166059320373, 0.005635273659488148, 0.005598197720478721, 0.00557969512744412, 0.005570043788415912, 0.005571719928608071, 0.005583741283162423, 0.005606897345237971, 0.0056386186100743, 0.005680979257062336, 0.005742146253326264, 0.005798784362859748, 0.005870563857815834, 0.005958641403270033, 0.006029457615736634, 0.0063683433148160615, 0.0067726154457943945, 0.006927129393879053, 0.007136508419763993, 0.008060162361264996, 0.00852974277055327, 0.008785608144514201, 0.009023417426706495, 0.009332153986076881]
forward = 356.73063159822254
tte = 1.5917808219178082
strikesf, pricesf = filterConvexCallPrices(strikes, prices, weights, forward, tol = 1e-6)
rmse = rmsd(pricesf .* weights, prices .* weights)
@benchmark strikesf, pricesf = filterConvexCallPrices(strikes, prices, weights, forward, tol = 1e-6)
```

The benchmark results are:

|Solver|RMSE|Time (ms)|
|:---|---:|---:|
|GoldfarbIdnaniSolver | 0.0031130349998388157 |  0.205|
|COSMO                | 0.0031130309597602370 | 21.485|
|SCS                  | 0.0031130429769795193 |  8.381|
|ECOS                 | 0.0031131478052242090 |  2.430|
