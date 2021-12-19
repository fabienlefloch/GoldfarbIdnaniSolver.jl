using GoldfarbIdnaniSolver, Test
using LinearAlgebra, SparseArrays, StatsBase

@testset "Qpgen1" begin
    dmat = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    dvec = [0.0, 5.0, 0.0]
    Amat = [-4.0 2.0 -2.0; -3.0 1.0 1.0]
    Aind = [2 2 2; 1 1 2; 2 2 3]
    bvec = [-8.0, 2.0, 0.0]
    sol, lagr, crval, iact, nact, iter = solveQPcompact(dmat, dvec, Amat, Aind, bvec, factorized = false)
    println(sol, " ", lagr, " ", crval, " ", iact, " ", nact, " ", iter)
    solRef = [0.4761905, 1.0476190, 2.0952381]
    for (a, b) in zip(solRef, sol)
        @test isapprox(a, b, atol = 1e-6)
    end
    @test isapprox(-2.380952, crval, atol = 1e-6)
    
end

@testset "Gwen" begin
    Dmat = [4.000 -1.224000 0.000 0.000000 0.000 0.000000 0 0e+00 0e+00
        -1.224 0.417898 0.000 0.000000 0.000 0.000000 0 0e+00 0e+00
        0.000 0.000000 49.000 -0.156000 0.000 0.000000 0 0e+00 0e+00
        0.000 0.000000 -0.156 0.109192 0.000 0.000000 0 0e+00 0e+00
        0.000 0.000000 0.000 0.000000 2.000 0.518000 0 0e+00 0e+00
        0.000 0.000000 0.000 0.000000 0.518 0.137524 0 0e+00 0e+00
        0.000 0.000000 0.000 0.000000 0.000 0.000000 1 0e+00 0e+00
        0.000 0.000000 0.000 0.000000 0.000 0.000000 0 1e-08 0e+00
        0.000 0.000000 0.000 0.000000 0.000 0.000000 0 0e+00 1e-08]
    dvec = [0.53741123, 0.40486447, 25.93321349, -0.05075975, 0.90951388, 0.25524170, 0.48459807, 0.00000000, 0.00000000]
    Amat = Float64.([1.000 0.0000 0.000 0 0 0 0 0 0 0 0 0
        -0.163 0.0000 0.000 1 0 0 -1 1 0 1 0 0
        -1.000 1.0000 0.000 0 0 0 0 0 0 0 0 0
        0.163 0.2045 0.000 0 1 0 -1 -1 1 -1 1 0
        0.000 -1.0000 1.000 0 0 0 0 0 0 0 0 0
        0.000 -0.2045 0.511 0 0 1 -1 0 -1 0 -1 0
        0.000 0.0000 -1.000 0 0 0 0 0 0 0 0 0
        0.000 0.0000 0.000 0 0 0 0 1 0 -1 0 -1
        0.000 0.0000 0.000 0 0 0 0 0 1 0 -1 -1])
    bvec = [0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, -1e+01, 0e+00, 0e+00, 0e+00, 0e+00, -1e-04]
    amat, aind = convertSparse(sparse(Amat))
    sol, lagr, crval, iact, nact, iter = solveQPcompact(copy(Dmat), copy(dvec), amat, aind, copy(bvec), meq=3)
    println(sol, " ", crval, " ", iter)
    @test iter[1] == 7
    @test isapprox(22.204332981569845, crval, atol = 1e-6)

    sol, lagr, crval, iact, nact, iter = solveQP(Dmat, dvec, Amat, bvec, meq=3)
    println(sol, " ", crval, " ", iter)
    @test iter[1] == 7
    @test isapprox(22.204332981569845, crval, atol = 1e-6)

end



@testset "test1" begin
    Dmat = [1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0]
    dvec = [0.0, 5.0, 0.0]
    Aind = [2 2 2
        1 1 2
        2 2 3]
    Amat = Float64.([-4 2 -2
        -3 1 1])
    bvec = [-8.0, 2.0, 0.0]

    sol, lagr, crval, iact, nact, iter = solveQPcompact(Dmat, dvec, Amat, Aind, bvec)
    println(sol, " ", crval, " ", iter)
    @test iter[1] == 3

    Dmat = Float64.([4 -2
        -2 4])
    Amat = Float64.([1 1 1
        1 1 1])
    Aind = [1 1 2
        1 2 1
        0 0 2]
    bvec = [0.0, 0.0, 2.0]
    dvec = [-6.0, 0.0]
    sol, lagr, crval, iact, nact, iter = solveQPcompact(Dmat, dvec, Amat, Aind, bvec)
    println(sol, " ", crval, " ", iter)
    @test iter[1] == 2

end
include("smallrandom.jl")
include("convexcallslarge.jl")
include("convexcalls.jl")