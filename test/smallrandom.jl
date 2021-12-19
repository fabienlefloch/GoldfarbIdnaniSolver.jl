@testset "Tyner" begin
    n = 66
    X = fill(1e-20, (n,n))
    X[diagind(X)] .= 1.0
    Dmat = X*X
    y = collect(1:n)
    dvec = X*y
    Amat = diagm(ones(n))
    bvec = y + rand(n)
    amat, aind = convertSparse(sparse(Amat))
    sol, lagr, crval, iact, nact, iter = solveQPcompact(copy(Dmat), copy(dvec), copy(amat), copy(aind), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    amat, aind = convertSparse(sparse(Amat))
    sol, lagr, crval, iact, nact, iter = solveQPcompact(copy(Dmat), copy(dvec), copy(amat), copy(aind), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    amat, aind = convertSparse(sparse(Amat))
    sol, lagr, crval, iact, nact, iter = solveQPcompact(copy(Dmat), copy(dvec), copy(amat), copy(aind), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    amat, aind = convertSparse(sparse(Amat))
    sol, lagr, crval, iact, nact, iter = solveQPcompact(copy(Dmat), copy(dvec), copy(amat), copy(aind), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    amat, aind = convertSparse(sparse(Amat))
    sol, lagr, crval, iact, nact, iter = solveQPcompact(copy(Dmat), copy(dvec), copy(amat), copy(aind), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
end

@testset "TynerDense" begin
    n = 66
    X = fill(1e-20, (n,n))
    X[diagind(X)] .= 1.0
    Dmat = X*X
    y = collect(1:n)
    dvec = X*y
    Amat = diagm(ones(n))
    bvec = y + rand(n)
    sol, lagr, crval, iact, nact, iter = solveQP(copy(Dmat), copy(dvec), copy(Amat), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    sol, lagr, crval, iact, nact, iter = solveQP(copy(Dmat), copy(dvec), copy(Amat), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    sol, lagr, crval, iact, nact, iter = solveQP(copy(Dmat), copy(dvec), copy(Amat), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    sol, lagr, crval, iact, nact, iter = solveQP(copy(Dmat), copy(dvec), copy(Amat), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
    bvec = y + rand(n)
    sol, lagr, crval, iact, nact, iter = solveQP(copy(Dmat), copy(dvec), copy(Amat), copy(bvec), meq=n)
    println(sol, " ", crval, " ", iter)
    @test length(findall( x -> isnan(x),  sol)) == 0
end