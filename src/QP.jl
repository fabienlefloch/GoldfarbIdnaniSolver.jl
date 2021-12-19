using LinearAlgebra
export solveQP

#  This routine implements the dual method of Goldfarb and Idnani (1982, 1983) for solving quadratic programming problems of the form
# \eqn{\min(-d^T b + 1/2 b^T D b)}{min(-d^T b + 1/2 b^T D b)} with the
# constraints \eqn{A^T b >= b_0}.
function solveQP(dmat::AbstractMatrix{T}, dvec::AbstractArray{T},
    Amat::AbstractMatrix{T}, bvec::AbstractArray{T};
    meq::Int = 0, factorized::Bool = false)::Tuple{AbstractArray{T},AbstractArray{T},T,AbstractArray{Int},Int,AbstractArray{Int}} where {T} #sol, lagr, crval, iact, nact, iter
    n = size(dmat, 1)
    q = 0
    if size(Amat, 1) > 0
        q = size(Amat, 2)
    end

    anrow = size(Amat, 1)
    if n != size(dmat, 2)
        throw(error("Dmat is not symmetric!"))
    end
    if n != length(dvec)
        throw(error("Dmat and dvec are incompatible!"))
    end
    if (n != anrow)
        throw(error("Incorrect number of rows. Amat amd dvec are incompatible!"))
    end
    if (q != length(bvec))
        throw(error("Incorrect number of columns. Amat, bvec incompatible!"))
    end
    if (meq > q) || (meq < 0)
        throw(error("Value of meq is invalid!"))
    end
    r = min(n, q)
    work = zeros(T, 2 * n + trunc(Int, r * (r + 5) / 2) + 2 * q + 1)
    sol, lagr, crval, iact, nact, iter, ierr = qpgen2(dmat, dvec, n, n, Amat, bvec, anrow, q, meq, factorized, work)
    if ierr == 1
        throw(error("constraints are inconsistent, no solution!"))
    elseif ierr == 2
        throw(error("matrix D in quadratic function is not positive definite!"))
    end
    return sol, lagr, crval, iact, nact, iter
end

#  Julia Port
#  Copyright (C) 2021 Fabien Le Floc'h <fabien@2ipi.com>
#  Original Fortran Code
#  Copyright (C) 1995-2010 Berwin A. Turlach <Berwin.Turlach@gmail.com>
# 
# 
#   this routine uses the Goldfarb/Idnani algorithm to solve the
#   following minimization problem:
# 
#         minimize  -d^T x + 1/2 *  x^T D x
#         where   A1^T x  = b1
#                 A2^T x >= b2
# 
#   the matrix D is assumed to be positive definite.  Especially,
#   w.l.o.g. D is assumed to be symmetric.
#   
#   Input parameter:
#   dmat   nxn matrix, the matrix D from above (dp)
#          *** WILL BE DESTROYED ON EXIT ***
#          The user has two possibilities:
#          a) Give D (ierr=0), in this case we use routines from LINPACK
#             to decompose D.
#          b) To get the algorithm started we need R^-1, where D=R^TR.
#             So if it is cheaper to calculate R^-1 in another way (D may
#             be a band matrix) then with the general routine, the user
#             may pass R^{-1}.  Indicated by ierr not equal to zero.
#   dvec   nx1 vector, the vector d from above (dp)
#          *** WILL BE DESTROYED ON EXIT ***
#          contains on exit the solution to the initial, i.e.,
#          unconstrained problem
#   fddmat scalar, the leading dimension of the matrix dmat
#   n      the dimension of dmat and dvec (int)
#   amat   nxq matrix, the matrix A from above (dp) [ A=(A1 A2)^T ]
#          *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE
#              CHANGED SIGNES ON EXIT ***
#   bvec   qx1 vector, the vector of constants b in the constraints (dp)
#          [ b = (b1^T b2^T)^T ]
#          *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE
#              CHANGED SIGNES ON EXIT ***
#   fdamat the first dimension of amat as declared in the calling program. 
#          fdamat >= n !!
#   q      integer, the number of constraints.
#   meq    integer, the number of equality constraints, 0 <= meq <= q.
#   ierr   integer, code for the status of the matrix D:
#             ierr =  0, we have to decompose D
#             ierr != 0, D is already decomposed into D=R^TR and we were
#                        given R^{-1}.
# 
#   Output parameter:
#   sol   nx1 the final solution (x in the notation above)
#   lagr  qx1 the final Lagrange multipliers
#   crval scalar, the value of the criterion at the minimum      
#   iact  qx1 vector, the constraints which are active in the final
#         fit (int)
#   nact  scalar, the number of constraints active in the final fit (int)
#   iter  2x1 vector, first component gives the number of "main" 
#         iterations, the second one says how many constraints were
#         deleted after they became active
#   ierr  integer, error code on exit, if
#            ierr = 0, no problems
#            ierr = 1, the minimization problem has no solution
#            ierr = 2, problems with decomposing D, in this case sol
#                      contains garbage!!
# 
#   Working space:
#   work  vector with length at least 2*n+r*(r+5)/2 + 2*q +1
#         where r=min(n,q)
# 
function qpgen2(dmat::AbstractMatrix{T}, dvec::AbstractArray{T}, fddmat::Int, n::Int, amat::AbstractMatrix{T},
   bvec::AbstractArray{T}, fdamat::Int, q::Int, meq::Int, factorized::Bool, work::AbstractArray{T})::Tuple{AbstractArray{T},AbstractArray{T},T,AbstractArray{Int},Int,AbstractArray{Int},Int} where {T} # sol, lagr, crval, iact, nact, iter, ierr

   sol = zeros(T, n)
   lagr = zeros(T, q)
   iact = zeros(Int, q)
   iter = zeros(Int, 2)
   local t1inf::Bool, t2min::Bool
    local it1::Int, iwzv::Int, iwrv::Int, iwrm::Int, iwsv::Int, iwuv::Int, nvl::Int, iwnbv::Int, l1::Int
    local temp::T, sum::T, t1::T, tt::T, gc::T, gs::T, nu::T, vsmall::T, tmpa::T, tmpb::T
    r = min(n, q)
    l = 2 * n + trunc(Int, (r * (r + 5)) / 2) + 2 * q + 1
    crval = zero(T)
    nact = 0
    vsmall = 2 * eps(T) #we deviate here from original Fortran code based on Powell guess

      #
    # store the initial dvec to calculate below the unconstrained minima of
    # the critical value.
    #
    work[1:n] = dvec[1:n]
    for i = n+1:l
        work[i] = zero(T)
    end
    for i = 1:q
        iact[i] = 0
        lagr[i] = zero(T)
    end
      #
    # get the initial solution
    #
    if !factorized
        info = dpofa(dmat, fddmat, n)
        if info != 0
            ierr = 2
            return (sol, lagr, crval, iact, nact, iter, ierr)
        end
        dposl(dmat, fddmat, n, dvec)
        dpori(dmat, fddmat, n)
    else
               #
        # Matrix D is already factorized, so we have to multiply d first with
        # R^-T and then with R^-1.  R^-1 is stored in the upper half of the
        # array dmat.
        #
        for j = 1:n
            sol[j] = zero(T)
            for i = 1:j
                sol[j] += dmat[i, j] * dvec[i]
            end
        end
        for j = 1:n
            dvec[j] = zero(T)
            for i = j:n
                dvec[j] += dmat[j, i] * sol[i]
            end
        end
    end
     #
    # set lower triangular of dmat to zero, store dvec in sol and
    # calculate value of the criterion at unconstrained minima
    #
    crval = zero(T)
    for j = 1:n
        sol[j] = dvec[j]
        crval += work[j] * sol[j]
        work[j] = zero(T)
        for i = j+1:n
            dmat[i, j] = zero(T)
        end
    end
    crval = -crval / 2
    ierr = 0
       #
    # calculate some constants, i.e., from which index on the different
    # quantities are stored in the work matrix
    #
    iwzv = n
    iwrv = iwzv + n
    iwuv = iwrv + r
    iwrm = iwuv + r + 1
    iwsv = iwrm + trunc(Int, (r * (r + 1)) / 2)
    iwnbv = iwsv + q
    #
    # calculate the norm of each column of the A matrix
    #
    for i = 1:q
        sum = zero(T)
        for j = 1:n
            sum += amat[j, i]^2
        end
        work[iwnbv+i] = sqrt(sum)
        if isnan(sum)
            throw(DomainError(sum))
        end
    end
    iter[1] = 0
    iter[2] = 0
    #
    # start a new iteration
    #
    @label L50
    iter[1] += 1
  #
    # calculate all constraints and check which are still violated
    # for the equality constraints we have to check whether the normal
    # vector has to be negated (as well as bvec in that case)
    #
    l = iwsv
    for i = 1:q
        l += 1
        sum = -bvec[i]
        for j = 1:n
            sum += amat[j, i] * sol[j]
        end
        if abs(sum) < vsmall
            sum = zero(T)
        end
        if i > meq
            work[l] = sum
        else
            work[l] = -abs(sum)
            if sum > zero(T)
                for j = 1:n
                    amat[j, i] = -amat[j, i]
                end
                bvec[i] = -bvec[i]
            end
        end
    end
    #
    # as safeguard against rounding errors set already active constraints
    # explicitly to zero
    #
    for i = 1:nact
        work[iwsv+iact[i]] = zero(T)
    end
      #
    # we weight each violation by the number of non-zero elements in the
    # corresponding row of A. then we choose the violated constraint which
    # has maximal absolute value, i.e., the minimum.
    # by obvious commenting and uncommenting we can choose the strategy to
    # take always the first constraint which is violated. ;-)
    #
    nvl = 0
    temp = zero(T)
    for i = 1:q
        if work[iwsv+i] < temp * work[iwnbv+i]
            nvl = i
            temp = work[iwsv+i] / work[iwnbv+i]
            if work[iwsv+i] == zero(T) #fortran 0 behavior
                temp = zero(T)
            end
        end
    end
    if nvl == 0
        for i = 1:nact
            lagr[iact[i]] = work[iwuv+i]
        end
        return (sol, lagr, crval, iact, nact, iter, ierr)
    end
        #
    # calculate d=J^Tn^+ where n^+ is the normal vector of the violated
    # constraint. J is stored in dmat in this implementation!!
    # if we drop a constraint, we have to jump back here.
    #
    @label L55
    for i = 1:n
        sum = zero(T)
        for j = 1:n
            sum += dmat[j, i] * amat[j, nvl]
        end
        if isnan(sum)
            throw(DomainError(sum))
        end
        work[i] = sum
    end
    #
    # Now calculate z = J_2 d_2
    #
    l1 = iwzv
    for i = 1:n
        work[l1+i] = zero(T)
    end
    for j = nact+1:n
        for i = 1:n
            work[l1+i] += dmat[i, j] * work[j]
        end
    end
  #
    # and r = R^{-1} d_1, check also if r has positive elements (among the
    # entries corresponding to inequalities constraints).
    #
    t1inf = true
    for i = nact:-1:1
        sum = work[i]
        l = iwrm + trunc(Int, (i * (i + 3)) / 2)
        l1 = l - i
        for j = i+1:nact
            sum -= work[l] * work[iwrv+j]
            l += j
        end
        if sum != zero(T)
            sum = sum / work[l1]
        end
        if isnan(sum)
            throw(DomainError(sum))
        end
        work[iwrv+i] = sum
        if iact[i] <= meq
            continue
        end
        if sum <= zero(T)
            continue
        end
        t1inf = false
        it1 = i
    end
  #
    # if r has positive elements, find the partial step length t1, which is
    # the maximum step in dual space without violating dual feasibility.
    # it1  stores in which component t1, the min of u/r, occurs.
    #
    if !t1inf
        t1 = work[iwuv+it1] / work[iwrv+it1]
        if work[iwuv+it1] == zero(T) #fortran 0
            t1 = 0
        end
        for i = 1:nact
            if iact[i] <= meq
                continue
            end
            if work[iwrv+i] <= zero(T)
                continue
            end
            temp = work[iwuv+i] / work[iwrv+i]
            if work[iwuv+i] == zero(T) #fortran 0 behavior
                temp = zero(T)
            end
            if temp < t1
                t1 = temp
                it1 = i
            end
        end
    end
    #
   #
    # test if the z vector is equal to zero
    #
    sum = zero(T)
    for i = iwzv+1:iwzv+n
        sum += work[i]^2
    end
    #  println(iter, " ",sum)
    if sum <= vsmall
        #
        # No step in primal space such that the new constraint becomes
        # feasible. Take step in dual space and drop a constant.
        #
        if t1inf
            #
            # No step in dual space possible either, problem is not solvable
            #
            ierr = 1
            return (sol, lagr, crval, iact, nact, iter, ierr)
        else
            # we take a partial step in dual space and drop constraint it1,
            # that is, we drop the it1-th active constraint.
            # then we continue at step 2(a) (marked by label 55)
            #
            for i = 1:nact
                work[iwuv+i] -= t1 * work[iwrv+i]
            end
            work[iwuv+nact+1] += t1
            @goto L700
        end
    else
        #
        # compute full step length t2, minimum step in primal space such that
        # the constraint becomes feasible.
        # keep sum (which is z^Tn^+) to update crval below!
        #
        sum = zero(T)
        for i = 1:n
            sum += work[iwzv+i] * amat[i, nvl]
        end
        tt = -work[iwsv+nvl] / sum
        if work[iwsv+nvl] == zero(T) #fortran 0 behavior
            tt = zero(T)
        end
        t2min = true
        if !t1inf
            if t1 < tt
                tt = t1
                t2min = false
            end
        end
        #
        # take step in primal and dual space
        #
        for i = 1:n
            sol[i] += tt * work[iwzv+i]
        end
        crval += tt * sum * (tt / 2 + work[iwuv+nact+1])
        for i = 1:nact
            work[iwuv+i] -= tt * work[iwrv+i]
        end
        work[iwuv+nact+1] += tt
        #
        # if it was a full step, then we check wheter further constraints are
        # violated otherwise we can drop the current constraint and iterate once
        # more
        if t2min
            #
            # we took a full step. Thus add constraint nvl to the list of active
            # constraints and update J and R
            #
            nact += 1
            iact[nact] = nvl
            #
            # to update R we have to put the first nact-1 components of the d vector
            # into column (nact) of R
            #
            l = iwrm + trunc(Int, ((nact - 1) * nact) / 2) + 1
            for i = 1:nact-1
                work[l] = work[i]
                l += 1
            end
            #
            # if now nact=n, then we just have to add the last element to the new
            # row of R.
            # Otherwise we use Givens transformations to turn the vector d(nact:n)
            # into a multiple of the first unit vector. That multiple goes into the
            # last element of the new row of R and J is accordingly updated by the
            # Givens transformations.
            #
            if nact == n
                work[l] = work[n]
            else
                for i = n:-1:nact+1
                    #
                    # we have to find the Givens rotation which will reduce the element
                    # (l1) of d to zero.
                    # if it is already zero we don't have to do anything, except of
                    # decreasing l1
                    #
                    if work[i] == zero(T)
                        continue
                    end
                    gc = max(abs(work[i-1]), abs(work[i]))
                    gs = min(abs(work[i-1]), abs(work[i]))
                    #term = gc * sqrt(1 + (gs*gs)/(gc  * gc))
                    #if isnan(term)
                    #    term = gc
                    #end
                    term = max(gc, sqrt(gs^2 + gc^2))
                    temp = copysign(term, work[i-1])
                    # println(iter, " ",i," ",gc, " ",gs," ",term, " ",work[i-1]," ",work[i])

                    if isnan(temp)
                        println(gc, " ", gs, " ", term, " ", work[i-1], " ", work[i])
                        throw(DomainError(temp))
                    end
                    # if gc == zero(T) #fortran 0
                    #     temp = zero(T)
                    # end
                    if work[i-1] == zero(T) #fortran 0
                        gc = zero(T)
                    else
                        gc = work[i-1] / temp
                    end

                    if work[i] == zero(T) #fortran 0
                        gs = zero(T)
                    else
                        gs = work[i] / temp
                    end
                    #
                    # The Givens rotation is done with the matrix (gc gs, gs -gc).
                    # If gc is one, then element (i) of d is zero compared with element
                    # (l1-1). Hence we don't have to do anything.
                    # If gc is zero, then we just have to switch column (i) and column (i-1)
                    # of J. Since we only switch columns in J, we have to be careful how we
                    # update d depending on the sign of gs.
                    # Otherwise we have to apply the Givens rotation to these columns.
                    # The i-1 element of d has to be updated to temp.
                    #
                    if gc == one(T)
                        continue
                    end
                    if gc == zero(T)
                        work[i-1] = temp * sign(gs) #important to use the sign here, instead of gs to avoid some infinite loops
                        for j = 1:n
                            temp = dmat[j, i-1]
                            dmat[j, i-1] = dmat[j, i]
                            dmat[j, i] = temp
                        end
                    else
                        work[i-1] = temp
                        nu = gs / (one(T) + gc)
                        for j = 1:n
                            temp = gc * dmat[j, i-1] + gs * dmat[j, i]
                            dmat[j, i] = nu * (dmat[j, i-1] + temp) - dmat[j, i]
                            dmat[j, i-1] = temp
                        end
                    end

                end
                #
                # l is still pointing to element (nact,nact) of the matrix R.
                # So store d(nact) in R(nact,nact)
                work[l] = work[nact]
            end
        else
            #
            # we took a partial step in dual space. Thus drop constraint it1,
            # that is, we drop the it1-th active constraint.
            # then we continue at step 2(a) (marked by label 55)
            # but since the fit changed, we have to recalculate now "how much"
            # the fit violates the chosen constraint now.
            #
            sum = -bvec[nvl]
            for j = 1:n
                sum = sum + sol[j] * amat[j, nvl]
            end
            if nvl > meq
                work[iwsv+nvl] = sum
            else
                work[iwsv+nvl] = -abs(sum)
                if sum > zero(T)
                    for j = 1:n
                        amat[j, nvl] = -amat[j, nvl]
                    end
                    bvec[nvl] = -bvec[nvl]
                end
            end
            @goto L700
        end
    end
    @goto L50
    #
    # Drop constraint it1
    #
    @label L700
    #
    # if it1 = nact it is only necessary to update the vector u and nact
    #
    if it1 == nact
        @goto L799
    end
    #
    # After updating one row of R (column of J) we will also come back here
    #
    @label L797
    #
    # we have to find the Givens rotation which will reduce the element
    # (it1+1,it1+1) of R to zero.
    # if it is already zero we don't have to do anything except of updating
    # u, iact, and shifting column (it1+1) of R to column (it1)
    # l  will point to element (1,it1+1) of R
    # l1 will point to element (it1+1,it1+1) of R
    #
    l = iwrm + trunc(Int, (it1 * (it1 + 1)) / 2) + 1
    l1 = l + it1
    if work[l1] == zero(T)
        @goto L798
    end
    gc = max(abs(work[l1-1]), abs(work[l1]))
    gs = min(abs(work[l1-1]), abs(work[l1]))
    term = max(gc, sqrt(gs^2 + gc^2))
    temp = copysign(term, work[l1-1])
    # temp = copysign(gc * sqrt(1 + (gs / gc) * (gs / gc)), work[l1-1])

    if work[l1-1] == zero(T) #fortran 0
        gc = zero(T)
    else
        gc = work[l1-1] / temp
    end

    if work[l1] == zero(T) #fortran 0
        gs = zero(T)
    else
        gs = work[l1] / temp
    end
    if isnan(gs) || isnan(gc) || isnan(temp)
        throw(DomainError(temp))
    end
    #
    # The Givens rotatin is done with the matrix (gc gs, gs -gc).
    # If gc is one, then element (it1+1,it1+1) of R is zero compared with
    # element (it1,it1+1). Hence we don't have to do anything.
    # if gc is zero, then we just have to switch row (it1) and row (it1+1)
    # of R and column (it1) and column (it1+1) of J. Since we swithc rows in
    # R and columns in J, we can ignore the sign of gs.
    # Otherwise we have to apply the Givens rotation to these rows/columns.
    #
    if gc == one(T)
        @goto L798
    end
    if gc == zero(T)
        for i = it1+1:nact
            temp = work[l1-1]
            work[l1-1] = work[l1]
            work[l1] = temp
            l1 = l1 + i
        end
        for i = 1:n
            temp = dmat[i, it1]
            dmat[i, it1] = dmat[i, it1+1]
            dmat[i, it1+1] = temp
        end
    else
        nu = gs / (one(T) + gc)
        if gs == zero(T)
            nu = zero(T)
        end
        for i = it1+1:nact
            temp = gc * work[l1-1] + gs * work[l1]
            work[l1] = nu * (work[l1-1] + temp) - work[l1]
            work[l1-1] = temp
            l1 = l1 + i
        end
        for i = 1:n
            temp = gc * dmat[i, it1] + gs * dmat[i, it1+1]
            dmat[i, it1+1] = nu * (dmat[i, it1] + temp) - dmat[i, it1+1]
            dmat[i, it1] = temp
        end
    end
    #
    # shift column (it1+1) of R to column (it1) (that is, the first it1
    # elements). The posit1on of element (1,it1+1) of R was calculated above
    # and stored in l.
    #
    @label L798
    l1 = l - it1
    for i = 1:it1
        work[l1] = work[l]
        l = l + 1
        l1 = l1 + 1
    end
    #
    # update vector u and iact as necessary
    # Continue with updating the matrices J and R
    #
    work[iwuv+it1] = work[iwuv+it1+1]
    iact[it1] = iact[it1+1]
    it1 += 1
    if it1 < nact
        @goto L797
    end
    @label L799
    work[iwuv+nact] = work[iwuv+nact+1]
    work[iwuv+nact+1] = zero(T)
    iact[nact] = 0
    nact -= 1
    iter[2] += 1
    @goto L55
    return (sol, lagr, crval, iact, nact, iter, ierr)
end

