using LinearAlgebra, SparseArrays
export solveQPcompact,convertSparse

#  This routine implements the dual method of Goldfarb and Idnani (1982, 1983) for solving quadratic programming problems of the form
# \eqn{\min(-d^T b + 1/2 b^T D b)}{min(-d^T b + 1/2 b^T D b)} with the
# constraints \eqn{A^T b >= b_0}.
function solveQPcompact(dmat::AbstractMatrix{T}, dvec::AbstractArray{T},
    Amat::AbstractMatrix{T}, Aind::AbstractMatrix{Int}, bvec::AbstractArray{T},
    meq::Int; factorized::Bool=false)::Tuple{AbstractArray{T},AbstractArray{T},T,AbstractArray{Int},Int,AbstractArray{Int}} where {T} #sol, lagr, crval, iact, nact, iter
    n = size(dmat, 1)
    q = 0
    if size(Amat, 1) > 0
        q = size(Amat, 2)
    end
    
    anrow = size(Amat, 1)
    if size(dmat, 1) > 0 && n != size(dmat, 2)
        throw(error("Dmat is not symmetric!"))
    end
    if n != length(dvec)
        throw(error("Dmat and dvec are incompatible!"))
    end
    if (anrow + 1 != size(Aind, 1)) || (q != size(Aind, 2)) || (q != length(bvec))
        throw(error("Amat, Aind and bvec are incompatible!"))
    end
    if (meq > q) || (meq < 0)
        throw(error("Value of meq is invalid!"))
    end
    r = min(n, q)
    work = zeros(T, 2 * n + trunc(Int,r * (r + 5) / 2) + 2 * q + 1)
    sol, lagr, crval, iact, nact, iter, ierr = qpgen1(dmat, dvec, n, n, Amat, Aind, bvec, anrow, q, meq, factorized, work)
    if ierr == 1
        throw(error("constraints are inconsistent, no solution!"))
    elseif ierr == 2
        throw(error("matrix D in quadratic function is not positive definite!"))
    end
    return sol, lagr, crval, iact, nact, iter
end

function convertSparse(dmat::AbstractMatrix{T})::Tuple{AbstractMatrix{Int},AbstractMatrix{T}} where {T} #aind, amat
    #  amat   lxq matrix (dp)
    #         *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE
    #             CHANGED SIGNES ON EXIT ***
    #  iamat  (l+1)xq matrix (int)
    #         these two matrices store the matrix A in compact form. the format
    #         is: [ A=(A1 A2)^T ]
    #           iamat(1,i) is the number of non-zero elements in column i of A
    #           iamat(k,i) for k>=2, is equal to j if the (k-1)-th non-zero
    #                      element in column i of A is A(i,j)
    #            amat(k,i) for k>=1, is equal to the k-th non-zero element
    #                      in column i of A.
    dmatCsr = copy(dmat')
    maxnnz = 0
    cols = rowvals(dmatCsr) #vector of raw indices
    vals = nonzeros(dmatCsr)
    n = size(dmatCsr,2)
    for i = 1:n
        nzr = nzrange(dmatCsr, i)
        maxnnz = max(maxnnz, length(nzr))
    end
    aind = zeros(Int, (maxnnz + 1, n))
    amat = zeros(T, (maxnnz, n))
    for j = 1:n
        aind[1, j] = length(nzrange(dmatCsr, j))
    end
    for i = 1:n
        k = 1
        for j in nzrange(dmatCsr, i)
            aind[k+1, i] = cols[j]
            amat[k, i] = vals[j]
            k+=1
        end
    end
    return aind, amat
end


#  Julia Port
#  Copyright (C) 2021 Fabien Le Floc'h <fabien@2ipi.com>
#  Original Fortran Code
#  Copyright (C) 1995-2010 Berwin A. Turlach <Berwin.Turlach@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
#  USA.
#
#  this routine uses the Goldfarb/Idnani algorithm to solve the
#  following minimization problem:
#
#        minimize  -d^T x + 1/2 *  x^T D x
#        where   A1^T x  = b1
#                A2^T x >= b2
#
#  the matrix D is assumed to be positive definite.  Especially,
#  w.l.o.g. D is assumed to be symmetric.
#
#  Input parameter:
#  dmat   nxn matrix, the matrix D from above (dp)
#         *** WILL BE DESTROYED ON EXIT ***
#         The user has two possibilities:
#         a) Give D (ierr=0), in this case we use routines from LINPACK
#            to decompose D.
#         b) To get the algorithm started we need R^-1, where D=R^TR.
#            So if it is cheaper to calculate R^-1 in another way (D may
#            be a band matrix) then with the general routine, the user
#            may pass R^{-1}.  Indicated by ierr not equal to zero.
#  dvec   nx1 vector, the vector d from above (dp)
#         *** WILL BE DESTROYED ON EXIT ***
#         contains on exit the solution to the initial, i.e.,
#         unconstrained problem
#  fddmat scalar, the leading dimension of the matrix dmat
#  n      the dimension of dmat and dvec (int)
#  amat   lxq matrix (dp)
#         *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE
#             CHANGED SIGNES ON EXIT ***
#  iamat  (l+1)xq matrix (int)
#         these two matrices store the matrix A in compact form. the format
#         is: [ A=(A1 A2)^T ]
#           iamat(1,i) is the number of non-zero elements in column i of A
#           iamat(k,i) for k>=2, is equal to j if the (k-1)-th non-zero
#                      element in column i of A is A(i,j)
#            amat(k,i) for k>=1, is equal to the k-th non-zero element
#                      in column i of A.
#
#  bvec   qx1 vector, the vector of constants b in the constraints (dp)
#         [ b = (b1^T b2^T)^T ]
#         *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE
#             CHANGED SIGNES ON EXIT ***
#  fdamat the first dimension of amat as declared in the calling program.
#         fdamat >= n (and iamat must have fdamat+1 as first dimension)
#  q      integer, the number of constraints.
#  meq    integer, the number of equality constraints, 0 <= meq <= q.
#  ierr   integer, code for the status of the matrix D:
#            ierr =  0, we have to decompose D
#            ierr != 0, D is already decomposed into D=R^TR and we were
#                       given R^{-1}.
#
#  Output parameter:
#  sol   nx1 the final solution (x in the notation above)
#  lagr  qx1 the final Lagrange multipliers
#  crval scalar, the value of the criterion at the minimum
#  iact  qx1 vector, the constraints which are active in the final
#        fit (int)
#  nact  scalar, the number of constraints active in the final fit (int)
#  iter  2x1 vector, first component gives the number of "main"
#        iterations, the second one says how many constraints were
#        deleted after they became active
#  ierr  integer, error code on exit, if
#           ierr = 0, no problems
#           ierr = 1, the minimization problem has no solution
#           ierr = 2, problems with decomposing D, in this case sol
#                     contains garbage!!
#
#  Working space:
#  work  vector with length at least 2*n+r*(r+5)/2 + 2*q +1
#        where r=min(n,q)
#
function qpgen1(dmat::AbstractMatrix{T}, dvec::AbstractArray{T}, fddmat::Int, n::Int, amat::AbstractMatrix{T},
    iamat::AbstractMatrix{Int}, bvec::AbstractArray{T}, fdamat::Int, q::Int, meq::Int, factorized::Bool, work::AbstractArray{T})::Tuple{AbstractArray{T},AbstractArray{T},T,AbstractArray{Int},Int,AbstractArray{Int},Int} where {T} # sol, lagr, crval, iact, nact, iter, ierr

    sol = zeros(T, n)
    lagr = zeros(T, q)
    iact = zeros(Int, q)
    iter = zeros(Int, 2)
    local t1inf::Bool, t2min::Bool
    local it1::Int, iwzv::Int, iwrv::Int, iwrm::Int, iwsv::Int, iwuv::Int, nvl::Int, iwnbv::Int, l1::Int
    local temp::T, sum::T, t1::T, tt::T, gc::T, gs::T, nu::T, vsmall::T, tmpa::T, tmpb ::T
    r = min(n, q)
    l = 2 * n + trunc(Int,(r * (r + 5)) / 2) + 2 * q + 1
    #
    #     code gleaned from Powell's ZQPCVX routine to determine a small
    #     number  that can be assumed to be an upper bound on the relative
    #     precision of the computer arithmetic.
    #
    vsmall = 1.0e-60
    tmpa = 1.0
    tmpb = 1.0
    while tmpa <= 1 || tmpb <= 1
        vsmall = vsmall + vsmall
        tmpa = 1.0 + 0.1 * vsmall
        tmpb = 1.0 + 0.2 * vsmall
    end

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
        for j = 1:iamat[1, i]
            sum += amat[j, i]^2
        end
        work[iwnbv+i] = sqrt(sum)
        if isnan(sum)
            throw(DomainError(sum))
        end
    end
    nact = 0
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
        for j = 1:iamat[1, i]
            sum += amat[j, i] * sol[iamat[j+1, i]]
        end
        if abs(sum) < vsmall
            sum = zero(T)
        end
        if i > meq
            work[l] = sum
        else
            work[l] = -abs(sum)
            if sum > zero(T)
                for j = 1:iamat[1, i]
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
        for j = 1:iamat[1, nvl]
            sum += dmat[iamat[j+1, nvl], i] * amat[j, nvl]
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
            if isnan(temp)
                throw(DomainError(temp))
            end
            if temp < t1
                t1 = temp
                it1 = i
            end
        end
    end
    #
    # test if the z vector is equal to zero
    #
    sum = zero(T)
    for i = iwzv+1:iwzv+n
        sum += work[i]^2
    end
    if abs(sum) <= vsmall
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
        for i = 1:iamat[1, nvl]
            sum += work[iwzv+iamat[i+1, nvl]] * amat[i, nvl]
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
                    temp = copysign(gc * sqrt(1 + gs * gs / (gc * gc)), work[i-1])
                    if gc == zero(T) #fortran 0
                        temp = zero(T)
                    end
                    if work[i-1] == zero(T) #fortran 0
                        gc = zero(T)
                    end

                    if work[i] == zero(T) #fortran 0
                        gs = zero(T)
                    end
                    gc = work[i-1] / temp
                    gs = work[i] / temp
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
                        work[i-1] = gs * temp
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
            for j = 1:iamat[1, nvl]
                sum = sum + sol[iamat[j+1, nvl]] * amat[j, nvl]
            end
            if nvl > meq
                work[iwsv+nvl] = sum
            else
                work[iwsv+nvl] = -abs(sum)
                if sum > zero(T)
                    for j = 1:iamat[1, nvl]
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
    temp = copysign(gc * sqrt(1 + (gs / gc) * (gs / gc)), work[l1-1])

    if gc == zero(T) #fortran 0
        temp = zero(T)
    end
    if work[l1-1] == zero(T) #fortran 0
        gc = zero(T)
    end

    if work[l1] == zero(T) #fortran 0
        gs = zero(T)
    end

    gc = work[l1-1] / temp
    gs = work[l1] / temp
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

function ddot(n::Int, dx::AbstractArray{T}, incx::Int, dy::AbstractArray{T}, incy::Int)::T where {T}

    #  Purpose
    #  =======
    #
    #     DDOT forms the dot product of two vectors.
    #     uses unrolled loops for increments equal to one.
    #
    dtemp = zero(T)
    if n <= 0
        return dtemp
    end
    if incx == 1 && incy == 1
        m = n % 5
        if m == 0
            for i = 1:m
                dtemp += dx[i] * dy[i]
            end
            if (n .< 5)
                return dtemp
            end
        end
        mp1 = m + 1
        for i = mp1:5:n
            dtemp += dx[i] * dy[i] + dx[i+1] * dy[i+1] + dx[i+2] * dy[i+2] + dx[i+3] * dy[i+3] + dx[i+4] * dy[i+4]
        end
    else
        ix = 1
        iy = 1
        if (inx < 0)
            ix = (-n + 1) * incx + 1
        end
        if (incy .< 0)
            iy = (-n + 1) * incy + 1
        end
        for i = 1:n
            dtemp += dx[ix] * dy[iy]
            ix += incx
            iy += incy
        end
    end
    return dtemp
end


function ddot(nrow::Int, x::AbstractMatrix{T}, beginI::Int, j1::Int, j2::Int)::T where {T}
    coldot = zero(T)
    if nrow <= 0
        return coldot
    end
    m = nrow % 5
    mpbegin = m + beginI
    endI = beginI + nrow - 1
    for i = beginI:mpbegin-1
        coldot += x[i, j1] * x[i, j2]
    end

    for i = mpbegin:5:endI
        coldot += x[i, j1] * x[i, j2] + x[i+1, j1] * x[i+1, j2] + x[i+2, j1] * x[i+2, j2] + x[i+3, j1] * x[i+3, j2] + x[i+4, j1] * x[i+4, j2]
    end
    return coldot
end


function ddotv(nrow::Int, x::AbstractMatrix{T}, beginI::Int, j1::Int, v::AbstractArray{T})::T where {T}
    coldot = zero(T)
    if nrow <= 0
        return coldot
    end
    m = nrow % 5
    mpbegin = m + beginI
    endI = beginI + nrow - 1
    for i = beginI:mpbegin-1
        coldot += x[i, j1] * v[i]
    end

    for i = mpbegin:5:endI
        coldot += x[i, j1] * v[i] + x[i+1, j1] * v[i+1] + x[i+2, j1] * v[i+2] + x[i+3, j1] * v[i+3] + x[i+4, j1] * v[i+4]
    end
    return coldot
end

function dscal(nrow::Int, a::T, x::AbstractMatrix{T}, beginI::Int, j::Int) where {T}
    if nrow <= 0
        return
    end
    m = nrow % 5
    mpbegin = m + beginI
    endI = beginI + nrow - 1

    for i = beginI:mpbegin-1
        x[i, j] *= a
    end
    for i = mpbegin:5:endI
        x[i, j] *= a
        x[i+1, j] *= a
        x[i+2, j] *= a
        x[i+3, j] *= a
        x[i+4, j] *= a
    end
end

function dpofa(a::AbstractMatrix{T}, lda::Int, n::Int)::Int where {T}
    for j = 1:n
        info = j
        s = 0.0
        jm1 = j - 1
        for k = 1:jm1
            t = a[k, j] - ddot(k-1, a, 1, k, j)
            t /= a[k, k]
            a[k, j] = t
            s = s + t * t
        end
        s = a[j, j] - s
        if s <= zero(T)
            return info
        end
        a[j, j] = sqrt(s)
    end
    return 0
end

function daxpyv(nrow::Int, a::Float64, x::AbstractMatrix{T}, beginI::Int, j1::Int, v::AbstractArray{T}) where {T}
    #This method multiplies a constant times a portion of a column
    #of a matrix and adds the product to the corresponding portion
    #of another column of the matrix --- a portion of col2 is
    #  replaced by the corresponding portion of a*col1 + col2.
    #It uses unrolled loops.
    #It is a modification of the LINPACK subroutine
    #DAXPY.  In the LINPACK listing DAXPY is attributed to Jack Dongarra
    #with a date of 3/11/78.

    if nrow <= 0
        return
    end
    if a == zero(T)
        return
    end

    m = nrow % 4
    mpbegin = m + beginI
    endI = beginI + nrow - 1

    for i = beginI:mpbegin-1
        v[i] += a * x[i, j1]
    end

    for i = mpbegin:4:endI
        v[i] += a * x[i, j1]
        v[i+1] += a * x[i+1, j1]
        v[i+2] += a * x[i+2, j1]
        v[i+3] += a * x[i+3, j1]
    end

    return
end

function daxpy(nrow::Int, a::T, x::AbstractMatrix{T}, beginI::Int, j1::Int, j2::Int) where {T}
    #This method multiplies a constant times a portion of a column
    #  *of a matrix and adds the product to the corresponding portion
    #  *of another column of the matrix --- a portion of col2 is
    #  replaced by the corresponding portion of a*col1 + col2.
    #  *It uses unrolled loops.
    #  *It is a modification of the LINPACK subroutine
    #  *DAXPY.  In the LINPACK listing DAXPY is attributed to Jack Dongarra
    #  *with a date of 3/11/78.

    if nrow <= 0
        return
    end
    if a == zero(T)
        return
    end

    m = nrow % 4
    mpbegin = m + beginI
    endI = beginI + nrow - 1

    for i = beginI:mpbegin-1
        x[i, j2] += a * x[i, j1]
    end

    for i = mpbegin:4:endI
        x[i, j2] += a * x[i, j1]
        x[i+1, j2] += a * x[i+1, j1]
        x[i+2, j2] += a * x[i+2, j1]
        x[i+3, j2] += a * x[i+3, j1]
    end
end

function dpori(a::AbstractMatrix{T}, lda::Int, n::Int) where {T}
    #     dpori computes the inverse of the factor of a
    #     double precision symmetric positive definite matrix
    #     using the factors computed by dpofa.
    #
    #     modification of dpodi by BaT 05/11/95
    #
    #     on entry
    #
    #        a       double precision(lda, n)
    #                the output  a  from dpofa
    #
    #        lda     integer
    #                the leading dimension of the array  a .
    #
    #        n       integer
    #                the order of the matrix  a .
    #
    #     on return
    #
    #        a       if dpofa was used to factor  a  then
    #                dpodi produces the upper half of inverse(a) .
    #                elements of  a  below the diagonal are unchanged.
    #
    #     error condition
    #
    #        a division by zero will occur if the input factor contains
    #        a zero on the diagonal and the inverse is requested.
    #        it will not occur if the subroutines are called correctly
    #        and if dpoco or dpofa has set info .eq. 0 .
    #
    #     linpack.  this version dated 08/14/78 .
    #     cleve moler, university of new mexico, argonne national lab.
    #     modified by Berwin A. Turlach 05/11/95
    #
    #     subroutines and functions
    #
    #     blas daxpy,dscal
    #     fortran mod
    #
    #     internal variables
    #
    for k = 1:n
        a[k, k] = one(T) / a[k, k]
        t = -a[k, k]
        dscal(k, t, a, 1, k)
        kp1 = k + 1
        if n >= kp1
            for j = kp1:n
                t = a[k, j]
                a[k, j] = zero(T)
                daxpy(k, t, a, 1, k, j)
            end
        end
    end
end

function dposl(a::AbstractMatrix{T}, lda::Int, n::Int, b::AbstractArray{T}) where {T}
    #
    #     dposl solves the double precision symmetric positive definite
    #     system a * x = b
    #     using the factors computed by dpoco or dpofa.
    #
    #     on entry
    #
    #        a       double precision(lda, n)
    #                the output from dpoco or dpofa.
    #
    #        lda     integer
    #                the leading dimension of the array  a .
    #
    #        n       integer
    #                the order of the matrix  a .
    #
    #        b       double precision(n)
    #                the right hand side vector.
    #
    #     on return
    #
    #        b       the solution vector  x .
    #
    #     error condition
    #
    #        a division by zero will occur if the input factor contains
    #        a zero on the diagonal.  technically this indicates
    #        singularity but it is usually caused by improper subroutine
    #        arguments.  it will not occur if the subroutines are called
    #        correctly and  info .eq. 0 .
    #
    #     to compute  inverse(a) * c  where  c  is a matrix
    #     with  p  columns
    #           call dpoco(a,lda,n,rcond,z,info)
    #           if (rcond is too small .or. info .ne. 0) go to ...
    #           do 10 j = 1, p
    #              call dposl(a,lda,n,c(1,j))
    #        10 continue
    #
    #     linpack.  this version dated 08/14/78 .
    #     cleve moler, university of new mexico, argonne national lab.
    #
    #     subroutines and functions
    #
    #     blas daxpy,ddot
    #
    #     internal variables
    #
    #
    #     solve trans(r)*y = b
    #
    for k = 1:n
        t = ddotv(k-1, a, 1, k, b)
        b[k] = (b[k] - t) / a[k, k]
    end
    #
    #     solve r*x = y
    #
    for kb = 1:n
        k = n + 1 - kb
        b[k] = b[k] / a[k, k]
        t = -b[k]
        daxpyv(k-1, t, a, 1, k, b)
    end
end
