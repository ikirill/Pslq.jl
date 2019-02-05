module Pslq

# Parallel Integer Relation Detection: Techniques and Applications
#     David H. Bailey
#     David J. Broadhurst
# https://doi.org/10.1090/S0025-5718-00-01278-3
# file:///Users/kirill/Documents/Papers/Parallel%20Integer%20Relation%20Detection%20PSLQ%20-%20Bailey%20-%20Broadhurst.pdf

# Goals
# 1. multi-level pslq, with support for parallelizing matrix operations
# 2. parallel pslq
# 3. support different floating-point types, both BigFloat and ArbFloat
# 4. Medium-strength testing, at least the examples in the paper

# DONE B should be stored as its transpose, for consistency
# DONE Rescaling
# DONE multi-level pslq
# DONE multi-pair pslq
# DONE Handle intermediate precisions: Float64, Float128, Double-Double, Quad-Quad, and intermediate BigFloat/ArbFloat
# TODO It's 2 times faster than the results reported in Bailey-Broadhurst, how is that even possible?
# DONE If the problem is that BigFloat matrix multiplications are too expensive, the call to sub-precision routines should be recursive: BigFloat -> lower precision -> Double64 -> Float64

# TODO Avoid storing A and updating it at every iteration

# TODO Support for Double64

# FIXED:
# I think what's going on is that sometimes the multi-pair
# iteration fails (as described in the paper, but I haven't
# checked it's the same reason), and in that case the (A1,B1)
# matrices are bogus---they do not increase 1/max(abs(diag(H)))
# or reduce minimum(abs(y)).
# -> I'm not sure how to detect this situation
# -> It depends on rounding errors too:
# -> Pslq.test_pslq(2,3,51) != Pslq.test_pslq(2,3,53)
# -> Pslq.test_pslq(2,3,80) != Pslq.test_pslq(2,3,81)

# The issue is: in the update step H <- lq(A1*H),
# norm(diag(H),Inf) is supposed to be reduced because that is
# the goal of the iteration. But after multiple iterations, with
# different precision levels, we get H ←
# lq(A1(m)*A1(m-1)*⋯*A1(1)*H), and somehow norm(diag(H),Inf)
# blows up completely. This isn't addressed in the paper, I
# don't think they implemented this.

# FIXME What is the point of updating A, if all we care about is
# FIXME \|A\|, which is \|B^-1\| or κ(B).

# Performance:
# julia> @time Pslq.bench_pslq()
#     old: 170s
#     9a1d08b (2019/01/23): 160s

using LinearAlgebra
using Printf
using Test

using DoubleFloats


struct IntegerRelation{T<:AbstractFloat}
    reliable :: Bool
    status :: Symbol
    x :: Array{T}
    relation :: Array{BigInt}
    residual :: T
    coefbound :: Float64
    confidence :: T
end

function Base.show(io::IO, ::MIME"text/plain", s::IntegerRelation)
    @printf io "IntegerRelation on %d numbers\n" length(s.x)
    @printf io "    reliable: %s\n" s.reliable
    @printf io "    status: %s\n" s.status
    @printf io "    residual: %.2e\n" s.residual
    @printf io "    coefbound: %.2e\n" s.coefbound
    @printf io "    confidence: %.2e\n" s.confidence
    @printf io "    relation: %s\n" (s.reliable ? s.relation : "⋯")
end

## {{{ Misc

nint(x::AbstractFloat) = let y = round(x, RoundNearestTiesAway); (y, BigInt(y)); end

y_confidence(y::Vector{<:AbstractFloat}) = let (ymin, ymax) = extrema(abs.(y)); ymin/ymax; end

function rescale_numbers(x::Vector{T}) where {T<:AbstractFloat}
    y = Rational{BigInt}[(2//1)^(-floor(log2(x[i]))) for i=eachindex(x)]
    return (x .* T.(y), y)
end

function unrescale_numbers!(relation::Vector{BigInt}, scale::Vector{Rational{BigInt}})
    y = relation .* scale
    s = reduce(lcm, denominator.(y))
    relation .= BigInt.(y .* s)
end

struct ContractException <: Exception
    msg :: String
end
Base.show(io::IO, e::ContractException) = print(io, "Contract violated: $(e.msg)")
macro check(cond)
    :(if !($(esc(cond))); throw(ContractException($(string(cond)))); end)
end

## }}}
## {{{ PSLQ (1)

function reduce_H(m::Union{Nothing,Int}, y::Vector{T}, H::Matrix{T}, A::Matrix{BigInt}, B::Matrix{BigInt}) where {T<:AbstractFloat}
    n = length(y)
    for i in (m === nothing ? 2 : m+1):n, j in (m === nothing ? i-1 : min(i-1, m+1)):-1:1
        t, t1 = nint(H[i,j] / H[j,j])
        y[j] += t * y[i]
        H[i,1:j] .-= t * H[j,1:j]
        A[i,1:n] .-= t1 * A[j,1:n]
        B[j,1:n] .+= t1 * B[i,1:n]
    end
    @assert (A * B' == I) "invariant B == inv(A)"
end

function init_H(y::Vector{T}, x::Vector{T}) where {T<:AbstractFloat}
    n = length(y)
    # s = [norm(x[k:n]) for k=1:n]
    s = sqrt.(reverse(cumsum(reverse(x.^2)))) ./ norm(x)
    H = zeros(T, n, n-1)
    for j in 1:n-1
        H[j,j] = s[j+1] / s[j]
        H[j+1:end,j] .= -y[j+1:end] .* (y[j] / (s[j] * s[j+1]))
    end
    H
end

function pick_and_swap_one_row(γ::Real, y::Vector{T}, H::Matrix{T}, A::Matrix{BigInt}, B::Matrix{BigInt}) where {T<:AbstractFloat}
    n = length(y)
    # 1.
    m = argmax([γ^i * abs(H[i,i]) for i in 1:n-1])
    # 2.
    y[m], y[m+1] = y[m+1], y[m]
    A[m,:], A[m+1,:] = A[m+1,:], A[m,:]
    H[m,:], H[m+1,:] = H[m+1,:], H[m,:]
    B[m,:], B[m+1,:] = B[m+1,:], B[m,:]
    return m
end

function pslq_simple(x::Array{T};
              γ::Real=1.25,
              coefbound::Real=floatmax(Float64),
              # threshold::Real=sqrt(eps(T)),
              tol::Real=eps(T)^0.9,
              verbose::Bool=true) where {T<:AbstractFloat}
    γ > sqrt(4/3) || error("invalid γ: !($γ > √4/3)")
    isbitstype(T) && @warn "Using a low-precision type T=$T in pslq is usually a mistake"
    n = length(x)
    # This is a direct transcription of the algorithm from pp.2–3
    # 1, 2, 3, 4.
    A, B = Matrix{BigInt}(I, n, n), Matrix{BigInt}(I, n, n)
    y = x ./ norm(x)
    H = init_H(y, x)
    reduce_H(nothing, y, H, A, B)
    @show tol

    status = :unknown
    for itcount in 1:typemax(Int)
        # m = pick_and_swap_one_row(γ, y, H, A, B)
        # 1.
        m = argmax([γ^i * abs(H[i,i]) for i in 1:n-1])
        # 2.
        y[m], y[m+1] = y[m+1], y[m]
        A[m,:], A[m+1,:] = A[m+1,:], A[m,:]
        H[m,:], H[m+1,:] = H[m+1,:], H[m,:]
        B[m,:], B[m+1,:] = B[m+1,:], B[m,:]

        # 3.
        if m ≤ n-2
            G, _ = givens(H[m,m], H[m,m+1], m, m+1)
            H = H * G'
            @assert abs(H[m,m+1]) < 4eps()
            H[m,m+1] = 0
        end

        # 4.
        reduce_H(m, y, H, A, B)

        if verbose && itcount%8 == 0
            @printf stderr "[%4d]: M=%8.1e ymin=%8.1e ymin/ymax=%.1e ABmax=%8.1e/%.1e\n" itcount 1/maximum(abs.(diag(H))) minimum(abs.(y)) y_confidence(y) norm(A,Inf) norm(B,Inf)
        end

        # 5.
        M = 1 / maximum(abs(H[j,j]) for j in 1:n-1)
        if M > coefbound
            status = :coefbound
            println("Reached coefbound: $M > $coefbound")
            break
        end

        # 6.
        if (Amax = norm(A, Inf)) > 1/tol
            status = :tol
            println("Reached max precision Amax=$(Float64(Amax))")
            break
        end
        if minimum(abs.(y)) < tol
            status = :finished
            break
        end

    end

    # Relation
    m = argmin(abs.(y))
    relation = B[m,:]
    @show relation

    # Done
    confidence = y_confidence(y)
    bitlength = sum(log2(max(1, abs(b))) for b in relation)
    accept_confidence = confidence < max(1e-100, tol^0.1)
    accept_bitlength = bitlength < 0.95 * log2(1/tol)
    @show Float64(bitlength)
    @show Float64(log2(1/tol))
    reliable = accept_confidence && accept_bitlength
    residual = dot(x, relation)
    coefbound = Float64(1 / maximum(abs.(diag(H))))
    IntegerRelation(reliable, status, x, relation, residual, coefbound, confidence)
end

function test1(m::Integer, n::Integer; kwargs...)
    setprecision(BigFloat, max(256, 20*m*n))
    x = 2^(big"1.0"/m) + 3^(big"1.0"/n)
    pslq([x^i for i in 0:m*n]; kwargs...)
end

## }}}
## {{{ PSLQ multi-pair iteration

function remove_corner(H::Matrix{<:AbstractFloat}, m::Int)
    @boundscheck (let n = size(H,1); size(H) == (n, n-1); end)
    (m ≤ size(H,1)-2 && H[m,m+1] != 0) || return
    G, _ = givens(H[m,m], H[m,m+1], m, m+1)
    rmul!(H, G')
    @assert H[m,m+1] < 4eps(eltype(H))
    H[m,m+1] = 0
end

function remove_corners!(H::Matrix{<:AbstractFloat})
    @inbounds for m=1:size(H,1)-2
        remove_corner(H, m)
    end
end

large_enough_int_type(::Type{Float64}) = Int64
large_enough_int_type(::Type{Double64}) = Int128
large_enough_int_type(::Type{BigFloat}) = BigInt

function swap_rows!(m::Int, y::Vector, H::Matrix, A::Matrix, B::Matrix)
    @inbounds begin
        y[m], y[m+1] = y[m+1], y[m]
        A[m,:], A[m+1,:] = A[m+1,:], A[m,:]
        H[m,:], H[m+1,:] = H[m+1,:], H[m,:]
        B[m,:], B[m+1,:] = B[m+1,:], B[m,:]
    end
end

function update_A!(A::Matrix, t::Matrix)
    n = size(A,1)
    @assert size(A) == (n,n) && size(t) == (n,n-1)
    @inbounds for k=1:n, j=1:n-1
        Ajk = A[j,k]
        for i=j+1:n
            A[i,k] -= t[i,j] * Ajk
        end
    end
end

function reduce_H!(H::Matrix{T}, t::UnitLowerTriangular{T}) where {T<:AbstractFloat}
    n = size(H,1)
    @check size(H) == (n, n-1) && size(t) == (n, n)
    @inbounds @fastmath for i in 2:n, j in 1:n-i+1
        l = i + j - 1
        Hlj = H[l,j]
        for k in j+1:l-1
            # NOTE this isn't very clear
            # NOTE but this doesn't use any t[l,k]'s that haven't been initialized yet
            Hlj -= t[l,k] * H[k,j]
        end
        t[l,j] = round(Hlj / H[j,j])
        H[l,j] = Hlj - t[l,j] * H[j,j]
    end
    @assert all(isfinite, H)
end

"""
    pslq_mp(typ, y, H)

Full multi-pair pslq iteration.

Because it is expected `pslq_mp` will be called with low levels of
precision (usually typ=Float64), I omitted the coefbound parameter for
now.

"""
function pslq_mp(::Type{T}, y0::Vector{BigFloat}, H0::Matrix{BigFloat};
          γ::Real=1.25,
          β0::Real=0.4,
          tol::Real=eps(T)^0.7,
          maxiter::Integer=precision(T),
          verbose::Bool=false
          ) where {T<:AbstractFloat}
    β = β0

    @assert γ > sqrt(4/3)
    @assert 0 ≤ β ≤ 0.5
    n = length(y0)
    tol = max(tol / norm(y0, Inf), eps(T)^0.7)
    y, H = T.(y0 ./ norm(y0, Inf)), T.(H0 ./ norm(H0, Inf))
    Z = large_enough_int_type(T)

    # FIXME I don't totally understand why we update (A,B) at every
    # FIXME step instead of accumulating a product of T's. Answer: the
    # FIXME rows of (A,B) are being swapped all the time, the matrix
    # FIXME that accumulates T's is actually just B.

    # FIXME In the paper, the matrix is T, but T is the name of the
    # FIXME floating point type here, so the matrix is t. I'll rename
    # FIXME it later.

    t = UnitLowerTriangular(Matrix{T}(I, n, n))

    A, B = Matrix{T}(I, n, n), Matrix{T}(I, n, n)
    A1 = Matrix{T}(undef, n, n)
    ymin_prev = T(+Inf)

    for itcount=1:maxiter
        # identify pairs to swap
        mp = sortperm(γ.^(1:n-1) .* abs.(diag(H)), rev=true)
        indices = Int[]
        for m in mp
            length(indices) > 2β*n && break
            (m ∈ indices || m+1 ∈ indices) && continue
            push!(indices, m, m+1)
            swap_rows!(m, y, H, A, B)
        end

        remove_corners!(H)
        reduce_H!(H, t)
        y = t' * y

        A1 .= A
        ldiv!(t, A)
        if norm(A,Inf) > ldexp(one(T), precision(T))
            A .= A1
            break
        end

        lmul!(t', B)

        # FIXME This bit doesn't seem to improve anything
        # ymin = minimum(abs.(y))
        # β = min(β0, ymin ≤ ymin_prev ? 2β : 0.5β)
        # ymin_prev = ymin

        # @printf "norm(A): %8.1e : %8.1e : %8.1e : %8.1e\n" Float64(norm(A, Inf)) Float64(opnorm(A,1)) Float64(cond(B,1)/opnorm(B,1)) Float64(norm(UnitLowerTriangular(B)\normalize(randn(n), 1), 1))
        done = minimum(abs.(y)) < tol || maximum(abs.(A)) > 1/tol
        done && break
    end

    verbose && @printf "%10s: M=%.1e ymin=%8.1e ymin/ymax=%.1e ABmax=%.1e/%.1e\n" T 1/maximum(abs.(diag(H))) minimum(abs.(y)) y_confidence(y) norm(A,Inf) norm(B,Inf)

    # NOTE invariant:
    # @assert big.(A) * big.(B') == I "in pslq_mp"

    A1, B1 = round.(Z, A), round.(Z, B)
    @assert A1 == A

    # NOTE When the iteration fails (as it says in the paper when it
    # NOTE loops selecting the pairs), the coefficients of A1 will
    # NOTE still grow. I don't understand why that is, tbh.

    @assert !isdiag(A1)
    success = !isdiag(A1) && A1 == A && B1 == B
    @assert success
    (success, A1, B1)
end

function withprecision(@nospecialize(f), ::Type{BigFloat}, newprec::Integer)
    oldprec = precision(BigFloat)
    setprecision(BigFloat, newprec)
    try
        f()
    finally
        setprecision(BigFloat, oldprec)
    end
end

halve_precision(@nospecialize(f), ::Type{BigFloat}; round::Integer=1) = withprecision(f, BigFloat, round*ceil(Int, precision(BigFloat)/2round))

function check_done(y::Vector{T}, H::Matrix{T}, A::Matrix;
             coefbound::Real,
             tol::Real) where {T<:AbstractFloat}
    reached_coefbound = 1/maximum(abs.(diag(H))) > coefbound
    finished = minimum(abs.(y)) < tol
    reached_tol = maximum(abs.(A)) > 1/tol
    return (reached_coefbound || finished || reached_tol,
            finished ? :finished : reached_tol ? :tol : reached_coefbound ? :coefbound : :working)
end

default_tol(::Type{BigFloat}) = eps(BigFloat)^0.9

function update_yHAB!(y,H,A,B,A1,B1)
    # FIXME I don't know why, but this invariant sometimes fails now
    # FIXME without affecting the final answer
    1/norm(diag(H), Inf) ≥ 0.1 || error("invariant failed: 1/norm(diag(H),∞) ≥ 1")
    # @assert norm(y, Inf) ≤ 1

    y .= B1 * y
    A .= A1 * A
    B .= B1 * B
    H .= qr((A1*H)').R'

    if 1/norm(diag(H), Inf) < 1
        @warn "Unexpected (harmless?) invariant violation: 1/norm(diag(H), Inf) = $(1/norm(diag(H), Inf)) < 1"
    end
    1/norm(diag(H), Inf) ≥ 0.1 || error("invariant failed: 1/norm(diag(H),∞) ≥ 1")
    # @assert norm(y, Inf) ≤ 1
    # @assert A * B' == I
end

"""
    pslq_recur(y, H; coefbound, verbose)

Run multi-pair pslq at different levels of precision, recurring into lower levels.
"""
function pslq_recur(y::Vector{BigFloat}, H::Matrix{BigFloat};
             coefbound::Real,
             level::Integer=0,
             verbose::Bool=true,
             tol::Real)
    @check coefbound ≥ 1
    n = length(y)
    @check size(H) == (n, n-1)

    # NOTE I think I figured it out: at intermediate precisions, it is
    # NOTE very important that the iteration doesn't "overfit" to the
    # NOTE roundoff errors present in the data. The same outcome you
    # NOTE get when it fails to find a relation (very large,
    # NOTE random-looking integers in A and B) can happen just the
    # NOTE same in the intermediate iteration leading to meaningless
    # NOTE (A1,B1). My solution is to use a lower tolerance.
    # FIXED I could also propagate tol explicitly to make sure we never go below it.
    tol = max(tol / norm(y, Inf), eps(BigFloat)^(level == 0 ? 0.9 : 0.7))

    y, H = copy(y), copy(H)
    coefbound *= norm(H, Inf)
    y ./= norm(y, Inf)
    H ./= norm(H, Inf)
    A, B = Matrix{BigInt}(I, n, n), Matrix{BigInt}(I, n, n)
    oldprec = precision(BigFloat)

    done = false
    for itcount=1:2*precision(BigFloat)
        if precision(BigFloat) ≤ 256
            success, A1, B1 = pslq_mp(Float64, y, H; tol=tol)
        else
            success, A1, B1 =
                (withprecision(BigFloat, 64 * round(Int, precision(BigFloat) / 2 / 64)) do
                 pslq_recur(y, H; coefbound=coefbound, level=level+1, tol=tol)
                 end)
        end
        if !success
            # NOTE This shouldn't happen a lot, but run_hard_tests triggers it
            @warn "!success"
            success, A1, B1 = pslq_mp(BigFloat, y, H; tol=tol)
        end

        @assert success
        update_yHAB!(y,H,A,B,A1,B1)
        verbose && @printf "[%4d/%3d]: M=%.1e ymin=%8.1e ymin/ymax=%.1e ABmax=%.1e/%.1e\n" precision(BigFloat) itcount 1/maximum(abs.(diag(H))) minimum(abs.(y)) y_confidence(y) norm(A,Inf) norm(B,Inf)
        done, _ = check_done(y, H, A; coefbound=coefbound, tol=tol)
        done && break
    end

    if !done
        display(A)
        display(B)
    end

    return (done && A != I, A, B)
end

function pslq(x::Array{T};
        γ::Real=1.25,
        coefbound::Real=floatmax(Float64),
        # threshold::Real=sqrt(eps(T)),
        tol::Real=eps(T)^0.9,
        verbose::Bool=true,
        rescale::Bool=false) where {T<:AbstractFloat}
    γ > sqrt(4/3) || error("invalid γ: !($γ > √4/3)")
    isbitstype(T) && @warn "Using a low-precision type T=$T in pslq is usually a mistake"

    if rescale
        @warn "Rescaling isn't actually a good idea, it seems!"
        x_orig = copy(x)
        x, x_scale = rescale_numbers(x)
    end

    n = length(x)
    A, B = Matrix{BigInt}(I, n, n), Matrix{BigInt}(I, n, n)
    y = x ./ norm(x)
    H = init_H(y, x)
    reduce_H(nothing, y, H, A, B)

    println(stderr, "Running PSLQ on $(length(x)) numbers with $(floor(Int,precision(BigFloat)/log2(10))) digits ($(precision(BigFloat)) bits), eps=$(@sprintf "%.4e" eps(BigFloat))")
    success, A1, B1 = pslq_recur(y, H; coefbound=coefbound, tol=eps(BigFloat)^0.9)
    @assert success
    update_yHAB!(y, H, A, B, A1, B1)

    _, status = check_done(y, H, A; coefbound=coefbound, tol=tol)

    relation = B[argmin(abs.(y)),:]
    if rescale
        unrescale_numbers!(relation, x_scale)
        x = x_orig
    end

    @show relation
    confidence = y_confidence(y)
    bitlength = sum(log2(max(1, abs(b))) for b in relation)
    accept_confidence = confidence < max(1e-100, tol^0.1)
    accept_bitlength = bitlength < 0.95 * log2(1/tol)
    @show Float64(bitlength)
    @show Float64(log2(1/tol))
    residual = dot(x, relation)
    # When testing the residual I think I can't assume that the
    # numbers x are accurate to all the given digits, so it's eps^0.5
    accept_residual = abs(residual) < eps(BigFloat)^0.5 * dot(abs.(x), abs.(relation))
    reliable = accept_confidence && accept_bitlength && accept_residual
    coefbound = Float64(1 / maximum(abs.(diag(H))))
    ans = IntegerRelation(reliable, status, x, relation, residual, coefbound, confidence)
    return ans
end

function test_pslq(m::Int, n::Int, prec::Union{Nothing,Integer}=nothing; kwargs...)
    setprecision(BigFloat, prec === nothing ? max(256, 40*m*n) : prec)
    x = 2^(big"1.0"/m) + 3^(big"1.0"/n)
    ans = pslq([x^i for i in 0:m*n]; kwargs...)
end

function runtests()
    @testset "pslq" begin
        @test test_pslq(2,3,64).relation == [-1, 36, -12, 6, 6, 0, -1]
        @test test_pslq(2,3,61).relation == [-1, 36, -12, 6, 6, 0, -1]
        @test test_pslq(2,2).reliable
        @test test_pslq(2,3).reliable
        @test test_pslq(2,4).reliable
        @test test_pslq(2,5).reliable
        @test test_pslq(2,6).reliable
        @test test_pslq(3,2).reliable
        @test test_pslq(3,3).reliable
        @test test_pslq(3,4).reliable
        @test test_pslq(3,5).reliable
        @test test_pslq(3,6).reliable
        @test test_pslq(4,2).reliable
        @test test_pslq(4,3).reliable
        @test test_pslq(4,4).reliable
        @test test_pslq(4,5).reliable
        @test test_pslq(4,6).reliable
        @test test_pslq(5,2).reliable
        @test test_pslq(5,3).reliable
        @test test_pslq(5,4).reliable
        @test test_pslq(5,5).reliable
        @test test_pslq(5,6).reliable
        @test test_pslq(6,2).reliable
        @test test_pslq(6,3).reliable
        @test test_pslq(6,4).reliable
        @test test_pslq(6,5).reliable
        @test test_pslq(6,6).reliable
    end
end

function bench_pslq()
    m, n = 8, 9
    setprecision(BigFloat, 64 * round(Int, 1200 * log2(10) / 64))
    x = 2^(big"1.0"/m) + 3^(big"1.0"/n)
    @time ans = pslq([x^i for i=0:m*n])
    display(ans)
    @testset "bench_pslq" begin
        @test ans.reliable
        # NOTE norm(ans.relation, Inf) == 2.571450903936e+12
        @test ans.relation == BigInt[-6049, 314928, -5773680, 50435136, -213194592, 391474944, -232581888, 19768320, -2304, 17496, 254356848, 21560180832, 224490118464, 375285778560, 77183767296, 664164864, 4608, 0, -20412, 2197672560, -423140585040, 2571450903936, -654822107712, 2847405312, -5376, 0, 0, 13608, 2925062928, 396356701248, 440657318016, 2932219008, 4032, 0, 0, 0, -5670, 901180080, -25623471312, 837561024, -2016, 0, 0, 0, 0, 1512, 63051696, 59488992, 672, 0, 0, 0, 0, 0, -252, 617328, -144, 0, 0, 0, 0, 0, 0, 24, 18, 0, 0, 0, 0, 0, 0, 0, -1]
    end
end

## }}}
## {{{ Various tests

"""
    run_hard_tests()

Run several tests of PSLQ on non-trivial problems with many digits. Sourced from the literature.

FIXME This isn't finished, add more examples.
"""
function run_hard_tests()
    @testset "pslq/hard" begin
        setprecision(BigFloat, floor(Int, 1000*log2(10) / 128) * 128)
        let α = big"1.176280818259917506544070338474035050693415806564695259830106347029688376548549962096830115581815394659207181379347681765627142993904690801894802523160077596570546062418875048962325907177334571567548096997559812677289401128791972456983735177677402547018406627860300931538336962607762681991597046834646632323107126561241422300847509827575317881149483168558685352483943243465069411489835604855670999941131248924651646199928894650701513975703312904628596531623403673087035935060381181206190204300924108552383983021499538728761959520567397158867506611293458075757439806512470474122134188106798291251486337803701296891625290465195911765657939458514754860892416697489181607020418800779527382130329176339909818744646931915542209759675861811791455556642983564965565963860450434719067256426322958012208664666341022433004123110637753690615489280426703078222637302770682587714578677367444532900397553752134439695945298280306674326082073171899434532475289250584159739440408254461851691378006656323698981137666295"
            @test !pslq([log(α); [log(α^k - 1) for k in [5,14,15,35,1,2,3,90,126,210,315,610]]]).reliable
            ans = pslq([log(α); [log(α^k - 1) for k in [5,14,15,35,1,2,3,90,126,210,315,630]]])
            display(ans)
            @test ans.reliable
        end
    end
end

## }}}

end
