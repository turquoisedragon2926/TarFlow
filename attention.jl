using Flux
using Statistics
using ParametricOperators

import ParametricOperators: init!, Parameters

struct Attention
    qkv::ParMatrix
    proj::ParMatrix
    num_heads::Int
    head_dim::Int
end

"""
Create a multi-head attention module.
- in_channels: embedding size C
- head_dim: per-head dimension D; num_heads = C ÷ D
"""
function Attention(in_channels::Int, head_dim::Int)
    @assert in_channels % head_dim == 0
    num_heads = in_channels ÷ head_dim
    qkv = ParMatrix(Float32, in_channels, 3 * in_channels)
    proj = ParMatrix(Float32, in_channels, in_channels)
    return Attention(qkv, proj, num_heads, head_dim)
end

function init!(A::Attention, θ::Parameters)
    init!(A.qkv, θ)
    # zero-init output projection like python for stability
    init!(A.proj, θ; w_init_fn=zeros)
    return θ
end

"""
Simple, stateless LayerNorm over the last dimension (no affine params for now).
"""
layernorm(x::AbstractArray) = begin
    μ = mean(x; dims=3)
    σ2 = mean((x .- μ) .^ 2; dims=3)
    (x .- μ) ./ sqrt.(σ2 .+ 1f-5)
end

"""
Scaled dot-product attention using ParametricOperators.
Input x: (B, T, C). Returns (B, T, C).
"""

function forward(A::Attention, θ::Parameters, x::AbstractArray{<:Real,3};
                 mask::AbstractMatrix{<:Real}=ones(Bool, size(x,2), size(x,2)),
                 temp::Float32=1.0f0)

    B, T, C = size(x)
    x̂ = layernorm(x)  # assumes LN over last dim

    H = A.num_heads
    D = A.head_dim
    @assert C == H*D "C must equal num_heads*head_dim"

    # (B,T,C) -> (B*T,C) * (C,3C) -> (B,T,3,H,D)
    qkv  = reshape(reshape(x̂, :, C) * A.qkv(θ), B, T, 3, H, D)

    q = @view qkv[:, :, 1, :, :]
    k = @view qkv[:, :, 2, :, :]
    v = @view qkv[:, :, 3, :, :]

    # match python sqrt_scale**2 / temp where sqrt_scale = D**(-0.25)
    s = Float32(D^-0.5 / temp)

    # Flatten heads: (B*H, T, D)
    q2 = reshape(q, B*H, T, D)
    k2 = reshape(k, B*H, T, D)
    v2 = reshape(v, B*H, T, D)

    # q2,k2,v2 are (BH, T, D)
    qBL = permutedims(q2, (2,3,1))            # (T, D, BH)
    kBL = permutedims(k2, (2,3,1))            # (T, D, BH)

    # scores: (T,T,BH) = (T,D,BH) × (D,T,BH)
    scoresBL = batched_mul(qBL, permutedims(kBL, (2,1,3))) .* s  # (T, T, BH)

    # mask (T,T) → (T,T,1) and broadcast; no mutation
    if !isnothing(mask)
        @assert size(mask) == (size(scoresBL,1), size(scoresBL,2))
        m = reshape(mask .!= 0, size(scoresBL,1), size(scoresBL,2), 1)
        neginf = convert(eltype(scoresBL), -Inf)     # e.g., -Inf32 on Float32, works on GPU too
        scoresBL = ifelse.(m, scoresBL, neginf)      # elementwise selection, no mutation
    end

    # softmax over keys (the 2nd dim in (Tq,Tk,BH))
    scoresBL = Flux.softmax(scoresBL; dims=2)  # (T, T, BH)

    # context: (T,D,BH) = (T,T,BH) × (T,D,BH)
    vBL  = permutedims(v2, (2,3,1))            # (T, D, BH)
    ctxBL = batched_mul(scoresBL, vBL)         # (T, D, BH)

    # back to (BH, T, D)
    ctx = permutedims(ctxBL, (3,1,2))          # (BH, T, D)


    # Merge heads and project
    x_out = reshape(ctx, B, T, H, D)
    x_out = reshape(x_out, B, T, C)
    x_out = reshape(reshape(x_out, :, C) * A.proj(θ), B, T, :)
    return x_out
end

struct MLP
    fc1::ParMatrix
    fc2::ParMatrix
    expansion::Int
end

function MLP(channels::Int, expansion::Int)
    fc1 = ParMatrix(Float32, channels, channels * expansion)
    fc2 = ParMatrix(Float32, channels * expansion, channels)
    return MLP(fc1, fc2, expansion)
end

function init!(m::MLP, θ::Parameters)
    init!(m.fc1, θ)
    init!(m.fc2, θ)
    return θ
end

function forward(m::MLP, θ::Parameters, x::AbstractArray{<:Real,3})
    x̂ = layernorm(x)
    h = x̂ * m.fc1(θ)
    h = Flux.gelu.(h)
    y = h * m.fc2(θ)
    return y
end

struct AttentionBlock
    attn::Attention
    mlp::MLP
end

function AttentionBlock(channels::Int, head_dim::Int, expansion::Int)
    return AttentionBlock(Attention(channels, head_dim), MLP(channels, expansion))
end

function init!(b::AttentionBlock, θ::Parameters)
    init!(b.attn, θ)
    init!(b.mlp, θ)
    return θ
end

function forward(b::AttentionBlock, θ::Parameters, x::AbstractArray{<:Real,3}; mask::AbstractMatrix{<:Real}=ones(Bool, size(x,2), size(x,2)))
    x = x .+ forward(b.attn, θ, x; mask=mask)
    x = x .+ forward(b.mlp, θ, x)
    return x
end
