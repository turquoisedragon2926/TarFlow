using Flux
using ParametricOperators
using ChainRulesCore
import ChainRulesCore: rrule, NoTangent, ZeroTangent

import ParametricOperators: init!, Parameters
include("attention.jl")
include("utils.jl")

"""
Closed-form autoregressive affine transform with logdet and custom rrule.
Masks the first token to keep it unchanged (triangular Jacobian).
Inputs: x,a,b are (B,T,C); mask (B,T,1) with mask[:,1,:]=0, others 1.
Returns (y, logdet::Vector{eltype(x)} of length B).
"""
function affine_transform(x::AbstractArray{<:Real,3},
                          a::AbstractArray{<:Real,3},
                          b::AbstractArray{<:Real,3})
    # clamp the log-scale to avoid exp overflow/underflow during training
    a_clamped = clamp.(a, -20f0, 20f0)
    E = exp.(-a_clamped)
    y = ((x .- b) .* E)
    # log|det J| = -sum(a) over masked positions and channels, per batch
    ld = -mean(a_clamped; dims=(2,3)) # NOTE: maybe mean to stabilize?
    ld = vec(ld)  # (B,)
    return y, ld
end

# function rrule(::typeof(affine_transform),
#                               x::AbstractArray{<:Real,3},
#                               a::AbstractArray{<:Real,3},
#                               b::AbstractArray{<:Real,3},
#                               mask::AbstractArray{<:Real,3})
#     y, ld = affine_transform(x, a, b, mask)
#     function pullback(ȳ_ld)
#         ȳ, ȴ = ȳ_ld
#         # Ensure shapes
#         if ȴ isa Number
#             ȴ = fill(ȴ, size(x,1))
#         end
#         ȴ = reshape(ȴ, size(x,1), 1, 1)
#         E = exp.(-a)
#         masked = mask
#         # dy/dx = masked .* E + (1-masked)
#         gx = ȳ .* (masked .* E .+ (1 .- masked))
#         # dy/da = masked .* (-(x-b) .* E)
#         ga = ȳ .* (masked .* (-(x .- b) .* E))  .+  (-masked) .* ȴ
#         # dy/db = masked .* ( -E )
#         gb = ȳ .* (masked .* (-E))
#         gmask = ZeroTangent()
#         return NoTangent(), gx, ga, gb, gmask
#     end
#     return (y, ld), pullback
# end

"""
A transformer-style block group with input/output projections and positional embedding.
Works on sequences shaped (B, T, C). Returns (y, logdet).
"""
struct MetaBlock
    proj_in::ParMatrix
    proj_out::ParMatrix
    pos_embed::ParMatrix        # added as x + pos_embed(θ)
    attn_blocks::Vector{AttentionBlock}
    nvp::Bool
    permutation::Vector{Int}
    inv_permutation::Vector{Int}
end

function MetaBlock(in_channels::Int, channels::Int, num_patches::Int;
                   attn_layers_per_block::Int=1, head_dim::Int=64, expansion::Int=4, nvp::Bool=true, permutation::Vector{Int}=collect(1:num_patches))
    proj_in = ParMatrix(Float32, in_channels, channels)
    out_dim = nvp ? 2*in_channels : in_channels
    proj_out = ParMatrix(Float32, channels, out_dim)
    pos_embed = ParMatrix(Float32, num_patches, channels)
    blocks = [AttentionBlock(channels, head_dim, expansion) for _ in 1:attn_layers_per_block]
    inv_permutation = invperm(permutation)
    return MetaBlock(proj_in, proj_out, pos_embed, blocks, nvp, permutation, inv_permutation)
end

function init!(m::MetaBlock, θ::Parameters)
    init!(m.proj_in, θ)
    init!(m.proj_out, θ; w_init_fn=zeros)
    # Small positional embeddings like python (std ~1e-2)
    init!(m.pos_embed, θ; w_init_fn=(T,mn,nn)->(0.01f0 .* randn(T, mn, nn)))
    for b in m.attn_blocks
        init!(b, θ)
    end
    return θ
end

function forward(m::MetaBlock, θ::Parameters, x::AbstractArray{<:Real,3})
    # x: (B, T, Cin)
    B, T, Cin = size(x)
    x_in = x[:, m.permutation, :]
    
    x = x_in * m.proj_in(θ)               # (B, T, C)
    x = x + m.pos_embed(θ)
    attn_mask = tril_mask(T)
    for b in m.attn_blocks
        x = forward(b, θ, x; mask=attn_mask)
    end
    h = x * m.proj_out(θ)                 # (B, T, out_dim)

    # shift h to the right by one token
    h = cat(zeros(eltype(h), B, 1, size(h,3)), h[:,1:end-1,:], dims=2)

    if m.nvp
        a = @view h[:,:,1:Cin]
        b = @view h[:,:,(Cin+1):Cin*2]
    else
        a = zeros(eltype(h), B, T, Cin)
        b = h
    end
    y, ld = affine_transform(x_in, a, b)
    
    # Apply inverse permutation at end
    y = y[:, m.inv_permutation, :]
    
    return y, ld
end

function backward(m::MetaBlock, θ::Parameters, x::AbstractArray{<:Real,3})
    # x is (B, T, Cin). At entry, x holds z (latents) for this block.
    # We will reconstruct in place: token 1 is identity; tokens 2..T get inverted.
    B, T, Cin = size(x)

    pos = ParametricOperators.params(m.pos_embed(θ))        # (T, C)
    Win = m.proj_in(θ)          # (Cin, C)
    Wout = m.proj_out(θ)        # (C, 2Cin) if nvp else (C, Cin)
    

    # loop over prefixes; at step i we produce params for token i+1 from prefix 1:i
    @inbounds for i in 1:(T-1)
        # prefix in *data space*: after step i-1, x[:, 1:i, :] are already reconstructed
        # encoder + pos
        h = x[:, 1:i, :] * Win                          # (B, i, C)
        h = h .+ reshape(pos[1:i, :], 1, i, size(pos, 2))                # broadcast over batch

        # causal attention over the prefix        
        for b in m.attn_blocks
            h = forward(b, θ, h; mask=tril_mask(i))       # (B, i, C)
        end

        # head to params
        h = h * Wout                              # (B, i, 2Cin) or (B, i, Cin)
        hlast = @view h[:, i, :]                  # (B, 2Cin) or (B, Cin)


        if m.nvp
            a_last = reshape(hlast[:, 1:Cin],  B, 1, Cin)        # logσ_{i+1}
            b_last = reshape(hlast[:, Cin+1:2Cin], B, 1, Cin)    # μ_{i+1}
        else
            a_last = zeros(eltype(h), B, 1, Cin)                       # logσ=0
            b_last = reshape(hlast, B, 1, Cin)                         # μ
        end
        # invert affine for token i+1: x_{i+1} = z_{i+1} * exp(a) + b
        # clamp exponent for numerical stability during sampling
        exp_a = exp.(clamp.(a_last, -20f0, 20f0))
        x[:, i+1:i+1, :] .= x[:, i+1:i+1, :] .* exp_a .+ b_last
    end

    return x
end
