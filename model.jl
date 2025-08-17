using Flux
using ParametricOperators

import ParametricOperators: init!, Parameters
include("block.jl")
include("utils.jl")

struct TarFlow
    patch_config::PatchConfig
    blocks::Vector{MetaBlock}
    nvp::Bool
end

function TarFlow(in_channels::Int, H::Int, W::Int, patch_size::Int,
                 channels::Int, num_blocks::Int; attn_layers_per_block::Int=1, head_dim::Int=64, expansion::Int=4, nvp::Bool=true)
    num_patches = (H*W) ÷ patch_size^2
    cfg = PatchConfig(H, W, patch_size, H, W)
    blocks = [MetaBlock(in_channels * patch_size^2, channels, num_patches;
                        attn_layers_per_block=attn_layers_per_block, 
                        head_dim=head_dim, expansion=expansion, nvp=nvp,
                        permutation=(k % 2 == 1 ? collect(1:num_patches) : collect(num_patches:-1:1)))
              for k in 1:num_blocks]
    return TarFlow(cfg, blocks, nvp)
end

function init!(m::TarFlow, θ::Parameters)
    for b in m.blocks
        init!(b, θ)
    end
    return θ
end

"""
Forward over image batch x of shape (B, C, H, W).
Returns (z, outputs, logdet) where z has same shape as x and logdet is (B,).
"""
function forward(model::TarFlow, θ::Any, x::AbstractArray{<:Real,4})
    patches = patch_data(model.patch_config, x)  # (B, T, Cin)
    attn_mask = tril_mask(size(patches,2))
    outputs = [x]
    lds = Float32[]
    for block in model.blocks
        patches, ld = forward(block, θ, patches; attn_mask=attn_mask)
        push!(outputs, unpatch_data(model.patch_config, patches))
        push!(lds, ld...)
    end
    z = unpatch_data(model.patch_config, patches)
    # Sum per-batch logdets across blocks
    logdets = sum(reshape(lds, :, length(lds)÷size(x,1)); dims=2)
    logdets = vec(logdets)
    return z, outputs, logdets
end

function backward(model::TarFlow, θ::Any, x::AbstractArray{<:Real,4})
    B, C, H, W = size(x)

    patches = patch_data(model.patch_config, x)  # (B, T, Cin)
    var = ones(Float32, size(patches, 2), C*model.patch_config.patch_size^2)

    if !model.nvp
        z2 = mean(patches.^2, dims=1)[1,:,:]  # Mean across batch dimension
        var = var * 0.99f0 + z2 * 0.01f0  # Exponential moving average with 0.01 learning rate
    end

    outputs = [x]
    patches = reshape(reshape(patches, :, size(patches, 3)) .* sqrt.(var), B, :, size(patches, 3))

    for block in reverse(model.blocks)
        patches = backward(block, θ, patches)
        push!(outputs, unpatch_data(model.patch_config, patches))
    end
    return outputs
end

# TODO: support forward and backward on GPU
# TODO: support gradient for patching operation: low priority
