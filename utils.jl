using LinearAlgebra

struct PatchConfig
    original_nx::Int
    original_ny::Int
    patch_size::Int
    padded_nx::Int
    padded_ny::Int
end

"""
Convert (B, C, H, W) into sequence of patches (B, T, C*ps^2) with stride=patch_size.
We assume H and W are divisible by patch_size.
"""
function patch_data(config::PatchConfig, sample::AbstractArray{<:Real,4})
    B, C, H, W = size(sample)
    ps = config.patch_size
    @assert H % ps == 0 && W % ps == 0
    Th = H ÷ ps
    Tw = W ÷ ps
    T = Th * Tw
    out = Array{eltype(sample)}(undef, B, T, C*ps*ps)
    t = 1
    @inbounds for i in 1:ps:H, j in 1:ps:W
        # block: (B, C, ps, ps)
        blk = sample[:, :, i:(i+ps-1), j:(j+ps-1)]
        out[:, t, :] = reshape(blk, B, :)
        t += 1
    end
    return out
end

"""
Inverse of patch_data to go back to (B, C, H, W).
"""
function unpatch_data(config::PatchConfig, patches::AbstractArray{<:Real,3})
    B, T, Cps = size(patches)
    ps = config.patch_size
    C = Cps ÷ (ps*ps)
    H = config.padded_nx
    W = config.padded_ny
    Th = H ÷ ps
    Tw = W ÷ ps
    @assert T == Th*Tw
    out = zeros(eltype(patches), B, C, H, W)
    t = 1
    @inbounds for i in 1:ps:H, j in 1:ps:W
        blk = reshape(patches[:, t, :], B, C, ps, ps)
        out[:, :, i:(i+ps-1), j:(j+ps-1)] .= blk
        t += 1
    end
    return out[:, :, 1:config.original_nx, 1:config.original_ny]
end

"""
Create a lower triangular mask for a given sequence length.
"""
tril_mask(n::Int) = (tril(ones(Bool, n, n)) .== 1)
