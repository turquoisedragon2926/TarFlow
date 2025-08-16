using Random
using Images

abstract type AbstractDataset end

"""
Random image dataset producing (B, C, H, W) samples in [0,1].
"""
struct RandomDataset <: AbstractDataset
    nc::Int
    nx::Int
    ny::Int
end

function get_data(dataset::RandomDataset, sample_index::Int; batch::Int=8)
    Random.seed!(sample_index)
    return rand(Float32, batch, dataset.nc, dataset.nx, dataset.ny)
end

struct SingleCatDataset <: AbstractDataset
    img::AbstractArray{<:Real,4}
end

function SingleCatDataset()
    img = load("cat.jpg")
    img = imresize(img, (64, 64))
    img_array = Float32.(channelview(img))
    return SingleCatDataset(reshape(img_array, 1, size(img_array)...))
end

function get_data(dataset::SingleCatDataset, sample_index::Int; batch::Int=8)   
    return repeat(dataset.img, batch, 1, 1, 1)
end
