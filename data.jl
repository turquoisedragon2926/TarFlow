using Random
using Images
using JLD2

abstract type AbstractDataset end

"""
Random image dataset producing (B, C, H, W) samples in [0,1].
"""
struct RandomDataset <: AbstractDataset
    nc::Int
    nx::Int
    ny::Int
end

function get_data(dataset::RandomDataset, batch_index::Int; batch::Int=8)
    Random.seed!(batch_index)
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

function get_data(dataset::SingleCatDataset, batch_index::Int; batch::Int=8)   
    return repeat(dataset.img, batch, 1, 1, 1)
end

struct VelocityModelDataset <: AbstractDataset
    x_data::AbstractArray{<:Real,4}
end

function VelocityModelDataset(path)
    x_data = JLD2.jldopen(path, "r")["x_data"][:,end:-1:1, :, 1:1000];
    x_data = permutedims(x_data, (4, 3, 1, 2));
    x_data = x_data .- mean(x_data, dims=(4)) ./ std(x_data, dims=(4))
    return VelocityModelDataset(x_data)
end

function get_data(dataset::VelocityModelDataset, batch_index::Int; batch::Int=8)
    return dataset.x_data[1+ batch*(batch_index-1):batch*batch_index, :, :, :]
end
