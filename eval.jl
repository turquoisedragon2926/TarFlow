# To visualize samples
using Images, Plots
using JLD2
using Flux
using LinearAlgebra
using Random
using ParametricOperators
using ProgressMeter

include("data.jl")
include("utils.jl")
include("attention.jl")
include("block.jl")
include("model.jl")
include("utils.jl")


data = JLD2.load("ckpts/checkpoint_1.jld2")
model = data["model"]
θ = data["θ_cpu"]

dataset = VelocityModelDataset("data/x_data_no_zero.jld2")
x = get_data(dataset, 1; batch=6)
x = x .+ 0.2f0 .* randn(Float32, size(x))
x = clamp.(x, -5f0, 5f0)
heatmap(x[1, 1, :, :]')

z, outs, ld = forward(model, θ, x)
heatmap(z[1, 1, :, :]')

x_back = backward(model, θ, z[1:1, :, :, :]); # TODO: fix to work with multiple batches
heatmap(x_back[end][1, 1, :, :]')

norm(x_back[end][1, 1, :, :] .- x[1, 1, :, :])
