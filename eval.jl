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

broken_down = []
x = get_data(SingleCatDataset(), 1; batch=8)
x = x .+ 0.05f0 .* randn(Float32, size(x))
x = clamp.(x, -1f0, 1f0)

for num in [100, 200, 300, 400, 500]
    data = JLD2.load("ckpts/checkpoint_$(num).jld2")
    samples = data["samples"];
    sample_idx = 5
    colorview(RGB, samples[sample_idx][2][1, :, :, :])

    plot(data["l2_terms"], label="l2")
    plot!(data["ld_terms"], label="ld")
    plot!(data["losses"], label="loss")

    # Look at a few forward passes
    model = data["model"]
    θ = data["θ"]

    z, outs, ld = forward(model, θ, x)
    push!(broken_down, z)
end

colorview(RGB, broken_down[1][2, :, :, :])
colorview(RGB, broken_down[5][2, :, :, :])

data = JLD2.load("ckpts/checkpoint_500.jld2")
model = data["model"]
θ = data["θ"]

x = get_data(SingleCatDataset(), 1; batch=8)
x = x .+ 0.05f0 .* randn(Float32, size(x))

x = clamp.(x, -1f0, 1f0)
z, outs, ld = forward(model, θ, x)

colorview(RGB, outs[5][1, :, :, :])
mean(z[1, :, :, :].^2)

t = randn(Float32, size(z[1, :, :, :]))
colorview(RGB, t)
mean(t.^2)

test = backward(model, θ, z[1:1, :, :, :])
colorview(RGB, test[5][1, :, :, :])
