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
x = x .+ 0.1f0 .* randn(Float32, size(x))
x = clamp.(x, -1f0, 1f0)
colorview(RGB, x[1, :, :, :])

for num in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    data = JLD2.load("ckpts/checkpoint_$(num).jld2")
    model = data["model"]
    θ = data["θ_cpu"]
    z, outs, ld = forward(model, θ, x)
    push!(broken_down, z)
end

colorview(RGB, broken_down[1][2, :, :, :])
colorview(RGB, broken_down[2][2, :, :, :])
colorview(RGB, broken_down[3][2, :, :, :])
colorview(RGB, broken_down[4][2, :, :, :])
colorview(RGB, broken_down[5][2, :, :, :])
colorview(RGB, broken_down[6][2, :, :, :])
colorview(RGB, broken_down[7][2, :, :, :])
colorview(RGB, broken_down[8][2, :, :, :])
colorview(RGB, broken_down[9][2, :, :, :])
colorview(RGB, broken_down[10][2, :, :, :])

for i in 1:length(broken_down)
    println("energy: ", mean(broken_down[i][1, :, :, :].^2))
end

data = JLD2.load("ckpts/checkpoint_500.jld2")
model = data["model"]
θ = data["θ_cpu"]
losses = data["losses"]
l2_terms = data["l2_terms"]
ld_terms = data["ld_terms"]

plot(l2_terms, label="l2_terms", title="Losses", xlabel="Step", ylabel="Loss")
plot!(ld_terms, label="ld_terms")
plot!(losses, label="losses")
savefig("losses.png")

# Load a new image
x = get_data(SingleCatDataset(), 1; batch=8)
x = x .+ 0.1f0 .* randn(Float32, size(x))
x = clamp.(x, -1f0, 1f0)
colorview(RGB, x[1, :, :, :])

# Forward pass
z, outs, ld = forward(model, θ, x)
mean(z.^2)
colorview(RGB, z[1, :, :, :])

# Sample random noise to compare energy to noise
t = randn(Float32, size(z[1:1, :, :, :]))
mean(t.^2)
colorview(RGB, t[1, :, :, :])

# backward pass
test2 = backward(model, θ, t[1:1, :, :, :])

colorview(RGB, test2[1][1, :, :, :])
colorview(RGB, test2[2][1, :, :, :])
colorview(RGB, test2[3][1, :, :, :])
colorview(RGB, test2[4][1, :, :, :])
colorview(RGB, test2[5][1, :, :, :])

# Plot difference
colorview(RGB, test2[5][1, :, :, :] - x[1, :, :, :])

# Sanity check invertibility
xtest = x[1:1, :, :, :]
colorview(RGB, xtest[1, :, :, :])

ztest = forward(model, θ, xtest)[1]
mean(ztest[1, :, :, :].^2)
colorview(RGB, ztest[1, :, :, :])

xback = backward(model, θ, ztest)
colorview(RGB, xback[end][1, :, :, :])

norm(xback[end][1, :, :, :] .- xtest[1, :, :, :])
colorview(RGB, xback[end][1, :, :, :] .- xtest[1, :, :, :])
