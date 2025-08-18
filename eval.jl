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


data = JLD2.load("vel_ckpts/checkpoint_10.jld2")
model = data["model"];
θ = data["θ_cpu"];

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

samples = JLD2.load("vel_ckpts_new/checkpoint_15.jld2", "samples");
using Statistics

for sample_idx in 1:10
    anim = @animate for i in 1:size(samples[sample_idx], 1)
        heatmap(samples[sample_idx][i][1, 1, :, :]', title="Flow $(i)")
    end
    gif(anim, "plots/samples_$(sample_idx).gif", fps=1)
end

# Create animation of samples
anim = @animate for sample_idx in 1:10
    heatmap(samples[sample_idx][end][1, 1, :, :]', title="Sample $(sample_idx)")
end
gif(anim, "plots/all_samples.gif", fps=1)

losses = data["losses"]
l2_terms = data["l2_terms"]
ld_terms = data["ld_terms"]

plot(l2_terms, label="l2_terms", title="Losses", xlabel="Step", ylabel="Loss")
plot!(ld_terms, label="ld_terms")
plot!(losses, label="losses")
savefig("plots/losses.png")

# Create combined plot with samples and hyperparameters
param_text = """
Hyperparameters:
epochs: 10
steps_per_epoch: 20
batch: 32
channels: 256 
patch: 64
H: 512
W: 256
C: 1
save_every: 10
attn_layers_per_block: 8
num_blocks: 32
head_dim: 8
expansion: 4
noise_std: 0.2
"""

# Create 3-panel plot
l = @layout [a b c]
p1 = heatmap(samples[1][end][1, 1, :, :]', title="Sample Animation")
p2 = plot(title="Configuration", grid=false, showaxis=false, ticks=false)
annotate!(p2, 0.5, 0.5, text(param_text, :left, 10))
p3 = plot(l2_terms, label="l2_terms", title="Losses", xlabel="Step", ylabel="Loss")
plot!(p3, ld_terms, label="ld_terms")
plot!(p3, losses, label="losses")

# Animate the left panel while keeping right panels static
anim = @animate for sample_idx in 1:10
    p1 = heatmap(samples[sample_idx][end][1, 1, :, :]', title="Sample $(sample_idx)")
    plot(p1, p2, p3, layout=l, size=(1500,400))
end
gif(anim, "plots/details.gif", fps=1)

dataset = VelocityModelDataset("data/x_data_no_zero.jld2", subsample_size_x=8, subsample_size_y=4)
x = get_data(dataset, 1; batch=10)
anim = @animate for i in 1:size(x, 1)
    heatmap(x[i, 1, :, :]', title="Sample $(i)")
end
gif(anim, "plots/x_data.gif", fps=1)
