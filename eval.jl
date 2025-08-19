# To visualize samples
using Images, Plots, Statistics
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

plots_dir = "plots_64"
data = JLD2.load("vel_ckpts_new/checkpoint_20.jld2")
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

data = JLD2.load("vel_ckpts_new/checkpoint_30.jld2")

samples = data["samples"];
hp = data["hp"];

for sample_idx in 1:10
    anim = @animate for i in 1:size(samples[sample_idx], 1)
        heatmap(samples[sample_idx][i][1, 1, :, :]', title="Flow $(i)")
    end
    gif(anim, "$(plots_dir)/samples_$(sample_idx).gif", fps=1)
end

# Create animation of samples
anim = @animate for sample_idx in 1:10
    heatmap(samples[sample_idx][end][1, 1, :, :]', title="Sample $(sample_idx)")
end
gif(anim, "$(plots_dir)/all_samples.gif", fps=1)

losses = data["losses"]
l2_terms = data["l2_terms"]
ld_terms = data["ld_terms"]

plot(l2_terms, label="l2_terms", title="Losses", xlabel="Step", ylabel="Loss")
plot!(ld_terms, label="ld_terms")
plot!(losses, label="losses")
savefig("$(plots_dir)/losses.png")

# Create combined plot with samples and hyperparameters
param_text = """
Hyperparameters:
epochs: 20
steps_per_epoch: $(hp.steps_per_epoch)
batch: $(hp.batch)
channels: $(hp.channels)
patch: $(hp.patch)
H: $(hp.H)
W: $(hp.W)
C: $(hp.C)
save_every: $(hp.save_every)
attn_layers_per_block: $(hp.attn_layers_per_block)
num_blocks: $(hp.num_blocks)
head_dim: $(hp.head_dim)
expansion: $(hp.expansion)
noise_std: $(hp.noise_std)
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
gif(anim, "$(plots_dir)/details.gif", fps=1)

dataset = VelocityModelDataset("data/x_data_no_zero.jld2", subsample_size_x=8, subsample_size_y=4)
x = get_data(dataset, 1; batch=10)
anim = @animate for i in 1:size(x, 1)
    heatmap(x[i, 1, :, :]', title="Sample $(i)")
end
gif(anim, "$(plots_dir)/x_data.gif", fps=1)
