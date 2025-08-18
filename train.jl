using Flux
using LinearAlgebra
using JLD2
using Random
using ParametricOperators
using ProgressMeter
import ParametricOperators: gpu, cpu

include("data.jl")
include("utils.jl")
include("attention.jl")
include("block.jl")
include("model.jl")

@kwdef struct HyperParams
    epochs::Int=2 # number of epochs to train for
    steps_per_epoch::Int=10 # number of kinda batches ? idk, lets change this
    batch::Int=8 # number of images in a batch
    C::Int=3 # number of channels in the input image per pixel
    H::Int=64 # number of pixels per dimension
    W::Int=64 # number of pixels per dimension
    patch::Int=8 # patch size per dimension
    channels::Int=64 # number of channels per pixel
    head_dim::Int=16 # number of channels per attention head
    num_blocks::Int=2 # number of meta blocks in the model
    expansion::Int=4 # expansion factor for the MLP in the attention block
    save_every::Int=10 # save checkpoint every n epochs
    attn_layers_per_block::Int=2 # number of attention layers per meta block
    noise_std::Float32=0.05f0 # Gaussian input noise std
    clip_norm::Float32=5.0f0  # gradient clipping (global norm)
    warmup_steps::Int=500     # LR warmup steps
end

function main(hp::HyperParams=HyperParams(), dataset::AbstractDataset=SingleCatDataset(); device=cpu, ckpts_dir="ckpts")
    Random.seed!(123) # To recreate model weights & load ckpts
    model = TarFlow(hp.C, hp.H, hp.W, hp.patch, hp.channels, hp.num_blocks; attn_layers_per_block=hp.attn_layers_per_block, head_dim=hp.head_dim, expansion=hp.expansion)
    θ = Dict{ParametricOperators.ParOperator,Any}()
    init!(model, θ)
    θ = θ |> device

    base_lr = 1e-4
    opt = Flux.Optimisers.AdamW(base_lr, (0.9, 0.95), 1e-4)
    state = Flux.Optimisers.setup(opt, θ)
    
    # Create cosine learning rate schedule with warmup
    total_steps = hp.epochs * hp.steps_per_epoch
    function lr_schedule(t)
        if t <= hp.warmup_steps
            return base_lr * (t / max(hp.warmup_steps, 1))
        else
            # cosine from base_lr to 1e-6 after warmup
            t2 = t - hp.warmup_steps
            T2 = max(total_steps - hp.warmup_steps, 1)
            return 1e-6 + 0.5 * (base_lr - 1e-6) * (1 + cos(π * t2 / T2))
        end
    end
    
    losses = []
    l2_terms = []
    ld_terms = []
    meter = Progress(hp.epochs * hp.steps_per_epoch)
    
    step_count = 0
    for epoch in 1:hp.epochs
        total = 0.0
        for step in 1:hp.steps_per_epoch
            step_count += 1
            
            x = get_data(dataset, step; batch=hp.batch)
            # Add Gaussian input noise and clamp to [0,1]
            x = x .+ hp.noise_std .* randn(Float32, size(x))
            x = Float32.(clamp.(x, -5f0, 5f0))
            p = patch_data(model.patch_config, x) |> device
            attn_mask = tril_mask(size(p,2)) |> device
            global ld_term = 0.0f0
            global l2_term = 0.0f0
            # Compute in patch space to avoid non-differentiable unpatch
            loss, grads = Flux.withgradient(θ, p) do θp, p
                ld_acc = 0.0f0
                for b in model.blocks
                    p, ld = forward(b, θp, p; attn_mask=attn_mask)
                    ld_acc += sum(ld)
                end
                global l2_term = 0.5f0 * mean(p.^2)
                global ld_term = ld_acc/length(model.blocks)
                l2_term - ld_term
            end
            push!(losses, loss)
            push!(l2_terms, l2_term)
            push!(ld_terms, ld_term)
            total += loss
            # Gradient clipping and LR scheduling via gradient scaling
            gdict = grads[1]
            # compute global norm
            total_sq = 0.0
            for gv in values(gdict)
                if gv isa AbstractArray
                    total_sq += sum(abs2, gv)
                end
            end
            gnorm = sqrt(total_sq + 1e-12)
            clip_scale = gnorm > hp.clip_norm ? (hp.clip_norm / gnorm) : 1.0
            current_lr = lr_schedule(step_count)
            lr_scale = current_lr / base_lr
            scale = clip_scale * lr_scale
            if scale != 1.0
                for (k, gv) in gdict
                    if gv isa AbstractArray
                        gdict[k] = gv .* scale
                    end
                end
            end
            state, θ = Flux.Optimisers.update(state, θ, gdict)
            θ = Dict{ParOperator, Any}(θ)

            meter.desc = "epoch=$epoch step=$step loss=$(round(total/hp.steps_per_epoch, digits=4))"
            next!(meter)
        end
        if epoch % hp.save_every == 0
            samples = []
            θ_cpu = θ |> cpu
            for _ in 1:10
                noise = randn(Float32, 1, hp.C, hp.H, hp.W)
                x_recon = backward(model, θ_cpu, noise)
                push!(samples, x_recon)
            end
            @info "Saving checkpoint"
            JLD2.@save "$(ckpts_dir)/checkpoint_$(epoch).jld2" model θ_cpu samples losses l2_terms ld_terms hp
        end
    end
end


# #### To Train on Single Cat Dataset ####
# hp = HyperParams(epochs=500, 
#                 steps_per_epoch=2, 
#                 batch=32, 
#                 channels=128,
#                 H=64, 
#                 C=3, 
#                 save_every=50,
#                 attn_layers_per_block=4,
#                 num_blocks=4,
#                 head_dim=8,
#                 expansion=4,
#                 noise_std=0.1)
# dataset = SingleCatDataset()
# main(hp, dataset; device=gpu)

# #### To Train on Velocity Model Dataset ####
# dataset = VelocityModelDataset("data/x_data_no_zero.jld2")
# hp = HyperParams(epochs=500,
#                 steps_per_epoch=20,
#                 batch=32, 
#                 channels=256,
#                 patch=64,
#                 H=512, 
#                 W=256, 
#                 C=1, 
#                 save_every=10,
#                 attn_layers_per_block=8,
#                 num_blocks=32,
#                 head_dim=8,
#                 expansion=4,
#                 noise_std=0.2)
# main(hp, dataset; device=gpu, ckpts_dir="vel_ckpts")


#### To Train on Velocity Model Dataset ####
dataset = VelocityModelDataset("data/x_data_no_zero.jld2", subsample_size_x=8, subsample_size_y=4)
hp = HyperParams(epochs=500,
                steps_per_epoch=60,
                batch=16, 
                channels=96,
                patch=2,
                H=64, 
                W=64, 
                C=1, 
                save_every=5,
                attn_layers_per_block=8,
                num_blocks=8,
                head_dim=96,
                expansion=4,
                noise_std=0.05)
main(hp, dataset; device=gpu, ckpts_dir="vel_ckpts_new")
