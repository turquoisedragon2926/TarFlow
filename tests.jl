using ParametricOperators
using ChainRulesCore
include("utils.jl")
include("attention.jl")
include("block.jl")
include("model.jl")
include("data.jl")

function test_patch_roundtrip()
    B, C, H, W = 2, 3, 32, 32
    x = rand(Float32, B, C, H, W)
    cfg = PatchConfig(H, W, 8, H, W)
    patches = patch_data(cfg, x)
    @assert size(patches) == (B, (H÷8)*(W÷8), C*8*8)
    xr = unpatch_data(cfg, patches)
    @assert size(xr) == size(x)
    @assert maximum(abs, xr .- x) < 1f-5
    println("Patch roundtrip OK")
end

function test_forward_shapes()
    B, C, H, W = 4, 3, 32, 32
    x = rand(Float32, B, C, H, W)
    model = TarFlow(C, H, W, 8, 64, 2; attn_layers_per_block=1, head_dim=64, expansion=4)
    θ = ParametricOperators.init!(model, Dict{ParOperator,Any}())
    z, outs, ld = forward(model, θ, x)
    @assert size(z) == size(x)
    @assert length(outs) == 2 + 1
    @assert length(ld) == B
    println("Forward shapes OK")
end

function test_metablock_gradients()
    dataset = SingleCatDataset()
    m = MetaBlock(3*8*8, 64, 64; attn_layers_per_block=2, head_dim=16, expansion=4)
    θ = ParametricOperators.init!(m, Dict{ParametricOperators.ParOperator,Any}())
    x = get_data(dataset, 1; batch=1)
    p = patch_data(PatchConfig(64, 64, 8, 64, 64), x)
    p, ld = forward(m, θ, p)
    
    grads = Flux.gradient(θ) do θp
        z, ld = forward(m, θp, p)
        0.5f0 * mean(z.^2) - mean(ld)
    end
    @assert !isnothing(grads)
    println("MetaBlock gradients OK")
end

function test_attention_block_gradients()
    b = AttentionBlock(64, 16, 4)
    θ = ParametricOperators.init!(b, Dict{ParametricOperators.ParOperator,Any}())
    x = rand(Float32, 1, 64, 64)
    z = forward(b, θ, x)
    grads = Flux.gradient(θ) do θp
        z = forward(b, θp, x)
        0.5f0 * mean(z.^2)
    end
    @assert !isnothing(grads)
    println("AttentionBlock gradients OK")
end

function test_model_gradients()
    dataset = SingleCatDataset()
    model = TarFlow(3, 64, 64, 8, 64, 2; attn_layers_per_block=2, head_dim=16, expansion=4)
    θ = ParametricOperators.init!(model, Dict{ParametricOperators.ParOperator,Any}())
    x = get_data(dataset, 1; batch=1)
    p = patch_data(model.patch_config, x)
    
    loss, grads = Flux.withgradient(θ, p) do θp, p
        ld_mean_sum = 0.0f0
        for b in model.blocks
            p, ld = forward(b, θp, p)
            ld_mean_sum += mean(ld)
        end
        0.5f0 * mean(p.^2) - ld_mean_sum/length(model.blocks)
    end
    @assert !isnothing(grads)
    println("Model gradients OK")
end

function test_gradient_update()
    dataset = SingleCatDataset()
    model = TarFlow(3, 64, 64, 8, 64, 2; attn_layers_per_block=2, head_dim=16, expansion=4)
    θ = ParametricOperators.init!(model, Dict{ParametricOperators.ParOperator,Any}())
    x = get_data(dataset, 1; batch=1)
    p = patch_data(model.patch_config, x)
    
    # Get initial output
    p_init = copy(p)
    for b in model.blocks
        p_init, _ = forward(b, θ, p_init)
    end
    
    # Do gradient update
    opt = Flux.Optimisers.Adam(1e-3)
    state = Flux.Optimisers.setup(opt, θ)
    
    loss, grads = Flux.withgradient(θ, p) do θp, p
        ld_acc_vec = zeros(Float32, size(p,1))
        for b in model.blocks
            p, ld = forward(b, θp, p)
            ld_acc_vec = ld_acc_vec + ld
        end
        0.5f0 * mean(p.^2) - mean(ld_acc_vec)
    end

    state, θ = Flux.Optimisers.update(state, θ, grads[1])
    θ = Dict{ParOperator, Any}(θ)
    
    # Get output after update
    p_after = copy(p)
    for b in model.blocks
        p_after, _ = forward(b, θ, p_after)
    end
    
    # Check outputs are different
    @assert maximum(abs, p_after .- p_init) > 1e-6 "Gradient update did not change outputs"
    println("Gradient update OK")
end

function test_backward_roundtrip()
    dataset = SingleCatDataset()
    model = TarFlow(3, 64, 64, 8, 64, 2; attn_layers_per_block=2, head_dim=16, expansion=4)
    θ = ParametricOperators.init!(model, Dict{ParametricOperators.ParOperator,Any}())
    x = get_data(dataset, 1; batch=1)
    p = patch_data(model.patch_config, x)

    # Forward pass
    z = copy(p)
    for b in model.blocks
        z, _ = forward(b, θ, z)
    end

    # Backward pass
    x_recon = copy(z)
    for b in reverse(model.blocks)
        x_recon = backward(b, θ, x_recon)
    end

    # Check reconstruction error
    @assert maximum(abs, x_recon .- p) < 1e-5 "Backward pass failed to reconstruct input"
    println("Backward roundtrip OK")
end

function test_block_invertibility()
    # Test single MetaBlock invertibility
    B, T, C = 2, 64, 192  # B=batch, T=sequence length, C=channels
    x = rand(Float32, B, T, C)
    block = MetaBlock(C, 64, T; attn_layers_per_block=2, head_dim=16, expansion=4)
    θ = ParametricOperators.init!(block, Dict{ParOperator,Any}())
    
    # Forward pass
    z, _ = forward(block, θ, x)
    
    # Backward pass
    x_recon = backward(block, θ, z)
    
    # Check reconstruction matches input
    @assert maximum(abs, x_recon .- x) < 1e-5 "MetaBlock failed invertibility test"
    println("Block invertibility OK")
end

function test_flow_invertibility()
    # Test full TarFlow invertibility
    B, C, H, W = 1, 3, 64, 64
    x = rand(Float32, B, C, H, W)
    model = TarFlow(C, H, W, 8, 64, 3; attn_layers_per_block=2, head_dim=16, expansion=4)
    θ = ParametricOperators.init!(model, Dict{ParOperator,Any}())
    
    # Forward pass through full model
    z, _, _ = forward(model, θ, x)
    
    # Backward pass through full model 
    x_recon = backward(model, θ, z)[end]
    
    # Check reconstruction matches input
    @assert maximum(abs, x_recon .- x) < 1e-5 "TarFlow failed invertibility test"
    println("Flow invertibility OK")
end

function test_attention_causal_consistency()
    # Test that causal attention gives same output for prefix when run on full sequence vs prefix only
    Random.seed!(100)
    B, T, C = 1, 2, 6  # B=batch, T=sequence length, C=channels
    x = rand(Float32, B, T, C)
    attn = Attention(C, C÷2)
    θ = ParametricOperators.init!(attn, Dict{ParOperator,Any}())
    
    # Run on full sequence with causal mask
    full_out = forward(attn, θ, x; mask=tril_mask(T))
    
    # Run on just first token 
    prefix_out = forward(attn, θ, x[:, 1:1, :]; mask=tril_mask(1))
    
    # First token output should match between both runs
    @assert norm(full_out[1, 1, :] - prefix_out[1, 1, :]) < 1e-5 "Attention failed causal consistency test"
    println("Attention causal consistency OK")
end


test_patch_roundtrip()
test_forward_shapes()
test_metablock_gradients()
test_attention_block_gradients() 
test_model_gradients()
test_gradient_update()
test_backward_roundtrip()
test_block_invertibility()
test_flow_invertibility()
test_attention_causal_consistency()
