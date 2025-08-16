using Test
using Flux
using Statistics
using ParametricOperators
include("attention.jl")

@testset "Attention Tests" begin
    @testset "LayerNorm" begin
        # Test basic layernorm functionality
        x = rand(Float32, 2, 3, 4)  # (B, T, C)
        normalized = layernorm(x)
        
        # Test output shape
        @test size(normalized) == size(x)
        
        # Test statistics along last dimension
        means = mean(normalized; dims=3)
        # Use same variance computation as in layernorm implementation
        vars = mean((normalized .- mean(normalized; dims=3)) .^ 2; dims=3)
        @test all(isapprox.(means, 0.0f0; atol=1f-6))
        @test all(isapprox.(vars, 1.0f0; atol=1f-6))

        # Test with different input sizes
        x2 = rand(Float32, 5, 10, 8)  # Different dimensions
        normalized2 = layernorm(x2)
        means2 = mean(normalized2; dims=3)
        vars2 = mean((normalized2 .- mean(normalized2; dims=3)) .^ 2; dims=3)
        @test all(isapprox.(means2, 0.0f0; atol=1f-6))
        @test all(isapprox.(vars2, 1.0f0; atol=1f-6))

        # Test with edge case values
        x3 = ones(Float32, 2, 3, 4)  # Constant input
        normalized3 = layernorm(x3)
        @test all(isapprox.(normalized3, 0.0f0; atol=1f-6))  # Should normalize to 0

        # Test with small values
        x4 = rand(Float32, 2, 3, 4) .* 1f-4
        normalized4 = layernorm(x4)
        means4 = mean(normalized4; dims=3)
        vars4 = mean((normalized4 .- mean(normalized4; dims=3)) .^ 2; dims=3)
        @test all(isapprox.(means4, 0.0f0; atol=1f-6))
        @test all(isapprox.(vars4, 1.0f0; atol=1f-6))
    end

    @testset "Attention Initialization" begin
        in_channels = 64
        head_dim = 32
        attn = Attention(in_channels, head_dim)

        # Test parameter initialization
        θ = Dict{ParOperator,Any}()
        init!(attn, θ)
        @test haskey(θ, attn.qkv)
        @test haskey(θ, attn.proj)
        
        # Test structure
        @test attn.num_heads == 2  # 64/32
        @test attn.head_dim == head_dim
        @test size(ParametricOperators.params(attn.qkv(θ))) == (in_channels, 3 * in_channels)
        @test size(ParametricOperators.params(attn.proj(θ))) == (in_channels, in_channels)
    end

    @testset "Attention Forward" begin
        B, T, C = 2, 4, 64  # batch, sequence length, channels
        head_dim = 32
        x = rand(Float32, B, T, C)
        attn = Attention(C, head_dim)
        θ = Dict{ParOperator,Any}()
        init!(attn, θ)
        
        # Test basic forward pass
        output = forward(attn, θ, x)
        @test size(output) == (B, T, C)
        
        # Test with different batch sizes
        x_large = rand(Float32, 8, T, C)
        output_large = forward(attn, θ, x_large)
        @test size(output_large) == (8, T, C)
        
        # Test with different sequence lengths
        x_long = rand(Float32, B, 8, C)
        output_long = forward(attn, θ, x_long)
        @test size(output_long) == (B, 8, C)
    end

    @testset "Attention Masking" begin
        B, T, C = 2, 4, 64
        head_dim = 32
        x = rand(Float32, B, T, C)
        attn = Attention(C, head_dim)
        θ = Dict{ParOperator,Any}()
        init!(attn, θ)
        
        # Create causal mask (upper triangular)
        mask = Matrix{Bool}(undef, T, T)
        for i in 1:T, j in 1:T
            mask[i,j] = i >= j
        end
        
        # Test with mask
        output_masked = forward(attn, θ, x; mask=mask)
        @test size(output_masked) == (B, T, C)
        
        # Test without mask
        output_unmasked = forward(attn, θ, x)
        @test size(output_unmasked) == (B, T, C)
        
        # Outputs should be different with and without mask
        @test !isapprox(output_masked, output_unmasked)
    end

    @testset "MLP" begin
        C = 64
        expansion = 4
        mlp = MLP(C, expansion)
        θ = Dict{ParOperator,Any}()
        init!(mlp, θ)
        
        # Test structure
        @test size(ParametricOperators.params(mlp.fc1(θ))) == (C, C * expansion)
        @test size(ParametricOperators.params(mlp.fc2(θ))) == (C * expansion, C)
        
        # Test forward pass
        B, T = 2, 4
        x = rand(Float32, B, T, C)
        
        output = forward(mlp, θ, x)
        @test size(output) == (B, T, C)
    end

    @testset "AttentionBlock" begin
        C = 64
        head_dim = 32
        expansion = 4
        block = AttentionBlock(C, head_dim, expansion)
        
        # Test initialization
        θ = Dict{ParOperator,Any}()
        init!(block, θ)
        @test haskey(θ, block.attn.qkv)
        @test haskey(θ, block.attn.proj)
        @test haskey(θ, block.mlp.fc1)
        @test haskey(θ, block.mlp.fc2)
        
        # Test forward pass
        B, T = 2, 4
        x = rand(Float32, B, T, C)
        output = forward(block, θ, x)
        @test size(output) == (B, T, C)
        
        # Test residual connections
        # The output should not be identical to input due to the transformations
        @test !isapprox(output, x)
    end
end
