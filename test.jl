using ParametricOperators
import ParametricOperators: gpu, cpu

x = randn(Float32, 10, 10)
m = ParMatrix(Float32, 10, 10)

θ = init(m)
θ = Dict{ParOperator,Any}()
init!(m, θ)
