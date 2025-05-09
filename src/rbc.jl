module RBC

using DynamicProgrammingGPU

@kwdef struct Parameters{F} <: ModelParameters
    β::F = 0.99
    δ::F = 0.01
    α::F = 0.35
    γ::F = 1.2
    ρ::F = 0.95
    σ::F = 0.005
    min::F = 1e-3
    max::F = 1e3
end

c(u,s,p) = (one(s[1])-p.δ)*s[1] - u[1] + exp(s[2]) * s[1]^p.α
f(u,s,v,p) = ((1-p.β) * c(u,s,p)^(1-p.γ) + p.β * v(u[1])^(1-p.γ))^(1/(1-p.γ))
bounds(b,s,v,p) = (
    zero(s[1]),
    (one(s[1])-p.δ)*s[1] + exp(s[2]) * s[1]^p.α - 1f-10
)
v0(s,p) = ((s[1])^(1-p.γ))^(1/(1-p.γ))

function init(p, n; m=3)
    
    grid = Grid((p.min,), (p.max,), (n[1],))
    prob = ValueFunction(
        UnivariateOptimizationProblem(
            f,
            bounds,
            GoldenSection(),
        ),
        Val(3),
        Tauchen(p.ρ, p.σ, m, n[2])
    )
    DynamicProgrammingGPU.init(prob, grid, Base.Fix2(v0, p))
    
end

end
