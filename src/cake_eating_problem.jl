module CakeEatingProblem

using DynamicProgrammingGPU

@kwdef struct Parameters{F} <: ModelParameters
    β::F = 0.99
    γ::F = 1.5
    min::F = 1e-3
    max::F = 1e3
end

c(u,s,p) = s[1] - u[1]
f(u,s,v,p) = ((1-p.β) * c(u,s,p)^(1-p.γ) + p.β * v(u[1])^(1-p.γ))^(1/(1-p.γ))
bounds(b,s,v,p) = (zero(s[1]), s[1])
v0(s,p) = ((s[1])^(1-p.γ) + one(p.γ))^(1/(1-p.γ))

function init(p, n)

    grid = Grid((p.min,), (p.max,), n)
    prob = ValueFunction(
        UnivariateOptimizationProblem(
            f,
            bounds,
            GoldenSection()
        ),
        Val(3),
    )
    DynamicProgrammingGPU.init(prob, grid, Base.Fix2(v0,p))

end

end
