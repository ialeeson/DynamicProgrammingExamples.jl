using CUDA, BenchmarkTools, DataFrames, Adapt, FileIO, Dates, CSV
import CommonSolve.solve!, CommonSolve.init

include("DynamicProgrammingGPU.jl")
using .DynamicProgrammingGPU



function push_time!(filename, t, date, name, n, nsteps)
    cpu = @belapsed(solve!(x, p),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n)
        )
    )
    gpu = @belapsed(solve!(x_cuda, p_cuda),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n);
            p_cuda = cu(p);
            x_cuda = cu(x)
        )
    )
    tex = @belapsed(solve!(x_tex, p_tex),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n);
            p_tex = cutex(p);
            x_tex = cutex(x)
        )
    )
    push!(t, (; date, name, n=string(n), cpu, gpu, tex))
    CSV.write(filename, t)
end

filename, date = ("times.csv", now())
types = Dict(1 => DateTime, 2 => Symbol, 4 => Float64, 5 => Float64, 6 => Float64)
t = isfile(filename) ? CSV.read(filename, DataFrame; types) : DataFrame(
    date = DateTime[], name = Symbol[], n = String[],
    cpu = Float64[], gpu = Float64[], tex = Float64[]
)
[push_time!(filename, t, date, name, (n,), 10^2) for n in 32 .* 2 .^ (0:0),
    name in (:CP,)]
[push_time!(filename, t, date, name, (n,m), 10^2) for n in 32 .* 2 .^ (0:0), m in (8,), name in (:RBC_QUADRATURE,)]
