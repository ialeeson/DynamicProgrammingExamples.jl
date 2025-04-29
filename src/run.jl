using Metal, BenchmarkTools, DataFrames
import CommonSolve.solve!, CommonSolve.init

import DynamicProgrammingExamples.CakeEatingProblem as CP
import DynamicProgrammingExamples.RBC 

function push_time!(t, name, n, nsteps)
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
            p_cuda = mtl(p);
            x_cuda = mtl(x)
        )
    )
    # tex = @belapsed(solve!(x_tex, p_tex),
    #     setup = (
    #         p = getproperty(eval($name), :Parameters)();
    #         x = getproperty(eval($name), :init)(p,$n);
    #         p_tex = mtl(p);
    #         x_tex = mtl(x)
    #     )
    # )
    push!(t, (; name, n, cpu, gpu))
end

t = DataFrame(name = Symbol[], n = Tuple[], cpu = Float64[], gpu = Float64[])
[push_time!(t, name, (n,), 10^2) for n in 32 .* 2 .^ (0:5),
    name in (:CP,)]
[push_time!(t, name, (n,m), 10^2) for n in 32 .* 2 .^ (0:5), m in (8,),
    name in (:RBC,)]
