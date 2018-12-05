include("mcmc_ising.jl")

using .MCMCIsing

N = 2
cgf = 2
# N_large = cgf * N
println("Size of decimated spin system = ", N)

beta = 0.4406868

coefTerm = [beta,0,0]

using DelimitedFiles
using Printf

# All 4x4 configurations
# ss_traj = zeros(Int64,N_large*N_large,2^(N_large*N_large))
# energy_traj = zeros(2^(N_large*N_large))
# for sample_count = 1 : 2^(N_large*N_large)
#     ss_traj[:,sample_count] = parse.(Int, collect(string(sample_count-1, base=2, pad=N_large*N_large)))
#     for iter in eachindex(ss_traj[:,sample_count])
#         if ss_traj[iter,sample_count] == 0
#             ss_traj[iter,sample_count] = -1
#         end
#     end
#     energy_traj[sample_count] = MCMCIsing.Hamltn(reshape(ss_traj[:,sample_count],N_large,N_large), coefTerm)
# end

function CGHamltn(ss, N, coef, cgf)
  # CG Hamiltonian for Ising model. Twice decimated
    exp_energy_sum = 0.0

    for sample_count = 1 : 2^((cgf*cgf - 1)*N*N)
        inst_ss = parse.(Int, collect(string(sample_count-1, base=2, pad=(cgf*cgf - 1)*N*N)))
        for iter in eachindex(inst_ss)
            if inst_ss[iter] == 0
                inst_ss[iter] = -1
            end
        end
        for ssidx = 1 : N*N
            instidx = cgf * cgf * N * floor(Int, (ssidx - 1)/N) + cgf * Int(mod(ssidx - 1, N)) + 1
            splice!(inst_ss, instidx+1:instidx, ss[ssidx])
        end
        inst_energy = MCMCIsing.Hamltn(reshape(inst_ss, cgf*N, cgf*N), coef)
        exp_energy_sum = exp_energy_sum + exp(-inst_energy)
    end

    return -log(exp_energy_sum)
end

# function CGHamltn2(ss, N, coef)
#   # CG Hamiltonian for Ising model. Twice decimated
#     exp_energy_sum = 0.0
#
#     for sample_count = 1 : 2^(4*N*N)
#         extract_ss = ss_traj[[1,3,9,11],sample_count]
#         if extract_ss[:] == ss[:]
#             inst_energy = energy_traj[sample_count]
#             exp_energy_sum = exp_energy_sum + exp(-inst_energy)
#         end
#     end
#
#     return -log(exp_energy_sum)
# end

function gen_all_ising(coef, N, cgf)

    cgss_traj = zeros(Int64,N*N,2^(N*N))
    cgenergy_traj = zeros(2^(N*N))

    for sample_count = 1 : 2^(N*N)
        cgss_traj[:,sample_count] = parse.(Int, collect(string(sample_count-1, base=2, pad=N*N)))
        for iter in eachindex(cgss_traj[:,sample_count])
            if cgss_traj[iter,sample_count] == 0
                cgss_traj[iter,sample_count] = -1
            end
        end
        cgenergy_traj[sample_count] = CGHamltn(reshape(cgss_traj[:,sample_count],N,N), N, coef, cgf)
    end

    return (cgss_traj, cgenergy_traj)
end

println("Generating all CG configurations...")
@time outobj = gen_all_ising(coefTerm[1], N, cgf)

cgss_traj = outobj[1]
cgenergy_traj = outobj[2]

writedlm("./data/ssL$(N)b$(@sprintf("%.4e", beta))_cgdeci$(cgf).dat", transpose(cgss_traj))
writedlm("./data/EL$(N)b$(@sprintf("%.4e", beta))_cgdeci$(cgf).dat", cgenergy_traj)
