include("mcmc_ising.jl")

using .MCMCIsing

N = 2
N_large = 2*N
println("Size of original spin system = ", N)

beta = 0.4406868

coefTerm = [beta,0,0]

using DelimitedFiles
using Printf

# All 4x4 configurations
ss_traj = zeros(Int64,N_large*N_large,2^(N_large*N_large))
energy_traj = zeros(2^(N_large*N_large))
for sample_count = 1 : 2^(N_large*N_large)
    ss_traj[:,sample_count] = parse.(Int, collect(bin(sample_count-1,N_large*N_large)))
    for iter in eachindex(ss_traj[:,sample_count])
        if ss_traj[iter,sample_count] == 0
            ss_traj[iter,sample_count] = -1
        end
    end
    energy_traj[sample_count] = MCMCIsing.Hamltn(reshape(ss_traj[:,sample_count],N_large,N_large), coefTerm)

function CGHamltn(ss, coef, ss_traj, energy_traj)
  # CG Hamiltonian for Ising model. Twice decimated
    exp_energy_sum = 0.0

    for sample_count = 1 : 2^(N_large*N_large)
        if ss_traj[sample_count,[1,3,9,11]] == ss[:]
            exp_energy_sum = exp_energy_sum + exp(-energy_traj[sample_count])
        end
    end

    return -log(exp_energy_sum)
end

function gen_all_ising(coef, N, ss_traj, energy_traj)

    cgss_traj = zeros(Int64,N*N,2^(N*N))
    cgenergy_traj = zeros(2^(N*N))

    for sample_count = 1 : 2^(N*N)
        cgss_traj[:,sample_count] = parse.(Int, collect(bin(sample_count-1,N*N)))
        for iter in eachindex(cgss_traj[:,sample_count])
            if cgss_traj[iter,sample_count] == 0
                cgss_traj[iter,sample_count] = -1
            end
        end
        cgenergy_traj[sample_count] = CGHamltn(reshape(cgss_traj[:,sample_count],N,N), coef, ss_traj, energy_traj)
    end

    return (cgss_traj, cgenergy_traj)
end

println("Generating all configurations...")
@time outobj = gen_all_ising(coefTerm[1], N, ss_traj, energy_traj)

cgss_traj = outobj[1]
cgenergy_traj = outobj[2]

writedlm("./data/ssL$(N)b$(@sprintf("%.4e", beta))_cgdeci.dat", transpose(cgss_traj))
writedlm("./data/EL$(N)b$(@sprintf("%.4e", beta))_cgdeci.dat",cgenergy_traj)
