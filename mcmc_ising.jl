module MCMCIsing

function Hamltn(ss, coef)
  # Hamiltonian for Ising model. Only works with nearest neighbor
  N     = length(ss[1,:])
  idxL = [N; collect(1:(N-1))]
  idxR = [collect(2:N); 1]
  ham = zeros(size(coef))
  ham[1] = sum(ss.*(ss[idxL,:]+ss[idxR,:]+ss[:,idxL]+ss[:,idxR])) * 0.5
  engy = -coef[1] * ham[1]
  return engy
end


function HamltnTerm(ss, nterm)
  # Hamiltonian for Ising model. Only works with nearest neighbor
  N     = length(ss[1,:])
  idxL = [N; collect(1:(N-1))]
  idxR = [collect(2:N); 1]
  idxL2 = [N-1;N; collect(1:(N-2))]
  idxR2 = [collect(3:N);1;2]
  assert(nterm==3)
  ham = zeros(nterm)
  ham[1] = sum(ss.*(ss[idxL,:]+ss[idxR,:]+
                    ss[:,idxL]+ss[:,idxR])) * 0.5
  ham[2] = sum(ss.*(ss[idxL,idxL]+ss[idxL,idxR]+
                    ss[idxR,idxL]+ss[idxR,idxR])) * 0.5
  ham[3] = sum(ss.*(ss[idxL2,:]+ss[idxR2,:]+
                    ss[:,idxL2]+ss[:,idxR2])) * 0.5

  return ham
end

# Subroutines used by Swendsen-Wang algorithm
function GetClusterNumber(index,cluster)
  ii = index
  while( ii != cluster[ii] )
    ii = cluster[ii]
  end
  return ii
end


function ConnectBond!(N, i1, j1, i2, j2, prob, ss, cluster)
  if( ss[i1,j1] * ss[i2,j2] < 0 )
    return
  end
  if( rand() >= prob )
    return
  end
  ii1 = i1 + (j1-1)*N
  ii2 = i2 + (j2-1)*N
  c1 = GetClusterNumber(ii1, cluster)
  c2 = GetClusterNumber(ii2, cluster)
  if( c1 < c2 )
    cluster[c2] = c1
  else
    cluster[c1] = c2
  end
  return
end


# MCMC using Swendsen-Wang algorithm. Sampling only.
# Allows the output of full trajectory
function ising_2d_sw(coef,tN,ss0,nSample;is_trajectory=false)

  N     = length(ss0[1,:])
  nterm = length(coef)

  Magavg  = zeros(size(ss0))
  Coravg  = zeros(size(ss0))
  ttmag   = zeros(floor(Int, div(tN,nSample)))

  ss_traj  = zeros(Int64,N*N,floor(Int, div(tN,nSample)))
  energy_traj = zeros(floor(Int, div(tN,nSample)))

  acceptCount = 0
  ss  = copy(ss0)                # current spin system   NxN

  engy = Hamltn(ss,coef)
  dHam = zeros(nterm)

  idxL = [N; collect(1:(N-1))]
  idxR = [collect(2:N); 1]

  # cluster is implemented as a linked chain
  cluster = zeros(Int64,N*N)
  flip = zeros(Int64,N*N)

  prob = 1.0 - exp(-2.0*coef[1])

  # Number of sweeps
  cSample = 0
  for nstep=1:tN
    cluster = collect(1:N*N)
    flip = rand([-1,1],N*N)
    for j = 1 : N
      for i = 1 : N
        # Modifies the cluster variable
        ConnectBond!( N, i, j, idxR[i], j, prob, ss, cluster )
        ConnectBond!( N, i, j, i, idxR[j], prob, ss, cluster )
      end
    end
    for ii = 1 : N*N
      ss[ii] = flip[GetClusterNumber(ii, cluster)]
    end

    # Record the observable
    if( mod(nstep,nSample) == 0 )
      cSample += 1

      Magavg += ss
      ttmag[cSample] = sum(ss)
      Coravg += ss[1,1]*ss

      if( is_trajectory )
        ss_traj[:,cSample] = ss[:]
        energy_traj[cSample] = Hamltn(ss, coef)
      end

    end
    #    println(ss)
    #    sleep(1)
  end       # end of MCMC

  # engy mag and cor
  Magavg = Magavg/cSample
  Coravg = Coravg/cSample

  return (Magavg, Coravg, ttmag,  ss, ss_traj, energy_traj)
end


end # module MCMCIsing
