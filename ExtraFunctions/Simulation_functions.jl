using SparseArrays, CSV, DataFrames, Statistics, NaNStatistics, LaTeXStrings, StatsBase, FFTW, NPZ

# list of all parameters with accompanying number of integrator entry
p_dict = Dict("sites"=>1, "n_avg"=>2,"M"=>3,"B_21"=>4,"κ"=>5,
"J"=>6,"Δ"=>7,"γ"=>8,"T"=>9,"V"=>10,"ρ"=>11)

# list of output sequence of extract data function
o_dict = Dict("t_vec"=>1, "X_data"=>2)

function classicalField_JJ(du, u, p, t)
    # DE of full classical field model for 2 sites integrating both ψ and X fields
    sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ = p
    
    # Left site
    du[1] = -im*(-(1-im*κ)*J*u[2]+im*0.5*(B_21*(u[3]-abs2(u[1]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[1].+V[1]*(1-im*κ)*u[1])
    du[3] = -2*J*imag(conj(u[1])*u[2]) - γ*(abs2(u[1])-n_avg*ρ[1])
    
    # Right site
    du[2] = -im*(-(1-im*κ)*J*u[1]+im*0.5*(B_21*(u[4]-abs2(u[2]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[2].+V[2]*(1-im*κ)*u[2])
    du[4] = -2*J*imag(conj(u[2])*u[1]) - γ*(abs2(u[2])-n_avg*ρ[2])
end

function σ_classicalField_JJ_add(du, u, p, t)
    # Additive noise term for SDE of full classical field model for 2 sites integrating both ψ and X fields
    sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength = p
    
    du[1] = noise_strength*sqrt(B_21*M*exp(Δ/T))
    du[2] = noise_strength*sqrt(B_21*M*exp(Δ/T))
    du[3] = 0.0 # No noise on X fields
    du[4] = 0.0
end

function classicalField_MultiSite(du, u, p, t)
    # DE of full classical field model for N sites integrating both ψ and X fields
    sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ = p

    # Left site
    du[1] = -im*(-(1-im*κ)*J*(u[2]-u[1])+im*0.5*(B_21*(u[sites+1]-abs2(u[1]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[1]+V[1]*(1-im*κ)*u[1]) # Add potential shift to left site
    du[sites+1] = -2*J*imag(conj(u[1])*u[2]) - γ*(abs2(u[1])-n_avg*ρ[1])

    # Inner sites
    for i in 2:sites-1
        du[i] = -im*(-(1-im*κ)*J*(u[i-1]+u[i+1]-2*u[i])+im*0.5*(B_21*(u[sites+i]-abs2(u[i]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[i]+V[i]*(1-im*κ)*u[i])
        du[sites+i] = -2*J*imag(conj(u[i])*(u[i-1]+u[i+1])) - γ*(abs2(u[i])-n_avg*ρ[i])
    end
    # Right site
    du[sites] = -im*(-(1-im*κ)*J*(u[sites-1]-u[sites])+im*0.5*(B_21*(u[2*sites]-abs2(u[sites]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[sites]+V[sites]*(1-im*κ)*u[sites]) # Add potential shift to right site
    du[2*sites] = -2*J*imag(conj(u[sites])*u[sites-1]) - γ*(abs2(u[sites])-n_avg*ρ[sites])
end

function σ_classicalField_MultiSite_add(du, u, p, t)
    # Additive noise term for SDE of full classical field model for N sites integrating both ψ and X fields
    sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength = p
    for i in 1:sites
        du[i] = noise_strength*sqrt(B_21*M*exp(Δ/T))
    end
    for i in sites+1:2*sites
        du[i] = 0.0 # No noise on X fields
    end
end

function classicalField(du, u, p, t)
    # DE of full classical field model integrating both ψ and X fields
    # Unified function for both 2-site and N-site logic
    sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength = p

    if sites == 2
        # --- 2-Site Logic ---
        # Left site
        du[1] = -im*(-(1-im*κ)*J*u[2]+im*0.5*(B_21*(u[3]-abs2(u[1]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[1] + V[1]*(1-im*κ)*u[1])
        du[3] = -2*J*imag(conj(u[1])*u[2]) - γ*(abs2(u[1])-n_avg*ρ[1])
        
        # Right site
        du[2] = -im*(-(1-im*κ)*J*u[1]+im*0.5*(B_21*(u[4]-abs2(u[2]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[2] + V[2]*(1-im*κ)*u[2])
        du[4] = -2*J*imag(conj(u[2])*u[1]) - γ*(abs2(u[2])-n_avg*ρ[2])
    else
        # --- N-Site Logic ---
        # Left site
        du[1] = -im*(-(1-im*κ)*J*(u[2]-u[1])+im*0.5*(B_21*(u[sites+1]-abs2(u[1]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[1] + V[1]*(1-im*κ)*u[1]) # Add potential shift to left site
        du[sites+1] = -2*J*imag(conj(u[1])*u[2]) - γ*(abs2(u[1])-n_avg*ρ[1])

        # Inner sites
        for i in 2:sites-1
            du[i] = -im*(-(1-im*κ)*J*(u[i-1]+u[i+1]-2*u[i])+im*0.5*(B_21*(u[sites+i]-abs2(u[i]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[i] + V[i]*(1-im*κ)*u[i])
            du[sites+i] = -2*J*imag(conj(u[i])*(u[i-1]+u[i+1])) - γ*(abs2(u[i])-n_avg*ρ[i])
        end
        
        # Right site
        du[sites] = -im*(-(1-im*κ)*J*(u[sites-1]-u[sites])+im*0.5*(B_21*(u[2*sites]-abs2(u[sites]))*(1+exp(Δ/T))-B_21*exp(Δ/T)*M-γ)*u[sites] + V[sites]*(1-im*κ)*u[sites]) # Add potential shift to right site
        du[2*sites] = -2*J*imag(conj(u[sites])*u[sites-1]) - γ*(abs2(u[sites])-n_avg*ρ[sites])
    end
end

function σ_classicalField_add(du, u, p, t)
    # Unified additive noise term for SDE of full classical field model
    sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength = p
    
    # Add noise to the ψ fields
    for i in 1:sites
        du[i] = noise_strength*sqrt(B_21*M*exp(Δ/T))
    end
    
    # Set noise to zero for the X fields
    for i in sites+1:2*sites
        du[i] = 0.0 # No noise on X fields
    end
end

"""Extract data after run to vectors."""
function extractSolDataX(sol_vec, p_struct)
    t_X_vec = sol_vec.t
    X_mat = zeros(ComplexF64, 2*p_struct.sites, length(t_X_vec))
    X_mat[:,:] .= Array(sol_vec)
    return t_X_vec, X_mat
end