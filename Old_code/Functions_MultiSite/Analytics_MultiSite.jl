using NLsolve, LinearAlgebra

function calculate_steady_state_multisite(sites, J, γ, V, B_21=1e-7, M=1e8, T=25.0, Δ=-60.0, κ=0.5 * B_21 * M * exp(Δ/T) / T, n_avg=3.0e3; guess_n=nothing, guess_phi=nothing)
    
    # Calculate derived parameters
    ΔVs = diff(V)
    
    # Use the previous solution as the guess if provided (continuation method)
    x0 = zeros(Float64, 2*sites - 1)
    if !isnothing(guess_n) && !isnothing(guess_phi)
        x0[1:sites-1] .= guess_phi
        x0[sites:end] .= guess_n
    else
        # Fallback for the first iteration
        x0[1:sites-1] .= 0.1
        x0[sites:end] .= n_avg
    end

    c_alpha = 2 * J / γ

    function f!(F, x)
        @inbounds begin
            # Offset for density indices: n[i] corresponds to x[offset + i]
            offset = sites - 1

            # --- 1. Phase Dynamics ---
            # Left Boundary (i=1)
            # n[1] -> x[offset+1], phi[1] -> x[1]
            n1, n2 = max(x[offset+1], 1e-9), max(x[offset+2], 1e-9)
            p1, p2 = x[1], x[2]
            
            term1 = J * (sqrt(n1/n2) - sqrt(n2/n1))
            term2 = J * κ * (sqrt(n1/n2) + sqrt(n2/n1))
            term3 = J * sqrt(max(x[offset+3], 1e-9)/n2) * (cos(p2) + κ*sin(p2))
            
            F[1] = term1*cos(p1) - term2*sin(p1) + term3 - J - ΔVs[1]

            # Inner Sites
            for i in 2:sites-2
                ni_prev = max(x[offset+i-1], 1e-9)
                ni      = max(x[offset+i], 1e-9)
                ni_next = max(x[offset+i+1], 1e-9)
                ni_next2= max(x[offset+i+2], 1e-9)
                
                pi_prev, pi, pi_next = x[i-1], x[i], x[i+1]

                sq_fwd = sqrt(ni/ni_next)
                sq_bwd = sqrt(ni_next/ni)

                next_term = J * sqrt(ni_next2/ni_next) * (cos(pi_next) + κ*sin(pi_next))
                prev_term = J * sqrt(ni_prev/ni)       * (cos(pi_prev) - κ*sin(pi_prev))

                F[i] = J*(sq_fwd - sq_bwd)*cos(pi) - 
                        J*κ*(sq_fwd + sq_bwd)*sin(pi) + 
                        next_term - prev_term - ΔVs[i]
            end

            # Right Boundary
            n_last_prev = max(x[offset+sites-2], 1e-9)
            n_last      = max(x[offset+sites-1], 1e-9)
            n_last_next = max(x[offset+sites], 1e-9)
            
            p_last_prev, p_last = x[sites-2], x[sites-1]

            term_r1 = sqrt(n_last/n_last_next)
            term_r2 = sqrt(n_last_next/n_last)
            prev_term_r = J * sqrt(n_last_prev/n_last) * (cos(p_last_prev) - κ*sin(p_last_prev))
            
            F[sites-1] = J*(term_r1 - term_r2)*cos(p_last) - 
                            J*κ*(term_r1 + term_r2)*sin(p_last) - 
                            prev_term_r + J - ΔVs[sites-1]

            # --- 2. Density Equations ---
            # We calculate alpha on the fly to avoid allocating a vector
            # Left
            a1 = c_alpha * sin(x[1])
            F[sites] = max(x[offset+1], 1e-9) - (n_avg - a1 * sqrt(max(x[offset+1], 1e-9) * max(x[offset+2], 1e-9)))

            # Inner
            for i in 2:sites-1
                a_prev = c_alpha * sin(x[i-1])
                a_curr = c_alpha * sin(x[i])
                
                nm = max(x[offset+i-1], 1e-9) # n minus 1
                nc = max(x[offset+i], 1e-9)   # n current
                np = max(x[offset+i+1], 1e-9) # n plus 1
                
                F[sites + i - 1] = nc - (n_avg + a_prev * sqrt(nm * nc) - a_curr * sqrt(nc * np))
            end

            # Right
            a_last = c_alpha * sin(x[sites-1])
            # n[sites] corresponds to x[offset+sites]
            F[2*sites-1] = max(x[offset+sites], 1e-9) - (n_avg + a_last * sqrt(max(x[offset+sites-1], 1e-9) * max(x[offset+sites], 1e-9)))
        end
    end

    # Solve (3*sites-1)D system
    res = nlsolve(f!, x0, ftol=1e-8, autodiff=:forward)

    φ = res.zero[1:sites-1]
    n = max.(res.zero[sites:2*sites-1], 1e-9)

    # Calculate M2 from n and φ
    M_2 = zeros(Float64, sites)

    # Left site
    M_2[1] = 1/(1+exp(Δ/T))*(M*exp(Δ/T) +2/B_21*(J*sqrt(n[2]/n[1])*(sin(φ[1])-κ*cos(φ[1])) + J*κ + γ/2 + κ*V[1]))

    # Inner sites
    for i in 2:sites-1
        M_2[i] = 1/(1+exp(Δ/T))*(M*exp(Δ/T) +2/B_21*(J*sqrt(n[i+1]/n[i])*(sin(φ[i])-κ*cos(φ[i])) - J*sqrt(n[i-1]/n[i])*(sin(φ[i-1])+κ*cos(φ[i-1])) + 2*J*κ + γ/2 + κ*V[i]))
    end

    # Right site
    M_2[sites] = 1/(1+exp(Δ/T))*(M*exp(Δ/T) +2/B_21*( -J*sqrt(n[sites-1]/n[sites])*(sin(φ[sites-1])+κ*cos(φ[sites-1])) + J*κ + γ/2 + κ*V[sites]))
    
    return (
        converged = converged(res),
        residual = res.residual_norm,
        φ = φ,
        n = n,
        M_2 = M_2
    )
end

using Roots, NLsolve, Optim, CairoMakie, LaTeXStrings

function calculate_critical_ΔV_multisite(sites, J, γ, B_21=1.0e-7, M=1.0e8, T=25.0, Δ=-60.0, κ=0.5*B_21*M*exp(Δ/T)/T, n_avg=3.0e3)
    # Pre-allocate A_loc matrix and V vector
    A_loc = zeros(Float64, 3*sites-1, 3*sites-1)
    V = zeros(Float64, sites)

    # Prepare storage variables
    max_real_eigenvals = Vector{Float64}()
    ΔV_scanned = Vector{Float64}()
    crossing_matrix_A = Matrix{Float64}(undef, 0, 0)
    crossing_found = false
    
    # Define parameters for the scan
    start_ΔV = 1e-6      
    stop_ΔV  = 100.0   
    rel_prec = 1e-3      

    n_points = ceil(Int, log(stop_ΔV / start_ΔV) / log(1 + rel_prec))
    log_ΔV = 10 .^ range(log10(start_ΔV), log10(stop_ΔV), length=n_points)
    ΔV_range = [0.0; log_ΔV]

    ΔV_stop_threshold = Inf
    ΔV_start = first(ΔV_range)

    # Store the solution from the previous step to use as a guess for the next
    prev_n = nothing
    prev_phi = nothing

    for ΔV in ΔV_range
        # Update Potentials
        V_sites = LinRange(0, ΔV, sites)
        V .= V_sites

        # 2. Define Steady State parameters
        if isnothing(prev_n)
             _, _, φ_loc, n_loc, M_2_loc = calculate_steady_state_multisite(sites, J, γ, V, B_21, M, T, Δ, κ, n_avg)
        else
             # Use previous result as guess
             _, _, φ_loc, n_loc, M_2_loc = calculate_steady_state_multisite(sites, J, γ, V, B_21, M, T, Δ, κ, n_avg, guess_n=prev_n, guess_phi=prev_phi)
        end

        # Update the guess for the next iteration
        prev_n = n_loc
        prev_phi = φ_loc

        # 3. Define derivative matrix
        # We define the matrix that gives the matrix equation dX/dt = A*X
        # where X = [δn_loc_1, δX_loc_1, δφ_loc_1, δn_loc_2, δX_loc_2, δφ_loc_2, ..., δn_loc_N-1, δX_loc_N-1, δφ_loc_N-1, δn_loc_N, δX_loc_N]
        # The elements of A are derived from linearizing the system around the steady state
        
        # Left site
        A_loc[1,1] = J*sqrt(n_loc[2]/n_loc[1])*(κ*cos(φ_loc[1])-sin(φ_loc[1])) + B_21*((1+exp(Δ/T))*(M_2_loc[1]-n_loc[1]) - exp(Δ/T)*M)-(2*J*κ+γ+2*κ*V[1]) # A_n1_n1
        A_loc[1,2] = B_21*(1+exp(Δ/T))*n_loc[1] # A_n1_X1
        A_loc[1,3] = -2*J*sqrt(n_loc[1]*n_loc[2])*(κ*sin(φ_loc[1])+cos(φ_loc[1])) # A_n1_φ1
        A_loc[1,4] = J*sqrt(n_loc[1]/n_loc[2])*(κ*cos(φ_loc[1])-sin(φ_loc[1])) # A_n1_n2

        A_loc[2,1] = -J*sqrt(n_loc[2]/n_loc[1])*sin(φ_loc[1]) - γ # A_X1_n1
        A_loc[2,3] = -2*J*sqrt(n_loc[1]*n_loc[2])*cos(φ_loc[1]) # A_X1_φ1
        A_loc[2,4] = -J*sqrt(n_loc[1]/n_loc[2])*sin(φ_loc[1]) # A_X1_n2

        A_loc[3,1] = (J/2*(1/(sqrt(n_loc[1]*n_loc[2])) + sqrt(n_loc[2]/n_loc[1])*1/(n_loc[1]))*cos(φ_loc[1]) - J*κ/2*(1/(sqrt(n_loc[1]*n_loc[2])) - sqrt(n_loc[2]/n_loc[1])*1/(n_loc[1]))*sin(φ_loc[1])) # A_φ1_n1
        A_loc[3,3] = -(J*(sqrt(n_loc[1]/n_loc[2]) - sqrt(n_loc[2]/n_loc[1]))*sin(φ_loc[1]) + J*κ*(sqrt(n_loc[1]/n_loc[2]) + sqrt(n_loc[2]/n_loc[1]))*cos(φ_loc[1])) # A_φ1_φ1
        A_loc[3,4] = -(J/2*(1/(sqrt(n_loc[1]*n_loc[2])) + sqrt(n_loc[1]/n_loc[2])*1/(n_loc[2]))*cos(φ_loc[1]) + J*κ/2*(1/(sqrt(n_loc[1]*n_loc[2])) - sqrt(n_loc[1]/n_loc[2])*1/(n_loc[2]))*sin(φ_loc[1])) - J/2*(sqrt(n_loc[3]/n_loc[2])*1/n_loc[2]*(cos(φ_loc[2]) + κ*sin(φ_loc[2]))) # A_φ1_n2
        A_loc[3,6] = J*sqrt(n_loc[3]/n_loc[2])*(κ*cos(φ_loc[2]) -sin(φ_loc[2])) # A_φ1_φ2
        A_loc[3,7] = J/2*1/sqrt(n_loc[2]*n_loc[3])*(cos(φ_loc[2]) + κ*sin(φ_loc[2])) # A_φ1_n3

        # Inner sites
        for i in 2:sites-1
            A_loc[3*i-2, 3*i-5] = J*sqrt(n_loc[i]/n_loc[i-1])*(κ*cos(φ_loc[i-1])+sin(φ_loc[i-1])) # A_nj_n(j-1)
            A_loc[3*i-2, 3*i-3] = 2*J*sqrt(n_loc[i-1]*n_loc[i])*(cos(φ_loc[i-1])-κ*sin(φ_loc[i-1])) # A_nj_φ(j-1)
            A_loc[3*i-2, 3*i-2] = J*sqrt(n_loc[i+1]/n_loc[i])*(κ*cos(φ_loc[i])-sin(φ_loc[i])) + J*sqrt(n_loc[i-1]/n_loc[i])*(sin(φ_loc[i-1])+κ*cos(φ_loc[i-1])) + B_21*((1+exp(Δ/T))*(M_2_loc[i]-n_loc[i]) - exp(Δ/T)*M)-(4*J*κ+γ+2*κ*V[i]) # A_nj_nj
            A_loc[3*i-2, 3*i-1] = B_21*(1+exp(Δ/T))*n_loc[i] # A_nj_Xj
            A_loc[3*i-2, 3*i] = -2*J*sqrt(n_loc[i]*n_loc[i+1])*(κ*sin(φ_loc[i])+cos(φ_loc[i])) # A_nj_φj
            A_loc[3*i-2, 3*i+1] = J*sqrt(n_loc[i]/n_loc[i+1])*(κ*cos(φ_loc[i])-sin(φ_loc[i])) # A_nj_n(j+1)
        end
        for i in 2:sites-1
            A_loc[3*i-1, 3*i-5] = J*sqrt(n_loc[i]/n_loc[i-1])*sin(φ_loc[i-1]) # A_Xj_n(j-1)
            A_loc[3*i-1, 3*i-3] = 2*J*sqrt(n_loc[i-1]*n_loc[i])*cos(φ_loc[i-1]) # A_Xj_φ(j-1)
            A_loc[3*i-1, 3*i-2] = J*sqrt(n_loc[i-1]/n_loc[i])*sin(φ_loc[i-1]) - J*sqrt(n_loc[i+1]/n_loc[i])*sin(φ_loc[i]) - γ # A_Xj_nj
            A_loc[3*i-1, 3*i] = -2*J*sqrt(n_loc[i]*n_loc[i+1])*cos(φ_loc[i]) # A_Xj_φj
            A_loc[3*i-1, 3*i+1] = -J*sqrt(n_loc[i]/n_loc[i+1])*sin(φ_loc[i]) # A_Xj_n(j+1)
        end
        for i in 2:sites-2
            A_loc[3*i, 3*i-5] = -J/2*1/(sqrt(n_loc[i-1]*n_loc[i]))*(cos(φ_loc[i-1]) - κ*sin(φ_loc[i-1])) # A_φj_n(j-1)
            A_loc[3*i, 3*i-3] = J*sqrt(n_loc[i-1]/n_loc[i])*(sin(φ_loc[i-1]) + κ*cos(φ_loc[i-1])) # A_φj_φ(j-1)
            A_loc[3*i, 3*i-2] = J/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) + sqrt(n_loc[i+1]/n_loc[i])*1/(n_loc[i]))*cos(φ_loc[i]) - J*κ/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) - sqrt(n_loc[i+1]/n_loc[i])*1/(n_loc[i]))*sin(φ_loc[i]) + J/2*sqrt(n_loc[i-1]/n_loc[i])*1/n_loc[i]*(cos(φ_loc[i-1]) - κ*sin(φ_loc[i-1])) # A_φj_nj
            A_loc[3*i, 3*i] = -(J*(sqrt(n_loc[i]/n_loc[i+1]) - sqrt(n_loc[i+1]/n_loc[i]))*sin(φ_loc[i]) + J*κ*(sqrt(n_loc[i]/n_loc[i+1]) + sqrt(n_loc[i+1]/n_loc[i]))*cos(φ_loc[i])) # A_φj_φj
            A_loc[3*i, 3*i+1] = -(J/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) + sqrt(n_loc[i]/n_loc[i+1])*1/(n_loc[i+1]))*cos(φ_loc[i]) + J*κ/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) - sqrt(n_loc[i]/n_loc[i+1])*1/(n_loc[i+1]))*sin(φ_loc[i])) - J/2*sqrt(n_loc[i+2]/n_loc[i+1])*1/n_loc[i+1]*(cos(φ_loc[i+1]) + κ*sin(φ_loc[i+1])) # A_φj_n(j+1)
            A_loc[3*i, 3*i+3] = J*sqrt(n_loc[i+2]/n_loc[i+1])*(κ*cos(φ_loc[i+1]) - sin(φ_loc[i+1])) # A_φj_φ(j+1)
            A_loc[3*i, 3*i+4] = J/2*1/sqrt(n_loc[i+1]*n_loc[i+2])*(cos(φ_loc[i+1]) + κ*sin(φ_loc[i+1])) # A_φj_n(j+2)
        end

        # Right site
        A_loc[3*sites-3, 3*sites-8] = J/2*(1/(sqrt(n_loc[sites-2]*n_loc[sites-1])))*(κ*sin(φ_loc[sites-2])-cos(φ_loc[sites-2])) # A_φ(N-1)_n(N-2)
        A_loc[3*sites-3, 3*sites-6] = J*sqrt(n_loc[sites-2]/n_loc[sites-1])*(sin(φ_loc[sites-2]) + κ*cos(φ_loc[sites-2])) # A_φ(N-1)_φ(N-2)
        A_loc[3*sites-3, 3*sites-5] = J/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) + sqrt(n_loc[sites]/n_loc[sites-1])*1/(n_loc[sites-1]))*cos(φ_loc[sites-1]) - J*κ/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) - sqrt(n_loc[sites]/n_loc[sites-1])*1/(n_loc[sites-1]))*sin(φ_loc[sites-1]) + J/2*sqrt(n_loc[sites-2]/n_loc[sites-1])*1/n_loc[sites-1]*(cos(φ_loc[sites-2]) - κ*sin(φ_loc[sites-2])) # A_φ(N-1)_n(N-1)
        A_loc[3*sites-3, 3*sites-3] = -(J*(sqrt(n_loc[sites-1]/n_loc[sites]) - sqrt(n_loc[sites]/n_loc[sites-1]))*sin(φ_loc[sites-1]) + J*κ*(sqrt(n_loc[sites-1]/n_loc[sites]) + sqrt(n_loc[sites]/n_loc[sites-1]))*cos(φ_loc[sites-1])) # A_φ(N-1)_φ(N-1)
        A_loc[3*sites-3, 3*sites-2] = -(J/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) + sqrt(n_loc[sites-1]/n_loc[sites])*1/(n_loc[sites]))*cos(φ_loc[sites-1]) + J*κ/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) - sqrt(n_loc[sites-1]/n_loc[sites])*1/(n_loc[sites]))*sin(φ_loc[sites-1])) # A_φ(N-1)_n(N)

        A_loc[3*sites-2, 3*sites-5] = J*sqrt(n_loc[sites]/n_loc[sites-1])*(sin(φ_loc[sites-1])+ κ*cos(φ_loc[sites-1])) # A_nN_n(N-1)
        A_loc[3*sites-2, 3*sites-3] = -2*J*sqrt(n_loc[sites-1]*n_loc[sites])*(κ*sin(φ_loc[sites-1])-cos(φ_loc[sites-1])) # A_nN_φ(N-1)
        A_loc[3*sites-2, 3*sites-2] = J*sqrt(n_loc[sites-1]/n_loc[sites])*(sin(φ_loc[sites-1])+ κ*cos(φ_loc[sites-1])) + B_21*((1+exp(Δ/T))*(M_2_loc[sites]-n_loc[sites]) - exp(Δ/T)*M)-(2*J*κ+γ+2*κ*V[sites]) # A_nN_nN
        A_loc[3*sites-2, 3*sites-1] = B_21*(1+exp(Δ/T))*n_loc[sites] # A_nN_XN

        A_loc[3*sites-1, 3*sites-5] = J*sqrt(n_loc[sites]/n_loc[sites-1])*sin(φ_loc[sites-1]) # A_XN_n(N-1)
        A_loc[3*sites-1, 3*sites-3] = 2*J*sqrt(n_loc[sites-1]*n_loc[sites])*cos(φ_loc[sites-1]) # A_XN_φ(N-1)
        A_loc[3*sites-1, 3*sites-2] = J*sqrt(n_loc[sites-1]/n_loc[sites])*sin(φ_loc[sites-1]) - γ # A_XN_nN
        
        # 4. Calculate Eigenvalues
        if any(isnan, A_loc) || any(isinf, A_loc)
            current_max_eig = -Inf
        else
            current_max_eig = maximum(real.(eigvals(A_loc)))
        end
        
        push!(max_real_eigenvals, current_max_eig)
        push!(ΔV_scanned, ΔV)

        # Check for crossing
        if !crossing_found && current_max_eig > 0
            crossing_found = true
            
            # Save the matrix at the moment of crossing
            crossing_matrix_A = copy(A_loc)
            
            # Calculate stop threshold
            range_covered = ΔV - ΔV_start
            ΔV_stop_threshold = ΔV + 0.1 * range_covered
        end

        # Stop logic
        if crossing_found && ΔV >= ΔV_stop_threshold
            break
        end
    end

    # --- Plotting Logic (Outside Loop) ---
    if crossing_found
        # Calculate the crossing eigenvectors for the final plot
        crossing_vals, crossing_vecs = eigen(crossing_matrix_A)
        # Find the vector corresponding to the max real eigenvalue
        crossing_eigenvecs_A = crossing_vecs[:, argmax(real.(crossing_vals))]
    else
        println("No stability crossing found in range.")
        return nothing
    end
    
    # Return reduced dataset
    crossing_idx = findfirst(x -> x > 0, max_real_eigenvals)
    return ΔV_scanned[crossing_idx], ΔV_scanned, max_real_eigenvals, crossing_matrix_A, crossing_eigenvecs_A
end