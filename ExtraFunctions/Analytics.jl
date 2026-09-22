using Roots, NLsolve, Optim, LinearAlgebra

# ==============================================================================
# Unified Steady State Calculator for 2-site and N-site logic
# ==============================================================================
function calculate_steady_state(sites::Int; J=1e-1, γ=1.0e-3, V=zeros(Float64, sites),
                                B_21=1.0e-7, M=1.0e8, T=25.0, Δ=-60.0,
                                κ=0.5*B_21*M*exp(Δ/T)/T, n_avg=3.0e3,
                                δ=0.0, guess_n=nothing, guess_phi=nothing)
   
    if sites == 2
        # --- 2-Site (Josephson Junction) Logic ---
        Vl, Vr = V[1], V[2]
        ΔV = Vr - Vl
       
        function dynamics(φ)
            # Analytical expression for the effective phase dynamics in the 2-site case,
            # derived from the steady-state conditions.
            # Calculates the ε q(φ) term in the Adler-equation for a phase difference φ.
            α = (2*J/γ)*sin(φ)
            Q = max(0.0, 1 + α^2 - δ^2) # Safety clamp
            S = (δ + α * sqrt(Q)) / (1 + α^2)

            nl_temp = (1-S)*n_avg # Left site density
            nr_temp = (1+S)*n_avg # Right site density

            term1 = sqrt(nl_temp/nr_temp)
            term2 = sqrt(nr_temp/nl_temp)
           
            return J*(term1-term2)*cos(φ) - J*κ*(term1+term2)*sin(φ)
        end

        # Find Critical Limits, if ΔV is outside these limits, the system cannot be locked
        opt_max = optimize(φ -> -dynamics(φ), 0.0, 2π) # Find the maximum of dynamics(φ) to determine the upper stability limit
        max_force = -Optim.minimum(opt_max)
        opt_min = optimize(dynamics, 0.0, 2π) # Find the minimum of dynamics(φ) to determine the lower stability limit
        min_force = Optim.minimum(opt_min)

        if ΔV > max_force + 1e-9 || ΔV < min_force - 1e-9 # If ΔV is outside the range of possible locking forces, the system cannot be locked
            # Desynchronized
            return (converged = false, residual = NaN, φ = [NaN], n = [NaN, NaN], M_2 = [NaN, NaN])
        else
            # Locked
            try
                φ_star = find_zero(φ -> dynamics(φ) - ΔV, 0.0) # Solve for the phase difference φ_star that satisfies the steady-state condition given ΔV
               
                α_star = (2*J/γ)*sin(φ_star)
                Q_star = max(0.0, 1 + α_star^2 - δ^2)
                S_star = (δ + α_star*sqrt(Q_star)) / (1 + α_star^2)
               
                # Calculate steady-state densities and M_2 values using the solved φ_star
                nl_steady = (1-S_star)*n_avg
                nr_steady = (1+S_star)*n_avg

                denom = 1 + exp(Δ/T)
                m2l = (M*exp(Δ/T) - 2*κ*J/B_21*sqrt(nr_steady/nl_steady)*cos(φ_star) + γ*(1-δ)*n_avg/(B_21*nl_steady) + 2*κ*Vl/B_21) / denom
                m2r = (M*exp(Δ/T) - 2*κ*J/B_21*sqrt(nl_steady/nr_steady)*cos(φ_star) + γ*(1+δ)*n_avg/(B_21*nr_steady) + 2*κ*Vr/B_21) / denom
               
                # Return the results for the locked state
                return (converged = true, residual = 0.0, φ = [φ_star], n = [nl_steady, nr_steady], M_2 = [m2l, m2r])
            catch
                # If the root-finding fails for any reason, return a non-converged result
                return (converged = false, residual = NaN, φ = [NaN], n = [NaN, NaN], M_2 = [NaN, NaN])
            end
        end

    else
        # --- N-Site Multisite Logic ---
        # The steady-state conditions for the N-site case are more complex and require solving a system of nonlinear equations.
        ΔVs = diff(V)
        x0 = zeros(Float64, 2*sites - 1)

        # If guess values for n and φ are provided, use them to initialize the solver; otherwise, use default guesses.
        if !isnothing(guess_n) && !isnothing(guess_phi)
            x0[1:sites-1] .= guess_phi
            x0[sites:end] .= guess_n
        else
            x0[1:sites-1] .= 0.1
            x0[sites:end] .= n_avg
        end

        c_alpha = 2 * J / γ

        function f!(F, x)
            # Define the system of nonlinear equations for the N-site case, derived from the steady-state conditions of the classical field model.
            # F will contain the residuals of the equations, and x contains the variables (phase differences and densities) that we are solving for.
            @inbounds begin
                offset = sites - 1

                # Clamp densities to avoid numerical issues with square roots
                # Left Boundary
                n1, n2 = max(x[offset+1], 1e-9), max(x[offset+2], 1e-9)
                p1, p2 = x[1], x[2]
                F[1] = J*(sqrt(n1/n2) - sqrt(n2/n1))*cos(p1) - J*κ*(sqrt(n1/n2) + sqrt(n2/n1))*sin(p1) + J*sqrt(max(x[offset+3], 1e-9)/n2)*(cos(p2) + κ*sin(p2)) - J - ΔVs[1]

                # Inner Sites
                for i in 2:sites-2
                    ni_prev, ni = max(x[offset+i-1], 1e-9), max(x[offset+i], 1e-9)
                    ni_next, ni_next2 = max(x[offset+i+1], 1e-9), max(x[offset+i+2], 1e-9)
                    pi_prev, pi, pi_next = x[i-1], x[i], x[i+1]

                    sq_fwd, sq_bwd = sqrt(ni/ni_next), sqrt(ni_next/ni)
                    F[i] = J*(sq_fwd - sq_bwd)*cos(pi) - J*κ*(sq_fwd + sq_bwd)*sin(pi) + J*sqrt(ni_next2/ni_next)*(cos(pi_next) + κ*sin(pi_next)) - J*sqrt(ni_prev/ni)*(cos(pi_prev) - κ*sin(pi_prev)) - ΔVs[i]
                end

                # Right Boundary
                n_last_prev, n_last, n_last_next = max(x[offset+sites-2], 1e-9), max(x[offset+sites-1], 1e-9), max(x[offset+sites], 1e-9)
                p_last_prev, p_last = x[sites-2], x[sites-1]
                term_r1, term_r2 = sqrt(n_last/n_last_next), sqrt(n_last_next/n_last)

                F[sites-1] = J*(term_r1 - term_r2)*cos(p_last) - J*κ*(term_r1 + term_r2)*sin(p_last) - J*sqrt(n_last_prev/n_last)*(cos(p_last_prev) - κ*sin(p_last_prev)) + J - ΔVs[sites-1]

                # Density Equations
                a1 = c_alpha * sin(x[1])
                F[sites] = max(x[offset+1], 1e-9) - (n_avg - a1 * sqrt(max(x[offset+1], 1e-9) * max(x[offset+2], 1e-9)))

                for i in 2:sites-1
                    a_prev, a_curr = c_alpha * sin(x[i-1]), c_alpha * sin(x[i])
                    nm, nc, np = max(x[offset+i-1], 1e-9), max(x[offset+i], 1e-9), max(x[offset+i+1], 1e-9)
                    F[sites + i - 1] = nc - (n_avg + a_prev * sqrt(nm * nc) - a_curr * sqrt(nc * np))
                end

                a_last = c_alpha * sin(x[sites-1])
                F[2*sites-1] = max(x[offset+sites], 1e-9) - (n_avg + a_last * sqrt(max(x[offset+sites-1], 1e-9) * max(x[offset+sites], 1e-9)))
            end
        end

        # Solve the system of nonlinear equations using a robust solver. The result will give us the steady-state phase differences and densities for the N-site case.
        res = nlsolve(f!, x0, ftol=1e-8, autodiff=:forward)
        φ = res.zero[1:sites-1]
        n = max.(res.zero[sites:2*sites-1], 1e-9)

        # Calculate M_2 values based on the solved φ and n values using the steady-state conditions for the X fields.
        M_2 = zeros(Float64, sites)
        M_2[1] = 1/(1+exp(Δ/T))*(M*exp(Δ/T) +2/B_21*(J*sqrt(n[2]/n[1])*(sin(φ[1])-κ*cos(φ[1])) + J*κ + γ/2 + κ*V[1]))

        for i in 2:sites-1
            M_2[i] = 1/(1+exp(Δ/T))*(M*exp(Δ/T) +2/B_21*(J*sqrt(n[i+1]/n[i])*(sin(φ[i])-κ*cos(φ[i])) - J*sqrt(n[i-1]/n[i])*(sin(φ[i-1])+κ*cos(φ[i-1])) + 2*J*κ + γ/2 + κ*V[i]))
        end
        M_2[sites] = 1/(1+exp(Δ/T))*(M*exp(Δ/T) +2/B_21*( -J*sqrt(n[sites-1]/n[sites])*(sin(φ[sites-1])+κ*cos(φ[sites-1])) + J*κ + γ/2 + κ*V[sites]))   

        return (converged = converged(res), residual = res.residual_norm, φ = φ, n = n, M_2 = M_2)
    end
end

# ==============================================================================
# Unified Critical ΔV Calculator
# ==============================================================================
function calculate_critical_ΔV(sites::Int; J=1e-1, γ=1.0e-3,
                               B_21=1.0e-7, M=1.0e8, T=25.0, Δ=-60.0, 
                               κ=0.5*B_21*M*exp(Δ/T)/T, n_avg=3.0e3, δ=0.0)
    # Calculate the critical ΔV at which the system transitions from stable to unstable by scanning over a range of ΔV values 
    # and analyzing the eigenvalues of the Jacobian matrix of the linearized system around the steady state.

    # Define the range of ΔV to scan and the resolution of the scan
    start_ΔV = 1e-8      
    stop_ΔV  = 10000.0   
    rel_prec = 1e-4      
    n_points = ceil(Int, log(stop_ΔV / start_ΔV) / log(1 + rel_prec))
    ΔV_range = [0.0; 10 .^ range(log10(start_ΔV), log10(stop_ΔV), length=n_points)]

    # Initialize storage for results
    eigenvals = Vector{Vector{ComplexF64}}()
    max_real_eigenvals = Vector{Float64}()
    ΔV_scanned = Vector{Float64}()
    
    crossing_found = false
    ΔV_stop_threshold = Inf
    ΔV_start = first(ΔV_range)
    
    crossing_matrix = nothing
    crossing_eigenvecs = nothing

    simple_eigenvals = Vector{Vector{ComplexF64}}()
    analytical_eigenvals = Vector{Vector{ComplexF64}}()

    if sites == 2
        # 2-site logic
        for ΔV in ΔV_range
            # Calculate the steady state for the current ΔV to get the necessary parameters for constructing the Jacobian matrix.
            V = [0.0, ΔV]
            sol = calculate_steady_state(2, J=J, γ=γ, V=V, B_21=B_21, M=M, T=T, Δ=Δ, κ=κ, n_avg=n_avg, δ=δ)
            
            if !sol.converged break end
            
            φ_loc = sol.φ[1]
            nl_loc, nr_loc = sol.n[1], sol.n[2]
            M_2_l_loc, M_2_r_loc = sol.M_2[1], sol.M_2[2]

            A_φφ = -(J*(sqrt(nl_loc/nr_loc) - sqrt(nr_loc/nl_loc))*sin(φ_loc)+J*κ*(sqrt(nl_loc/nr_loc) + sqrt(nr_loc/nl_loc))*cos(φ_loc))
            A_φnl = (J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*cos(φ_loc) - J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*sin(φ_loc))
            A_φnr = -(J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*cos(φ_loc) + J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*sin(φ_loc))
            A_nlφ = -2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)+cos(φ_loc))
            A_nlnl = (J*sqrt(nr_loc/nl_loc)*(κ*cos(φ_loc)-sin(φ_loc)) + B_21*((1+exp(Δ/T))*(M_2_l_loc-nl_loc) - exp(Δ/T)*M)-γ-2*κ*0.0)
            A_nlnr = J*sqrt(nl_loc/nr_loc)*(κ*cos(φ_loc)-sin(φ_loc))
            A_nlXl = B_21*(1+exp(Δ/T))*nl_loc
            A_nrφ = -2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)-cos(φ_loc))
            A_nrnl = J*sqrt(nr_loc/nl_loc)*(sin(φ_loc)+κ*cos(φ_loc))
            A_nrnr = (J*sqrt(nl_loc/nr_loc)*(sin(φ_loc)+κ*cos(φ_loc)) + B_21*((1+exp(Δ/T))*(M_2_r_loc-nr_loc) - exp(Δ/T)*M)-γ-2*κ*ΔV)
            A_nrXr = B_21*(1+exp(Δ/T))*nr_loc
            A_Xlφ = -2*J*sqrt(nl_loc*nr_loc)*cos(φ_loc)
            A_Xlnl = -J*sqrt(nr_loc/nl_loc)*sin(φ_loc)-γ
            A_Xlnr = -J*sqrt(nl_loc/nr_loc)*sin(φ_loc)
            A_Xrφ = 2*J*sqrt(nl_loc*nr_loc)*cos(φ_loc)
            A_Xrnl = J*sqrt(nr_loc/nl_loc)*sin(φ_loc)
            A_Xrnr = J*sqrt(nl_loc/nr_loc)*sin(φ_loc)-γ

            A_loc = [A_φφ A_φnl A_φnr 0 0;
                     A_nlφ A_nlnl A_nlnr A_nlXl 0;
                     A_nrφ A_nrnl A_nrnr 0 A_nrXr;
                     A_Xlφ A_Xlnl A_Xlnr 0 0;
                     A_Xrφ A_Xrnl A_Xrnr 0 0]

            # Check for NaN or Inf values in the Jacobian matrix before computing eigenvalues to avoid errors. 
            # If such values are present, we can assign a default value to indicate instability.
            if any(isnan, A_loc) || any(isinf, A_loc)
                push!(eigenvals, fill(-Inf + 0im, 5))
                push!(max_real_eigenvals, -Inf)
            else
                evs = eigvals(A_loc)
                push!(max_real_eigenvals, maximum(real.(evs)))
                push!(eigenvals, evs)
            end

            push!(ΔV_scanned, ΔV)

            # Check for the first crossing of the real part of any eigenvalue from negative to positive, which indicates the onset of instability.
            if !crossing_found && max_real_eigenvals[end] > 0
                crossing_found = true
                ΔV_stop_threshold = ΔV + 0.1 * (ΔV - ΔV_start)
                crossing_matrix = copy(A_loc)
                crossing_eigenvecs = eigvecs(A_loc)
            end

            # For the 2-site case, we can also compute a simplified 2x2 Jacobian matrix that captures the essential dynamics of the phase and density differences,
            # and compare its eigenvalues to the analytical expressions derived from the Adler-like equation for the phase difference.
            θ_dot = J*sqrt(nr_loc/nl_loc)*(cos(φ_loc)+κ*sin(φ_loc))
            
            C_11 = im./2*(B_21*((1+exp(Δ/T))*M_2_l_loc - exp(Δ/T)*M) - γ) + θ_dot
            C_12 = -J*(1-im*κ)
            C_21 = -J*(1-im*κ)
            C_22 = im./2*(B_21*((1+exp(Δ/T))*M_2_r_loc - exp(Δ/T)*M) - γ) + (1 - im*κ)*ΔV + θ_dot

            C_loc = [C_11 C_12; C_21 C_22]

            evs_simple = eigvals(C_loc)

            eig_analytical_1 = J*(1-im*κ)*((sqrt(nr_loc/nl_loc)+sqrt(nl_loc/nr_loc))*cos(φ_loc) + im*(sqrt(nr_loc/nl_loc)-sqrt(nl_loc/nr_loc))*sin(φ_loc))
            eig_analytical_2 = 0


            push!(simple_eigenvals, evs_simple)
            push!(analytical_eigenvals, [eig_analytical_1, eig_analytical_2])

            if crossing_found && ΔV >= ΔV_stop_threshold break end
        end
    else
        # N-site logic
        A_loc = zeros(Float64, 3*sites-1, 3*sites-1)
        V = zeros(Float64, sites)
        prev_n, prev_phi = nothing, nothing

        for ΔV in ΔV_range
            # Calculate the steady state for the current ΔV to get the necessary parameters for constructing the Jacobian matrix.
            V .= LinRange(0, ΔV, sites)
            sol = calculate_steady_state(sites, J=J, γ=γ, V=V, B_21=B_21, M=M, T=T, Δ=Δ, κ=κ, n_avg=n_avg, guess_n=prev_n, guess_phi=prev_phi)
            
            if !sol.converged break end

            n_loc, φ_loc, M_2_loc = sol.n, sol.φ, sol.M_2
            prev_n, prev_phi = n_loc, φ_loc
            A_loc .= 0.0 
            
            # Left site
            A_loc[1,1] = J*sqrt(n_loc[2]/n_loc[1])*(κ*cos(φ_loc[1])-sin(φ_loc[1])) + B_21*((1+exp(Δ/T))*(M_2_loc[1]-n_loc[1]) - exp(Δ/T)*M)-(2*J*κ+γ+2*κ*V[1]) 
            A_loc[1,2] = B_21*(1+exp(Δ/T))*n_loc[1] 
            A_loc[1,3] = -2*J*sqrt(n_loc[1]*n_loc[2])*(κ*sin(φ_loc[1])+cos(φ_loc[1])) 
            A_loc[1,4] = J*sqrt(n_loc[1]/n_loc[2])*(κ*cos(φ_loc[1])-sin(φ_loc[1])) 
            A_loc[2,1] = -J*sqrt(n_loc[2]/n_loc[1])*sin(φ_loc[1]) - γ 
            A_loc[2,3] = -2*J*sqrt(n_loc[1]*n_loc[2])*cos(φ_loc[1]) 
            A_loc[2,4] = -J*sqrt(n_loc[1]/n_loc[2])*sin(φ_loc[1]) 
            A_loc[3,1] = (J/2*(1/(sqrt(n_loc[1]*n_loc[2])) + sqrt(n_loc[2]/n_loc[1])*1/(n_loc[1]))*cos(φ_loc[1]) - J*κ/2*(1/(sqrt(n_loc[1]*n_loc[2])) - sqrt(n_loc[2]/n_loc[1])*1/(n_loc[1]))*sin(φ_loc[1])) 
            A_loc[3,3] = -(J*(sqrt(n_loc[1]/n_loc[2]) - sqrt(n_loc[2]/n_loc[1]))*sin(φ_loc[1]) + J*κ*(sqrt(n_loc[1]/n_loc[2]) + sqrt(n_loc[2]/n_loc[1]))*cos(φ_loc[1])) 
            A_loc[3,4] = -(J/2*(1/(sqrt(n_loc[1]*n_loc[2])) + sqrt(n_loc[1]/n_loc[2])*1/(n_loc[2]))*cos(φ_loc[1]) + J*κ/2*(1/(sqrt(n_loc[1]*n_loc[2])) - sqrt(n_loc[1]/n_loc[2])*1/(n_loc[2]))*sin(φ_loc[1])) - J/2*(sqrt(n_loc[3]/n_loc[2])*1/n_loc[2]*(cos(φ_loc[2]) + κ*sin(φ_loc[2]))) 
            A_loc[3,6] = J*sqrt(n_loc[3]/n_loc[2])*(κ*cos(φ_loc[2]) -sin(φ_loc[2])) 
            A_loc[3,7] = J/2*1/sqrt(n_loc[2]*n_loc[3])*(cos(φ_loc[2]) + κ*sin(φ_loc[2])) 

            # Inner sites
            for i in 2:sites-1
                A_loc[3*i-2, 3*i-5] = J*sqrt(n_loc[i]/n_loc[i-1])*(κ*cos(φ_loc[i-1])+sin(φ_loc[i-1])) 
                A_loc[3*i-2, 3*i-3] = 2*J*sqrt(n_loc[i-1]*n_loc[i])*(cos(φ_loc[i-1])-κ*sin(φ_loc[i-1])) 
                A_loc[3*i-2, 3*i-2] = J*sqrt(n_loc[i+1]/n_loc[i])*(κ*cos(φ_loc[i])-sin(φ_loc[i])) + J*sqrt(n_loc[i-1]/n_loc[i])*(sin(φ_loc[i-1])+κ*cos(φ_loc[i-1])) + B_21*((1+exp(Δ/T))*(M_2_loc[i]-n_loc[i]) - exp(Δ/T)*M)-(4*J*κ+γ+2*κ*V[i]) 
                A_loc[3*i-2, 3*i-1] = B_21*(1+exp(Δ/T))*n_loc[i] 
                A_loc[3*i-2, 3*i] = -2*J*sqrt(n_loc[i]*n_loc[i+1])*(κ*sin(φ_loc[i])+cos(φ_loc[i])) 
                A_loc[3*i-2, 3*i+1] = J*sqrt(n_loc[i]/n_loc[i+1])*(κ*cos(φ_loc[i])-sin(φ_loc[i])) 
                
                A_loc[3*i-1, 3*i-5] = J*sqrt(n_loc[i]/n_loc[i-1])*sin(φ_loc[i-1]) 
                A_loc[3*i-1, 3*i-3] = 2*J*sqrt(n_loc[i-1]*n_loc[i])*cos(φ_loc[i-1]) 
                A_loc[3*i-1, 3*i-2] = J*sqrt(n_loc[i-1]/n_loc[i])*sin(φ_loc[i-1]) - J*sqrt(n_loc[i+1]/n_loc[i])*sin(φ_loc[i]) - γ 
                A_loc[3*i-1, 3*i] = -2*J*sqrt(n_loc[i]*n_loc[i+1])*cos(φ_loc[i]) 
                A_loc[3*i-1, 3*i+1] = -J*sqrt(n_loc[i]/n_loc[i+1])*sin(φ_loc[i]) 
            end
            for i in 2:sites-2
                A_loc[3*i, 3*i-5] = -J/2*1/(sqrt(n_loc[i-1]*n_loc[i]))*(cos(φ_loc[i-1]) - κ*sin(φ_loc[i-1])) 
                A_loc[3*i, 3*i-3] = J*sqrt(n_loc[i-1]/n_loc[i])*(sin(φ_loc[i-1]) + κ*cos(φ_loc[i-1])) 
                A_loc[3*i, 3*i-2] = J/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) + sqrt(n_loc[i+1]/n_loc[i])*1/(n_loc[i]))*cos(φ_loc[i]) - J*κ/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) - sqrt(n_loc[i+1]/n_loc[i])*1/(n_loc[i]))*sin(φ_loc[i]) + J/2*sqrt(n_loc[i-1]/n_loc[i])*1/n_loc[i]*(cos(φ_loc[i-1]) - κ*sin(φ_loc[i-1])) 
                A_loc[3*i, 3*i] = -(J*(sqrt(n_loc[i]/n_loc[i+1]) - sqrt(n_loc[i+1]/n_loc[i]))*sin(φ_loc[i]) + J*κ*(sqrt(n_loc[i]/n_loc[i+1]) + sqrt(n_loc[i+1]/n_loc[i]))*cos(φ_loc[i])) 
                A_loc[3*i, 3*i+1] = -(J/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) + sqrt(n_loc[i]/n_loc[i+1])*1/(n_loc[i+1]))*cos(φ_loc[i]) + J*κ/2*(1/(sqrt(n_loc[i]*n_loc[i+1])) - sqrt(n_loc[i]/n_loc[i+1])*1/(n_loc[i+1]))*sin(φ_loc[i])) - J/2*sqrt(n_loc[i+2]/n_loc[i+1])*1/n_loc[i+1]*(cos(φ_loc[i+1]) + κ*sin(φ_loc[i+1])) 
                A_loc[3*i, 3*i+3] = J*sqrt(n_loc[i+2]/n_loc[i+1])*(κ*cos(φ_loc[i+1]) - sin(φ_loc[i+1])) 
                A_loc[3*i, 3*i+4] = J/2*1/sqrt(n_loc[i+1]*n_loc[i+2])*(cos(φ_loc[i+1]) + κ*sin(φ_loc[i+1])) 
            end

            # Right site
            A_loc[3*sites-3, 3*sites-8] = J/2*(1/(sqrt(n_loc[sites-2]*n_loc[sites-1])))*(κ*sin(φ_loc[sites-2])-cos(φ_loc[sites-2])) 
            A_loc[3*sites-3, 3*sites-6] = J*sqrt(n_loc[sites-2]/n_loc[sites-1])*(sin(φ_loc[sites-2]) + κ*cos(φ_loc[sites-2])) 
            A_loc[3*sites-3, 3*sites-5] = J/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) + sqrt(n_loc[sites]/n_loc[sites-1])*1/(n_loc[sites-1]))*cos(φ_loc[sites-1]) - J*κ/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) - sqrt(n_loc[sites]/n_loc[sites-1])*1/(n_loc[sites-1]))*sin(φ_loc[sites-1]) + J/2*sqrt(n_loc[sites-2]/n_loc[sites-1])*1/n_loc[sites-1]*(cos(φ_loc[sites-2]) - κ*sin(φ_loc[sites-2])) 
            A_loc[3*sites-3, 3*sites-3] = -(J*(sqrt(n_loc[sites-1]/n_loc[sites]) - sqrt(n_loc[sites]/n_loc[sites-1]))*sin(φ_loc[sites-1]) + J*κ*(sqrt(n_loc[sites-1]/n_loc[sites]) + sqrt(n_loc[sites]/n_loc[sites-1]))*cos(φ_loc[sites-1])) 
            A_loc[3*sites-3, 3*sites-2] = -(J/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) + sqrt(n_loc[sites-1]/n_loc[sites])*1/(n_loc[sites]))*cos(φ_loc[sites-1]) + J*κ/2*(1/(sqrt(n_loc[sites-1]*n_loc[sites])) - sqrt(n_loc[sites-1]/n_loc[sites])*1/(n_loc[sites]))*sin(φ_loc[sites-1])) 
            A_loc[3*sites-2, 3*sites-5] = J*sqrt(n_loc[sites]/n_loc[sites-1])*(sin(φ_loc[sites-1])+ κ*cos(φ_loc[sites-1])) 
            A_loc[3*sites-2, 3*sites-3] = -2*J*sqrt(n_loc[sites-1]*n_loc[sites])*(κ*sin(φ_loc[sites-1])-cos(φ_loc[sites-1])) 
            A_loc[3*sites-2, 3*sites-2] = J*sqrt(n_loc[sites-1]/n_loc[sites])*(sin(φ_loc[sites-1])+ κ*cos(φ_loc[sites-1])) + B_21*((1+exp(Δ/T))*(M_2_loc[sites]-n_loc[sites]) - exp(Δ/T)*M)-(2*J*κ+γ+2*κ*V[sites]) 
            A_loc[3*sites-2, 3*sites-1] = B_21*(1+exp(Δ/T))*n_loc[sites] 
            A_loc[3*sites-1, 3*sites-5] = J*sqrt(n_loc[sites]/n_loc[sites-1])*sin(φ_loc[sites-1]) 
            A_loc[3*sites-1, 3*sites-3] = 2*J*sqrt(n_loc[sites-1]*n_loc[sites])*cos(φ_loc[sites-1]) 
            A_loc[3*sites-1, 3*sites-2] = J*sqrt(n_loc[sites-1]/n_loc[sites])*sin(φ_loc[sites-1]) - γ 
                        
            # Check for NaN or Inf values in the Jacobian matrix before computing eigenvalues to avoid errors.
            if any(isnan, A_loc) || any(isinf, A_loc)
                push!(eigenvals, fill(-Inf + 0im, 3*sites-1))
                push!(max_real_eigenvals, -Inf)
            else
                evs = eigvals(A_loc)
                push!(eigenvals, evs)
                push!(max_real_eigenvals, maximum(real.(evs)))
            end
            
            push!(ΔV_scanned, ΔV)

            if !crossing_found && max_real_eigenvals[end] > 0
                crossing_found = true
                ΔV_stop_threshold = ΔV + 0.1 * (ΔV - ΔV_start)
                crossing_matrix = copy(A_loc)
            end

            if crossing_found && ΔV >= ΔV_stop_threshold break end
        end
    end

    if crossing_found && !isnothing(crossing_matrix)
        crossing_vals, crossing_vecs = eigen(crossing_matrix)
        crossing_eigenvecs = crossing_vecs[:, argmax(real.(crossing_vals))]
        crossing_idx = findfirst(x -> x > 0, max_real_eigenvals)
        
        return (critical_ΔV = ΔV_scanned[crossing_idx], 
                ΔV_scanned = ΔV_scanned, 
                eigenvals = im.*eigenvals,                      # Return im.*eigenvals to match the convention used in the analytical eigenvalues
                crossing_matrix = im.*crossing_matrix,          # Return im.*crossing_matrix to match the convention used in the analytical eigenvalues
                crossing_eigenvecs = im.*crossing_eigenvecs,    # Return im.*crossing_eigenvecs to match the convention used in the analytical eigenvalues
                simple_eigenvals = simple_eigenvals,
                analytical_eigenvals = analytical_eigenvals)
    else
        println("No stability crossing found in range.")
        return (critical_ΔV = nothing, 
                ΔV_scanned = ΔV_scanned, 
                eigenvals = im.*eigenvals,                      # Return im.*eigenvals to match the convention used in the analytical eigenvalues
                crossing_matrix = nothing, 
                crossing_eigenvecs = nothing,
                simple_eigenvals = simple_eigenvals,
                analytical_eigenvals = analytical_eigenvals)
    end
end