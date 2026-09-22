using Roots
using Optim
using CairoMakie
using LinearAlgebra

function calculate_steady_state_JJ(J, γ, Vl, Vr, B_21=1.0e-7, M=1.0e8, T=25.0, Δ=-60.0, κ=0.5*B_21*M*exp(Δ/T)/T, n_avg=3.0e3, δ = 0.0)
    # --- 1. System Parameters ---
    ΔV = Vr - Vl                      # potential difference [meV]
    
    # --- 2. Dynamics Function (Internal Force) ---
    function dynamics(φ)
        α = (2*J/γ)*sin(φ)

        Q = 1 + α^2 - δ^2
        if Q < 0
             Q = 0.0 # Safety clamp for unphysical parameters
        end
        S = (δ + α * sqrt(Q)) / (1 + α^2)

        nl_temp = (1-S)*n_avg
        nr_temp = (1+S)*n_avg

        # Avoid division by zero in square roots if n drops to 0 (unlikely here but safe)
        term1 = sqrt(nl_temp/nr_temp)
        term2 = sqrt(nr_temp/nl_temp)
        
        return J*(term1-term2)*cos(φ) - J*κ*(term1+term2)*sin(φ)
    end

    # --- 3. Pre-calculate Plotting Data & Limits ---
    # We do this early so we can return it even if the steady state fails
    φ_plot = range(-2π, 2π, length=1000)
    φ_dot_plot = dynamics.(φ_plot)

    # Find Critical Limits (Max/Min Restoring Forces)
    opt_max = optimize(φ -> -dynamics(φ), 0.0, 2π)
    max_force = -Optim.minimum(opt_max) 
    
    opt_min = optimize(dynamics, 0.0, 2π)
    min_force = Optim.minimum(opt_min)

    # Determine which limit applies to the current ΔV direction
    limit_val = (ΔV >= 0) ? max_force : min_force
    φ_crit_limit = (ΔV >= 0) ? Optim.minimizer(opt_max) : Optim.minimizer(opt_min)
    ΔV_abs_max = abs(limit_val)

    # --- 4. Helper to Calculate M2 Populations ---
    # Calculates Excited State populations given a specific phase and photon numbers
    function calc_M2(φ_in, nl_in, nr_in)
        # Denominator
        denom = 1+exp(Δ/T)
        
        # Left M2
        m2l = (M*exp(Δ/T) 
              -2*κ*J/B_21*sqrt(nr_in/nl_in)*cos(φ_in) 
              +γ*(1-δ)*n_avg/(B_21*nl_in) 
              +2*κ*Vl/B_21) / denom
              
        # Right M2
        m2r = (M*exp(Δ/T) 
              -2*κ*J/B_21*sqrt(nl_in/nr_in)*cos(φ_in) 
              +γ*(1+δ)n_avg/(B_21*nr_in)
              +2*κ*Vr/B_21) / denom
              
        return m2l, m2r
    end

    # --- 5. Main Solver Logic ---
    
    # CHECK: Adler Condition (Is ΔV outside the locking range?)
    if ΔV > max_force + 1e-9 || ΔV < min_force - 1e-9
        # --- CASE A: Desynchronized (Return Fallbacks) ---
        return (
            status     = "Desynchronised",
            φ_final    = NaN,        # No steady phase
            nl_final   = NaN,      # Fallback to average
            nr_final   = NaN,      # Fallback to average
            M_2_l      = NaN,        # Undefined without steady phase
            M_2_r      = NaN,        # Undefined without steady phase
            
            # Critical / Plotting Info (Always valid)
            ΔV_abs_max = ΔV_abs_max,
            φ_crit     = φ_crit_limit,
            plot_φ     = φ_plot,
            plot_φ_dot = φ_dot_plot,
            current_ΔV = ΔV
        )
    else
        # --- CASE B: Locked (Solve for Steady State) ---
        try
            # Find root for: dynamics(φ) - ΔV = 0
            φ_star = find_zero(φ -> dynamics(φ) - ΔV, 0.0)
            
            # Calculate final state properties
            α_star = (2*J/γ)*sin(φ_star)

            Q_star = 1 + α_star^2 - δ^2
            if Q_star < 0
                Q_star = 0.0
            end
            S_star = (δ + α_star*sqrt(Q_star)) / (1 + α_star^2)

            nl_steady = (1-S_star)*n_avg
            nr_steady = (1+S_star)*n_avg
            
            m2l, m2r = calc_M2(φ_star, nl_steady, nr_steady)
            
            
            return (
                status     = "Locked",
                φ_final    = φ_star,
                nl_final   = nl_steady,
                nr_final   = nr_steady,
                M_2_l      = m2l,
                M_2_r      = m2r,
                
                # Critical / Plotting Info
                ΔV_abs_max = ΔV_abs_max,
                φ_crit     = φ_crit_limit,
                plot_φ     = φ_plot,
                plot_φ_dot = φ_dot_plot,
                current_ΔV = ΔV
            )
        catch e
            # Fallback if solver fails numerically inside the range
            return (
                status     = "Solver Failed",
                φ_final    = NaN,
                nl_final   = NaN,
                nr_final   = NaN,
                M_2_l      = NaN,
                M_2_r      = NaN,
                ΔV_abs_max = ΔV_abs_max,
                φ_crit     = φ_crit_limit,
                plot_φ     = φ_plot,
                plot_φ_dot = φ_dot_plot,
                current_ΔV = ΔV
            )
        end
    end
end


function calculate_critical_ΔV_JJ(J, γ, B_21=1.0e-7, M=1.0e8, T=25.0, Δ=-60.0, κ=0.5*B_21*M*exp(Δ/T)/T, n_avg=3.0e3, δ=0.0)
    # Prepare storage for eigenvalues
    real_eigenvals = Vector{Vector{Float64}}()
    imag_eigenvals = Vector{Vector{Float64}}()
    max_real_eigenvals = Vector{Float64}()
    A_matrix = Vector{Matrix{Float64}}()
    eigenvecs = Vector{Matrix{ComplexF64}}()

    # Define the range of Vr values to scan
    start_ΔV = 1e-6      
    stop_ΔV  = 100.0   
    rel_prec = 1e-4      

    n_points = ceil(Int, log(stop_ΔV / start_ΔV) / log(1 + rel_prec))
    log_ΔV = 10 .^ range(log10(start_ΔV), log10(stop_ΔV), length=n_points)
    ΔV_range = [0.0; log_ΔV]
    # Storage for the specific Vr values we actually scan
    Vr_scanned = Vector{Float64}()

    # Variables to track the dynamic stopping condition
    crossing_found = false
    Vr_stop_threshold = Inf
    Vr_start = first(ΔV_range) # Typically 0.0

    for Vr in ΔV_range
        # Update Potentials
        Vl_loc = 0.0
        Vr_loc = Vr

        # --- 2. Define Steady State parameters ---
        ss_status, φ_loc, nl_loc, nr_loc, M_2_l_loc, M_2_r_loc, _, _, _, _, _ = calculate_steady_state_JJ(J, γ, Vl_loc, Vr_loc, B_21, M, T, Δ, κ, n_avg, δ)
        if ss_status != "Locked"
            break  # Skip non-locked states
        end
        # --- 3. Define derivative matrix ---
        # We define the matrix that gives the matrix equation dX/dt = A*X
        # where X = [δφ_loc, δnl_loc, δnr_loc, δXl_loc, δXr_loc]
        # The elements of A are derived from linearizing the system around the steady state
        # We define each element of the matrix A seperately by
        # A = [A_φφ, A_φnl, A_φnr, 0,0;
        #      A_nlφ, A_nlnl, A_nlnr, A_nlXl,0;
        #      A_nrφ, A_nrnl, A_nrnr, 0,A_nrXr;
        #      A_Xlφ, A_Xlnl, A_Xlnr, 0, 0;
        #      A_Xrφ, A_Xrnl, A_Xrnr, 0, 0]
        A_φφ = -(J*(sqrt(nl_loc/nr_loc) - sqrt(nr_loc/nl_loc))*sin(φ_loc)+J*κ*(sqrt(nl_loc/nr_loc) + sqrt(nr_loc/nl_loc))*cos(φ_loc))
        A_φnl = (J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*cos(φ_loc) - J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*sin(φ_loc))
        A_φnr = -(J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*cos(φ_loc) + J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*sin(φ_loc))

        A_nlφ = -2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)+cos(φ_loc))
        A_nlnl = (J*sqrt(nr_loc/nl_loc)*(κ*cos(φ_loc)-sin(φ_loc)) + B_21*((1+exp(Δ/T))*(M_2_l_loc-nl_loc) - exp(Δ/T)*M)-γ-2*κ*Vl_loc)
        A_nlnr = J*sqrt(nl_loc/nr_loc)*(κ*cos(φ_loc)-sin(φ_loc))
        A_nlXl = B_21*(1+exp(Δ/T))*nl_loc

        A_nrφ = -2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)-cos(φ_loc))
        A_nrnl = J*sqrt(nr_loc/nl_loc)*(sin(φ_loc)+κ*cos(φ_loc))
        A_nrnr = (J*sqrt(nl_loc/nr_loc)*(sin(φ_loc)+κ*cos(φ_loc)) + B_21*((1+exp(Δ/T))*(M_2_r_loc-nr_loc) - exp(Δ/T)*M)-γ-2*κ*Vr_loc)
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
        
        # Determine the matrix in the Y = [δφ_loc, δn+_loc, δn-_loc, δX+_loc, δX-_loc] basis
        # where δn+ = δnl + δnr and δn- = δnl - δnr (same for X)
        # This is done by applying the transformation matrix T and its inverse
        Q = [1 0 0 0 0;
             0 1 1 0 0;
             0 1 -1 0 0;
             0 0 0 1 1;
             0 0 0 1 -1]
        Q_inv = inv(Q)
        B_loc = Q * A_loc * Q_inv

        # --- 4. Calculate Eigenvalues ---
        if any(isnan, A_loc) || any(isinf, A_loc)
            # If A contains NaN or Inf, skip eigenvalue calculation
            # and assign a large negative value to indicate stability
            push!(max_real_eigenvals, -Inf)
            push!(real_eigenvals, fill(-Inf, 5))
            push!(A_matrix, A_loc)
            push!(eigenvecs, zeros(ComplexF64, 5, 5))
        else
            eigenvals = eigvals(A_loc)
            push!(max_real_eigenvals, maximum(real.(eigenvals)))
            push!(real_eigenvals, real.(eigenvals))
            push!(imag_eigenvals, imag.(eigenvals))
            push!(A_matrix, A_loc)
            push!(eigenvecs, eigvecs(A_loc))
        end

        # --- 5. Dynamic Range Check ---
        push!(Vr_scanned, Vr) # Save the current Vr value

        # Check if we have crossed the stability threshold (max eigenvalue > 0)
        current_max_eig = max_real_eigenvals[end]

        if !crossing_found && current_max_eig > 0
            # Crossing detected!
            crossing_found = true
            
            # Calculate the range covered so far (from start to crossing)
            range_covered = Vr - Vr_start
            
            # Set the stop threshold to current Vr + 10% of that covered range
            Vr_stop_threshold = Vr + 0.1 * range_covered

            # Determine the eigenvector of the eigenvalue with the largest real part at the crossing point
            crossing_matrix_A = A_loc
            crossing_eigenvecs_A = eigvecs(crossing_matrix_A)[:, argmax(real.(eigvals(crossing_matrix_A)))]
            crossing_matrix_B = B_loc
            crossing_eigenvecs_B = eigvecs(crossing_matrix_B)[:, argmax(real.(eigvals(crossing_matrix_B)))]

        end

        # Stop the loop if we exceed the calculated threshold
        if crossing_found && Vr >= Vr_stop_threshold
            break
        end
    end
    crossing_idx = findfirst(x -> x > 0, max_real_eigenvals)

    if isnothing(crossing_idx)
        println("No critical threshold found in the scanned range.")
        # Return sensible defaults, e.g., the last scanned value or nothing
        return nothing, Vr_scanned, real_eigenvals, imag_eigenvals, max_real_eigenvals, A_matrix, eigenvecs
    else
        return Vr_scanned[crossing_idx], Vr_scanned, real_eigenvals, imag_eigenvals, max_real_eigenvals, A_matrix, eigenvecs
    end
end


function calculate_critical_delta_JJ(J, γ, Vl, Vr, B_21=1.0e-7, M=1.0e8, T=25.0, Δ=-60.0, κ=0.5*B_21*M*exp(Δ/T)/T, n_avg=3.0e3)
    # Prepare storage
    max_real_eigenvals = Vector{Float64}()
    δ_scanned = Vector{Float64}()
    critical_δ = Vector{Float64}()
    
    # Define range
    δ_step = 0.001
    δ_range = -1.0:δ_step:1.0
    start_bound = first(δ_range)
    
    # State tracking: Initialize as nothing (no assumption)
    previous_stable = nothing

    for δ in δ_range

        # --- 1. Calculate Steady State ---
        ss = calculate_steady_state_JJ(J, γ, Vl, Vr, B_21, M, T, Δ, κ, n_avg, δ)
        
        # Determine stability for this specific delta
        is_stable_now = false
        max_eig = NaN

        if ss.status == "Locked"
            # Unpack values
            nl_loc = ss.nl_final
            nr_loc = ss.nr_final
            φ_loc = ss.φ_final
            M_2_l_loc = ss.M_2_l
            M_2_r_loc = ss.M_2_r

            # --- 2. Construct Jacobian ---
            a_loc = -(J*(sqrt(nl_loc/nr_loc) - sqrt(nr_loc/nl_loc))*sin(φ_loc)+J*κ*(sqrt(nl_loc/nr_loc) + sqrt(nr_loc/nl_loc))*cos(φ_loc))
            b_loc = (J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*cos(φ_loc) - J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*sin(φ_loc))
            c_loc = -(J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*cos(φ_loc) + J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*sin(φ_loc))
            d_loc = 0; e_loc = 0

            f_loc = -(2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)+cos(φ_loc)))
            g_loc = (J*sqrt(nr_loc/nl_loc)*(κ*cos(φ_loc)-sin(φ_loc)) + B_21*((1+exp(Δ/T))*M_2_l_loc - exp(Δ/T)*M)-γ-2*κ*Vl)
            h_loc = J*sqrt(nl_loc/nr_loc)*(κ*cos(φ_loc)-sin(φ_loc))
            i_loc = B_21*(1+exp(Δ/T))*nl_loc; j_loc = 0

            k_loc = -2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)-cos(φ_loc))
            l_loc = J*sqrt(nr_loc/nl_loc)*(sin(φ_loc)+κ*cos(φ_loc))
            m_loc = (J*sqrt(nl_loc/nr_loc)*(sin(φ_loc)+κ*cos(φ_loc)) + B_21*((1+exp(Δ/T))*M_2_r_loc - exp(Δ/T)*M)-γ-2*κ*Vr)
            n_loc = 0; o_loc = B_21*(1+exp(Δ/T))*nr_loc

            p_loc = 2*J*κ*sqrt(nl_loc*nr_loc)*sin(φ_loc)
            q_loc = -(B_21*((1+exp(Δ/T))*M_2_l_loc-exp(Δ/T)*M) + J*κ*sqrt(nr_loc/nl_loc)*cos(φ_loc) - 2*κ*Vl)
            r_loc = -J*κ*sqrt(nl_loc/nr_loc)*cos(φ_loc)
            s_loc = -B_21*(1+exp(Δ/T))*nl_loc; t_loc = 0

            u_loc = 2*J*κ*sqrt(nl_loc*nr_loc)*sin(φ_loc)
            v_loc = -J*κ*sqrt(nr_loc/nl_loc)*cos(φ_loc)
            w_loc = -(B_21*((1+exp(Δ/T))*M_2_r_loc-exp(Δ/T)*M) + J*κ*sqrt(nl_loc/nr_loc)*cos(φ_loc) - 2*κ*Vr)
            x_loc = 0; y_loc = -B_21*(1+exp(Δ/T))*nr_loc

            A_loc = [a_loc b_loc c_loc d_loc e_loc;
                     f_loc g_loc h_loc i_loc j_loc;
                     k_loc l_loc m_loc n_loc o_loc;
                     p_loc q_loc r_loc s_loc t_loc;
                     u_loc v_loc w_loc x_loc y_loc]
            
            # --- 3. Calculate Eigenvalues ---
            if !any(isnan, A_loc) && !any(isinf, A_loc)
                eigenvals = eigvals(A_loc)
                max_eig = maximum(real.(eigenvals))
                
                if max_eig < 0
                    is_stable_now = true
                end
            end
        end
        
        push!(max_real_eigenvals, max_eig)

        # --- 4. Transition Logic ---
        if isnothing(previous_stable)
            # First Iteration
            previous_stable = is_stable_now
            if is_stable_now
                push!(critical_δ, δ)
            end
        else
            # Check for State Change
            if previous_stable != is_stable_now
                
                # SNAP LOGIC: If we become stable very close to the start boundary (-1.0),
                # we assume the region extends to the limit (handling numerical failure at exactly -1.0)
                if is_stable_now && abs(δ - start_bound) <= (2 * δ_step)
                     # Only push -1.0 if we haven't already
                     if isempty(critical_δ) || critical_δ[end] != start_bound
                        push!(critical_δ, start_bound)
                     end
                else
                     push!(critical_δ, δ)
                end
                
                previous_stable = is_stable_now
            end
        end

        push!(δ_scanned, δ)
    end
    
    # --- 5. End Boundary Check ---
    # If stable at the end, close the range with the last scanned point
    if !isnothing(previous_stable) && previous_stable
        push!(critical_δ, last(δ_scanned))
    end

    return critical_δ, δ_scanned, max_real_eigenvals
end