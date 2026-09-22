using CairoMakie, GLMakie, LinearAlgebra, Printf, JLD2, FileIO

# =========================================================================
# SECTION 1: Load and Process Data
# =========================================================================
function load_data_JJ(filepath)
    println("Loading data from: $filepath")
    data = load(filepath)

    # 1. Extract Parameters
    pars = data["parameters"]
    sites = pars.sites
    J = pars.J
    M = pars.M
    Δ = pars.Δ
    γ = pars.γ
    n_avg = pars.n_avg
    B_21 = pars.B_21
    κ = pars.κ
    save_Δt = pars.save_Δt
    ρ = pars.ρ
    V = pars.V

    # Derived scalar parameters (needed for Table/Analysis)
    Vl = V[1]
    Vr = V[2]
    ΔV = Vr - Vl
    # Assuming rho[2] = 1.0 + δ
    δ = ρ[2] - 1.0

    # 2. Extract Time and State Data
    t_vec = data["t_vec"]
    X_data = data["X_data"]

    # 3. Process Arrays (Photon numbers, Populations, Phases)
    # Pre-allocate arrays
    len = length(t_vec)
    n = zeros(Float64, sites, len)
    M_2 = zeros(Float64, sites, len)
    M_1 = zeros(Float64, sites, len)
    theta = zeros(Float64, sites, len)
    φ = zeros(Float64, len)

    # Calculate values
    n[1,:] = abs2.(X_data[1,:])
    n[2,:] = abs2.(X_data[2,:])

    M_2[1,:] = real.(X_data[3,:]) .- n[1,:]
    M_2[2,:] = real.(X_data[4,:]) .- n[2,:]

    M_1[1,:] = M .- M_2[1,:]
    M_1[2,:] = M .- M_2[2,:]

    theta[1,:] = angle.(X_data[1,:])
    theta[2,:] = angle.(X_data[2,:])

    # Phase difference (θ1 - θ2)
    φ[:] = angle.(conj.(X_data[1,:]) .* X_data[2,:])

    println("Data loaded and processed successfully.")

    # Return everything as a NamedTuple
    return (
        t_vec=t_vec, X_data=X_data, J=J, M=M, Δ=Δ, γ=γ, n_avg=n_avg, B_21=B_21, κ=κ, 
        save_Δt=save_Δt, ρ=ρ, V=V, Vl=Vl, Vr=Vr, ΔV=ΔV, δ=δ,
        n=n, M_1=M_1, M_2=M_2, theta=theta, φ=φ
    )
end


# =========================================================================
# SECTION 2: Plot Results
# =========================================================================
function plot_results_JJ(t_vec, X_data, J, n, theta, φ, M_1, M_2, ss_result, plot_ΔV_range, plot_real_eigenvals, plot_imag_eigenvals, n_FT, φ_FT; save_path="Figures/ClassicalField_X_JJ.png")
    
    CairoMakie.activate!()

    f1 = Figure(size = (1000, 1200))

    # --- Plot 1: Photon Number ---
    ax1 = Axis(f1[1,1], title="Photon Number Time Series", xlabel="Time (1/meV)", ylabel="Photon Number")
    lines!(ax1, t_vec*J, n[1,:], label="Site 1")
    lines!(ax1, t_vec*J, n[2,:], label="Site 2")
    lines!(ax1, t_vec*J, (n[1,:].+n[2,:])./2, label="Average")
    axislegend(position = :rb)

    # --- Plot 2: Phase ---
    ax2 = Axis(f1[1,2], title="Phase Time Series", xlabel="Time (1/meV)", ylabel="Phase (rad)")
    lines!(ax2, t_vec*J, theta[1,:], label="Site 1")
    lines!(ax2, t_vec*J, theta[2,:], label="Site 2")
    axislegend(position = :rb)

    # --- Plot 3: Phase Difference ---
    ax3 = Axis(f1[2,1], title="Phase difference Time Series", xlabel="Time (1/meV)", ylabel="Phase difference (rad)")
    lines!(ax3, t_vec*J, φ, label="θ₁ - θ₂")

    # --- Plot 4: Excited Molecules ---
    ax4 = Axis(f1[2,2], title="Excited Molecule Number Time Series", xlabel="Time (1/meV)", ylabel="Number of Excited Molecules")
    lines!(ax4, t_vec*J, M_2[1,:], label="Site 1")
    lines!(ax4, t_vec*J, M_2[2,:], label="Site 2")
    axislegend(position = :rb)

    # --- Plot 5: Ground State Molecules ---
    ax5 = Axis(f1[3,1], title="Ground State Molecule Number Time Series", xlabel="Time (1/meV)", ylabel="Number of Ground State Molecules")
    lines!(ax5, t_vec*J, M_1[1,:], label="Site 1")
    lines!(ax5, t_vec*J, M_1[2,:], label="Site 2")
    axislegend(position = :rb)

    # --- Plot 6: Phase Space Dynamics ---
    ax6 = Axis(f1[3,2], title = "Phase Space Dynamics", xlabel = L"\phi", ylabel = L"\dot{\phi} \ \mathrm{(meV)}")
    lines!(ax6, ss_result.plot_φ, ss_result.plot_φ_dot, color = :blue, label=L"F(\phi)")
    hlines!(ax6, [ss_result.current_ΔV], color = :red, linestyle = :dash, label="ΔV = $(round(ss_result.current_ΔV, digits=3))")
    if ss_result.status == "Locked"
        scatter!(ax6, [ss_result.φ_final], [ss_result.current_ΔV], color = :red, markersize = 15, label="Steady State")
    end
    axislegend(position = :rb)

    display(f1)
    
    # Ensure directory exists before saving
    mkpath(dirname(save_path))
    save(save_path, f1)
    println("Plot saved to: $save_path")

    f2 = Figure(size = (1200, 600))

    # --- Plot 1: Real Part of Eigenvalues vs Potential difference ---
    ax1 = Axis(f2[1,1], title="Real Part of Eigenvalues vs Potential difference |ΔV|", xlabel="|ΔV| [meV]", ylabel="Real Eigenvalues", limits = (nothing, nothing, nothing, nothing))
    for i in 1:5
        lines!(ax1, plot_ΔV_range, getindex.(plot_real_eigenvals, i), label="Real Eigenvalue $i")
    end
    lines!(ax1, plot_ΔV_range, zeros(length(plot_ΔV_range)), color = :red, linestyle = :dash, label="Stability Threshold")
    
    # --- Plot 2: Imaginary Part of Eigenvalues vs Potential difference ---
    ax2 = Axis(f2[1,2], title="Imaginary Part of Eigenvalues vs Potential difference |ΔV|", xlabel="|ΔV| [meV]", ylabel="Imaginary Eigenvalues", limits = (nothing, nothing, nothing, nothing))
    for i in 1:5
        lines!(ax2, plot_ΔV_range, getindex.(plot_imag_eigenvals, i), label="Imaginary Eigenvalue $i")
    end
    display(f2)

    f3 = Figure(size = (1200, 600))

    # --- Plot 1: Fourier Transform of Site 1 Photon Number ---
    ax1 = Axis(f3[1, 1], title="Fourier Transform of Site 1 Photon Number", xlabel="Frequency (1/meV)", ylabel="FT(n)")
    lines!(ax1, 1:length(n_FT[1, 2:end]), abs.(n_FT[1, 2:end]), label="Site 1")

    # --- Plot 2: Fourier Transform of Site 2 Photon Number ---
    ax2 = Axis(f3[1, 2], title="Fourier Transform of Site 2 Photon Number", xlabel="Frequency (1/meV)", ylabel="FT(n)")
    lines!(ax2, 1:length(n_FT[2, 2:end]), abs.(n_FT[2, 2:end]), label="Site 2")

    # --- Plot 3: Fourier Transform of Phase ---
    ax3 = Axis(f3[2,1], title="Fourier Transform of Phase", xlabel="Frequency (1/meV)", ylabel="FT(φ)")
    lines!(ax3, 1:length(φ_FT[2:end]), abs.(φ_FT[2:end]), label="Phase")

    display(f3)
end

# =========================================================================
# SECTION 3: Stability Video Generation
# =========================================================================
function generate_stability_video(matrix_sequence; filename="stability_evolution.mp4")
    
    GLMakie.activate!()

    # 7.1 Helper Function (Internal)
    function get_data(A)
        F = eigen(A)
        p = sortperm(real.(F.values)) 
        sorted_vecs = F.vectors[:, p]
        return A', abs.(sorted_vecs') 
    end

    # 7.2 Calculate Global Symmetric Limits
    global_max_abs = maximum(maximum.(abs.(m) for m in matrix_sequence))
    limit_val = global_max_abs
    color_range_A = (-limit_val, limit_val) 

    # 7.3 Setup Figure
    f_video = Figure(size = (1200, 500))

    # -- Left Plot: Jacobian --
    ax1 = Axis(f_video[1,1], title = "Jacobian (Red=+, Blue=-)", xlabel = "Column", ylabel = "Row", yreversed = true)
    obs_matrix_A = Observable(matrix_sequence[1]') 

    hm1 = heatmap!(ax1, obs_matrix_A, colormap = :balance, colorrange = color_range_A)
    Colorbar(f_video[1,2], hm1, label = "Value")

    # -- Right Plot: Eigenvector Magnitudes --
    ax2 = Axis(f_video[1,3], title = "Eigenvector Magnitudes", yreversed = true)
    
    # Initialize with first frame
    F_init = eigen(matrix_sequence[1])
    p_init = sortperm(real.(F_init.values))
    obs_matrix_V = Observable(abs.(F_init.vectors[:, p_init]'))

    hm2 = heatmap!(ax2, obs_matrix_V, colormap = :plasma, colorrange = (0, 1.0))
    Colorbar(f_video[1,4], hm2, label = "|v|")

    # 7.4 Record
    step = 50 
    record(f_video, filename, 1:step:length(matrix_sequence); framerate = 60) do i
        A_curr = matrix_sequence[i]
        
        # Update Jacobian (Raw Signed Data)
        obs_matrix_A[] = A_curr'
        
        # Update Eigenvectors (Sorted & Magnitude)
        F = eigen(A_curr)
        p = sortperm(real.(F.values))
        obs_matrix_V[] = abs.(F.vectors[:, p]')
    end
    
    println("Video saved to: $filename")
    
    # Switch back to CairoMakie for safe static plotting later
    CairoMakie.activate!()
end

# =========================================================================
# SECTION 4: Comparison Table and System Status
# =========================================================================
function print_stats_JJ(ss_result, φ, n, M_2, J, γ, Vl, Vr, ΔV, δ, κ, ΔV_max_stability)
    
    println("\n\nComparison: Steady State vs Simulation")
    println("-------------------------------------------------------------------------------------------")
    @printf("%-10s | %-18s | %-18s | %-18s | %-18s\n", "Variable", "Steady State", "Simulation", "Abs. Diff.", "Rel. Diff.")
    println("-------------------------------------------------------------------------------------------")

    labels = ["φ", "nl", "nr", "M_2_l", "M_2_r"]

    ss_values = [
        ss_result.φ_final, 
        ss_result.nl_final, 
        ss_result.nr_final, 
        ss_result.M_2_l, 
        ss_result.M_2_r
    ]

    # Simulation statistics (Last 10%)
    n_steps = length(φ)
    start_idx = max(1, floor(Int, 0.9 * n_steps))
    avg_range = start_idx:n_steps
    calc_mean(v) = sum(v) / length(v)

    sim_values = [
        calc_mean(φ[avg_range]), 
        calc_mean(n[1, avg_range]), 
        calc_mean(n[2, avg_range]), 
        calc_mean(M_2[1, avg_range]), 
        calc_mean(M_2[2, avg_range])
    ]

    for i in eachindex(labels)
        ss_val = ss_values[i]
        sim_val = sim_values[i]
        abs_diff = abs(sim_val - ss_val)

        if abs(ss_val) > 1e-15
            rel_diff = abs((sim_val - ss_val) / ((sim_val +ss_val)/2))
        else
            rel_diff = abs(sim_val - ss_val)
        end

        @printf("%-10s | %-18.6e | %-18.6e | %-18.6e | %-18.6e\n", labels[i], ss_val, sim_val, abs_diff, rel_diff)
    end
    println("-------------------------------------------------------------------------------------------")

    # 8.2 System Status and Parameters
    println("\nSystem Status Analysis")
    println("----------------------")
    println("Parameters:")
    @printf("  J  = %.4e [meV]\n", J)
    @printf("  γ  = %.4e [meV]\n", γ)
    @printf("  Vl = %.4e [meV]\n", Vl)
    @printf("  Vr = %.4e [meV]\n", Vr)
    @printf("  ΔV = %.4e [meV]\n", ΔV)
    @printf("  δ  = %.4f\n", δ)
    @printf("  κ  = %.4e [meV]\n", κ)

    adler_limit_val = abs(ss_result.ΔV_abs_max)
    if isnothing(ΔV_max_stability)
        stability_limit_val = NaN # Treat as infinite stability if none found
        limit_str = "Not Found (> Scan Range)"
    else
        stability_limit_val = abs(ΔV_max_stability)
        limit_str = @sprintf("%.4f", stability_limit_val)
    end
    current_ΔV_ratio = abs(ΔV)

    @printf("\nSynchronization Boundaries |ΔV| \n")
    @printf("  Adler Limit     : %.4f\n", adler_limit_val)
    @printf("  Stability Limit : %.4f\n", stability_limit_val)

    println("\nFinal Status:")
    if current_ΔV_ratio > adler_limit_val
        println("  >> DESYNCHRONIZED (Adler Mechanism)")
        println("  Reason: The potential difference exceeds the fundamental phase-locking range.")
    elseif current_ΔV_ratio > stability_limit_val
        println("  >> DESYNCHRONIZED (Stability Mechanism)")
        println("  Reason: Inside Adler range, but the phase-locked state is dynamically unstable.")
    else
        println("  >> SYNCHRONIZED")
        println("  Reason: System is within both Adler and stability boundaries.")
    end
    println("-------------------------------------------------------------------------------------------")
end