using CairoMakie, GLMakie, LinearAlgebra, Printf, JLD2, FileIO, Statistics, LaTeXStrings


# =========================================================================
# SECTION 1: Load and Process Data (Multi-Site)
# =========================================================================
function load_data_multisite(filepath)
    println("Loading data from: $filepath")
    data = load(filepath)

    # 1. Extract Parameters
    pars = data["parameters"]
    # Extract specific fields
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

    # 2. Extract Time and State Data
    t_vec = data["t_vec"]
    X_data = data["X_data"]

    # 3. Process Arrays
    # Photon numbers (n)
    n = zeros(Float64, sites, length(t_vec))
    n[1:sites,:] .= abs2.(X_data[1:sites,:])

    # Excited molecules (M2) and Ground state (M1)
    M_2 = zeros(Float64, sites, length(t_vec))
    M_2[1:sites,:] = real.(X_data[sites+1:2*sites,:]) .- n[1:sites,:]
    
    M_1 = zeros(Float64, sites, length(t_vec))
    M_1[1:sites,:] = M .- M_2[1:sites,:]

    # Phases (theta)
    theta = zeros(Float64, sites, length(t_vec))
    theta[1:sites,:] = angle.(X_data[1:sites,:])

    # Phase differences (phi between site i and i+1)
    φ = zeros(Float64, sites-1, length(t_vec))
    φ[1:sites-1,:] = angle.(conj.(X_data[1:sites-1,:]) .* X_data[2:sites,:])

    println("Data loaded and processed successfully.")

    # Return as NamedTuple
    return (
        t_vec=t_vec, sites=sites, J=J, M=M, Δ=Δ, γ=γ, n_avg=n_avg, 
        B_21=B_21, κ=κ, save_Δt=save_Δt, ρ=ρ, V=V,
        n=n, M_1=M_1, M_2=M_2, theta=theta, φ=φ
    )
end

function print_stats_multisite(sim_data, ss_result, ΔV_max_stability)
    # --- 1. Extract Parameters ---
    sites = sim_data.sites
    J = sim_data.J
    γ = sim_data.γ
    V = sim_data.V
    n_avg = sim_data.n_avg

    # Simulation Data Statistics (Last 10% for steady state average)
    n_steps = size(sim_data.n, 2)
    start_idx = max(1, floor(Int, 0.9 * n_steps))
    avg_range = start_idx:n_steps
    calc_mean(v) = mean(v)

    # --- 2. Print System Parameters ---
    println("\n==================================================================================")
    println("SYSTEM PARAMETERS & COMPARISON")
    println("==================================================================================")
    @printf("  Sites : %d\n", sites)
    @printf("  J     : %.4e [meV]\n", J)
    @printf("  γ     : %.4e [meV]\n", γ)
    
    println("\n[Potentials]")
    for i in 1:sites
        @printf("  V_%-2d : %.4e [meV]\n", i, V[i])
    end

    # --- 3. Comparison Table (Steady State vs Simulation) ---
    println("\n\n[Results Comparison]")
    println("----------------------------------------------------------------------------------")
    @printf("%-10s | %-15s | %-15s | %-15s | %-15s\n", "Variable", "Steady State", "Simulation", "Abs. Diff.", "Rel. Diff.")
    println("----------------------------------------------------------------------------------")

    # --- Populations (n) ---
    for i in 1:sites
        ss_val = ss_result.n[i]
        sim_val = calc_mean(sim_data.n[i, avg_range])
        
        abs_diff = abs(sim_val - ss_val)
        mean_val = (abs(sim_val) + abs(ss_val)) / 2
        rel_diff = (mean_val > 1e-15) ? abs_diff / mean_val : 0.0

        @printf("n_%-2d       | %-15.4e | %-15.4e | %-15.4e | %-15.4e\n", i, ss_val, sim_val, abs_diff, rel_diff)
    end
    println("----------------------------------------------------------------------------------")

    # --- Phases (φ) ---
    # The steady state solver returns phases for links (sites-1)
    for i in 1:sites-1
        ss_val = ss_result.φ[i]
        sim_val = calc_mean(sim_data.φ[i, avg_range])
        
        abs_diff = abs(sim_val - ss_val)
        mean_val = (abs(sim_val) + abs(ss_val)) / 2
        rel_diff = (mean_val > 1e-15) ? abs_diff / mean_val : 0.0

        @printf("φ_%-2d       | %-15.4e | %-15.4e | %-15.4e | %-15.4e\n", i, ss_val, sim_val, abs_diff, rel_diff)
    end
    println("----------------------------------------------------------------------------------")

    # --- Excited Molecules (M_2) ---
    for i in 1:sites
        ss_val = ss_result.M_2[i]
        sim_val = calc_mean(sim_data.M_2[i, avg_range])
        
        abs_diff = abs(sim_val - ss_val)
        mean_val = (abs(sim_val) + abs(ss_val)) / 2
        rel_diff = (mean_val > 1e-15) ? abs_diff / mean_val : 0.0

        @printf("M_2_%-2d      | %-15.4e | %-15.4e | %-15.4e | %-15.4e\n", i, ss_val, sim_val, abs_diff, rel_diff)
    end
    println("----------------------------------------------------------------------------------")

    # --- 4. Solver Status Summary ---
    if ss_result.converged
        println("  Steady State Solver : CONVERGED (Residual: $(ss_result.residual))")
    else
        println("  Steady State Solver : FAILED    (Residual: $(ss_result.residual))")
    end

    # --- 5. Stability Threshold ---
    @printf("\n  Calculated ΔV Stability Threshold : %.4e [meV]\n", ΔV_max_stability)
    println("==================================================================================\n")
end

# =========================================================================
# SECTION 3: Plot Results
# =========================================================================
function plot_results_multisite(data, ss_result, ΔV_range, max_real_eigenvals, crossing_matrix_A, crossing_eigenvecs_A, n_FT, φ_FT; save_path="Figures/ClassicalField_X_triple.png")
    CairoMakie.activate!()
    
    # Unpack necessary data
    sites = data.sites
    t_vec = data.t_vec
    J = data.J
    n = data.n
    φ = data.φ
    M_1 = data.M_1
    M_2 = data.M_2
    V = data.V
    ρ = data.ρ
    ss_n = ss_result.n
    ss_φ = ss_result.φ
    ss_M2 = ss_result.M_2

    limits_large = (nothing, nothing, nothing, nothing)
    
    f1 = Figure(size = (1000, 1000))

    # --- Time Series Plots ---
    
    # ax1: Photon Numbers
    ax1 = Axis(f1[1,1], title="Photon Number Time Series", xlabel="Time (1/meV)", ylabel="Photon Number", limits=limits_large)
    for i in 1:sites
        lines!(ax1, t_vec*J, n[i,:], label="Site $i")
    end

    # ax2: Phase Differences
    ax2 = Axis(f1[1,2], title="Phase difference Time Series", xlabel="Time (1/meV)", ylabel="Phase difference (rad)", limits=limits_large)
    for i in 1:sites-1
        lines!(ax2, t_vec*J, φ[i,:], label="φ_$(i)")
    end

    # --- Distribution Plots (Averages) ---

    # ax3: Final Photon Distribution
    ax3 = Axis(f1[2,1], title="Final Photon Number Distribution", xlabel="Site", ylabel="Photon Number", xticks=1:sites, limits=(0, sites+1, nothing, nothing))
    for i in 1:sites
        scatter!(ax3, [i], [mean(n[i, :])], label="Site $i")
    end

    # ax4: Final Phase Difference Distribution
    ax4 = Axis(f1[2,2], title="Final Phase Difference Distribution", xlabel="Site Pair (i, i+1)", ylabel="Phase (rad)", xticks=1:sites-1, limits=(0, sites, nothing, nothing))
    for i in 1:sites-1
        scatter!(ax4, [i], [mean(φ[i, :])], label="φ_$(i)")
    end

    # ax5: Steady State Photon Distribution
    ax5 = Axis(f1[3,1], title="Steady State Photon Number Distribution", xlabel="Site", ylabel="Photon Number", xticks=1:sites, limits=(0, sites+1, nothing, nothing))
    for i in 1:sites
        scatter!(ax5, [i], [ss_n[i]], label="Site $i")
    end

    # ax6: Steady State Phase Difference Distribution
    ax6 = Axis(f1[3,2], title="Steady State Phase DifferenceDistribution", xlabel="Site Pair (i, i+1)", ylabel="Phase (rad)", xticks=1:sites-1, limits=(0, sites, nothing, nothing))
    for i in 1:sites-1
        scatter!(ax6, [i], [ss_φ[i]], label="φ_$(i)")
    end

    # ax7: Final Total Excitations Distribution
    ax7 = Axis(f1[4,1], title="Final Total Excitations Distribution", xlabel="Site", ylabel="Total Excitations", xticks=1:sites, limits=(0, sites+1, nothing, nothing))
    for i in 1:sites
        scatter!(ax7, [i], [mean(n[i, :]) + mean(M_2[i, :])], label="Site $i")
    end

    # ax8: Final Excited Molecule Distribution
    ax8 = Axis(f1[4,2], title="Final Excited Molecule Distribution", xlabel="Site", ylabel="Excited Molecules", xticks=1:sites, limits=(0, sites+1, nothing, nothing))
    for i in 1:sites
        scatter!(ax8, [i], [mean(M_2[i, :])], label="Site $i")
    end

    # ax9: Potential and Pump Profiles
    ax9 = Axis(f1[5,1], title="Potential and Pump Profiles", xlabel="Site", ylabel="Value", xticks=1:sites, limits=(0, sites+1, nothing, nothing))
    scatter!(ax9, 1:sites, V, label="Potential V")
    # Handle scaling safely
    ρ_max = maximum(ρ)
    V_max = maximum(V)
    scale_factor = (ρ_max != 0 && V_max != 0) ? (V_max / ρ_max) : 1.0
    scatter!(ax9, 1:sites, ρ .* scale_factor, label="Pump ρ (scaled)")
    axislegend(ax9, position=:rb)

    # ax10: Plot just the first and last photon number time series for clarity
    ax10 = Axis(f1[5,2], title="Photon Number Time Series (First and Last Sites)", xlabel="Time (1/meV)", ylabel="Photon Number", limits=limits_large)
    lines!(ax10, t_vec*J, n[1,:], label="Site 1")
    lines!(ax10, t_vec*J, n[sites,:], label="Site $sites")
    axislegend(ax10, position=:rt)

    # ax11: Stability analysis vs ΔV
    limits = (nothing, nothing, minimum(max_real_eigenvals)*(1+0.1), -minimum(max_real_eigenvals)*(1+0.1))
    ax11 = Axis(f1[6,1], title="Stability Analysis vs ΔV", xlabel="ΔV [meV]", ylabel="Max Real Eigenvalue", limits=limits)
    for i in 1:3*sites-1
        lines!(ax11, ΔV_range, max_real_eigenvals, label="Eigenvalue $i")
    end
    hlines!(ax11, [0.0], color=:red, linestyle=:dash, label="Stability Threshold")
    display(f1)

    f2 = Figure(size=(1200, 800))

    # ax12: Fourier Transform of Phase Difference
    ax12 = Axis(f2[1, 1], title="Fourier Transform of Phase Difference", xlabel="Frequency (meV)", ylabel="Magnitude")
    for i in 1:sites-1
        lines!(ax12, abs.(φ_FT[i, 2:end]), label="|FT(Phase Difference Site $i)|")
    end
    axislegend(ax12, position=:rt)


    # ax13: Fourier Transform of Site Photon Numbers
    ax13 = Axis(f2[1, 2], title="Fourier Transform of Site Photon Numbers", xlabel="Frequency (meV)", ylabel="Magnitude")
    for i in 1:sites
        lines!(ax13, abs.(n_FT[i, 2:end]), label="|FT(Site $i Photon Number)|")
    end
    axislegend(ax13, position=:rt)

    display(f2)

    # --- Heatmap of A Matrix at Stability Crossing ---
    # Extract the crossing Matrix
    ΔV = ΔV_range[findfirst(x -> x > 0, max_real_eigenvals)]
    crossing_matrix_A = crossing_matrix_A
    crossing_eigenvecs_A = crossing_eigenvecs_A

    # 1. Generate Custom Labels
    # Pattern: n, X, φ repeating. Index increments every 3 variables.
    data = crossing_matrix_A
    rows, cols = size(data)
    var_symbols = ["n", "X", "\\phi"] 

    function get_param_label(i)
        sym = var_symbols[mod1(i, 3)]   
        sub = div(i - 1, 3) + 1         
        return latexstring("$(sym)_{$(sub)}")
    end

    param_labels_x = [get_param_label(j) for j in 1:cols]
    param_labels_y = [get_param_label(i) for i in 1:rows]

    f3 = Figure(size=(800, 800))
    # 2. Create the Axis
    ax12 = Axis(f3[1, 1], 
        title = latexstring("Stability Matrix at Crossing (\$\\Delta V_{max}\$ = $(round(ΔV, digits=4)) meV)"),        
        titlesize = 24,
        xticks = (1:cols, param_labels_x),
        yticks = (1:rows, param_labels_y),
        yreversed = true,        # Row 1 at the top
        xaxisposition = :top,    
        xticklabelrotation = 0   
    )

    # --- NEW: Replace exact 0.0 with NaN for plotting ---
    # We use a copy to avoid permanently altering your original matrix
    plot_data = copy(data)
    plot_data[plot_data .== 0.0] .= NaN

    # Calculate limits based ONLY on finite values (ignoring -Inf and NaN)
    finite_data = filter(isfinite, plot_data)
    if isempty(finite_data)
        # Fallback if matrix is all zeros (which are now all NaNs)
        clims = (0.0, 1.0)
    else
        clims = (minimum(finite_data), maximum(finite_data))
    end

    # 3. Plot Heatmap
    # colorrange: sets the scale to the finite data limits
    # lowclip: sets a specific color for values below the range (e.g., -Inf)
    # nan_color: sets a specific color for NaN values (our exact zeros)
    hm = heatmap!(ax12, 1:cols, 1:rows, plot_data', 
        colormap = :thermal,
        colorrange = clims, 
        lowclip = :white,
        nan_color = :white  
    )

    # 4. Add Colorbar
    Colorbar(f3[1, 2], hm, label = "Magnitude")

    display(f3)
    # Save
    mkpath(dirname("Figures/A_Matrix_Heatmap.png"))
    save("Figures/A_Matrix_Heatmap.png", f3)
    println("Summary plot saved to Figures/A_Matrix_Heatmap.png")



    # # 5. Overlay Values
    # # for i in 1:rows
    # #     for j in 1:cols
    # #         val = data[i, j] # Keep precision
            
    # #         # Only display text if value is finite (skips -Inf)
    # #         if isfinite(val)
    # #             # Dynamic text color: White text on dark background, Black on light
    # #             # We compare val to the calculated range to decide contrast
    # #             normalized_val = (val - clims[1]) / (clims[2] - clims[1])
    # #             textcolor = normalized_val < 0.5 ? :white : :black
                
    # #             text!(ax, string(round(val, digits=2)), 
    # #                 position = (j, i), 
    # #                 align = (:center, :center), 
    # #                 color = textcolor, 
    # #                 fontsize = 12
    # #             )
    # #         end
    # #     end
    # # end
    
    # Save
    mkpath(dirname(save_path))
    save(save_path, f1)
    println("Summary plot saved to $save_path")
end
