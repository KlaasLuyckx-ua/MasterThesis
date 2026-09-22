# ==============================================================================
# UNIFIED ANALYSIS SCRIPT (2-sites & N-sites, Deterministic & Stochastic)
# ==============================================================================
using JLD2, CairoMakie, GLMakie, Printf, Statistics, FFTW, LaTeXStrings

include("../ExtraFunctions/Analytics.jl")
include("../ExtraFunctions/Structs.jl")
include("../ExtraFunctions/Simulation_functions.jl")

# --- Run Configuration ---
run_name = "sites_2_det"  # Update this to match the run you want to analyze
filepath = "ClassicalField/Data/$(run_name).jld2"
output_dir = "ClassicalField/Plots/$(run_name)"
mkpath(output_dir)

# --- Choose which plots to generate ---
eigenval_plot = true   # Set to true to generate eigenvalue evolution plots
heatmap = false         # Set to true to generate stability matrix heatmap

# =========================================================================
# 1. Load and Process Data
# =========================================================================
println("Loading data from: $filepath")
data = load(filepath)

# Extract Data
t_vec = data["t_vec"]::Vector{Float64}
X_data = data["X_data"]::Matrix{ComplexF64}
pars = data["parameters"]
ss_result = data["ss_result"]
crit_result = data["crit_result"]
ΔV = data["ΔV"]
noise_strength = data["noise_strength"]


# Extract Parameters
sites = pars.sites
J = pars.J
M = pars.M
γ = pars.γ
n_avg = pars.n_avg
κ = pars.κ
V = pars.V

len = length(t_vec)

# Pre-allocate State Arrays
n = zeros(Float64, sites, len)
M_2 = zeros(Float64, sites, len)
M_1 = zeros(Float64, sites, len)
theta = zeros(Float64, sites, len)
φ = zeros(Float64, sites-1, len)

# Process State Arrays
for i in 1:sites
    n[i,:] = abs2.(X_data[i,:])
    M_2[i,:] = real.(X_data[sites+i,:]) .- n[i,:]
    M_1[i,:] = M .- M_2[i,:]
    theta[i,:] = angle.(X_data[i,:])
end

for i in 1:sites-1
    φ[i,:] = angle.(conj.(X_data[i,:]) .* X_data[i+1,:])
end

# Calculate Fast Fourier Transforms (excluding the DC component 1)
n_FT = zeros(ComplexF64, sites, len)
φ_FT = zeros(ComplexF64, sites-1, len)

for i in 1:sites
    n_FT[i,:] = fft(n[i,:])
end
for i in 1:sites-1
    φ_FT[i,:] = fft(φ[i,:])
end

# Calculate Spatial Coherence relative to Site 1
avg_range = floor(Int, 0.5 * len):len
g1_coherence = zeros(Float64, sites)
I1 = mean(abs2.(X_data[1, avg_range])) # Time-averaged intensity of Site 1

for i in 1:sites
    Ii = mean(abs2.(X_data[i, avg_range]))
    cross_term = mean(conj.(X_data[1, avg_range]) .* X_data[i, avg_range])
    
    # Normalized first-order coherence
    g1_coherence[i] = abs(cross_term) / sqrt(I1 * Ii)
end

# Calculate Frequencies
save_Δt = pars.save_Δt
fs = 1.0 / save_Δt 
freqs = fftfreq(len, fs)

# Calculate time-averaged frequency (∂_t θ) for each site
mean_freq = zeros(Float64, sites)

for i in 1:sites
    # Using angle(X_t* * X_{t+1}) to cleanly calculate Δθ without phase wrapping issues
    phase_diffs = angle.(conj.(X_data[i, avg_range[1:end-1]]) .* X_data[i, avg_range[2:end]])
    
    # Frequency is the mean phase difference divided by the time step
    mean_freq[i] = mean(phase_diffs) / save_Δt
end

mean_phase = zeros(Float64, sites)
for i in 1:sites
    # Average the complex correlation first, then extract the angle
    complex_corr = mean(conj.(X_data[1, avg_range]) .* X_data[i, avg_range])
    mean_phase[i] = angle(complex_corr)
end

println("Data processing complete.")

# =========================================================================
# 2. Print Summary Statistics
# =========================================================================
println("\n==================================================================================")
println("SYSTEM PARAMETERS & COMPARISON ($sites SITES | Noise: $noise_strength)")
println("==================================================================================")
# Print key parameters with formatting
@printf("  J     : %.4e [meV]\n", J)
@printf("  γ     : %.4e [meV]\n", γ)
@printf("  κ     : %.4e [meV]\n", κ)
@printf("  ΔV    : %.4e [meV]\n", ΔV)

# Time Series Statistics (Last 10%)
start_idx = max(1, floor(Int, 0.9 * len))
avg_range = start_idx:len

println("\n[Results Comparison: Steady State vs Simulation]")
println("----------------------------------------------------------------------------------")
@printf("%-10s | %-15s | %-15s | %-15s | %-15s\n", "Variable", "Steady State", "Simulation", "Abs. Diff.", "Rel. Diff.")
println("----------------------------------------------------------------------------------")
# Compare steady state results with time-averaged simulation results for n, φ, and M_2. Calculate absolute and relative differences.

for i in 1:sites
    ss_val = ss_result.n[i]
    sim_val = mean(n[i, avg_range])
    abs_diff = abs(sim_val - ss_val)
    mean_val = (abs(sim_val) + abs(ss_val)) / 2
    rel_diff = (mean_val > 1e-15) ? abs_diff / mean_val : 0.0
    @printf("n_%-2d      | %-15.4e | %-15.4e | %-15.4e | %-15.4e\n", i, ss_val, sim_val, abs_diff, rel_diff)
end
println("----------------------------------------------------------------------------------")
for i in 1:sites-1
    ss_val = ss_result.φ[i]
    sim_val = mean(φ[i, avg_range])
    abs_diff = abs(sim_val - ss_val)
    mean_val = (abs(sim_val) + abs(ss_val)) / 2
    rel_diff = (mean_val > 1e-15) ? abs_diff / mean_val : 0.0
    @printf("φ_%-2d      | %-15.4e | %-15.4e | %-15.4e | %-15.4e\n", i, ss_val, sim_val, abs_diff, rel_diff)
end
println("----------------------------------------------------------------------------------")
for i in 1:sites
    ss_val = ss_result.M_2[i]
    sim_val = mean(M_2[i, avg_range])
    abs_diff = abs(sim_val - ss_val)
    mean_val = (abs(sim_val) + abs(ss_val)) / 2
    rel_diff = (mean_val > 1e-15) ? abs_diff / mean_val : 0.0
    @printf("M_2_%-2d    | %-15.4e | %-15.4e | %-15.4e | %-15.4e\n", i, ss_val, sim_val, abs_diff, rel_diff)
end

println("----------------------------------------------------------------------------------")
if ss_result.converged
    println("  Steady State Solver : CONVERGED (Residual: $(ss_result.residual))")
else
    println("  Steady State Solver : FAILED    (Residual: $(ss_result.residual))")
end
stability_limit = isnothing(crit_result.critical_ΔV) ? NaN : crit_result.critical_ΔV
@printf("\n  Calculated ΔV Stability Threshold : %.4e [meV]\n", stability_limit)
println("==================================================================================\n")

# =========================================================================
# 3. Generating Plots
# =========================================================================
CairoMakie.activate!()
f1 = Figure(size = (1200, 1500))
title_str = "Dynamics of $sites Sites (ΔV = $ΔV meV | Noise = $noise_strength)"
Label(f1[0, 1:2], title_str, fontsize=40, font=:bold)

# --- ROW 1 ---
# Plot 1: Photon Number Time Series
ax1 = Axis(f1[1,1], title="Photon Number Time Series", xlabel="Time (1/meV)", ylabel="Photon Number", titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites
    lines!(ax1, t_vec*J, n[i,:], label="Site $i")
end
if sites == 2
    lines!(ax1, t_vec*J, (n[1,:].+n[2,:])./2, label="Average", color=:black, linestyle=:dash)
end
if sites <= 5 axislegend(ax1, position=:lb, labelsize=20) end

# Plot 2: Phase Difference Time Series
ax2 = Axis(f1[1,2], title="Phase Difference Time Series", xlabel="Time (1/meV)", ylabel="Phase Difference (rad)", titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites-1
    lines!(ax2, t_vec*J, φ[i,:], label="φ_$(i)")
end
if sites <= 5 axislegend(ax2, position=:rb, labelsize=20) end

# --- ROW 2 ---
# Plot 3: Excited Molecule Time Series
ax3 = Axis(f1[2,1], title="Excited Molecules Time Series", xlabel="Time (1/meV)", ylabel="Number of Excited Molecules", titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites
    lines!(ax3, t_vec*J, M_2[i,:], label="Site $i")
end
if sites <= 5 axislegend(ax3, position=:rb, labelsize=20) end

# Plot 4: FT of Phase Difference
ax4 = Axis(f1[2, 2], title="Fourier Transform of Phase Difference", xlabel="Frequency (meV)", ylabel="Magnitude", titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites-1
    lines!(ax4, freqs[2:end], abs.(φ_FT[i, 2:end]), label="|FT(φ_$(i))|")
end
if sites <= 5 axislegend(ax4, position=:rt, labelsize=20) end

# --- ROW 3 ---
# Plot 5: Final Photon Distribution
ax5 = Axis(f1[3,1], title="Time-Averaged Photon Distribution", xlabel="Site", ylabel="Photon Population (n)", xticks=1:sites, titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites
    scatter!(ax5, [i], [mean(n[i, avg_range])], label="Photons (Site $i)", marker=:circle, markersize=12)
end

# Plot 6: Final Phase Difference Distribution
ax6 = Axis(f1[3,2], title="Time-Averaged Phase Differences", xlabel="Site", ylabel="Phase Difference (rad)", xticks=1:sites, titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites-1
    scatter!(ax6, [i], [mean(φ[i, avg_range])], label="Phase Difference (Site $i)", marker=:star5, markersize=12)
end

# --- ROW 4 ---
# Plot 7: FT of Photon Numbers
ax7 = Axis(f1[4, 1], title="Fourier Transform of Photon Numbers", xlabel="Frequency (meV)", ylabel="Magnitude", titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
for i in 1:sites
    lines!(ax7, freqs[2:end], abs.(n_FT[i, 2:end]), label="|FT(n_$(i))|")
end
if sites <= 5 axislegend(ax7, position=:rt, labelsize=20) end

# Plot 8: Spatial Coherence Decay
ax8 = Axis(f1[4, 2], title="Spatial Coherence |g⁽¹⁾(1, i)|", xlabel="Site (i)", ylabel="Coherence", xticks=1:sites, titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
scatterlines!(ax8, 1:sites, g1_coherence, color=:purple, linewidth=2, marker=:diamond, markersize=14, label="Relative to Site 1")
ylims!(ax8, 0.0, 1.05)
axislegend(ax8, position=:lb, labelsize=20)


# --- ROW 5 ---
# Plot 9: Time-Averaged Frequency Distribution (∂_t θ)
ax9 = Axis(f1[5, 1],
    title="Time-Averaged Frequency (∂_t θ_j)", 
    xlabel="Site", 
    ylabel="Frequency", 
    xticks=1:sites, 
    titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)

for i in 1:sites
    scatter!(ax9, [i], [mean_freq[i]], label="Frequency (Site $i)", marker=:utriangle, markersize=14, color=:orange)
end

ax10 = Axis(f1[5, 2],
    title="Time-Averaged Phase (θ_j-θ_1)",
    xlabel="Site", 
    ylabel="Phase (rad)", 
    xticks=1:sites, 
    titlesize=25, xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)

for i in 1:sites
    scatter!(ax10, [i], [mean_phase[i]], label="Phase (Site $i)", marker=:circle, markersize=14, color=:blue)
end

display(f1)
plot_file = joinpath(output_dir, "$(run_name)_Overview.png")
save(plot_file, f1)
println("Plot successfully generated and saved to: $plot_file")

# =========================================================================
# 4. Optional: Eigenvalue Evolution Plots (Real and Imaginary Parts)
# =========================================================================
if eigenval_plot == true
    println("Generating Eigenvalue Evolution Plots...")
    
    f_eigen = Figure(size = (800, 2000))
    
    ΔV_scan = crit_result.ΔV_scanned
    num_steps = length(ΔV_scan)
    
    # 1. Reconstruct complex eigenvalues for tracking
    raw_complex_eigs = crit_result.eigenvals
    num_eigs = length(raw_complex_eigs[1])
    
    # 2. Track eigenvalues by minimizing distance in the complex plane
    tracked_eigs = similar(raw_complex_eigs)
    tracked_eigs[1] = raw_complex_eigs[1] # First step remains unchanged as the baseline
    
    for step in 2:num_steps
        prev_eigs = tracked_eigs[step-1]
        curr_eigs = copy(raw_complex_eigs[step])
        
        matched_curr = similar(curr_eigs)
        available = trues(num_eigs)
        
        for i in 1:num_eigs
            # Find the available eigenvalue in the current step closest to prev_eigs[i]
            best_dist = Inf
            best_idx = -1
            for j in 1:num_eigs
                if available[j]
                    dist = abs(curr_eigs[j] - prev_eigs[i])
                    if dist < best_dist
                        best_dist = dist
                        best_idx = j
                    end
                end
            end
            matched_curr[i] = curr_eigs[best_idx]
            available[best_idx] = false # Mark as used so it isn't assigned twice
        end
        tracked_eigs[step] = matched_curr
    end

    # --- Plotting ---
    
    # ROW 1: Plot ALL tracked eigenvalues together
    ax_real_all = Axis(f_eigen[1, 1], 
                       title="Real Eigenvalues vs ΔV", 
                       xlabel="ΔV [meV]", 
                       ylabel="Real Eigenvalues",
                       titlesize=20, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)
                       
    ax_imag_all = Axis(f_eigen[1, 2], 
                       title=LaTeXString("Imaginaire eigenwaarden"), 
                       xlabel=LaTeXString("ΔV [meV]"), 
                       ylabel=LaTeXString("\\Im(λ)"),
                       titlesize=20, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)

    for i in 1:num_eigs
        r_vals = [real(tracked_eigs[step][i]) for step in 1:num_steps]
        i_vals = [imag(tracked_eigs[step][i]) for step in 1:num_steps]
        
        lines!(ax_real_all, ΔV_scan, r_vals, linewidth=1.5)
        lines!(ax_imag_all, ΔV_scan, i_vals, linewidth=1.5)
    end
    
    hlines!(ax_imag_all, [0.0], color=:red, linestyle=:dash, label="Stability Threshold")
    
    # ROWS 2 to 6: Plot individual tracked eigenvalues (only the first few for clarity)
    num_rows_individual = min(5, num_eigs)
    
    for i in 1:num_rows_individual
        row_idx = i + 1 # Offset by 1 because Row 1 is taken by the collective plot
        
        ax_real_ind = Axis(f_eigen[row_idx, 1], 
                           title="Real Part Eigenvalue $i vs ΔV", 
                           xlabel="ΔV [meV]", 
                           ylabel="Re(λ_$i)")
                           
        ax_imag_ind = Axis(f_eigen[row_idx, 2], 
                           title="Imaginary Part Eigenvalue $i vs ΔV", 
                           xlabel="ΔV [meV]", 
                           ylabel="Im(λ_$i)")

        # Extract the tracked i-th eigenvalue across all scan points
        r_vals = [real(tracked_eigs[step][i]) for step in 1:num_steps]
        i_vals = [imag(tracked_eigs[step][i]) for step in 1:num_steps]
        
        lines!(ax_real_ind, ΔV_scan, r_vals, linewidth=1.5)
        lines!(ax_imag_ind, ΔV_scan, i_vals, linewidth=1.5)
        
        hlines!(ax_imag_ind, [0.0], color=:red, linestyle=:dash)
    end
    
    display(f_eigen)
    
    eigen_plot_file = joinpath(output_dir, "$(run_name)_Eigenvalues.png")
    save(eigen_plot_file, f_eigen)
    println("Eigenvalue plots successfully saved to: $eigen_plot_file")

    if sites == 2
        println("Generating Simple Eigenvalue Evolution Plots...")
        
        ΔV_scan = crit_result.ΔV_scanned
        num_steps = length(ΔV_scan)
        
        # 1. Reconstruct complex eigenvalues for distance tracking
        simple_eigs = crit_result.simple_eigenvals
        num_eigs = length(simple_eigs[1]) # Should be 2 for the 2x2 system
        
        # 2. Track eigenvalues by minimizing distance in the complex plane
        tracked_simple_eigs = similar(simple_eigs)
        tracked_simple_eigs[1] = simple_eigs[1] # Baseline
        
        for step in 2:num_steps
            prev_eigs = tracked_simple_eigs[step-1]
            curr_eigs = copy(simple_eigs[step])
            
            matched_curr = similar(curr_eigs)
            available = trues(num_eigs)
            
            for i in 1:num_eigs
                best_dist = Inf
                best_idx = -1
                for j in 1:num_eigs
                    if available[j]
                        dist = abs(curr_eigs[j] - prev_eigs[i])
                        if dist < best_dist
                            best_dist = dist
                            best_idx = j
                        end
                    end
                end
                matched_curr[i] = curr_eigs[best_idx]
                available[best_idx] = false
            end
            tracked_simple_eigs[step] = matched_curr
        end

        # --- Plotting ---
        f_simple = Figure(size=(900, 800))
        
        # ROW 1: Plot ALL tracked simple eigenvalues together
        ax_simple_real_all = Axis(f_simple[1, 1], 
                                title="All Real 2x2 Eigenvalues vs ΔV", 
                                xlabel="ΔV [meV]", 
                                ylabel="Real Eigenvalues")
                                
        ax_simple_imag_all = Axis(f_simple[1, 2], 
                                title="All Imaginary 2x2 Eigenvalues vs ΔV", 
                                xlabel="ΔV [meV]", 
                                ylabel="Imaginary Eigenvalues")

        for i in 1:num_eigs
            r_vals = [real(tracked_simple_eigs[step][i]) for step in 1:num_steps]
            i_vals = [imag(tracked_simple_eigs[step][i]) for step in 1:num_steps]
            
            lines!(ax_simple_real_all, ΔV_scan, r_vals, linewidth=2)
            lines!(ax_simple_imag_all, ΔV_scan, i_vals, linewidth=2)
        end
        
        hlines!(ax_simple_imag_all, [0.0], color=:red, linestyle=:dash, label="Stability Threshold")

        # ROWS 2 to 3: Plot individual tracked eigenvalues
        for i in 1:num_eigs
            row_idx = i + 1
            
            ax_real_ind = Axis(f_simple[row_idx, 1], 
                            title="Real Part 2x2 Eigenvalue $i vs ΔV", 
                            xlabel="ΔV [meV]", 
                            ylabel="Re(λ_$i)")
                            
            ax_imag_ind = Axis(f_simple[row_idx, 2], 
                            title="Imaginary Part 2x2 Eigenvalue $i vs ΔV", 
                            xlabel="ΔV [meV]", 
                            ylabel="Im(λ_$i)")

            # Extract the tracked i-th eigenvalue across all scan points
            r_vals = [real(tracked_simple_eigs[step][i]) for step in 1:num_steps]
            i_vals = [imag(tracked_simple_eigs[step][i]) for step in 1:num_steps]
            
            lines!(ax_real_ind, ΔV_scan, r_vals, linewidth=2)
            lines!(ax_imag_ind, ΔV_scan, i_vals, linewidth=2)
            
            hlines!(ax_imag_ind, [0.0], color=:red, linestyle=:dash)
        end
        
        display(f_simple)
        
        simple_plot_file = joinpath(output_dir, "$(run_name)_Simple_Eigenvalues.png")
        save(simple_plot_file, f_simple)
        println("Simple Eigenvalue plots successfully saved to: $simple_plot_file")

        f_eigenval_analytical = Figure(size=(900, 400))
        # Plot the analytical eigenvalues from the critical result
        ax_analytical_real = Axis(f_eigenval_analytical[1, 1], 
                                title="Real Analytical Eigenvalue", 
                                xlabel="ΔV [meV]", 
                                ylabel="Real Eigenvalue", 
                                limits=(nothing, nothing, -0.5, 0.5),
                                titlesize=20, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)
        ax_analytical_imag = Axis(f_eigenval_analytical[1, 2], 
                                title=LaTeXString("Imaginaire Analytische Eigenwaarde"), 
                                xlabel=LaTeXString("ΔV [meV]"), 
                                ylabel=LaTeXString("\\Im(λ)"), 
                                titlesize=20, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)
        r_vals_analytical_1 = [real(crit_result.analytical_eigenvals[step][1]) for step in 1:num_steps]
        i_vals_analytical_1 = [imag(crit_result.analytical_eigenvals[step][1]) for step in 1:num_steps]
        r_vals_analytical_2 = [real(crit_result.analytical_eigenvals[step][2]) for step in 1:num_steps]
        i_vals_analytical_2 = [imag(crit_result.analytical_eigenvals[step][2]) for step in 1:num_steps]
        lines!(ax_analytical_real, ΔV_scan, r_vals_analytical_2, label="Re(λ_analytical_2)", linewidth=2)
        lines!(ax_analytical_imag, ΔV_scan, i_vals_analytical_2, label="Im(λ_analytical_2)", linewidth=2)
        lines!(ax_analytical_real, ΔV_scan, r_vals_analytical_1, label="Re(λ_analytical_1)", linewidth=2)
        lines!(ax_analytical_imag, ΔV_scan, i_vals_analytical_1, label="Im(λ_analytical_1)", linewidth=2)
        hlines!(ax_analytical_imag, [0.0], color=:red, linestyle=:dash, label="Stability Threshold")
        display(f_eigenval_analytical)
        save(joinpath(output_dir, "$(run_name)_Analytical_Eigenvalues.png"), f_eigenval_analytical)
        println("Analytical Eigenvalue plots successfully saved to: $(joinpath(output_dir, "$(run_name)_Analytical_Eigenvalues.png"))")
    end
end


# =========================================================================
# 5. Optional: Stability Matrix Heatmap
# =========================================================================
# If the stability crossing was detected and the critical Jacobian matrix is available, we can visualize its structure using a heatmap. 
# This can provide insights into which variables and interactions are most influential in driving the instability.
if heatmap == true
    println("Stability crossing detected. Generating Heatmap...")
    f3 = Figure(size=(800, 800))
    data_A = imag.(crit_result.crossing_matrix)
    rows, cols = size(data_A)
    
    # Custom Labels
    var_symbols = ["n", "X", "\\phi"] 
    get_label(i) = latexstring("$(var_symbols[mod1(i, 3)])_{$(div(i - 1, 3) + 1)}")
    
    ax_hm = Axis(f3[1, 1], 
        title = "Stability Matrix at Critical ΔV (ΔV_max = $(round(stability_limit, digits=5)) meV)",        
        titlesize = 25,
        xticks = (1:cols, [get_label(j) for j in 1:cols]),
        yticks = (1:rows, [get_label(i) for i in 1:rows]),
        yreversed = true, xaxisposition = :top, xlabelsize=20, ylabelsize=20, xticklabelsize=18, yticklabelsize=18)

    data_A[data_A .== 0.0] .= NaN
    finite_data = filter(isfinite, data_A)
    clims = isempty(finite_data) ? (0.0, 1.0) : (minimum(finite_data), maximum(finite_data))

    hm = heatmap!(ax_hm, 1:cols, 1:rows, data_A', 
                  colormap = :thermal, colorrange = clims, 
                  lowclip = :white, nan_color = :white)
    Colorbar(f3[1, 2], hm, label = "Value", labelsize=25, ticklabelsize=20)
    
    display(f3)

    heatmap_file = joinpath(output_dir, "$(run_name)_Heatmap.png")
    save(heatmap_file, f3)
    println("Heatmap saved to: $heatmap_file")
end