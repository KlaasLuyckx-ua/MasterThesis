# ==============================================================================
# RIGOROUS 2D GRID SEARCH & MULTIVARIATE SCALING DISCOVERY
# ==============================================================================
using CSV, DataFrames, LsqFit, CairoMakie, Statistics

# Include your physics engine
include("../Functions_MultiSite/Analytics_MultiSite.jl")

sites = 5
grid_resolution = 10 

# --- 1. Define Static Parameters ---
γ_stat = 2.0e-4  # Anchor gamma. We will vary κ to change the κ/γ ratio.
n_avg = 3.0e2
B_21 = 1.0e-8
Δ = -100.0
T = 25.0

# --- 2. Define 2D Grid Ranges (Logarithmic spacing) ---
# Sweeping across bounds spanning both the small and large regimes
J_range = 10 .^ range(log10(1e-3), log10(1e-1), length=grid_resolution)
kg_range = 10 .^ range(log10(1e-2), log10(1e2), length=grid_resolution) # kg stands for κ/γ


# ==============================================================================
# PART 1: GENERATE 2D GRID DATA
# ==============================================================================
println("Starting 2D Grid Search (Total Simulations: $(grid_resolution^2))...")
t_start = time()

all_data = DataFrame[]

for J_loc in J_range
    for kg_loc in kg_range
        
        # Calculate the actual κ required to achieve this κ/γ ratio
        κ_loc = kg_loc * γ_stat
        
        # Back-calculate the M needed to achieve this exact κ_loc
        M_loc = κ_loc * 2.0 * T / (B_21 * exp(Δ/T))
        
        # Run simulation
        ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J_loc, γ_stat, B_21, M_loc, T, Δ, κ_loc, n_avg)
        
        push!(all_data, DataFrame(J=J_loc, kg=kg_loc, dV_max=ΔV_max))
    end
end

df_grid = vcat(all_data...)
println("Grid Search Complete! Elapsed time: $(round(time() - t_start, digits=2)) seconds.")


# ==============================================================================
# PART 2: BIVARIATE NON-LINEAR REGRESSION
# ==============================================================================
println("\nPerforming Unconstrained Bivariate Fit...")

# Prepare data arrays
X_matrix = hcat(df_grid.J, df_grid.kg) # 2-column matrix
Y_data = df_grid.dV_max

# Define the Unconstrained 2D Power Law Model
# p[1] = C (Prefactor)
# p[2] = b_J (J exponent)
# p[3] = b_kg (κ/γ exponent)
bivariate_model(X, p) = p[1] .* (X[:, 1].^p[2]) .* (X[:, 2].^p[3])

# Initial Guesses (C=1.0, b_J=1.5, b_kg=0.5)
p0 = [1.0, 1.5, 0.5]

# Run the Levenberg-Marquardt optimizer over the 2D surface
fit_2D = curve_fit(bivariate_model, X_matrix, Y_data, p0)
C_fit, bJ_fit, bkg_fit = fit_2D.param

# Calculate R² for the bivariate surface
Y_pred = bivariate_model(X_matrix, fit_2D.param)
r2_2D = 1.0 - sum((Y_data .- Y_pred).^2) / sum((Y_data .- mean(Y_data)).^2)

println("===========================================================================")
println(" 2D BIVARIATE FIT RESULTS (Sites = $sites)")
println("===========================================================================")
println(" Master Equation: ΔV_max = C * J^(b_J) * (κ/γ)^(b_kg)")
println("---------------------------------------------------------------------------")
println(rpad(" Prefactor C:", 25), round(C_fit, sigdigits=5))
println(rpad(" J Exponent (b_J):", 25), round(bJ_fit, digits=5))
println(rpad(" κ/γ Exponent (b_kg):", 25), round(bkg_fit, digits=5))
println(rpad(" Fit R² Score:", 25), round(r2_2D, digits=8))
println("===========================================================================\n")


# ==============================================================================
# PART 3: THE ULTIMATE DATA COLLAPSE PLOT
# ==============================================================================
println("Generating Master Data Collapse Plot...")

# Construct the discovered universal axis X
X_universal = (df_grid.J .^ bJ_fit) .* (df_grid.kg .^ bkg_fit)

fig = Figure(size = (1200, 600), fontsize = 16)
Label(fig[0, 1:2], "UNIVERSAL DATA COLLAPSE (2D GRID SEARCH)", fontsize = 24, font = :bold)

# Panel 1: Prediction vs Reality (Parity Plot)
ax_parity = Axis(fig[1, 1], title = "Model Accuracy (Predicted vs Actual)", 
                 xlabel = "Actual Simulated ΔV_max", ylabel = "Predicted ΔV_max",
                 xscale = log10, yscale = log10)

scatter!(ax_parity, Y_data, Y_pred, color = (:blue, 0.2), markersize = 6)
# Draw perfect y=x reference line
min_val, max_val = minimum(Y_data), maximum(Y_data)
lines!(ax_parity, [min_val, max_val], [min_val, max_val], color = :red, linewidth = 2, linestyle = :dash, label="Perfect Fit (y=x)")
axislegend(ax_parity, position=:lt)

# Panel 2: The Data Collapse
lbl_x = "J^($(round(bJ_fit, digits=3))) * (κ/γ)^($(round(bkg_fit, digits=3)))"
ax_collapse = Axis(fig[1, 2], title = "Data Collapse onto Master Equation", 
                   xlabel = lbl_x, ylabel = "ΔV_max")

scatter!(ax_collapse, X_universal, Y_data, color = (:purple, 0.2), markersize = 6, label="Grid Search Data")

# Draw the theoretical line C * X
x_dense = range(0, maximum(X_universal), length=200)
lines!(ax_collapse, x_dense, C_fit .* x_dense, color = :black, linewidth = 3, label="Master Fit (C = $(round(C_fit, sigdigits=4)))")
axislegend(ax_collapse, position=:lt)

display(fig)
save("Param_scan/Figures/Grid_2D_Data_Collapse_$(sites)Sites.png", fig)