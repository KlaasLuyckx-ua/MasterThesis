# ==============================================================================
# Data Generation and Storage
# Run this script once to generate all parameters and calculations.
# ==============================================================================

using CSV, DataFrames

t_start = time()

include("../Functions_MultiSite/Analytics_MultiSite.jl")
sites = 3
n_points = 10  # number of points in parameter scan

println("Starting calculations small...")

# --- 1. Define Static Parameters ---
J = 1.0e-2
γ = 2.0e-4
n_avg = 3.0e2
M = 1.0e7
B_21 = 1.0e-8
Δ = -100
T = 25.0
κ = 0.5*B_21*M*exp(Δ/T)/T
δ = 0.0

# --- 2. Define Scanning Parameter Ranges ---
J_range = 10 .^ range(log10(1e-3), log10(1e-1), length=n_points)
γ_range = 10 .^ range(log10(2e-5), log10(2e-3), length=n_points)
n_avg_range = 10 .^ range(log10(3e1), log10(3e3), length=n_points)
M_range = 10 .^ range(log10(1e7), log10(1e8), length=n_points)
B_21_range = 10 .^ range(log10(1e-9), log10(1e-7), length=n_points)
Δ_range = range(-140, -60, length=n_points)
T_range = range(10, 40, length=n_points)

# --- 3. Initialize Master Array ---
# We will create a massive array of DataFrames to stitch into one master table
all_data = DataFrame[]

println("Scanning J...")
for J_loc in J_range
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J_loc, γ, B_21, M, T, Δ, κ, n_avg)
    push!(all_data, DataFrame(ScanType="J", X_Value=J_loc, dV_Max=ΔV_max))
end

println("Scanning γ...")
for γ_loc in γ_range
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ_loc, B_21, M, T, Δ, κ, n_avg)
    push!(all_data, DataFrame(ScanType="γ", X_Value=γ_loc, dV_Max=ΔV_max))
end

println("Scanning n_avg...")
for n_avg_loc in n_avg_range
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T, Δ, κ, n_avg_loc)
    push!(all_data, DataFrame(ScanType="n_avg", X_Value=n_avg_loc, dV_Max=ΔV_max))
end

println("Scanning M...")
for M_loc in M_range
    κ_loc = 0.5*B_21*M_loc*exp(Δ/T)/T
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M_loc, T, Δ, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="M", X_Value=M_loc, dV_Max=ΔV_max))
end

println("Scanning B_21...")
for B_21_loc in B_21_range
    κ_loc = 0.5*B_21_loc*M*exp(Δ/T)/T
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21_loc, M, T, Δ, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="B_21", X_Value=B_21_loc, dV_Max=ΔV_max))
end

println("Scanning Δ...")
for Δ_loc in Δ_range
    κ_loc = 0.5*B_21*M*exp(Δ_loc/T)/T
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T, Δ_loc, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="Δ", X_Value=Δ_loc, dV_Max=ΔV_max))
end

println("Scanning T...")
for T_loc in T_range
    κ_loc = 0.5*B_21*M*exp(Δ/T_loc)/T_loc
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T_loc, Δ, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="T", X_Value=T_loc, dV_Max=ΔV_max))
end

# --- 4. Stitch Data & Save ---
master_df = vcat(all_data...)

output_file = "Param_scan/Data/Master_Stability_Results_$(sites)_Site_small.csv"
CSV.write(output_file, master_df)

println("\nCalculations complete.")
println("Data successfully saved to: $output_file")




println("Starting calculations large...")

# --- 1. Define Static Parameters ---
J = 1.0e-1
γ = 2.0e-3
n_avg = 3.0e3
M = 1.0e8
B_21 = 1.0e-7
Δ = -60
T = 25.0
κ = 0.5*B_21*M*exp(Δ/T)/T
δ = 0.0

# --- 2. Define Scanning Parameter Ranges ---
J_range = 10 .^ range(log10(1e-2), log10(1e0), length=n_points)
γ_range = 10 .^ range(log10(2e-4), log10(2e-2), length=n_points)
n_avg_range = 10 .^ range(log10(3e2), log10(3e4), length=n_points)
M_range = 10 .^ range(log10(1e7), log10(1e9), length=n_points)
B_21_range = 10 .^ range(log10(1e-8), log10(1e-6), length=n_points)
Δ_range = range(-100, -20, length=n_points)
T_range = range(10, 40, length=n_points)

# --- 3. Initialize Master Array ---
# We will create a massive array of DataFrames to stitch into one master table
all_data = DataFrame[]

println("Scanning J...")
for J_loc in J_range
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J_loc, γ, B_21, M, T, Δ, κ, n_avg)
    push!(all_data, DataFrame(ScanType="J", X_Value=J_loc, dV_Max=ΔV_max))
end

println("Scanning γ...")
for γ_loc in γ_range
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ_loc, B_21, M, T, Δ, κ, n_avg)
    push!(all_data, DataFrame(ScanType="γ", X_Value=γ_loc, dV_Max=ΔV_max))
end

println("Scanning n_avg...")
for n_avg_loc in n_avg_range
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T, Δ, κ, n_avg_loc)
    push!(all_data, DataFrame(ScanType="n_avg", X_Value=n_avg_loc, dV_Max=ΔV_max))
end

println("Scanning M...")
for M_loc in M_range
    κ_loc = 0.5*B_21*M_loc*exp(Δ/T)/T
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M_loc, T, Δ, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="M", X_Value=M_loc, dV_Max=ΔV_max))
end

println("Scanning B_21...")
for B_21_loc in B_21_range
    κ_loc = 0.5*B_21_loc*M*exp(Δ/T)/T
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21_loc, M, T, Δ, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="B_21", X_Value=B_21_loc, dV_Max=ΔV_max))
end

println("Scanning Δ...")
for Δ_loc in Δ_range
    κ_loc = 0.5*B_21*M*exp(Δ_loc/T)/T
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T, Δ_loc, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="Δ", X_Value=Δ_loc, dV_Max=ΔV_max))
end

println("Scanning T...")
for T_loc in T_range
    κ_loc = 0.5*B_21*M*exp(Δ/T_loc)/T_loc
    ΔV_max, _, _, _, _ = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T_loc, Δ, κ_loc, n_avg)
    push!(all_data, DataFrame(ScanType="T", X_Value=T_loc, dV_Max=ΔV_max))
end

# --- 4. Stitch Data & Save ---
master_df = vcat(all_data...)

output_file = "Param_scan/Data/Master_Stability_Results_$(sites)_Site_large.csv"
CSV.write(output_file, master_df)

println("\nCalculations complete.")
println("Data successfully saved to: $output_file")
println("Elapsed time: ", time() - t_start, " seconds")