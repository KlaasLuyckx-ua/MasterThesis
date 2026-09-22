using JLD2, DataFrames
include("../Analytics/Analytics.jl") 

# 1. Parse Command Line Arguments from the Cluster
if length(ARGS) != 4
    error("Usage: julia param_scan_worker.jl <sites> <regime> <param> <n_points>")
end

sites = parse(Int, ARGS[1])
regime = ARGS[2]
param = ARGS[3]
n_points = parse(Int, ARGS[4])

# 2. Setup Regime Parameters
if regime == "s" # 's' for small thermalisation regime
    J_base, γ_base, n_avg_base = 1.0e-2, 2.0e-4, 3.0e2
    M_base, B_21_base, Δ_base, T_base = 1.0e7, 1.0e-8, -100.0, 25.0
    
    scan_ranges = Dict(
        "J"     => 10 .^ range(log10(1e-3), log10(1e-1), length=n_points),
        "gamma" => 10 .^ range(log10(2e-5), log10(2e-3), length=n_points),
        "n_avg" => 10 .^ range(log10(3e1),  log10(3e3),  length=n_points),
        "M"     => 10 .^ range(log10(1e7),  log10(1e8),  length=n_points),
        "B_21"  => 10 .^ range(log10(1e-9), log10(1e-7), length=n_points),
        "Delta" => range(-140, -60, length=n_points),
        "T"     => range(10, 40, length=n_points)
    )
elseif regime == "l" # 'l' for large thermalisation regime
    J_base, γ_base, n_avg_base = 1.0e-1, 2.0e-3, 3.0e3
    M_base, B_21_base, Δ_base, T_base = 1.0e8, 1.0e-7, -60.0, 25.0
    
    scan_ranges = Dict(
        "J"     => 10 .^ range(log10(1e-2), log10(1e0),  length=n_points),
        "gamma" => 10 .^ range(log10(2e-4), log10(2e-2), length=n_points),
        "n_avg" => 10 .^ range(log10(3e2),  log10(3e4),  length=n_points),
        "M"     => 10 .^ range(log10(1e7),  log10(1e9),  length=n_points),
        "B_21"  => 10 .^ range(log10(1e-8), log10(1e-6), length=n_points),
        "Delta" => range(-100, -20, length=n_points),
        "T"     => range(10, 40, length=n_points)
    )
else
    error("Regime must be 's' or 'l'")
end

range_to_scan = scan_ranges[param]
all_data = DataFrame(ScanType=String[], X_Value=Float64[], dV_Max=Float64[])

println("Worker started: $sites Sites | Regime: $regime | Scanning: $param")
t_start = time()

# 3. Execute Scan
for val in range_to_scan
    # Reset to base parameters for each step
    J_c, γ_c, n_avg_c = J_base, γ_base, n_avg_base
    M_c, B_21_c, Δ_c, T_c = M_base, B_21_base, Δ_base, T_base
    scan_label = param

    # Map the current scan parameter
    if param == "J"
        J_c = val
    elseif param == "gamma"
        γ_c = val
        scan_label = "γ"
    elseif param == "n_avg"
        n_avg_c = val
    elseif param == "M"
        M_c = val
    elseif param == "B_21"
        B_21_c = val
    elseif param == "Delta"
        Δ_c = val
        scan_label = "Δ"
    elseif param == "T"
        T_c = val
    end

    # Calculate dependent kappa locally for this step
    κ_c = 0.5 * B_21_c * M_c * exp(Δ_c/T_c) / T_c
    
    # Call unified critical ΔV solver (δ defaults to 0.0 inside)
    sol = calculate_critical_ΔV(sites; J=J_c, γ=γ_c, B_21=B_21_c, M=M_c, T=T_c, Δ=Δ_c, κ=κ_c, n_avg=n_avg_c)
    
    # Handle NaN cases where solver might not cross stability threshold
    ΔV_max = isnothing(sol.critical_ΔV) ? NaN : sol.critical_ΔV
    
    push!(all_data, (scan_label, val, ΔV_max))
end

# 4. Save to a temporary JLD2 file
output_file = "Param_scan/Data/Results_$(sites)_$(regime)_$(param).jld2"
jldsave(output_file; df = all_data)

println("Worker finished in $(round(time() - t_start, digits=2)) seconds. Saved to $output_file")