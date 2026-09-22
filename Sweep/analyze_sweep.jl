# ==============================================================================
# DATA AGGREGATION & PLOTTING SCRIPT
# ==============================================================================
using JLD2, CairoMakie, Printf, Statistics, LaTeXStrings

# --- Configuration ---
run_name = "sites_2_1e7"
base_dir = joinpath("Sweep/Data", run_name)
output_plot_dir = joinpath("Sweep/Plots", run_name)

include("../ExtraFunctions/Analytics.jl") 
mkpath(output_plot_dir)

# Define the number of tasks and smoothing window sizes for the plots
num_tasks = 100
window_v_phi = 5
window_coh = 5

# --- 1. Data Aggregation ---
println("Aggregating data from $base_dir...")

ΔV_arr, v_phi_arr = Float64[], Float64[]
equal_time_coh_array = ComplexF64[]

for task_id in 1:num_tasks
    file = joinpath(base_dir, "result_$(task_id).jld2")
    if isfile(file)
        data_loc = load(file)
        push!(ΔV_arr, data_loc["dV"])
        push!(v_phi_arr, data_loc["v_phi"])
        push!(equal_time_coh_array, data_loc["equal_time_coh"])
    else
        @warn "File missing: $file"
    end
end

# Sort arrays by ΔV
perm = sortperm(ΔV_arr)
ΔV_sorted = ΔV_arr[perm]
v_phi_sorted = abs.(v_phi_arr[perm])
equal_time_coh_sorted = equal_time_coh_array[perm]

# --- 2. Outlier Filter (IQR Method on ΔV) ---
Q1, Q3 = quantile(ΔV_sorted, 0.25), quantile(ΔV_sorted, 0.75)
upper_bound = Q3 + 3.0 * (Q3 - Q1)
valid_indices = ΔV_sorted .<= upper_bound

if count(.!valid_indices) > 0
    println("Removed $(count(.!valid_indices)) outlier(s) (ΔV > $upper_bound)")
end

ΔV_sorted = ΔV_sorted[valid_indices]
v_phi_sorted = v_phi_sorted[valid_indices]
equal_time_coh_sorted = equal_time_coh_sorted[valid_indices]

# --- 3. Theoretical Calculations ---
data_gl = load(joinpath(base_dir, "result_1.jld2"))
p = data_gl["parameters"]

sites = haskey(p, :sites) ? p.sites : 2
δ = haskey(p, :δ) ? p.δ : 0.0

crit_res = calculate_critical_ΔV(sites, J=p.J, γ=p.γ, B_21=p.B_21, M=p.M, T=p.T, Δ=p.Δ, κ=p.κ, n_avg=p.n_avg, δ=δ)
ΔV_max_stability = isnothing(crit_res.critical_ΔV) ? NaN : crit_res.critical_ΔV

# --- Helper: Simple Moving Average ---
function simple_moving_average(data, window_size)
    half_window = window_size ÷ 2
    return [mean(@view data[max(1, i - half_window) : min(length(data), i + half_window)]) for i in 1:length(data)]
end

# --- 4. Plot 1: Phase Velocity (v_phi) ---
smoothed_v_phi = simple_moving_average(v_phi_sorted, window_v_phi)
dv_phi = diff(smoothed_v_phi) ./ diff(ΔV_sorted)
ΔV_mid = ΔV_sorted[1:end-1] .+ diff(ΔV_sorted) ./ 2.0

idx_peak = argmax(dv_phi)
final_ΔV_c = ΔV_mid[idx_peak]
@printf("\n[Phase Velocity] Final Critical ΔV: %.3f\n", final_ΔV_c)

fig_vphi = Figure(size = (700, 700)) 
ax_vphi = Axis(fig_vphi[1, 1], title = LaTeXString("Fasesnelheid"), xlabel = LaTeXString("ΔV/ΔV_{th}"), ylabel = LaTeXString("|v_ϕ|/ΔV_{th}"), titlesize=25, xlabelsize=25, xticklabelsize=20, ylabelsize=25, yticklabelsize=20)

scatterlines!(ax_vphi, ΔV_sorted/ΔV_max_stability, v_phi_sorted/ΔV_max_stability, label = LaTeXString("Data"), color = :green, linewidth = 2, marker = :circle, markersize = 8, strokewidth = 0)
if !isnan(ΔV_max_stability)
    vlines!(ax_vphi, [ΔV_max_stability]/ΔV_max_stability, color=:red, linestyle=:dash, label=LaTeXString(@sprintf("Theoretische ΔV_{max}: %.3f", ΔV_max_stability/ΔV_max_stability)))
end
lines!(ax_vphi, ΔV_sorted/ΔV_max_stability, ΔV_sorted/ΔV_max_stability, color=:orange, linestyle=:dash, linewidth=3, label=LaTeXString("|v_ϕ| = ΔV"))

y_val = v_phi_sorted[argmin(abs.(ΔV_sorted .- final_ΔV_c))]
scatter!(ax_vphi, [final_ΔV_c]/ΔV_max_stability, [y_val]/ΔV_max_stability, color=:purple, marker=:star5, markersize=15, label=LaTeXString(@sprintf("Gedetecteerde ΔV_{max}: %.3f", final_ΔV_c/ΔV_max_stability)))
axislegend(ax_vphi, position = :lt, labelsize=20)

ax_vphi_deriv = Axis(fig_vphi[2, 1], title = "First Derivative of Phase Velocity", xlabel = "ΔV/ΔV_{th}", ylabel = "d|v_ϕ|/dΔV", titlesize=25, xlabelsize=25, xticklabelsize=20, ylabelsize=25, yticklabelsize=20)
if !isnan(ΔV_max_stability)
    vlines!(ax_vphi_deriv, [ΔV_max_stability]/ΔV_max_stability, color=:red, linestyle=:dash, label=@sprintf("Theoretical ΔV_max: %.3f", ΔV_max_stability/ΔV_max_stability))
end
lines!(ax_vphi_deriv, ΔV_mid/ΔV_max_stability, dv_phi, color=:blue, linewidth=2.5, label="d|v_ϕ|/dΔV")
scatter!(ax_vphi_deriv, [final_ΔV_c]/ΔV_max_stability, [dv_phi[idx_peak]], color=:purple, marker=:star5, markersize=15, label=@sprintf("Max at %.3f", final_ΔV_c/ΔV_max_stability))
axislegend(ax_vphi_deriv, position=:lt, labelsize=20)

display(fig_vphi)
save(joinpath(output_plot_dir, "v_phi_$run_name.png"), fig_vphi)

# --- 5. Plot 2: Equal-Time Coherence ---
coh_abs = abs.(equal_time_coh_sorted)
smoothed_coh = simple_moving_average(coh_abs, window_coh)
d_coh = diff(smoothed_coh) ./ diff(ΔV_sorted)

idx_drop = argmin(d_coh)
final_ΔV_c_coh = ΔV_mid[idx_drop]
@printf("[Coherence] Final Critical ΔV: %.3f\n", final_ΔV_c_coh)

fig_coherence = Figure(size=(700, 700)) 
ax_coh = Axis(fig_coherence[1, 1], title=LaTeXString("Gelijktijdige coherentie"), xlabel=LaTeXString("ΔV/ΔV_{th}"), ylabel=LaTeXString("|g⁽¹⁾(0)|"), titlesize=25, xlabelsize=25, xticklabelsize=20, ylabelsize=25, yticklabelsize=20)

scatterlines!(ax_coh, ΔV_sorted/ΔV_max_stability, coh_abs, color=:green, linewidth=2, marker=:circle, markersize=8, strokewidth=0, label=LaTeXString("Data"))
if !isnan(ΔV_max_stability)
    vlines!(ax_coh, [ΔV_max_stability]/ΔV_max_stability, color=:red, linestyle=:dash, label=LaTeXString(@sprintf("Theoretische ΔV_{max}: %.3f", ΔV_max_stability/ΔV_max_stability)))
end

y_val_coh = coh_abs[argmin(abs.(ΔV_sorted .- final_ΔV_c_coh))]
scatter!(ax_coh, [final_ΔV_c_coh]/ΔV_max_stability, [y_val_coh], color=:purple, marker=:star5, markersize=15, label=LaTeXString(@sprintf("Gedetecteerde ΔV_{max}: %.3f", final_ΔV_c_coh/ΔV_max_stability)))
axislegend(ax_coh, position=:lb, labelsize=20)

ax_d_coh = Axis(fig_coherence[2, 1], title="First Derivative of Coherence", xlabel="ΔV/ΔV_{th}", ylabel="d|g⁽¹⁾(0)|/dΔV", titlesize=25, xlabelsize=25, xticklabelsize=20, ylabelsize=25, yticklabelsize=20)
if !isnan(ΔV_max_stability)
    vlines!(ax_d_coh, [ΔV_max_stability]/ΔV_max_stability, color=:red, linestyle=:dash, label=@sprintf("Theoretical ΔV_max: %.3f", ΔV_max_stability/ΔV_max_stability))
end
lines!(ax_d_coh, ΔV_mid/ΔV_max_stability, d_coh, color=:blue, linewidth=2.5, label="d|g⁽¹⁾(0)|/dΔV")
scatter!(ax_d_coh, [final_ΔV_c_coh]/ΔV_max_stability, [d_coh[idx_drop]], color=:purple, marker=:star5, markersize=15, label=@sprintf("Min at %.3f", final_ΔV_c_coh/ΔV_max_stability))
axislegend(ax_d_coh, position=:lb, labelsize=20)

display(fig_coherence)
save(joinpath(output_plot_dir, "Coherence_$run_name.png"), fig_coherence)

println("All plots generated and saved in $output_plot_dir")