# ==============================================================================
# UNIFIED MASTER SCRIPT: Parameter Scans, Local Fits, & Universal Bridge
# ==============================================================================
using CairoMakie, GLM, LsqFit, DataFrames, CSV, Statistics, Printf

# ------------------------------------------------------------------------------
# 1. SMALL REGIME: Load Data, Plot 12-Panels, and Local Fit
# ------------------------------------------------------------------------------
println("\n" * "="^80)
println("PROCESSING SMALL REGIME DATA")
println("="^80)
df_small = CSV.read("Param_scan/Data_Cluster/Master_Stability_Results_2_Site_s.csv", DataFrame)

J_s, γ_s, M_s, B_21_s, Δ_s, T_s = 1.0e-2, 2.0e-4, 1.0e7, 1.0e-8, -100.0, 25.0
κ_s = 0.5 * B_21_s * M_s * exp(Δ_s/T_s) / T_s

J_data_s    = filter(row -> row.ScanType == "J", df_small)
γ_data_s    = filter(row -> row.ScanType == "γ", df_small)
n_avg_data_s= filter(row -> row.ScanType == "n_avg", df_small)
M_data_s    = filter(row -> row.ScanType == "M", df_small)
B_21_data_s = filter(row -> row.ScanType == "B_21", df_small)
Δ_data_s    = filter(row -> row.ScanType == "Δ", df_small)
T_data_s    = filter(row -> row.ScanType == "T", df_small)

κ_M_s   = 0.5 .* B_21_s .* M_data_s.X_Value .* exp(Δ_s/T_s) ./ T_s
κ_B21_s = 0.5 .* B_21_data_s.X_Value .* M_s .* exp(Δ_s/T_s) ./ T_s
κ_Δ_s   = 0.5 .* B_21_s .* M_s .* exp.(Δ_data_s.X_Value ./ T_s) ./ T_s
κ_T_s   = 0.5 .* B_21_s .* M_s .* exp.(Δ_s ./ T_data_s.X_Value) ./ T_data_s.X_Value

# Helper function to force A × 10ᴮ formatting using standard fonts (Unicode)
function format_sci(xs)
    superscripts = Dict('-' => '⁻', '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', 
                        '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹')
    
    return map(xs) do x
        if x == 0.0
            return "0"
        else
            exponent = floor(Int, log10(abs(x)))
            mantissa = x / 10.0^exponent
            exp_str = join([superscripts[c] for c in string(exponent)])
            return @sprintf("%.1f×10%s", mantissa, exp_str)
        end
    end
end

# --- 1A. Small Regime 12-Panel Plot ---
println("Generating Small Regime 12-Panel Plot...")
fig_small = Figure(size = (1200, 850), fontsize = 16)
Label(fig_small[0, 1:4], "SMALL PARAMETER REGIME", fontsize = 24, font = :bold)

ax1 = Axis(fig_small[1, 1], xlabel = "J [meV]", ylabel = "ΔV_max / J")
ax2 = Axis(fig_small[1, 2], xlabel = "γ [meV]", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)
ax3 = Axis(fig_small[1, 3], xlabel = "1/γ [1/meV]", ylabel = "ΔV_max / J")
ax4 = Axis(fig_small[1, 4], xlabel = "n_avg", ylabel = "ΔV_max / J", yticks = LinearTicks(4), ytickformat = format_sci)
ax5 = Axis(fig_small[2, 1], xlabel = "M", ylabel = "ΔV_max / J", xticks = LinearTicks(3))
ax6 = Axis(fig_small[2, 2], xlabel = "B_21", ylabel = "ΔV_max / J")
ax7 = Axis(fig_small[2, 3], xlabel = "Δ", ylabel = "ΔV_max / J")
ax8 = Axis(fig_small[2, 4], xlabel = "T", ylabel = "ΔV_max / J")
ax9 = Axis(fig_small[3, 1], xlabel = "exp(Δ/T)/T", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)
ax10 = Axis(fig_small[3, 2], xlabel = "κ", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)
ax11 = Axis(fig_small[3, 3], xlabel = "κ/γ", ylabel = "ΔV_max / J")
ax12 = Axis(fig_small[3, 4], xlabel = "√(J*κ/γ)", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
    ax.yticks = LinearTicks(4)
end

lines!(ax1, J_data_s.X_Value, J_data_s.dV_Max ./ J_data_s.X_Value, color = :blue, linewidth = 2); scatter!(ax1, J_data_s.X_Value, J_data_s.dV_Max ./ J_data_s.X_Value, color = :blue, markersize = 8)
lines!(ax2, γ_data_s.X_Value, γ_data_s.dV_Max ./ J_s, color = :red, linewidth = 2); scatter!(ax2, γ_data_s.X_Value, γ_data_s.dV_Max ./ J_s, color = :red, markersize = 8)
lines!(ax3, 1 ./ γ_data_s.X_Value, γ_data_s.dV_Max ./ J_s, color = :red, linewidth = 2); scatter!(ax3, 1 ./ γ_data_s.X_Value, γ_data_s.dV_Max ./ J_s, color = :red, markersize = 8)
lines!(ax4, n_avg_data_s.X_Value, n_avg_data_s.dV_Max ./ J_s, color = :green, linewidth = 2); scatter!(ax4, n_avg_data_s.X_Value, n_avg_data_s.dV_Max ./ J_s, color = :green, markersize = 8)
lines!(ax5, M_data_s.X_Value, M_data_s.dV_Max ./ J_s, color = :purple, linewidth = 2); scatter!(ax5, M_data_s.X_Value, M_data_s.dV_Max ./ J_s, color = :purple, markersize = 8)
lines!(ax6, B_21_data_s.X_Value, B_21_data_s.dV_Max ./ J_s, color = :orange, linewidth = 2); scatter!(ax6, B_21_data_s.X_Value, B_21_data_s.dV_Max ./ J_s, color = :orange, markersize = 8)
lines!(ax7, Δ_data_s.X_Value, Δ_data_s.dV_Max ./ J_s, color = :cyan, linewidth = 2); scatter!(ax7, Δ_data_s.X_Value, Δ_data_s.dV_Max ./ J_s, color = :cyan, markersize = 8)
lines!(ax8, T_data_s.X_Value, T_data_s.dV_Max ./ J_s, color = :magenta, linewidth = 2); scatter!(ax8, T_data_s.X_Value, T_data_s.dV_Max ./ J_s, color = :magenta, markersize = 8)

lines!(ax9, exp.(Δ_data_s.X_Value ./ T_s) ./ T_s, Δ_data_s.dV_Max ./ J_s, color = :cyan, linewidth = 2); scatter!(ax9, exp.(Δ_data_s.X_Value ./ T_s) ./ T_s, Δ_data_s.dV_Max ./ J_s, color = :cyan, markersize = 8)
lines!(ax9, exp.(Δ_s ./ T_data_s.X_Value) ./ T_data_s.X_Value, T_data_s.dV_Max ./ J_s, color = :magenta, linewidth = 2); scatter!(ax9, exp.(Δ_s ./ T_data_s.X_Value) ./ T_data_s.X_Value, T_data_s.dV_Max ./ J_s, color = :magenta, markersize = 8)

lines!(ax10, κ_M_s, M_data_s.dV_Max ./ J_s, color = :purple, linewidth = 2, label = "M"); scatter!(ax10, κ_M_s, M_data_s.dV_Max ./ J_s, color = :purple, markersize = 8)
lines!(ax10, κ_B21_s, B_21_data_s.dV_Max ./ J_s, color = :orange, linewidth = 2, label = "B_21"); scatter!(ax10, κ_B21_s, B_21_data_s.dV_Max ./ J_s, color = :orange, markersize = 8)
lines!(ax10, κ_Δ_s, Δ_data_s.dV_Max ./ J_s, color = :cyan, linewidth = 2, label = "Δ"); scatter!(ax10, κ_Δ_s, Δ_data_s.dV_Max ./ J_s, color = :cyan, markersize = 8)
lines!(ax10, κ_T_s, T_data_s.dV_Max ./ J_s, color = :magenta, linewidth = 2, label = "T"); scatter!(ax10, κ_T_s, T_data_s.dV_Max ./ J_s, color = :magenta, markersize = 8); axislegend(ax10, position = :rb)

lines!(ax11, κ_s ./ γ_data_s.X_Value, γ_data_s.dV_Max ./ J_s, color = :red, linewidth = 2, label = "γ"); scatter!(ax11, κ_s ./ γ_data_s.X_Value, γ_data_s.dV_Max ./ J_s, color = :red, markersize = 8)
lines!(ax11, κ_M_s ./ γ_s, M_data_s.dV_Max ./ J_s, color = :purple, linewidth = 2, label = "M"); scatter!(ax11, κ_M_s ./ γ_s, M_data_s.dV_Max ./ J_s, color = :purple, markersize = 8)
lines!(ax11, κ_B21_s ./ γ_s, B_21_data_s.dV_Max ./ J_s, color = :orange, linewidth = 2, label = "B_21"); scatter!(ax11, κ_B21_s ./ γ_s, B_21_data_s.dV_Max ./ J_s, color = :orange, markersize = 8)
lines!(ax11, κ_Δ_s ./ γ_s, Δ_data_s.dV_Max ./ J_s, color = :cyan, linewidth = 2, label = "Δ"); scatter!(ax11, κ_Δ_s ./ γ_s, Δ_data_s.dV_Max ./ J_s, color = :cyan, markersize = 8)
lines!(ax11, κ_T_s ./ γ_s, T_data_s.dV_Max ./ J_s, color = :magenta, linewidth = 2, label = "T"); scatter!(ax11, κ_T_s ./ γ_s, T_data_s.dV_Max ./ J_s, color = :magenta, markersize = 8); axislegend(ax11, position = :rb)

lines!(ax12, sqrt.(J_data_s.X_Value .* (κ_s ./ γ_s)), J_data_s.dV_Max ./ J_data_s.X_Value, color = :blue, linewidth = 2, label = "J"); scatter!(ax12, sqrt.(J_data_s.X_Value .* (κ_s ./ γ_s)), J_data_s.dV_Max ./ J_data_s.X_Value, color = :blue, markersize = 8)
lines!(ax12, sqrt.(J_s .* (κ_s ./ γ_data_s.X_Value)), γ_data_s.dV_Max ./ J_s, color = :red, linewidth = 2, label = "γ"); scatter!(ax12, sqrt.(J_s .* (κ_s ./ γ_data_s.X_Value)), γ_data_s.dV_Max ./ J_s, color = :red, markersize = 8)
lines!(ax12, sqrt.(J_s .* (κ_M_s ./ γ_s)), M_data_s.dV_Max ./ J_s, color = :purple, linewidth = 2, label = "M"); scatter!(ax12, sqrt.(J_s .* (κ_M_s ./ γ_s)), M_data_s.dV_Max ./ J_s, color = :purple, markersize = 8)
lines!(ax12, sqrt.(J_s .* (κ_B21_s ./ γ_s)), B_21_data_s.dV_Max ./ J_s, color = :orange, linewidth = 2, label = "B_21"); scatter!(ax12, sqrt.(J_s .* (κ_B21_s ./ γ_s)), B_21_data_s.dV_Max ./ J_s, color = :orange, markersize = 8)
lines!(ax12, sqrt.(J_s .* (κ_Δ_s ./ γ_s)), Δ_data_s.dV_Max ./ J_s, color = :cyan, linewidth = 2, label = "Δ"); scatter!(ax12, sqrt.(J_s .* (κ_Δ_s ./ γ_s)), Δ_data_s.dV_Max ./ J_s, color = :cyan, markersize = 8)
lines!(ax12, sqrt.(J_s .* (κ_T_s ./ γ_s)), T_data_s.dV_Max ./ J_s, color = :magenta, linewidth = 2, label = "T"); scatter!(ax12, sqrt.(J_s .* (κ_T_s ./ γ_s)), T_data_s.dV_Max ./ J_s, color = :magenta, markersize = 8); axislegend(ax12, position = :rb)

save("Param_scan/Figures/Parameter_Scans_2_Site_small.png", fig_small)
display(fig_small)

# --- 1B. Small Regime Linear Regression (Y ~ √(J*κ/γ)) ---
X_lin_small = vcat(
    sqrt.(J_data_s.X_Value .* (κ_s ./ γ_s)),
    sqrt.(J_s .* (κ_s ./ γ_data_s.X_Value)),
    sqrt.(J_s .* (κ_M_s ./ γ_s)),
    sqrt.(J_s .* (κ_B21_s ./ γ_s)),
    sqrt.(J_s .* (κ_Δ_s ./ γ_s)),
    sqrt.(J_s .* (κ_T_s ./ γ_s))
)
Y_small = vcat(
    J_data_s.dV_Max ./ J_data_s.X_Value, 
    γ_data_s.dV_Max ./ J_s, 
    M_data_s.dV_Max ./ J_s, 
    B_21_data_s.dV_Max ./ J_s, 
    Δ_data_s.dV_Max ./ J_s, 
    T_data_s.dV_Max ./ J_s
)

println("\nPerforming Small Regime Linear Regression (Y ~ √(J*κ/γ))...")
df_reg_s = DataFrame(X_scaled = X_lin_small, Y = Y_small)
model_small = lm(@formula(Y ~ X_scaled), df_reg_s)
display(DataFrame(coeftable(model_small)))

α_small = coef(model_small)[1]
C1_iso  = coef(model_small)[2]
println("R² value (Small): ", r2(model_small))


# ------------------------------------------------------------------------------
# 2. LARGE REGIME: Load Data, Plot 12-Panels, and Local Fit
# ------------------------------------------------------------------------------
println("\n" * "="^80)
println("PROCESSING LARGE REGIME DATA")
println("="^80)
df_large = CSV.read("Param_scan/Data_Cluster/Master_Stability_Results_2_Site_l.csv", DataFrame)

J_l, γ_l, M_l, B_21_l, Δ_l, T_l = 1.0e-1, 2.0e-3, 1.0e8, 1.0e-7, -60.0, 25.0
κ_l = 0.5 * B_21_l * M_l * exp(Δ_l/T_l) / T_l

J_data_l    = filter(row -> row.ScanType == "J", df_large)
γ_data_l    = filter(row -> row.ScanType == "γ", df_large)
n_avg_data_l= filter(row -> row.ScanType == "n_avg", df_large)
M_data_l    = filter(row -> row.ScanType == "M", df_large)
B_21_data_l = filter(row -> row.ScanType == "B_21", df_large)
Δ_data_l    = filter(row -> row.ScanType == "Δ", df_large)
T_data_l    = filter(row -> row.ScanType == "T", df_large)

κ_M_l   = 0.5 .* B_21_l .* M_data_l.X_Value .* exp(Δ_l/T_l) ./ T_l
κ_B21_l = 0.5 .* B_21_data_l.X_Value .* M_l .* exp(Δ_l/T_l) ./ T_l
κ_Δ_l   = 0.5 .* B_21_l .* M_l .* exp.(Δ_data_l.X_Value ./ T_l) ./ T_l
κ_T_l   = 0.5 .* B_21_l .* M_l .* exp.(Δ_l ./ T_data_l.X_Value) ./ T_data_l.X_Value

# --- 2A. Large Regime 12-Panel Plot ---
println("Generating Large Regime 12-Panel Plot...")
fig_large = Figure(size = (1200, 850), fontsize = 16)
Label(fig_large[0, 1:4], "LARGE PARAMETER REGIME", fontsize = 24, font = :bold)

bx1 = Axis(fig_large[1, 1], xlabel = "J [meV]", ylabel = "ΔV_max / J")
bx2 = Axis(fig_large[1, 2], xlabel = "γ [meV]", ylabel = "ΔV_max / J", xticks = LinearTicks(3))
bx3 = Axis(fig_large[1, 3], xlabel = "1/γ [1/meV]", ylabel = "ΔV_max / J")
bx4 = Axis(fig_large[1, 4], xlabel = "n_avg", ylabel = "ΔV_max / J")
bx5 = Axis(fig_large[2, 1], xlabel = "M", ylabel = "ΔV_max / J")
bx6 = Axis(fig_large[2, 2], xlabel = "B_21", ylabel = "ΔV_max / J")
bx7 = Axis(fig_large[2, 3], xlabel = "Δ", ylabel = "ΔV_max / J")
bx8 = Axis(fig_large[2, 4], xlabel = "T", ylabel = "ΔV_max / J")
bx9 = Axis(fig_large[3, 1], xlabel = "exp(Δ/T)/T", ylabel = "ΔV_max / J")
bx10 = Axis(fig_large[3, 2], xlabel = "κ", ylabel = "ΔV_max / J", xticks = LinearTicks(3))
bx11 = Axis(fig_large[3, 3], xlabel = "κ/γ", ylabel = "ΔV_max / J")
bx12 = Axis(fig_large[3, 4], xlabel = "J*κ/γ", ylabel = "ΔV_max / J")

for bx in [bx1, bx2, bx3, bx4, bx5, bx6, bx7, bx8, bx9, bx10, bx11, bx12]
    bx.yticks = LinearTicks(4)
end

lines!(bx1, J_data_l.X_Value, J_data_l.dV_Max ./ J_data_l.X_Value, color = :blue, linewidth = 2); scatter!(bx1, J_data_l.X_Value, J_data_l.dV_Max ./ J_data_l.X_Value, color = :blue, markersize = 8)
lines!(bx2, γ_data_l.X_Value, γ_data_l.dV_Max ./ J_l, color = :red, linewidth = 2); scatter!(bx2, γ_data_l.X_Value, γ_data_l.dV_Max ./ J_l, color = :red, markersize = 8)
lines!(bx3, 1 ./ γ_data_l.X_Value, γ_data_l.dV_Max ./ J_l, color = :red, linewidth = 2); scatter!(bx3, 1 ./ γ_data_l.X_Value, γ_data_l.dV_Max ./ J_l, color = :red, markersize = 8)
lines!(bx4, n_avg_data_l.X_Value, n_avg_data_l.dV_Max ./ J_l, color = :green, linewidth = 2); scatter!(bx4, n_avg_data_l.X_Value, n_avg_data_l.dV_Max ./ J_l, color = :green, markersize = 8)
lines!(bx5, M_data_l.X_Value, M_data_l.dV_Max ./ J_l, color = :purple, linewidth = 2); scatter!(bx5, M_data_l.X_Value, M_data_l.dV_Max ./ J_l, color = :purple, markersize = 8)
lines!(bx6, B_21_data_l.X_Value, B_21_data_l.dV_Max ./ J_l, color = :orange, linewidth = 2); scatter!(bx6, B_21_data_l.X_Value, B_21_data_l.dV_Max ./ J_l, color = :orange, markersize = 8)
lines!(bx7, Δ_data_l.X_Value, Δ_data_l.dV_Max ./ J_l, color = :cyan, linewidth = 2); scatter!(bx7, Δ_data_l.X_Value, Δ_data_l.dV_Max ./ J_l, color = :cyan, markersize = 8)
lines!(bx8, T_data_l.X_Value, T_data_l.dV_Max ./ J_l, color = :magenta, linewidth = 2); scatter!(bx8, T_data_l.X_Value, T_data_l.dV_Max ./ J_l, color = :magenta, markersize = 8)

lines!(bx9, exp.(Δ_data_l.X_Value ./ T_l) ./ T_l, Δ_data_l.dV_Max ./ J_l, color = :cyan, linewidth = 2); scatter!(bx9, exp.(Δ_data_l.X_Value ./ T_l) ./ T_l, Δ_data_l.dV_Max ./ J_l, color = :cyan, markersize = 8)
lines!(bx9, exp.(Δ_l ./ T_data_l.X_Value) ./ T_data_l.X_Value, T_data_l.dV_Max ./ J_l, color = :magenta, linewidth = 2); scatter!(bx9, exp.(Δ_l ./ T_data_l.X_Value) ./ T_data_l.X_Value, T_data_l.dV_Max ./ J_l, color = :magenta, markersize = 8)

lines!(bx10, κ_M_l, M_data_l.dV_Max ./ J_l, color = :purple, linewidth = 2, label = "M"); scatter!(bx10, κ_M_l, M_data_l.dV_Max ./ J_l, color = :purple, markersize = 8)
lines!(bx10, κ_B21_l, B_21_data_l.dV_Max ./ J_l, color = :orange, linewidth = 2, label = "B_21"); scatter!(bx10, κ_B21_l, B_21_data_l.dV_Max ./ J_l, color = :orange, markersize = 8)
lines!(bx10, κ_Δ_l, Δ_data_l.dV_Max ./ J_l, color = :cyan, linewidth = 2, label = "Δ"); scatter!(bx10, κ_Δ_l, Δ_data_l.dV_Max ./ J_l, color = :cyan, markersize = 8)
lines!(bx10, κ_T_l, T_data_l.dV_Max ./ J_l, color = :magenta, linewidth = 2, label = "T"); scatter!(bx10, κ_T_l, T_data_l.dV_Max ./ J_l, color = :magenta, markersize = 8); axislegend(bx10, position = :rb)

lines!(bx11, κ_l ./ γ_data_l.X_Value, γ_data_l.dV_Max ./ J_l, color = :red, linewidth = 2, label = "γ"); scatter!(bx11, κ_l ./ γ_data_l.X_Value, γ_data_l.dV_Max ./ J_l, color = :red, markersize = 8)
lines!(bx11, κ_M_l ./ γ_l, M_data_l.dV_Max ./ J_l, color = :purple, linewidth = 2, label = "M"); scatter!(bx11, κ_M_l ./ γ_l, M_data_l.dV_Max ./ J_l, color = :purple, markersize = 8)
lines!(bx11, κ_B21_l ./ γ_l, B_21_data_l.dV_Max ./ J_l, color = :orange, linewidth = 2, label = "B_21"); scatter!(bx11, κ_B21_l ./ γ_l, B_21_data_l.dV_Max ./ J_l, color = :orange, markersize = 8)
lines!(bx11, κ_Δ_l ./ γ_l, Δ_data_l.dV_Max ./ J_l, color = :cyan, linewidth = 2, label = "Δ"); scatter!(bx11, κ_Δ_l ./ γ_l, Δ_data_l.dV_Max ./ J_l, color = :cyan, markersize = 8)
lines!(bx11, κ_T_l ./ γ_l, T_data_l.dV_Max ./ J_l, color = :magenta, linewidth = 2, label = "T"); scatter!(bx11, κ_T_l ./ γ_l, T_data_l.dV_Max ./ J_l, color = :magenta, markersize = 8); axislegend(bx11, position = :rb)

lines!(bx12, J_data_l.X_Value .* (κ_l ./ γ_l), J_data_l.dV_Max ./ J_data_l.X_Value, color = :blue, linewidth = 2, label = "J"); scatter!(bx12, J_data_l.X_Value .* (κ_l ./ γ_l), J_data_l.dV_Max ./ J_data_l.X_Value, color = :blue, markersize = 8)
lines!(bx12, J_l .* (κ_l ./ γ_data_l.X_Value), γ_data_l.dV_Max ./ J_l, color = :red, linewidth = 2, label = "γ"); scatter!(bx12, J_l .* (κ_l ./ γ_data_l.X_Value), γ_data_l.dV_Max ./ J_l, color = :red, markersize = 8)
lines!(bx12, J_l .* (κ_M_l ./ γ_l), M_data_l.dV_Max ./ J_l, color = :purple, linewidth = 2, label = "M"); scatter!(bx12, J_l .* (κ_M_l ./ γ_l), M_data_l.dV_Max ./ J_l, color = :purple, markersize = 8)
lines!(bx12, J_l .* (κ_B21_l ./ γ_l), B_21_data_l.dV_Max ./ J_l, color = :orange, linewidth = 2, label = "B_21"); scatter!(bx12, J_l .* (κ_B21_l ./ γ_l), B_21_data_l.dV_Max ./ J_l, color = :orange, markersize = 8)
lines!(bx12, J_l .* (κ_Δ_l ./ γ_l), Δ_data_l.dV_Max ./ J_l, color = :cyan, linewidth = 2, label = "Δ"); scatter!(bx12, J_l .* (κ_Δ_l ./ γ_l), Δ_data_l.dV_Max ./ J_l, color = :cyan, markersize = 8)
lines!(bx12, J_l .* (κ_T_l ./ γ_l), T_data_l.dV_Max ./ J_l, color = :magenta, linewidth = 2, label = "T"); scatter!(bx12, J_l .* (κ_T_l ./ γ_l), T_data_l.dV_Max ./ J_l, color = :magenta, markersize = 8); axislegend(bx12, position = :rb)

save("Param_scan/Figures/Parameter_Scans_2_Site_large.png", fig_large)
display(fig_large)

# --- 2B. Large Regime Linear Regression (Y ~ J*κ/γ) ---
X_lin_large = vcat(
    J_data_l.X_Value .* (κ_l ./ γ_l),
    J_l .* (κ_l ./ γ_data_l.X_Value),
    J_l .* (κ_M_l ./ γ_l),
    J_l .* (κ_B21_l ./ γ_l),
    J_l .* (κ_Δ_l ./ γ_l),
    J_l .* (κ_T_l ./ γ_l)
)
Y_large = vcat(
    J_data_l.dV_Max ./ J_data_l.X_Value, 
    γ_data_l.dV_Max ./ J_l, 
    M_data_l.dV_Max ./ J_l, 
    B_21_data_l.dV_Max ./ J_l, 
    Δ_data_l.dV_Max ./ J_l, 
    T_data_l.dV_Max ./ J_l
)

println("\nPerforming Large Regime Linear Regression (Y ~ J*κ/γ)...")
df_reg_l = DataFrame(X_scaled = X_lin_large, Y = Y_large)
model_large = lm(@formula(Y ~ X_scaled), df_reg_l)
display(DataFrame(coeftable(model_large)))

α_large = coef(model_large)[1]
C2_iso  = coef(model_large)[2]
println("R² value (Large): ", r2(model_large))


# ------------------------------------------------------------------------------
# 5. 2-PANEL PLOT: Visualizing the Linearized Scaling
# ------------------------------------------------------------------------------
println("\nGenerating 2-panel linearized comparison plot...")
fig_lin = Figure(size = (1200, 500), fontsize = 16)

ax_small_lin = Axis(fig_lin[1, 1], title = "Linearized Small Regime", xlabel = "√(J*κ/γ)", ylabel = "ΔV_max / J")
ax_large_lin = Axis(fig_lin[1, 2], title = "Linearized Large Regime", xlabel = "J*κ/γ", ylabel = "ΔV_max / J")

x_iso_s_range = range(0, maximum(X_lin_small), length=500)
x_iso_l_range = range(0, maximum(X_lin_large), length=500)

y_iso_s = C1_iso .* x_iso_s_range .+ α_small
y_iso_l = C2_iso .* x_iso_l_range .+ α_large
# -------------------------------------------------------

scatter!(ax_small_lin, X_lin_small, Y_small, color = (:blue, 0.6), markersize = 8, label = "Data")
scatter!(ax_large_lin, X_lin_large, Y_large, color = (:red, 0.6), markersize = 8, label = "Data")

lines!(ax_small_lin, x_iso_s_range, y_iso_s, color = :black, linewidth = 2, label = "Fit")
lines!(ax_large_lin, x_iso_l_range, y_iso_l, color = :black, linewidth = 2, label = "Fit")

# Annotation text logic
sign_small = α_small < 0 ? "-" : "+"
text_small = @sprintf("ΔV_max / J = %.4f * √(J*κ/γ) %s %.4f\nR² = %.4f", C1_iso, sign_small, abs(α_small), r2(model_small))
text_large = @sprintf("ΔV_max / J = %.4f * (J*κ/γ) + %.4f\nR² = %.4f", C2_iso, α_large, r2(model_large))

text!(ax_small_lin, text_small, position = (0.95, 0.05), align = (:right, :bottom), space = :relative, fontsize = 18, font = :bold)
text!(ax_large_lin, text_large, position = (0.95, 0.05), align = (:right, :bottom), space = :relative, fontsize = 18, font = :bold)

save("Param_scan/Figures/Linearized_Scaling_JJ.png", fig_lin)
display(fig_lin)

# ------------------------------------------------------------------------------
# 6. FORMULA SUMMARY TABLE
# ------------------------------------------------------------------------------
println("\n\n======================================================================================================")
println(rpad(" MODEL TYPE", 25), "| EXTRACTED FORMULA (Scaling relative to J*κ/γ)")
println("======================================================================================================")

eqn_small = "ΔV_max / J = ($(round(C1_iso, digits=5))) * √(J*κ/γ) + ($(round(α_small, digits=6)))"
println(rpad(" Isolated Power (Small)", 25), "| ", eqn_small)
println("------------------------------------------------------------------------------------------------------")

eqn_large = "ΔV_max / J = ($(round(C2_iso, digits=5))) * (J*κ/γ) + ($(round(α_large, digits=5)))"
println(rpad(" Isolated Square (Large)", 25), "| ", eqn_large)
println("------------------------------------------------------------------------------------------------------")

println("======================================================================================================\n")