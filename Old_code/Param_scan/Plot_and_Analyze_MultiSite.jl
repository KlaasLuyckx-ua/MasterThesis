# ==============================================================================
# UNIFIED MASTER SCRIPT: Parameter Scans, Local Fits, & Universal Bridge
# ==============================================================================
using CairoMakie, GLM, LsqFit, DataFrames, CSV, Statistics, Printf

sites_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
for sites in sites_range
    # ------------------------------------------------------------------------------
    # 1. SMALL REGIME: Load Data, Plot 12-Panels, and Local Fit
    # ------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("PROCESSING SMALL REGIME DATA")
    println("="^80)
    df_small = CSV.read("Param_scan/Data_Cluster/Master_Stability_Results_$(sites)_Site_s.csv", DataFrame)

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

    # --- Small Regime 12-Panel Plot ---
    println("Generating Small Regime 12-Panel Plot...")
    fig_small = Figure(size = (1200, 850), fontsize = 16)
    Label(fig_small[0, 1:4], "SMALL PARAMETER REGIME $(sites) SITES", fontsize = 24, font = :bold)

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

    save("Param_scan/Figures/Parameter_Scans_$(sites)_Site_small.png", fig_small)
    display(fig_small)


    # ------------------------------------------------------------------------------
    # 2. LARGE REGIME: Load Data, Plot 12-Panels, and Local Fit
    # ------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("PROCESSING LARGE REGIME DATA")
    println("="^80)
    df_large = CSV.read("Param_scan/Data_Cluster/Master_Stability_Results_$(sites)_Site_l.csv", DataFrame)

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

    # --- Large Regime 12-Panel Plot ---
    println("Generating Large Regime 12-Panel Plot...")
    fig_large = Figure(size = (1200, 850), fontsize = 16)
    Label(fig_large[0, 1:4], "LARGE PARAMETER REGIME $(sites) SITES", fontsize = 24, font = :bold)

    bx1 = Axis(fig_large[1, 1], xlabel = "J [meV]", ylabel = "ΔV_max / J")
    bx2 = Axis(fig_large[1, 2], xlabel = "γ [meV]", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)
    bx3 = Axis(fig_large[1, 3], xlabel = "1/γ [1/meV]", ylabel = "ΔV_max / J")
    bx4 = Axis(fig_large[1, 4], xlabel = "n_avg", ylabel = "ΔV_max / J", yticks = LinearTicks(4), ytickformat = format_sci)
    bx5 = Axis(fig_large[2, 1], xlabel = "M", ylabel = "ΔV_max / J", xticks = LinearTicks(3))
    bx6 = Axis(fig_large[2, 2], xlabel = "B_21", ylabel = "ΔV_max / J")
    bx7 = Axis(fig_large[2, 3], xlabel = "Δ", ylabel = "ΔV_max / J")
    bx8 = Axis(fig_large[2, 4], xlabel = "T", ylabel = "ΔV_max / J")
    bx9 = Axis(fig_large[3, 1], xlabel = "exp(Δ/T)/T", ylabel = "ΔV_max / J")
    bx10 = Axis(fig_large[3, 2], xlabel = "κ", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)
    bx11 = Axis(fig_large[3, 3], xlabel = "κ/γ", ylabel = "ΔV_max / J")
    bx12 = Axis(fig_large[3, 4], xlabel = "√(J*κ/γ)", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)

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

    lines!(bx12, sqrt.(J_data_l.X_Value .* (κ_l ./ γ_l)), J_data_l.dV_Max ./ J_data_l.X_Value, color = :blue, linewidth = 2, label = "J"); scatter!(bx12, sqrt.(J_data_l.X_Value .* (κ_l ./ γ_l)), J_data_l.dV_Max ./ J_data_l.X_Value, color = :blue, markersize = 8)
    lines!(bx12, sqrt.(J_l .* (κ_l ./ γ_data_l.X_Value)), γ_data_l.dV_Max ./ J_l, color = :red, linewidth = 2, label = "γ"); scatter!(bx12, sqrt.(J_l .* (κ_l ./ γ_data_l.X_Value)), γ_data_l.dV_Max ./ J_l, color = :red, markersize = 8)
    lines!(bx12, sqrt.(J_l .* (κ_M_l ./ γ_l)), M_data_l.dV_Max ./ J_l, color = :purple, linewidth = 2, label = "M"); scatter!(bx12, sqrt.(J_l .* (κ_M_l ./ γ_l)), M_data_l.dV_Max ./ J_l, color = :purple, markersize = 8)
    lines!(bx12, sqrt.(J_l .* (κ_B21_l ./ γ_l)), B_21_data_l.dV_Max ./ J_l, color = :orange, linewidth = 2, label = "B_21"); scatter!(bx12, sqrt.(J_l .* (κ_B21_l ./ γ_l)), B_21_data_l.dV_Max ./ J_l, color = :orange, markersize = 8)
    lines!(bx12, sqrt.(J_l .* (κ_Δ_l ./ γ_l)), Δ_data_l.dV_Max ./ J_l, color = :cyan, linewidth = 2, label = "Δ"); scatter!(bx12, sqrt.(J_l .* (κ_Δ_l ./ γ_l)), Δ_data_l.dV_Max ./ J_l, color = :cyan, markersize = 8)
    lines!(bx12, sqrt.(J_l .* (κ_T_l ./ γ_l)), T_data_l.dV_Max ./ J_l, color = :magenta, linewidth = 2, label = "T"); scatter!(bx12, sqrt.(J_l .* (κ_T_l ./ γ_l)), T_data_l.dV_Max ./ J_l, color = :magenta, markersize = 8); axislegend(bx12, position = :rb)

    save("Param_scan/Figures/Parameter_Scans_$(sites)_Site_large.png", fig_large)
    display(fig_large)



    # ------------------------------------------------------------------------------
    # 3. DECOUPLED FITS: Full Power Law Exponent Discovery
    # ------------------------------------------------------------------------------
    println("\n" * "="^80)
    println("PERFORMING DECOUPLED FITS (Power Law Exponent Discovery)")
    println("="^80)

    # --- Aggregating Data ---
    # Small Regime
    kg_s = vcat(κ_s ./ γ_data_s.X_Value, κ_M_s ./ γ_s, κ_B21_s ./ γ_s, κ_Δ_s ./ γ_s, κ_T_s ./ γ_s)
    Y_kg_s = vcat(γ_data_s.dV_Max ./ J_s, M_data_s.dV_Max ./ J_s, B_21_data_s.dV_Max ./ J_s, Δ_data_s.dV_Max ./ J_s, T_data_s.dV_Max ./ J_s)

    # Large Regime (Combined κ/γ)
    kg_l = vcat(κ_l ./ γ_data_l.X_Value, κ_M_l ./ γ_l, κ_B21_l ./ γ_l, κ_Δ_l ./ γ_l, κ_T_l ./ γ_l)
    Y_kg_l = vcat(γ_data_l.dV_Max ./ J_l, M_data_l.dV_Max ./ J_l, B_21_data_l.dV_Max ./ J_l, Δ_data_l.dV_Max ./ J_l, T_data_l.dV_Max ./ J_l)

    # Large Regime (Isolated 1/γ)
    inv_gamma_l_x = 1.0 ./ γ_data_l.X_Value
    Y_inv_gamma_l = γ_data_l.dV_Max ./ J_l

    # Large Regime (Isolated κ)
    kappa_l_x = vcat(κ_M_l, κ_B21_l, κ_Δ_l, κ_T_l)
    Y_kappa_l = vcat(M_data_l.dV_Max ./ J_l, B_21_data_l.dV_Max ./ J_l, Δ_data_l.dV_Max ./ J_l, T_data_l.dV_Max ./ J_l)


    # Define a safe Power Law model
    power_model(x, p) = p[1] .* (max.(x, 1e-10) .^ p[2])


    # --- 1. Small Regime Fits ---
    p0_J_s = [1.0, 0.5]
    Y_J_s_scaled = J_data_s.dV_Max ./ J_data_s.X_Value
    fit_J_s = curve_fit(power_model, J_data_s.X_Value, Y_J_s_scaled, p0_J_s)
    A_J_s, b_J_s = fit_J_s.param
    r2_J_s = 1.0 - sum((Y_J_s_scaled .- power_model(J_data_s.X_Value, fit_J_s.param)).^2) / sum((Y_J_s_scaled .- mean(Y_J_s_scaled)).^2)

    p0_kg_s = [1.0, 0.5]
    fit_kg_s = curve_fit(power_model, kg_s, Y_kg_s, p0_kg_s)
    A_kg_s, b_kg_s = fit_kg_s.param
    r2_kg_s = 1.0 - sum((Y_kg_s .- power_model(kg_s, fit_kg_s.param)).^2) / sum((Y_kg_s .- mean(Y_kg_s)).^2)


    # --- 2. Large Regime Fits ---
    p0_J_l = [1.0, 1.0]
    Y_J_l_scaled = J_data_l.dV_Max ./ J_data_l.X_Value
    fit_J_l = curve_fit(power_model, J_data_l.X_Value, Y_J_l_scaled, p0_J_l)
    A_J_l, b_J_l = fit_J_l.param
    r2_J_l = 1.0 - sum((Y_J_l_scaled .- power_model(J_data_l.X_Value, fit_J_l.param)).^2) / sum((Y_J_l_scaled .- mean(Y_J_l_scaled)).^2)

    p0_kg_l = [1.0, 1.0]
    fit_kg_l = curve_fit(power_model, kg_l, Y_kg_l, p0_kg_l)
    A_kg_l, b_kg_l = fit_kg_l.param
    r2_kg_l = 1.0 - sum((Y_kg_l .- power_model(kg_l, fit_kg_l.param)).^2) / sum((Y_kg_l .- mean(Y_kg_l)).^2)

    p0_inv_gamma_l = [1.0, 1.0] 
    fit_inv_gamma_l = curve_fit(power_model, inv_gamma_l_x, Y_inv_gamma_l, p0_inv_gamma_l)
    A_inv_gamma_l, b_inv_gamma_l = fit_inv_gamma_l.param
    r2_inv_gamma_l = 1.0 - sum((Y_inv_gamma_l .- power_model(inv_gamma_l_x, fit_inv_gamma_l.param)).^2) / sum((Y_inv_gamma_l .- mean(Y_inv_gamma_l)).^2)

    p0_kappa_l = [1.0, 1.0] 
    fit_kappa_l = curve_fit(power_model, kappa_l_x, Y_kappa_l, p0_kappa_l)
    A_kappa_l, b_kappa_l = fit_kappa_l.param
    r2_kappa_l = 1.0 - sum((Y_kappa_l .- power_model(kappa_l_x, fit_kappa_l.param)).^2) / sum((Y_kappa_l .- mean(Y_kappa_l)).^2)


    # --- 3. Construct Empirical Combined Variables ---
    # Small Regime Combined Arrays (Forced to √(J*κ/γ))
    J_full_s = vcat(J_data_s.X_Value, fill(J_s, length(Y_kg_s)))
    kg_full_s = vcat(fill(κ_s / γ_s, nrow(J_data_s)), kg_s)
    X_comb_s = sqrt.(J_full_s .* kg_full_s)
    Y_comb_s = vcat(Y_J_s_scaled, Y_kg_s)

    # Large Regime Combined Arrays (Forced to √(J*κ/γ))
    J_full_l = vcat(J_data_l.X_Value, fill(J_l, length(Y_kg_l)))
    kg_full_l = vcat(fill(κ_l / γ_l, nrow(J_data_l)), kg_l)
    X_comb_l = sqrt.(J_full_l .* kg_full_l)
    Y_comb_l = vcat(Y_J_l_scaled, Y_kg_l)

    # Linear fits for the combined empirical models (Y ~ 0 + X_comb)
    fit_comb_s = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_s, Y = Y_comb_s))
    C_comb_s = coef(fit_comb_s)[1]
    r2_comb_s = 1.0 - sum((Y_comb_s .- predict(fit_comb_s)).^2) / sum((Y_comb_s .- mean(Y_comb_s)).^2)

    fit_comb_l = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_l, Y = Y_comb_l))
    C_comb_l = coef(fit_comb_l)[1]
    r2_comb_l = 1.0 - sum((Y_comb_l .- predict(fit_comb_l)).^2) / sum((Y_comb_l .- mean(Y_comb_l)).^2)


    # --- 4. Plotting All Decoupled & Combined Fits ---
    println("Generating Extended Decoupled Fits Plot...")
    fig_dec = Figure(size = (1200, 1600), fontsize = 16)
    Label(fig_dec[0, 1:2], "DECOUPLED FITS & EMPIRICAL COMBINATION FOR $sites SITES", fontsize = 24, font = :bold)

    # Row 1: Small Regime (Isolated J and κ/γ)
    ax_J_s = Axis(fig_dec[1, 1], title = "Small Regime: J-part", xlabel = "J [meV]", ylabel = "ΔV_max / J")
    ax_kg_s = Axis(fig_dec[1, 2], title = "Small Regime: κ/γ-part", xlabel = "κ/γ", ylabel = "ΔV_max / J")

    # Row 2: Large Regime (Isolated J and combined κ/γ)
    ax_J_l = Axis(fig_dec[2, 1], title = "Large Regime: J-part", xlabel = "J [meV]", ylabel = "ΔV_max / J")
    ax_kg_l = Axis(fig_dec[2, 2], title = "Large Regime: Combined κ/γ-part", xlabel = "κ/γ", ylabel = "ΔV_max / J")

    # Row 3: Large Regime (Isolated 1/γ and Isolated κ)
    ax_inv_gamma_l = Axis(fig_dec[3, 1], title = "Large Regime: Isolated 1/γ-part", xlabel = "1/γ [1/meV]", ylabel = "ΔV_max / J")
    ax_kappa_l = Axis(fig_dec[3, 2], title = "Large Regime: Isolated κ-part", xlabel = "κ", ylabel = "ΔV_max / J")

    # Row 4: Empirical Combined Scans
    lbl_xlabel_s = "√(J*κ/γ)"
    lbl_xlabel_l = "√(J*κ/γ)"
    ax_comb_s = Axis(fig_dec[4, 1], title = "Small Regime: Empirical Master Curve", xlabel = lbl_xlabel_s, ylabel = "ΔV_max / J")
    ax_comb_l = Axis(fig_dec[4, 2], title = "Large Regime: Empirical Master Curve", xlabel = lbl_xlabel_l, ylabel = "ΔV_max / J")


    # Plot Scatter Data (Isolated)
    scatter!(ax_J_s, J_data_s.X_Value, Y_J_s_scaled, color = (:blue, 0.4), markersize=8, label = "J Scan Data")
    scatter!(ax_kg_s, kg_s, Y_kg_s, color = (:purple, 0.4), markersize=8, label = "γ, M, B21, Δ, T Data")

    scatter!(ax_J_l, J_data_l.X_Value, Y_J_l_scaled, color = (:blue, 0.4), markersize=8, label = "J Scan Data")
    scatter!(ax_kg_l, kg_l, Y_kg_l, color = (:purple, 0.4), markersize=8, label = "γ, M, B21, Δ, T Data")

    scatter!(ax_inv_gamma_l, inv_gamma_l_x, Y_inv_gamma_l, color = (:red, 0.4), markersize=8, label = "1/γ Data")
    scatter!(ax_kappa_l, kappa_l_x, Y_kappa_l, color = (:orange, 0.4), markersize=8, label = "κ Data")

    # Plot Scatter Data (Combined Master Curves)
    scatter!(ax_comb_s, X_comb_s, Y_comb_s, color = (:green, 0.4), markersize=6, label = "All Scans (Overlapped)")
    scatter!(ax_comb_l, X_comb_l, Y_comb_l, color = (:green, 0.4), markersize=6, label = "All Scans (Overlapped)")


    # Generate smooth evaluation arrays
    j_dense_s = range(0, maximum(J_data_s.X_Value), length=200)
    kg_dense_s = range(0, maximum(kg_s), length=200)
    j_dense_l = range(0, maximum(J_data_l.X_Value), length=200)
    kg_dense_l = range(0, maximum(kg_l), length=200)
    inv_gamma_dense_l = range(0, maximum(inv_gamma_l_x), length=200) 
    kappa_dense_l = range(0, maximum(kappa_l_x), length=200)
    comb_dense_s = range(0, maximum(X_comb_s), length=200)
    comb_dense_l = range(0, maximum(X_comb_l), length=200)

    # Dynamically formulate the legend labels (NOW WITH R²)
    lbl_J_s = "Fit: $(round(A_J_s, sigdigits=4)) * J ^ ($(round(b_J_s, digits=3)))  (R² = $(round(r2_J_s, digits=4)))"
    lbl_kg_s = "Fit: $(round(A_kg_s, sigdigits=4)) * (κ/γ) ^ ($(round(b_kg_s, digits=3)))  (R² = $(round(r2_kg_s, digits=4)))"
    lbl_J_l = "Fit: $(round(A_J_l, sigdigits=4)) * J ^ ($(round(b_J_l, digits=3)))  (R² = $(round(r2_J_l, digits=4)))"
    lbl_kg_l = "Fit: $(round(A_kg_l, sigdigits=4)) * (κ/γ) ^ ($(round(b_kg_l, digits=3)))  (R² = $(round(r2_kg_l, digits=4)))"
    lbl_inv_gamma_l = "Fit: $(round(A_inv_gamma_l, sigdigits=4)) * (1/γ) ^ ($(round(b_inv_gamma_l, digits=3)))  (R² = $(round(r2_inv_gamma_l, digits=4)))"
    lbl_kappa_l = "Fit: $(round(A_kappa_l, sigdigits=4)) * κ ^ ($(round(b_kappa_l, digits=3)))  (R² = $(round(r2_kappa_l, digits=4)))"
    lbl_comb_s = "Linear Fit: slope = $(round(C_comb_s, sigdigits=4))  (R² = $(round(r2_comb_s, digits=4)))"
    lbl_comb_l = "Linear Fit: slope = $(round(C_comb_l, sigdigits=4))  (R² = $(round(r2_comb_l, digits=4)))"

    # Plot Power Fits
    lines!(ax_J_s, j_dense_s, power_model(j_dense_s, fit_J_s.param), color = :black, linewidth=3, label=lbl_J_s)
    lines!(ax_kg_s, kg_dense_s, power_model(kg_dense_s, fit_kg_s.param), color = :black, linewidth=3, label=lbl_kg_s)
    lines!(ax_J_l, j_dense_l, power_model(j_dense_l, fit_J_l.param), color = :black, linewidth=3, label=lbl_J_l)
    lines!(ax_kg_l, kg_dense_l, power_model(kg_dense_l, fit_kg_l.param), color = :black, linewidth=3, label=lbl_kg_l)
    lines!(ax_inv_gamma_l, inv_gamma_dense_l, power_model(inv_gamma_dense_l, fit_inv_gamma_l.param), color = :black, linewidth=3, label=lbl_inv_gamma_l)
    lines!(ax_kappa_l, kappa_dense_l, power_model(kappa_dense_l, fit_kappa_l.param), color = :black, linewidth=3, label=lbl_kappa_l)

    # Plot Empirical Master Curve Fits
    lines!(ax_comb_s, comb_dense_s, C_comb_s .* comb_dense_s, color=:black, linewidth=3, label=lbl_comb_s)
    lines!(ax_comb_l, comb_dense_l, C_comb_l .* comb_dense_l, color=:black, linewidth=3, label=lbl_comb_l)

    axislegend(ax_J_s, position=:lt); axislegend(ax_kg_s, position=:lt)
    axislegend(ax_J_l, position=:lt); axislegend(ax_kg_l, position=:lt)
    axislegend(ax_inv_gamma_l, position=:lt); axislegend(ax_kappa_l, position=:lt)
    axislegend(ax_comb_s, position=:lt); axislegend(ax_comb_l, position=:lt)

    display(fig_dec)
    save("Param_scan/Figures/Decoupled_Fits_$(sites)_Site.png", fig_dec)


    # ------------------------------------------------------------------------------
    # 4. FORMULA SUMMARY TABLE
    # ------------------------------------------------------------------------------
    println("\n===========================================================================================================================")
    println(rpad(" DECOUPLED POWER FITS", 30), rpad("| EXTRACTED PREFACTORS & EXPONENTS", 68), "| R² VALUE")
    println("===========================================================================================================================")

    eqn_J_s = "ΔV_max / J = ($(round(A_J_s, sigdigits=5))) * J ^ ($(round(b_J_s, digits=4)))"
    println(rpad(" Small Regime: J-part", 30), "| ", rpad(eqn_J_s, 65), " | ", round(r2_J_s, digits=6))

    eqn_kg_s = "ΔV_max / J = ($(round(A_kg_s, sigdigits=5))) * (κ/γ) ^ ($(round(b_kg_s, digits=4)))"
    println(rpad(" Small Regime: κ/γ-part", 30), "| ", rpad(eqn_kg_s, 65), " | ", round(r2_kg_s, digits=6))

    println("---------------------------------------------------------------------------------------------------------------------------")

    eqn_J_l = "ΔV_max / J = ($(round(A_J_l, sigdigits=5))) * J ^ ($(round(b_J_l, digits=4)))"
    println(rpad(" Large Regime: J-part", 30), "| ", rpad(eqn_J_l, 65), " | ", round(r2_J_l, digits=6))

    eqn_kg_l = "ΔV_max / J = ($(round(A_kg_l, sigdigits=5))) * (κ/γ) ^ ($(round(b_kg_l, digits=4)))"
    println(rpad(" Large Regime: κ/γ comb.", 30), "| ", rpad(eqn_kg_l, 65), " | ", round(r2_kg_l, digits=6))

    eqn_inv_gamma_l = "ΔV_max / J = ($(round(A_inv_gamma_l, sigdigits=5))) * (1/γ) ^ ($(round(b_inv_gamma_l, digits=4)))"
    println(rpad(" Large Regime: 1/γ-part", 30), "| ", rpad(eqn_inv_gamma_l, 65), " | ", round(r2_inv_gamma_l, digits=6))

    eqn_kappa_l = "ΔV_max / J = ($(round(A_kappa_l, sigdigits=5))) * κ ^ ($(round(b_kappa_l, digits=4)))"
    println(rpad(" Large Regime: κ-part", 30), "| ", rpad(eqn_kappa_l, 65), " | ", round(r2_kappa_l, digits=6))

    println("\n===========================================================================================================================")
    println(rpad(" EMPIRICAL COMBINED FITS", 30), rpad("| MASTER EQUATION", 68), "| R² VALUE")
    println("===========================================================================================================================")

    eqn_comb_s = "ΔV_max / J = ($(round(C_comb_s, sigdigits=5))) * √(J*κ/γ)"
    println(rpad(" Small Regime Master", 30), "| ", rpad(eqn_comb_s, 65), " | ", round(r2_comb_s, digits=6))

    eqn_comb_l = "ΔV_max / J = ($(round(C_comb_l, sigdigits=5))) * √(J*κ/γ)"
    println(rpad(" Large Regime Master", 30), "| ", rpad(eqn_comb_l, 65), " | ", round(r2_comb_l, digits=6))
    println("===========================================================================================================================\n")
end