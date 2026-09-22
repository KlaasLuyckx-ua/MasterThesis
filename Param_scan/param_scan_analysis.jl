# ==============================================================================
# UNIFIED MASTER SCRIPT
# ==============================================================================
using CairoMakie, GLM, LsqFit, DataFrames, Statistics, Printf, JLD2, LaTeXStrings

mkpath("Param_scan/Plots")

# Formatting helper for scientific notation in plots
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

# Fit models
power_model(x, p) = p[1] .* (max.(x, 1e-10) .^ p[2])
power_model_2d(X, p) = p[1] .* (max.(X[:, 1], 1e-10) .^ p[2]) .* (max.(X[:, 2], 1e-10) .^ p[3])

# Base parameters for thermalization regimes
J_s, γ_s, M_s, B_21_s, Δ_s, T_s = 1.0e-2, 2.0e-4, 1.0e7, 1.0e-8, -100.0, 25.0
J_l, γ_l, M_l, B_21_l, Δ_l, T_l = 1.0e-1, 2.0e-3, 1.0e8, 1.0e-7, -60.0, 25.0

κ_s = 0.5 * B_21_s * M_s * exp(Δ_s/T_s) / T_s
κ_l = 0.5 * B_21_l * M_l * exp(Δ_l/T_l) / T_l

for sites in 2
    mkpath("Param_scan/Plots/$(sites)_Site")
    file_s = "Param_scan/Data/Master_Results_$(sites)_Site_s.jld2"
    file_l = "Param_scan/Data/Master_Results_$(sites)_Site_l.jld2"
    
    if !isfile(file_s) || !isfile(file_l)
        println("Data for $sites sites not found. Skipping...")
        continue
    end

    println("\n" * "="^100)
    println("PROCESSING DATA FOR N = $sites SITES")
    println("="^100)

    df_small = load(file_s)["df"]
    df_large = load(file_l)["df"]

    # --- 1. Filter Data & Calculate Kappas ---
    J_data_s     = filter(row -> row.ScanType == "J", df_small)
    γ_data_s     = filter(row -> row.ScanType == "γ", df_small)
    n_avg_data_s = filter(row -> row.ScanType == "n_avg", df_small)
    M_data_s     = filter(row -> row.ScanType == "M", df_small)
    B_21_data_s  = filter(row -> row.ScanType == "B_21", df_small)
    Δ_data_s     = filter(row -> row.ScanType == "Δ", df_small)
    T_data_s     = filter(row -> row.ScanType == "T", df_small)

    κ_M_s   = 0.5 .* B_21_s .* M_data_s.X_Value .* exp(Δ_s/T_s) ./ T_s
    κ_B21_s = 0.5 .* B_21_data_s.X_Value .* M_s .* exp(Δ_s/T_s) ./ T_s
    κ_Δ_s   = 0.5 .* B_21_s .* M_s .* exp.(Δ_data_s.X_Value ./ T_s) ./ T_s
    κ_T_s   = 0.5 .* B_21_s .* M_s .* exp.(Δ_s ./ T_data_s.X_Value) ./ T_data_s.X_Value

    J_data_l     = filter(row -> row.ScanType == "J", df_large)
    γ_data_l     = filter(row -> row.ScanType == "γ", df_large)
    n_avg_data_l = filter(row -> row.ScanType == "n_avg", df_large)
    M_data_l     = filter(row -> row.ScanType == "M", df_large)
    B_21_data_l  = filter(row -> row.ScanType == "B_21", df_large)
    Δ_data_l     = filter(row -> row.ScanType == "Δ", df_large)
    T_data_l     = filter(row -> row.ScanType == "T", df_large)

    κ_M_l   = 0.5 .* B_21_l .* M_data_l.X_Value .* exp(Δ_l/T_l) ./ T_l
    κ_B21_l = 0.5 .* B_21_data_l.X_Value .* M_l .* exp(Δ_l/T_l) ./ T_l
    κ_Δ_l   = 0.5 .* B_21_l .* M_l .* exp.(Δ_data_l.X_Value ./ T_l) ./ T_l
    κ_T_l   = 0.5 .* B_21_l .* M_l .* exp.(Δ_l ./ T_data_l.X_Value) ./ T_data_l.X_Value

    # --- 2. Generate 12-Panel Plots ---
    for (regime_name, J_val, γ_val, J_data, γ_data, n_avg_data, M_data, B_21_data, Δ_data, T_data, κ_M, κ_B21, κ_Δ, κ_T, κ_val, Δ_val, T_val, suffix) in [
        ("SMALL", J_s, γ_s, J_data_s, γ_data_s, n_avg_data_s, M_data_s, B_21_data_s, Δ_data_s, T_data_s, κ_M_s, κ_B21_s, κ_Δ_s, κ_T_s, κ_s, Δ_s, T_s, "s"),
        ("LARGE", J_l, γ_l, J_data_l, γ_data_l, n_avg_data_l, M_data_l, B_21_data_l, Δ_data_l, T_data_l, κ_M_l, κ_B21_l, κ_Δ_l, κ_T_l, κ_l, Δ_l, T_l, "l")
    ]
        println("Generating 12-Panel Plot for $regime_name Regime...")
        fig = Figure(size = (1200, 850), fontsize = 16)
        regime_nl = regime_name == "SMALL" ? "Weak" : "Strong"
        Label(fig[0, 1:4], "$regime_nl Thermalization ($sites Sites)", fontsize = 35, font = :bold)
        
        ax1 = Axis(fig[1, 1], xlabel = "J [meV]", ylabel = "ΔV_max / J")
        ax2 = Axis(fig[1, 2], xlabel = "γ [meV]", xticks = LinearTicks(3), xtickformat = format_sci)
        ax3 = Axis(fig[1, 3], xlabel = "1/γ [1/meV]")
        ax4 = Axis(fig[1, 4], xlabel = "n_avg", yticks = LinearTicks(4), ytickformat = format_sci)
        ax5 = Axis(fig[2, 1], xlabel = "M", ylabel = "ΔV_max / J", xticks = LinearTicks(3))
        ax6 = Axis(fig[2, 2], xlabel = "B_21")
        ax7 = Axis(fig[2, 3], xlabel = "Δ")
        ax8 = Axis(fig[2, 4], xlabel = "T")
        ax9 = Axis(fig[3, 1], xlabel = "exp(Δ/T)/T", ylabel = "ΔV_max / J", xticks = LinearTicks(3), xtickformat = format_sci)
        ax10 = Axis(fig[3, 2], xlabel = "κ", xticks = LinearTicks(3), xtickformat = format_sci)
        ax11 = Axis(fig[3, 3], xlabel = "κ/γ")
        ax12 = Axis(fig[3, 4], xlabel = "J*κ/γ", xticks = LinearTicks(3), xtickformat = format_sci)

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12] 
            ax.yticks = LinearTicks(4); ax.xlabelsize=25; ax.ylabelsize=25; ax.xticklabelsize=20; ax.yticklabelsize=20 
        end

        # Raw parameter dependencies
        lines!(ax1, J_data.X_Value, J_data.dV_Max ./ J_data.X_Value, color=:blue, linewidth=2); scatter!(ax1, J_data.X_Value, J_data.dV_Max ./ J_data.X_Value, color=:blue, markersize=8)
        lines!(ax2, γ_data.X_Value, γ_data.dV_Max ./ J_val, color=:red, linewidth=2); scatter!(ax2, γ_data.X_Value, γ_data.dV_Max ./ J_val, color=:red, markersize=8)
        lines!(ax3, 1 ./ γ_data.X_Value, γ_data.dV_Max ./ J_val, color=:red, linewidth=2); scatter!(ax3, 1 ./ γ_data.X_Value, γ_data.dV_Max ./ J_val, color=:red, markersize=8)
        lines!(ax4, n_avg_data.X_Value, n_avg_data.dV_Max ./ J_val, color=:green, linewidth=2); scatter!(ax4, n_avg_data.X_Value, n_avg_data.dV_Max ./ J_val, color=:green, markersize=8)
        lines!(ax5, M_data.X_Value, M_data.dV_Max ./ J_val, color=:purple, linewidth=2); scatter!(ax5, M_data.X_Value, M_data.dV_Max ./ J_val, color=:purple, markersize=8)
        lines!(ax6, B_21_data.X_Value, B_21_data.dV_Max ./ J_val, color=:orange, linewidth=2); scatter!(ax6, B_21_data.X_Value, B_21_data.dV_Max ./ J_val, color=:orange, markersize=8)
        lines!(ax7, Δ_data.X_Value, Δ_data.dV_Max ./ J_val, color=:cyan, linewidth=2); scatter!(ax7, Δ_data.X_Value, Δ_data.dV_Max ./ J_val, color=:cyan, markersize=8)
        lines!(ax8, T_data.X_Value, T_data.dV_Max ./ J_val, color=:magenta, linewidth=2); scatter!(ax8, T_data.X_Value, T_data.dV_Max ./ J_val, color=:magenta, markersize=8)

        # Derived groupings
        lines!(ax9, exp.(Δ_data.X_Value ./ T_val) ./ T_val, Δ_data.dV_Max ./ J_val, color=:cyan, linewidth=2); scatter!(ax9, exp.(Δ_data.X_Value ./ T_val) ./ T_val, Δ_data.dV_Max ./ J_val, color=:cyan, markersize=8)
        lines!(ax9, exp.(Δ_val ./ T_data.X_Value) ./ T_data.X_Value, T_data.dV_Max ./ J_val, color=:magenta, linewidth=2); scatter!(ax9, exp.(Δ_val ./ T_data.X_Value) ./ T_data.X_Value, T_data.dV_Max ./ J_val, color=:magenta, markersize=8)

        lines!(ax10, κ_M, M_data.dV_Max ./ J_val, color=:purple, linewidth=2, label="M"); scatter!(ax10, κ_M, M_data.dV_Max ./ J_val, color=:purple, markersize=8)
        lines!(ax10, κ_B21, B_21_data.dV_Max ./ J_val, color=:orange, linewidth=2, label="B_21"); scatter!(ax10, κ_B21, B_21_data.dV_Max ./ J_val, color=:orange, markersize=8)
        lines!(ax10, κ_Δ, Δ_data.dV_Max ./ J_val, color=:cyan, linewidth=2, label="Δ"); scatter!(ax10, κ_Δ, Δ_data.dV_Max ./ J_val, color=:cyan, markersize=8)
        lines!(ax10, κ_T, T_data.dV_Max ./ J_val, color=:magenta, linewidth=2, label="T"); scatter!(ax10, κ_T, T_data.dV_Max ./ J_val, color=:magenta, markersize=8); axislegend(ax10, position=:rb)

        lines!(ax11, κ_val ./ γ_data.X_Value, γ_data.dV_Max ./ J_val, color=:red, linewidth=2, label="γ"); scatter!(ax11, κ_val ./ γ_data.X_Value, γ_data.dV_Max ./ J_val, color=:red, markersize=8)
        lines!(ax11, κ_M ./ γ_val, M_data.dV_Max ./ J_val, color=:purple, linewidth=2, label="M"); scatter!(ax11, κ_M ./ γ_val, M_data.dV_Max ./ J_val, color=:purple, markersize=8)
        lines!(ax11, κ_B21 ./ γ_val, B_21_data.dV_Max ./ J_val, color=:orange, linewidth=2, label="B_21"); scatter!(ax11, κ_B21 ./ γ_val, B_21_data.dV_Max ./ J_val, color=:orange, markersize=8)
        lines!(ax11, κ_Δ ./ γ_val, Δ_data.dV_Max ./ J_val, color=:cyan, linewidth=2, label="Δ"); scatter!(ax11, κ_Δ ./ γ_val, Δ_data.dV_Max ./ J_val, color=:cyan, markersize=8)
        lines!(ax11, κ_T ./ γ_val, T_data.dV_Max ./ J_val, color=:magenta, linewidth=2, label="T"); scatter!(ax11, κ_T ./ γ_val, T_data.dV_Max ./ J_val, color=:magenta, markersize=8); axislegend(ax11, position=:rb)

        # Scaled specific arrays
        X12_J   = J_data.X_Value .* (κ_val ./ γ_val)
        X12_g   = J_val .* (κ_val ./ γ_data.X_Value)
        X12_M   = J_val .* (κ_M ./ γ_val)
        X12_B21 = J_val .* (κ_B21 ./ γ_val)
        X12_Δ   = J_val .* (κ_Δ ./ γ_val)
        X12_T   = J_val .* (κ_T ./ γ_val)

        lines!(ax12, X12_J, J_data.dV_Max ./ J_data.X_Value, color=:blue, linewidth=2, label="J"); scatter!(ax12, X12_J, J_data.dV_Max ./ J_data.X_Value, color=:blue, markersize=8)
        lines!(ax12, X12_g, γ_data.dV_Max ./ J_val, color=:red, linewidth=2, label="γ"); scatter!(ax12, X12_g, γ_data.dV_Max ./ J_val, color=:red, markersize=8)
        lines!(ax12, X12_M, M_data.dV_Max ./ J_val, color=:purple, linewidth=2, label="M"); scatter!(ax12, X12_M, M_data.dV_Max ./ J_val, color=:purple, markersize=8)
        lines!(ax12, X12_B21, B_21_data.dV_Max ./ J_val, color=:orange, linewidth=2, label="B_21"); scatter!(ax12, X12_B21, B_21_data.dV_Max ./ J_val, color=:orange, markersize=8)
        lines!(ax12, X12_Δ, Δ_data.dV_Max ./ J_val, color=:cyan, linewidth=2, label="Δ"); scatter!(ax12, X12_Δ, Δ_data.dV_Max ./ J_val, color=:cyan, markersize=8)
        lines!(ax12, X12_T, T_data.dV_Max ./ J_val, color=:magenta, linewidth=2, label="T"); scatter!(ax12, X12_T, T_data.dV_Max ./ J_val, color=:magenta, markersize=8); axislegend(ax12, position=:rb)

        display(fig)
        save("Param_scan/Plots/$(sites)_Site/Parameter_Scans_$(suffix)_$(sites)_Site.png", fig)
    end

    # --- 3. Decoupled Fits & Empirical Combination ---
    println("Performing Decoupled & Empirical Fits...")

    kg_s = vcat(κ_s ./ γ_data_s.X_Value, κ_M_s ./ γ_s, κ_B21_s ./ γ_s, κ_Δ_s ./ γ_s, κ_T_s ./ γ_s)
    Y_kg_s = vcat(γ_data_s.dV_Max ./ J_s, M_data_s.dV_Max ./ J_s, B_21_data_s.dV_Max ./ J_s, Δ_data_s.dV_Max ./ J_s, T_data_s.dV_Max ./ J_s)

    kg_l = vcat(κ_l ./ γ_data_l.X_Value, κ_M_l ./ γ_l, κ_B21_l ./ γ_l, κ_Δ_l ./ γ_l, κ_T_l ./ γ_l)
    Y_kg_l = vcat(γ_data_l.dV_Max ./ J_l, M_data_l.dV_Max ./ J_l, B_21_data_l.dV_Max ./ J_l, Δ_data_l.dV_Max ./ J_l, T_data_l.dV_Max ./ J_l)

    Y_J_s_scaled = J_data_s.dV_Max ./ J_data_s.X_Value
    Y_J_l_scaled = J_data_l.dV_Max ./ J_data_l.X_Value

    # Non-linear Power Fits (Decoupled)
    fit_J_s = curve_fit(power_model, J_data_s.X_Value, Y_J_s_scaled, [1.0, 0.5])
    r2_J_s = 1.0 - sum((Y_J_s_scaled .- power_model(J_data_s.X_Value, fit_J_s.param)).^2) / sum((Y_J_s_scaled .- mean(Y_J_s_scaled)).^2)

    fit_kg_s = curve_fit(power_model, kg_s, Y_kg_s, [1.0, 0.5])
    r2_kg_s = 1.0 - sum((Y_kg_s .- power_model(kg_s, fit_kg_s.param)).^2) / sum((Y_kg_s .- mean(Y_kg_s)).^2)

    fit_J_l = curve_fit(power_model, J_data_l.X_Value, Y_J_l_scaled, [1.0, 1.0])
    r2_J_l = 1.0 - sum((Y_J_l_scaled .- power_model(J_data_l.X_Value, fit_J_l.param)).^2) / sum((Y_J_l_scaled .- mean(Y_J_l_scaled)).^2)

    fit_kg_l = curve_fit(power_model, kg_l, Y_kg_l, [1.0, 1.0])
    r2_kg_l = 1.0 - sum((Y_kg_l .- power_model(kg_l, fit_kg_l.param)).^2) / sum((Y_kg_l .- mean(Y_kg_l)).^2)

    # Combined Variables
    Y_comb_s = vcat(Y_J_s_scaled, Y_kg_s)
    J_comb_s = vcat(J_data_s.X_Value, fill(J_s, length(Y_kg_s)))
    kg_comb_s = vcat(fill(κ_s / γ_s, nrow(J_data_s)), kg_s)
    X_raw_comb_s = J_comb_s .* kg_comb_s
    X_2d_s = hcat(J_comb_s, kg_comb_s)
    
    Y_comb_l = vcat(Y_J_l_scaled, Y_kg_l)
    J_comb_l = vcat(J_data_l.X_Value, fill(J_l, length(Y_kg_l)))
    kg_comb_l = vcat(fill(κ_l / γ_l, nrow(J_data_l)), kg_l)
    X_raw_comb_l = J_comb_l .* kg_comb_l
    X_2d_l = hcat(J_comb_l, kg_comb_l)

    # Independent 2D Power Fits
    fit_2d_s = curve_fit(power_model_2d, X_2d_s, Y_comb_s, [1.0, 0.5, 0.5])
    r2_2d_s = 1.0 - sum((Y_comb_s .- power_model_2d(X_2d_s, fit_2d_s.param)).^2) / sum((Y_comb_s .- mean(Y_comb_s)).^2)

    fit_2d_l = curve_fit(power_model_2d, X_2d_l, Y_comb_l, [1.0, 1.0, 1.0])
    r2_2d_l = 1.0 - sum((Y_comb_l .- power_model_2d(X_2d_l, fit_2d_l.param)).^2) / sum((Y_comb_l .- mean(Y_comb_l)).^2)

    # Combined 1D Power Fits
    fit_raw_comb_s = curve_fit(power_model, X_raw_comb_s, Y_comb_s, [1.0, 0.5])
    r2_raw_comb_s = 1.0 - sum((Y_comb_s .- power_model(X_raw_comb_s, fit_raw_comb_s.param)).^2) / sum((Y_comb_s .- mean(Y_comb_s)).^2)

    fit_raw_comb_l = curve_fit(power_model, X_raw_comb_l, Y_comb_l, [1.0, 1.0])
    r2_raw_comb_l = 1.0 - sum((Y_comb_l .- power_model(X_raw_comb_l, fit_raw_comb_l.param)).^2) / sum((Y_comb_l .- mean(Y_comb_l)).^2)

    # Empirical Combination Variables (Square root vs Linear)
    X_comb_s = sqrt.(X_raw_comb_s)
    X_comb_l = sites == 2 ? X_raw_comb_l : sqrt.(X_raw_comb_l)

    # Linear Fits for Master Arrays
    fit_comb_s = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_s, Y = Y_comb_s))
    C_comb_s = coef(fit_comb_s)[1]
    r2_comb_s = 1.0 - sum((Y_comb_s .- predict(fit_comb_s)).^2) / sum((Y_comb_s .- mean(Y_comb_s)).^2)

    fit_comb_l = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_l, Y = Y_comb_l))
    C_comb_l = coef(fit_comb_l)[1]
    r2_comb_l = 1.0 - sum((Y_comb_l .- predict(fit_comb_l)).^2) / sum((Y_comb_l .- mean(Y_comb_l)).^2)

    # --- 4. Plot Decoupled & Combined Fits ---
    fig_dec = Figure(size = (1200, 1600), fontsize = 16)
    Label(fig_dec[0, 1:2], "Fits for $sites Sites", fontsize = 35, font = :bold)

    ax_J_s = Axis(fig_dec[1, 1], title = "Weak Thermalization: J-part", xlabel = "J [meV]", ylabel = "ΔV_max / J")
    ax_kg_s = Axis(fig_dec[1, 2], title = "Weak Thermalization: κ/γ-part", xlabel = "κ/γ", ylabel = "ΔV_max / J")
    ax_J_l = Axis(fig_dec[2, 1], title = "Strong Thermalization: J-part", xlabel = "J [meV]", ylabel = "ΔV_max / J")
    ax_kg_l = Axis(fig_dec[2, 2], title = "Strong Thermalization: Combined κ/γ-part", xlabel = "κ/γ", ylabel = "ΔV_max / J")
    ax_comb_s = Axis(fig_dec[3, 1], title = "Weak Thermalization Master Curve", xlabel = "J*κ/γ", ylabel = "ΔV_max / J")
    ax_comb_l = Axis(fig_dec[3, 2], title = "Strong Thermalization Master Curve", xlabel = "J*κ/γ", ylabel = "ΔV_max / J")

    for ax in [ax_J_s, ax_kg_s, ax_J_l, ax_kg_l, ax_comb_s, ax_comb_l]
        ax.titlesize=25; ax.xlabelsize=20; ax.ylabelsize=20; ax.xticklabelsize=20; ax.yticklabelsize=20
    end

    scatter!(ax_J_s, J_data_s.X_Value, Y_J_s_scaled, color=(:blue, 0.4), markersize=8)
    scatter!(ax_kg_s, kg_s, Y_kg_s, color=(:purple, 0.4), markersize=8)
    scatter!(ax_J_l, J_data_l.X_Value, Y_J_l_scaled, color=(:blue, 0.4), markersize=8)
    scatter!(ax_kg_l, kg_l, Y_kg_l, color=(:purple, 0.4), markersize=8)
    scatter!(ax_comb_s, X_raw_comb_s, Y_comb_s, color=(:green, 0.4), markersize=8)
    scatter!(ax_comb_l, X_raw_comb_l, Y_comb_l, color=(:green, 0.4), markersize=8)

    # Trendlines
    j_dense_s = range(0, maximum(J_data_s.X_Value), length=200)
    kg_dense_s = range(0, maximum(kg_s), length=200)
    j_dense_l = range(0, maximum(J_data_l.X_Value), length=200)
    kg_dense_l = range(0, maximum(kg_l), length=200)
    raw_dense_s_plot = range(0, maximum(X_raw_comb_s), length=200)
    raw_dense_l_plot = range(0, maximum(X_raw_comb_l), length=200)

    fit_y_l = sites == 2 ? C_comb_l .* raw_dense_l_plot : C_comb_l .* sqrt.(raw_dense_l_plot)

    lines!(ax_J_s, j_dense_s, power_model(j_dense_s, fit_J_s.param), color=:black, linewidth=3, label="β = $(round(fit_J_s.param[2], digits=3)) \n R² = $(round(r2_J_s, digits=3))")
    lines!(ax_kg_s, kg_dense_s, power_model(kg_dense_s, fit_kg_s.param), color=:black, linewidth=3, label="β = $(round(fit_kg_s.param[2], digits=3)) \n R² = $(round(r2_kg_s, digits=3))")
    lines!(ax_J_l, j_dense_l, power_model(j_dense_l, fit_J_l.param), color=:black, linewidth=3, label="β = $(round(fit_J_l.param[2], digits=3)) \n R² = $(round(r2_J_l, digits=3))")
    lines!(ax_kg_l, kg_dense_l, power_model(kg_dense_l, fit_kg_l.param), color=:black, linewidth=3, label="β = $(round(fit_kg_l.param[2], digits=3)) \n R² = $(round(r2_kg_l, digits=3))")
    lines!(ax_comb_s, raw_dense_s_plot, C_comb_s .* sqrt.(raw_dense_s_plot), color=:black, linewidth=3, label="Prefactor = $(round(C_comb_s, digits=3)) \n R² = $(round(r2_comb_s, digits=3))")
    lines!(ax_comb_l, raw_dense_l_plot, fit_y_l, color=:black, linewidth=3, label="Prefactor = $(round(C_comb_l, digits=3)) \n R² = $(round(r2_comb_l, digits=3))")

    for ax in [ax_J_s, ax_kg_s, ax_J_l, ax_kg_l, ax_comb_s, ax_comb_l] axislegend(ax, position=:lt, labelsize=20) end
    
    display(fig_dec)
    save("Param_scan/Plots/$(sites)_Site/Decoupled_Fits_$(sites)_Site.png", fig_dec)

    # --- 5. Isolated Master Curve Plot ---
    println("Generating Isolated Master Curve Plot...")
    fig_master = Figure(size = (800, 300), fontsize = 16)
    
    ax_master_s = Axis(fig_master[1, 1], title = LaTeXString("Zwakke thermalisatie"), xlabel = LaTeXString("Jκ/γ"), ylabel = LaTeXString("\\Delta V_{max} / J"), titlesize=25, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)
    ax_master_l = Axis(fig_master[1, 2], title = LaTeXString("Sterke thermalisatie"), xlabel = LaTeXString("Jκ/γ"), ylabel = LaTeXString("\\Delta V_{max} / J"), titlesize=25, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)
    
    scatter!(ax_master_s, X_raw_comb_s, Y_comb_s, color = (:cyan, 0.4), markersize = 10, label = "Data")
    scatter!(ax_master_l, X_raw_comb_l, Y_comb_l, color = (:orange, 0.4), markersize = 10, label = "Data")
    
    lines!(ax_master_s, raw_dense_s_plot, C_comb_s .* sqrt.(raw_dense_s_plot), color = :black, linewidth = 2, label = "Fit")
    lines!(ax_master_l, raw_dense_l_plot, fit_y_l, color = :black, linewidth = 2, label = "Fit")
    
    val_small = @sprintf("%.2f", C_comb_s)
    text_small = rich(
        "ΔV", subscript("max"), " / J = ", 
        rich(val_small, color=:blue), 
        " ⋅ √(Jκ/γ)"
    )
    
    val_large = @sprintf("%.2f", C_comb_l)
    text_large = sites == 2 ? rich(
        "ΔV", subscript("max"), " / J = ", 
        rich(val_large, color=:red), 
        " ⋅ Jκ/γ"
    ) : rich(
        "ΔV", subscript("max"), " / J = ", 
        rich(val_large, color=:red), 
        " ⋅ √(Jκ/γ)"
    )

    text!(ax_master_s, 0.95, 0.05, text = text_small, align = (:right, :bottom), space = :relative, fontsize = 18, font = :bold)
    text!(ax_master_l, 0.95, 0.05, text = text_large, align = (:right, :bottom), space = :relative, fontsize = 18, font = :bold)

    save("Param_scan/Plots/$(sites)_Site/Master_Curve_$(sites)_Site.png", fig_master)
    display(fig_master)
    
    # --- 6. Isolated Combined Power Master Curve ---
    println("Generating Isolated Combined Power Master Curve Plot...")
    fig_power_master = Figure(size = (1200, 500), fontsize = 16)

    ax_pwr_master_s = Axis(fig_power_master[1, 1], title = "Weak Thermalization Power Master Curve", xlabel = "J*κ/γ", ylabel = "ΔV_max / J", titlesize=25, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)
    ax_pwr_master_l = Axis(fig_power_master[1, 2], title = "Strong Thermalization Power Master Curve", xlabel = "J*κ/γ", ylabel = "ΔV_max / J", titlesize=25, xlabelsize=20, ylabelsize=20, xticklabelsize=20, yticklabelsize=20)

    scatter!(ax_pwr_master_s, X_raw_comb_s, Y_comb_s, color = (:blue, 0.6), markersize = 8, label = "Data")
    scatter!(ax_pwr_master_l, X_raw_comb_l, Y_comb_l, color = (:red, 0.6), markersize = 8, label = "Data")

    raw_dense_s = range(0, maximum(X_raw_comb_s), length=500)
    raw_dense_l = range(0, maximum(X_raw_comb_l), length=500)

    lines!(ax_pwr_master_s, raw_dense_s, power_model(raw_dense_s, fit_raw_comb_s.param), color = :black, linewidth = 2, label = "Fit")
    lines!(ax_pwr_master_l, raw_dense_l, power_model(raw_dense_l, fit_raw_comb_l.param), color = :black, linewidth = 2, label = "Fit")

    text_pwr_small = @sprintf("ΔV_max / J = %.4f * (J*κ/γ)^%.4f\nR² = %.4f", fit_raw_comb_s.param[1], fit_raw_comb_s.param[2], r2_raw_comb_s)
    text_pwr_large = @sprintf("ΔV_max / J = %.4f * (J*κ/γ)^%.4f\nR² = %.4f", fit_raw_comb_l.param[1], fit_raw_comb_l.param[2], r2_raw_comb_l)

    text!(ax_pwr_master_s, text_pwr_small, position = (0.95, 0.05), align = (:right, :bottom), space = :relative, fontsize = 18, font = :bold)
    text!(ax_pwr_master_l, text_pwr_large, position = (0.95, 0.05), align = (:right, :bottom), space = :relative, fontsize = 18, font = :bold)

    save("Param_scan/Plots/$(sites)_Site/Power_Master_Curve_$(sites)_Site.png", fig_power_master)
    display(fig_power_master)
    
    # --- 7. Formula Summary Terminal Table ---
    println("\n===========================================================================================================================")
    println(rpad(" DECOUPLED POWER FITS", 30), rpad("| EXTRACTED PREFACTORS & EXPONENTS", 68), "| R² VALUE")
    println("===========================================================================================================================")

    println(rpad(" Weak Thermalization: J-part", 30), "| ", rpad("ΔV_max / J = ($(round(fit_J_s.param[1], sigdigits=5))) * J ^ ($(round(fit_J_s.param[2], digits=4)))", 65), " | ", round(r2_J_s, digits=6))
    println(rpad(" Weak Thermalization: κ/γ-part", 30), "| ", rpad("ΔV_max / J = ($(round(fit_kg_s.param[1], sigdigits=5))) * (κ/γ) ^ ($(round(fit_kg_s.param[2], digits=4)))", 65), " | ", round(r2_kg_s, digits=6))
    println("---------------------------------------------------------------------------------------------------------------------------")
    println(rpad(" Strong Thermalization: J-part", 30), "| ", rpad("ΔV_max / J = ($(round(fit_J_l.param[1], sigdigits=5))) * J ^ ($(round(fit_J_l.param[2], digits=4)))", 65), " | ", round(r2_J_l, digits=6))
    println(rpad(" Strong Thermalization: κ/γ comb.", 30), "| ", rpad("ΔV_max / J = ($(round(fit_kg_l.param[1], sigdigits=5))) * (κ/γ) ^ ($(round(fit_kg_l.param[2], digits=4)))", 65), " | ", round(r2_kg_l, digits=6))

    println("\n===========================================================================================================================")
    println(rpad(" COMBINED POWER FITS", 30), rpad("| MASTER POWER EQUATION", 68), "| R² VALUE")
    println("===========================================================================================================================")

    println(rpad(" Weak Thermalization Indep Power", 30), "| ", rpad("ΔV_max / J = ($(round(fit_2d_s.param[1], sigdigits=5))) * J ^ ($(round(fit_2d_s.param[2], digits=4))) * (κ/γ) ^ ($(round(fit_2d_s.param[3], digits=4)))", 65), " | ", round(r2_2d_s, digits=6))
    println(rpad(" Weak Thermalization Comb Power", 30), "| ", rpad("ΔV_max / J = ($(round(fit_raw_comb_s.param[1], sigdigits=5))) * (J*κ/γ) ^ ($(round(fit_raw_comb_s.param[2], digits=4)))", 65), " | ", round(r2_raw_comb_s, digits=6))
    println("---------------------------------------------------------------------------------------------------------------------------")
    println(rpad(" Strong Thermalization Indep Power", 30), "| ", rpad("ΔV_max / J = ($(round(fit_2d_l.param[1], sigdigits=5))) * J ^ ($(round(fit_2d_l.param[2], digits=4))) * (κ/γ) ^ ($(round(fit_2d_l.param[3], digits=4)))", 65), " | ", round(r2_2d_l, digits=6))
    println(rpad(" Strong Thermalization Comb Power", 30), "| ", rpad("ΔV_max / J = ($(round(fit_raw_comb_l.param[1], sigdigits=5))) * (J*κ/γ) ^ ($(round(fit_raw_comb_l.param[2], digits=4)))", 65), " | ", round(r2_raw_comb_l, digits=6))

    println("\n===========================================================================================================================")
    println(rpad(" EMPIRICAL COMBINED FITS", 30), rpad("| MASTER EQUATION", 68), "| R² VALUE")
    println("===========================================================================================================================")

    println(rpad(" Weak Thermalization Master", 30), "| ", rpad("ΔV_max / J = ($(round(C_comb_s, sigdigits=5))) * √(J*κ/γ)", 65), " | ", round(r2_comb_s, digits=6))
    eqn_comb_l = sites == 2 ? "ΔV_max / J = ($(round(C_comb_l, sigdigits=5))) * (J*κ/γ)" : "ΔV_max / J = ($(round(C_comb_l, sigdigits=5))) * √(J*κ/γ)"
    println(rpad(" Strong Thermalization Master", 30), "| ", rpad(eqn_comb_l, 65), " | ", round(r2_comb_l, digits=6))
    println("===========================================================================================================================\n")
end