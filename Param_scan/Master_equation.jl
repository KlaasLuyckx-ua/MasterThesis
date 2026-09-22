# ==============================================================================
# MULTI-SITE MASTER FIT DISCOVERY (All Sites)
# ==============================================================================
using GLM, LsqFit, DataFrames, JLD2, Statistics, CairoMakie, Printf, LaTeXStrings

mkpath("Param_scan/Plots")

function calculate_master_fits()
    power_model(x, p) = p[1] .* (max.(x, 1e-10) .^ p[2])
    results = []

    println("Processing datasets for Master Equation Discovery...\n")

    for sites in 4:16
        file_s = "Param_scan/Data/Master_Results_$(sites)_Site_s.jld2"
        file_l = "Param_scan/Data/Master_Results_$(sites)_Site_l.jld2"
        
        if !isfile(file_s) || !isfile(file_l)
            continue
        end

        # --- 1. Small Regime Processing ---
        df_small = load(file_s)["df"]
        
        J_s, γ_s, M_s, B_21_s, Δ_s, T_s = 1.0e-2, 2.0e-4, 1.0e7, 1.0e-8, -100.0, 25.0
        κ_s = 0.5 * B_21_s * M_s * exp(Δ_s/T_s) / T_s
        
        J_data_s = filter(row -> row.ScanType == "J", df_small)
        γ_data_s = filter(row -> row.ScanType == "γ", df_small)
        M_data_s = filter(row -> row.ScanType == "M", df_small)
        B_21_data_s = filter(row -> row.ScanType == "B_21", df_small)
        Δ_data_s = filter(row -> row.ScanType == "Δ", df_small)
        T_data_s = filter(row -> row.ScanType == "T", df_small)
        
        κ_M_s   = 0.5 .* B_21_s .* M_data_s.X_Value .* exp(Δ_s/T_s) ./ T_s
        κ_B21_s = 0.5 .* B_21_data_s.X_Value .* M_s .* exp(Δ_s/T_s) ./ T_s
        κ_Δ_s   = 0.5 .* B_21_s .* M_s .* exp.(Δ_data_s.X_Value ./ T_s) ./ T_s
        κ_T_s   = 0.5 .* B_21_s .* M_s .* exp.(Δ_s ./ T_data_s.X_Value) ./ T_data_s.X_Value
        
        kg_s = vcat(κ_s ./ γ_data_s.X_Value, κ_M_s ./ γ_s, κ_B21_s ./ γ_s, κ_Δ_s ./ γ_s, κ_T_s ./ γ_s)
        Y_J_s_scaled = J_data_s.dV_Max ./ J_data_s.X_Value
        Y_kg_s = vcat(γ_data_s.dV_Max ./ J_s, M_data_s.dV_Max ./ J_s, B_21_data_s.dV_Max ./ J_s, Δ_data_s.dV_Max ./ J_s, T_data_s.dV_Max ./ J_s)

        J_full_s = vcat(J_data_s.X_Value, fill(J_s, length(Y_kg_s)))
        kg_full_s = vcat(fill(κ_s / γ_s, nrow(J_data_s)), kg_s)
        X_base_s = J_full_s .* kg_full_s
        Y_comb_s = vcat(Y_J_s_scaled, Y_kg_s)

        fit_comb_power_s = curve_fit(power_model, X_base_s, Y_comb_s, [1.0, 0.5])
        b_comb_s = fit_comb_power_s.param[2]

        X_comb_s = sqrt.(X_base_s)
        fit_comb_s = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_s, Y = Y_comb_s))
        C_comb_s = coef(fit_comb_s)[1]
        r2_comb_s = 1.0 - sum((Y_comb_s .- predict(fit_comb_s)).^2) / sum((Y_comb_s .- mean(Y_comb_s)).^2)

        # --- 2. Large Regime Processing ---
        df_large = load(file_l)["df"]
        
        J_l, γ_l, M_l, B_21_l, Δ_l, T_l = 1.0e-1, 2.0e-3, 1.0e8, 1.0e-7, -60.0, 25.0
        κ_l = 0.5 * B_21_l * M_l * exp(Δ_l/T_l) / T_l
        
        J_data_l = filter(row -> row.ScanType == "J", df_large)
        γ_data_l = filter(row -> row.ScanType == "γ", df_large)
        M_data_l = filter(row -> row.ScanType == "M", df_large)
        B_21_data_l = filter(row -> row.ScanType == "B_21", df_large)
        Δ_data_l = filter(row -> row.ScanType == "Δ", df_large)
        T_data_l = filter(row -> row.ScanType == "T", df_large)
        
        κ_M_l   = 0.5 .* B_21_l .* M_data_l.X_Value .* exp(Δ_l/T_l) ./ T_l
        κ_B21_l = 0.5 .* B_21_data_l.X_Value .* M_l .* exp(Δ_l/T_l) ./ T_l
        κ_Δ_l   = 0.5 .* B_21_l .* M_l .* exp.(Δ_data_l.X_Value ./ T_l) ./ T_l
        κ_T_l   = 0.5 .* B_21_l .* M_l .* exp.(Δ_l ./ T_data_l.X_Value) ./ T_data_l.X_Value
        
        kg_l = vcat(κ_l ./ γ_data_l.X_Value, κ_M_l ./ γ_l, κ_B21_l ./ γ_l, κ_Δ_l ./ γ_l, κ_T_l ./ γ_l)
        Y_J_l_scaled = J_data_l.dV_Max ./ J_data_l.X_Value
        Y_kg_l = vcat(γ_data_l.dV_Max ./ J_l, M_data_l.dV_Max ./ J_l, B_21_data_l.dV_Max ./ J_l, Δ_data_l.dV_Max ./ J_l, T_data_l.dV_Max ./ J_l)

        J_full_l = vcat(J_data_l.X_Value, fill(J_l, length(Y_kg_l)))
        kg_full_l = vcat(fill(κ_l / γ_l, nrow(J_data_l)), kg_l)
        X_base_l = J_full_l .* kg_full_l
        Y_comb_l = vcat(Y_J_l_scaled, Y_kg_l)

        fit_comb_power_l = curve_fit(power_model, X_base_l, Y_comb_l, [1.0, 1.0])
        b_comb_l = fit_comb_power_l.param[2]

        X_comb_l = sqrt.(X_base_l)
        fit_comb_l = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_l, Y = Y_comb_l))
        C_comb_l = coef(fit_comb_l)[1]
        r2_comb_l = 1.0 - sum((Y_comb_l .- predict(fit_comb_l)).^2) / sum((Y_comb_l .- mean(Y_comb_l)).^2)

        # --- 3. Store Results ---
        push!(results, (
            sites = Float64(sites),
            C_s = C_comb_s, b_comb_s = b_comb_s,
            C_l = C_comb_l, b_comb_l = b_comb_l,
            eq_s = "ΔV_max / J = ($(round(C_comb_s, sigdigits=5))) * √(J*κ/γ)",
            r2_s = round(r2_comb_s, digits=6),
            eq_l = "ΔV_max / J = ($(round(C_comb_l, sigdigits=5))) * √(J*κ/γ)",
            r2_l = round(r2_comb_l, digits=6)
        ))
    end

    if isempty(results)
        println("No Master Stability files found to process.")
        return
    end

    # ==========================================================================
    # 4. FINITE-SIZE SCALING ANALYSIS (N -> ∞)
    # ==========================================================================
    println("\n" * "="^80)
    println("PERFORMING FINITE-SIZE SCALING ANALYSIS")
    println("="^80)

    # All data for Prefactors
    N_vals_pref = Float64[r.sites for r in results]
    C_s_vals = Float64[r.C_s for r in results]
    C_l_vals = Float64[r.C_l for r in results]

    # Bulk data (N >= 4) for asymptotic exponent analysis
    bulk_results = filter(r -> r.sites >= 4, results)
    N_vals_exp = Float64[r.sites for r in bulk_results]
    inv_N_exp = 1.0 ./ N_vals_exp
    bcomb_s_vals = Float64[r.b_comb_s for r in bulk_results]
    bcomb_l_vals = Float64[r.b_comb_l for r in bulk_results]

    # --- A. Fit Prefactors C(N) (Seed values) ---
    fit_pow_s = lm(@formula(logC ~ logN), DataFrame(logN=log.(N_vals_pref), logC=log.(C_s_vals)))
    A_pow_s, a_pow_s = exp(coef(fit_pow_s)[1]), -coef(fit_pow_s)[2]
    
    fit_pow_l = lm(@formula(logC ~ logN), DataFrame(logN=log.(N_vals_pref), logC=log.(C_l_vals)))
    A_pow_l, a_pow_l = exp(coef(fit_pow_l)[1]), -coef(fit_pow_l)[2]

    # --- B. Fit Exponents (N >= 4 ONLY) ---
    fit_bcomb_s = lm(@formula(b ~ invN), DataFrame(invN=inv_N_exp, b=bcomb_s_vals))
    fit_bcomb_l = lm(@formula(b ~ invN), DataFrame(invN=inv_N_exp, b=bcomb_l_vals))

    # --- C. Plotting Combined Power Law ---
    println("Generating Combined Power Law Plot...")
    fig_pow_comb = Figure(size = (600, 400), fontsize = 16)
    
    pow_model_fit(x, p) = p[1] .* (x .^ -p[2])
    
    # Use logarithmic scaling fits as initial guess for robust nonlinear fitting
    fit_comb = curve_fit(pow_model_fit, N_vals_pref, C_l_vals, [mean([A_pow_s, A_pow_l]), mean([a_pow_s, a_pow_l])])
    A_comb, a_comb = fit_comb.param

    ss_tot_c = sum((C_l_vals .- mean(C_l_vals)).^2)
    ss_res_c = sum((C_l_vals .- pow_model_fit(N_vals_pref, fit_comb.param)).^2)
    r2_c = 1.0 - (ss_res_c / ss_tot_c)

    N_dense_pref = range(minimum(N_vals_pref)-0.2, maximum(N_vals_pref)+0.2, length=200)

    ax_comb = Axis(fig_pow_comb[1, 1], title = LaTeXString("Schaling van voorfactor"), xlabel = LaTeXString("Sites (N)"), ylabel = LaTeXString("Voorfactor C_N"), titlesize=35, xlabelsize=25, xticklabelsize=20, ylabelsize=25, yticklabelsize=20, yscale=log10, xscale=log10, xticks = [4,6,8,10,12,14,16])
    scatter!(ax_comb, N_vals_pref, C_l_vals, color=(:red, 0.6), marker=:rect, markersize=14, label="Data")
    lines!(ax_comb, N_dense_pref, pow_model_fit(N_dense_pref, fit_comb.param), color=:black, linewidth=4, label="Fit")

    text_pow_c = LaTeXString(@sprintf("C_N = %.2f × N^{-%.2f}", A_comb, a_comb))
    text!(ax_comb, text_pow_c, position = (0.95, 0.90), align = (:right, :top), space = :relative, fontsize = 25, font = :bold)
    axislegend(ax_comb, position = :lb, labelsize=20)

    display(fig_pow_comb)
    save("Param_scan/Plots/Prefactor_PowerLaw_Combined.png", fig_pow_comb)

    # --- D. Plotting Thermodynamic Limit ---
    println("Generating Thermodynamic Limit Plot...")
    fig_b = Figure(size = (800, 800), fontsize = 16)
    Label(fig_b[0, 1], "Scaling of Exponent (N → ∞)", fontsize = 35, font = :bold)

    invN_dense = range(0.0, maximum(1.0 ./ N_vals_pref)*1.1, length=200)

    for (row, reg_nl, bcomb, fit_comb_exp) in [
        (1, "Small Regime", bcomb_s_vals, fit_bcomb_s),
        (2, "Large Regime", bcomb_l_vals, fit_bcomb_l)]
        
        ax_exp = Axis(fig_b[row, 1], title = "$reg_nl (J*κ/γ Exponent)", xlabel = "1 / N (Zero is Infinite Chain)", ylabel = "Exponent β", titlesize=25, xlabelsize=25, xticklabelsize=20, ylabelsize=25, yticklabelsize=20)

        scatter!(ax_exp, inv_N_exp, bcomb, color=:green, markersize=12, label="Bulk Data (N≥4)")
        lines!(ax_exp, invN_dense, predict(fit_comb_exp, DataFrame(invN=invN_dense)), color=:black, linewidth=3)
        scatter!(ax_exp, [0.0], [coef(fit_comb_exp)[1]], color=:red, marker=:star5, markersize=20, label="Thermodynamic Limit")
        
        if row == 1 axislegend(ax_exp, position=:lt, labelsize=20) end
    end
    display(fig_b)
    save("Param_scan/Plots/Exponent_Scaling.png", fig_b)
end 

calculate_master_fits()