# ==============================================================================
# MULTI-SITE MASTER FIT DISCOVERY
# ==============================================================================
using GLM, LsqFit, DataFrames, CSV, Statistics, CairoMakie, Printf

function calculate_master_fits()
    # Define the safe Power Law model locally
    power_model(x, p) = p[1] .* (max.(x, 1e-10) .^ p[2])

    results = []
    println("Processing datasets for multiple sites...\n")

    for sites in 4:32
        file_small = "Param_scan/Data_Cluster/Master_Stability_Results_$(sites)_Site_s.csv"
        file_large = "Param_scan/Data_Cluster/Master_Stability_Results_$(sites)_Site_l.csv"
        if !isfile(file_small) || !isfile(file_large)
            continue
        end

        # --------------------------------------------------------------------------
        # 1. SMALL REGIME PROCESSING
        # --------------------------------------------------------------------------
        df_small = CSV.read(file_small, DataFrame)
        
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
        
        # Target arrays scaled by J
        Y_J_s_scaled = J_data_s.dV_Max ./ J_data_s.X_Value
        Y_kg_s = vcat(γ_data_s.dV_Max ./ J_s, M_data_s.dV_Max ./ J_s, B_21_data_s.dV_Max ./ J_s, Δ_data_s.dV_Max ./ J_s, T_data_s.dV_Max ./ J_s)

        # Full combined arrays
        J_full_s = vcat(J_data_s.X_Value, fill(J_s, length(Y_kg_s)))
        kg_full_s = vcat(fill(κ_s / γ_s, nrow(J_data_s)), kg_s)
        X_base_s = J_full_s .* kg_full_s
        Y_comb_s = vcat(Y_J_s_scaled, Y_kg_s)

        # Combined exponent fit on scaled data
        fit_comb_power_s = curve_fit(power_model, X_base_s, Y_comb_s, [1.0, 0.5])
        b_comb_s = fit_comb_power_s.param[2]

        # Empirical Combined fit forced to √(J*κ/γ)
        X_comb_s = sqrt.(X_base_s)
        
        fit_comb_s = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_s, Y = Y_comb_s))
        C_comb_s = coef(fit_comb_s)[1]
        r2_comb_s = 1.0 - sum((Y_comb_s .- predict(fit_comb_s)).^2) / sum((Y_comb_s .- mean(Y_comb_s)).^2)

        # --------------------------------------------------------------------------
        # 2. LARGE REGIME PROCESSING
        # --------------------------------------------------------------------------
        df_large = CSV.read(file_large, DataFrame)
        
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
        
        # Target arrays scaled by J
        Y_J_l_scaled = J_data_l.dV_Max ./ J_data_l.X_Value
        Y_kg_l = vcat(γ_data_l.dV_Max ./ J_l, M_data_l.dV_Max ./ J_l, B_21_data_l.dV_Max ./ J_l, Δ_data_l.dV_Max ./ J_l, T_data_l.dV_Max ./ J_l)

        # Full combined arrays
        J_full_l = vcat(J_data_l.X_Value, fill(J_l, length(Y_kg_l)))
        kg_full_l = vcat(fill(κ_l / γ_l, nrow(J_data_l)), kg_l)
        X_base_l = J_full_l .* kg_full_l
        Y_comb_l = vcat(Y_J_l_scaled, Y_kg_l)

        # Combined exponent fit on scaled data
        fit_comb_power_l = curve_fit(power_model, X_base_l, Y_comb_l, [1.0, 1.0])
        b_comb_l = fit_comb_power_l.param[2]

        # Empirical Combined fit forced to √(J*κ/γ)
        X_comb_l = sqrt.(X_base_l)
        
        fit_comb_l = lm(@formula(Y ~ 0 + X_comb), DataFrame(X_comb = X_comb_l, Y = Y_comb_l))
        C_comb_l = coef(fit_comb_l)[1]
        r2_comb_l = 1.0 - sum((Y_comb_l .- predict(fit_comb_l)).^2) / sum((Y_comb_l .- mean(Y_comb_l)).^2)

        # --------------------------------------------------------------------------
        # 3. STORE RESULTS
        # --------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------
    # 4. PRINT FORMATTED MASTER TABLE
    # ------------------------------------------------------------------------------
    println("=========================================================================================================================")
    println(rpad(" SITES", 10), "| ", rpad("REGIME", 10), "| ", rpad("EMPIRICAL MASTER EQUATION", 75), "| R² VALUE")
    println("=========================================================================================================================")
    for res in results
        println(rpad(" $(Int(res.sites))", 10), "| ", rpad("Small", 10), "| ", rpad(res.eq_s, 75), "| ", res.r2_s)
        println(rpad(" ", 10), "| ", rpad("Large", 10), "| ", rpad(res.eq_l, 75), "| ", res.r2_l)
        println("-------------------------------------------------------------------------------------------------------------------------")
    end
    println("=========================================================================================================================\n")


    # ==============================================================================
    # 5. FINITE-SIZE SCALING ANALYSIS (N -> ∞)
    # ==============================================================================
    println("\n" * "="^80)
    println("PERFORMING FINITE-SIZE SCALING ANALYSIS (Geometry vs Dynamics)")
    println("="^80)

    # --- Separate Data Streams ---
    # 1. Prefactors use ALL data
    N_vals_pref = Float64[r.sites for r in results]
    C_s_vals = Float64[r.C_s for r in results]
    C_l_vals = Float64[r.C_l for r in results]

    # 2. Exponents use ONLY Bulk data (N >= 4)
    bulk_results = filter(r -> r.sites >= 4, results)

    N_vals_exp = Float64[r.sites for r in bulk_results]
    inv_N_exp  = 1.0 ./ N_vals_exp
    
    bcomb_s_vals = Float64[r.b_comb_s for r in bulk_results]
    bcomb_l_vals = Float64[r.b_comb_l for r in bulk_results]

    # --- A. Fit Prefactors C(N) using GLM Log-Transforms (ALL N) ---
    # Small Regime
    fit_exp_s = lm(@formula(logC ~ N), DataFrame(N=N_vals_pref, logC=log.(C_s_vals)))
    fit_pow_s = lm(@formula(logC ~ logN), DataFrame(logN=log.(N_vals_pref), logC=log.(C_s_vals)))
    A_exp_s, alpha_s = exp(coef(fit_exp_s)[1]), -coef(fit_exp_s)[2]
    A_pow_s, a_pow_s = exp(coef(fit_pow_s)[1]), -coef(fit_pow_s)[2]
    
    # Large Regime
    fit_exp_l = lm(@formula(logC ~ N), DataFrame(N=N_vals_pref, logC=log.(C_l_vals)))
    fit_pow_l = lm(@formula(logC ~ logN), DataFrame(logN=log.(N_vals_pref), logC=log.(C_l_vals)))
    A_exp_l, alpha_l = exp(coef(fit_exp_l)[1]), -coef(fit_exp_l)[2]
    A_pow_l, a_pow_l = exp(coef(fit_pow_l)[1]), -coef(fit_pow_l)[2]

    # --- B. Fit Exponents to Finite Size Scaling Law: b(N) = b_∞ + k(1/N) (N >= 4 ONLY) ---
    fit_bcomb_s = lm(@formula(b ~ invN), DataFrame(invN=inv_N_exp, b=bcomb_s_vals))
    fit_bcomb_l = lm(@formula(b ~ invN), DataFrame(invN=inv_N_exp, b=bcomb_l_vals))

    # --- C. Plotting the Scaling Laws ---
    println("Generating Prefactor Scaling Plot (All N)...")
    fig_C = Figure(size = (1400, 800), fontsize = 16)
    Label(fig_C[0, 1:3], "PREFACTOR SCALING: EXPONENTIAL VS POWER LAW (All N)", fontsize = 22, font = :bold)

    N_dense_pref = range(minimum(N_vals_pref)-0.2, maximum(N_vals_pref)+0.2, length=200)
    
    for (row, regime, C_vals, A_e, al, A_p, a_p) in [(1, "Small Regime", C_s_vals, A_exp_s, alpha_s, A_pow_s, a_pow_s), 
                                                     (2, "Large Regime", C_l_vals, A_exp_l, alpha_l, A_pow_l, a_pow_l)]
        # Linear Scale
        ax_lin = Axis(fig_C[row, 1], title = "$regime (Linear)", xlabel = "Sites (N)", ylabel = "Prefactor C")
        scatter!(ax_lin, N_vals_pref, C_vals, color=:black, markersize=12, label="Data (All N)")
        lines!(ax_lin, N_dense_pref, A_e .* exp.(-al .* N_dense_pref), color=:blue, linewidth=3)
        lines!(ax_lin, N_dense_pref, A_p .* (N_dense_pref .^ -a_p), color=:red, linewidth=3, linestyle=:dash)
        if row == 1 axislegend(ax_lin) end

        # Semi-Log
        ax_semi = Axis(fig_C[row, 2], title = "$regime (Semi-Log)", xlabel = "Sites (N)", ylabel = "Prefactor C (Log)", yscale=log10)
        scatter!(ax_semi, N_vals_pref, C_vals, color=:black, markersize=12)
        lines!(ax_semi, N_dense_pref, A_e .* exp.(-al .* N_dense_pref), color=:blue, linewidth=3)
        lines!(ax_semi, N_dense_pref, A_p .* (N_dense_pref .^ -a_p), color=:red, linewidth=3, linestyle=:dash)

        # Log-Log
        ax_log = Axis(fig_C[row, 3], title = "$regime (Log-Log)", xlabel = "Sites (Log)", ylabel = "Prefactor C (Log)", xscale=log10, yscale=log10)
        scatter!(ax_log, N_vals_pref, C_vals, color=:black, markersize=12)
        lines!(ax_log, N_dense_pref, A_e .* exp.(-al .* N_dense_pref), color=:blue, linewidth=3)
        lines!(ax_log, N_dense_pref, A_p .* (N_dense_pref .^ -a_p), color=:red, linewidth=3, linestyle=:dash)
    end
    display(fig_C)
    save("Param_scan/Figures/Prefactor_Scaling.png", fig_C)

    # --- D. Plotting ONLY the Power Law (Combined Data, Linear Scale) ---
    println("Generating Combined Power Law Plot...")
    fig_pow_comb = Figure(size = (900, 650), fontsize = 16)
    
    # 1. Combine the datasets
    N_combined = vcat(N_vals_pref, N_vals_pref)
    C_combined = vcat(C_s_vals, C_l_vals)

    # 2. Define the Power Law model and perform a global fit
    pow_model(x, p) = p[1] .* (x .^ -p[2])
    
    # Use the average of your previous separate fits as a robust initial guess
    p0_comb = [mean([A_pow_s, A_pow_l]), mean([a_pow_s, a_pow_l])]
    fit_comb = curve_fit(pow_model, N_combined, C_combined, p0_comb)
    A_comb, a_comb = fit_comb.param

    # 3. Calculate Global R²
    ss_tot_c = sum((C_combined .- mean(C_combined)).^2)
    ss_res_c = sum((C_combined .- pow_model(N_combined, fit_comb.param)).^2)
    r2_c = 1.0 - (ss_res_c / ss_tot_c)

    # 4. Create the plot
    ax_comb = Axis(fig_pow_comb[1, 1], 
                   title = "Prefactor Scaling: Power Law Fit", 
                   xlabel = "Sites (N)", 
                   ylabel = "Prefactor C")

    # Scatter both datasets to show how they align
    scatter!(ax_comb, N_vals_pref, C_s_vals, color=(:blue, 0.6), markersize=14, label="Small Regime Data")
    scatter!(ax_comb, N_vals_pref, C_l_vals, color=(:red, 0.6), marker=:rect, markersize=14, label="Large Regime Data")
    
    # Plot the universal fitted line
    lines!(ax_comb, N_dense_pref, pow_model(N_dense_pref, fit_comb.param), color=:black, linewidth=4, label="Combined Fit")

    # Add Equation and R² text (Top Right)
    text_pow_c = @sprintf("C = %.4f × N^{-%.4f}\nR² = %.4f", A_comb, a_comb, r2_c)
    text!(ax_comb, text_pow_c, position = (0.95, 0.95), align = (:right, :top), space = :relative, fontsize = 20, font = :bold)
    
    axislegend(ax_comb, position = :lb) # Placed Bottom Left to avoid text overlap

    display(fig_pow_comb)
    save("Param_scan/Figures/Prefactor_PowerLaw_Combined.png", fig_pow_comb)

    println("Generating Thermodynamic Limit Exponent Plot (Filtered)...")
    fig_b = Figure(size = (1000, 800), fontsize = 16)
    Label(fig_b[0, 1], "FINITE-SIZE SCALING OF COMBINED EXPONENT (N → ∞) [Filtered N≥4]", fontsize = 22, font = :bold)

    invN_dense = range(0.0, maximum(1.0 ./ N_vals_pref)*1.1, length=200)

    for (row, reg, bcomb, fit_comb_exp) in [
        (1, "Small Regime", bcomb_s_vals, fit_bcomb_s),
        (2, "Large Regime", bcomb_l_vals, fit_bcomb_l)]
        
        ax_exp = Axis(fig_b[row, 1], title = "$reg (J*κ/γ Exponent)", xlabel = "1 / N  (Zero is Infinite Chain)", ylabel = "Exponent b_comb")

        # Plot Bulk Exponents (N>=4) ONLY
        scatter!(ax_exp, inv_N_exp, bcomb, color=:green, markersize=12, label="Bulk Data (N≥4)")
        lines!(ax_exp, invN_dense, predict(fit_comb_exp, DataFrame(invN=invN_dense)), color=:black, linewidth=3)
        scatter!(ax_exp, [0.0], [coef(fit_comb_exp)[1]], color=:red, marker=:star5, markersize=20, label="Thermodynamic Limit")
        
        if row == 1 axislegend(ax_exp, position=:lt) end
    end
    display(fig_b)
    save("Param_scan/Figures/Exponent_Scaling.png", fig_b)


    # --- D. Print Scaling Summary Table ---
    println("\n=========================================================================================================")
    println(rpad(" SYSTEM GEOMETRY SCALING (All N)", 35), "| ", rpad("EXPONENTIAL DECAY (R²)", 30), "| ", "POWER LAW DECAY (R²)")
    println("=========================================================================================================")
    println(rpad(" Small Regime Prefactor C(N)", 35), "| ", rpad("C ∝ e^(-$(round(alpha_s, digits=3))N)  ($(round(r2(fit_exp_s), digits=4)))", 30), "| ", "C ∝ N^(-$(round(a_pow_s, digits=3)))  ($(round(r2(fit_pow_s), digits=4)))")
    println(rpad(" Large Regime Prefactor C(N)", 35), "| ", rpad("C ∝ e^(-$(round(alpha_l, digits=3))N)  ($(round(r2(fit_exp_l), digits=4)))", 30), "| ", "C ∝ N^(-$(round(a_pow_l, digits=3)))  ($(round(r2(fit_pow_l), digits=4)))")
    println("---------------------------------------------------------------------------------------------------------")
    
    println("\n=========================================================================================================")
    println(rpad(" BULK THERMODYNAMIC LIMITS (N≥4 → ∞)", 45), "| ", rpad("EXTRACTED ASYMPTOTE", 20))
    println("=========================================================================================================")
    println(rpad(" Small Regime Combined Exponent", 45), "| ", rpad("b_∞ = $(round(coef(fit_bcomb_s)[1], digits=4))", 20))
    println("---------------------------------------------------------------------------------------------------------")
    println(rpad(" Large Regime Combined Exponent", 45), "| ", rpad("b_∞ = $(round(coef(fit_bcomb_l)[1], digits=4))", 20))
    println("=========================================================================================================\n")

end 
calculate_master_fits()