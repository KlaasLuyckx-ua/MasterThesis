# --- 1. Import Packages, include Functions and set dataPath ---
using DifferentialEquations, JLD2, CairoMakie, Trapz, Printf, FFTW, LsqFit, DSP, TimerOutputs

# Initialize the TimerOutput object
const to = TimerOutput()

@timeit to "1. Setup" begin
include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../Functions_JJ/Analytics_JJ.jl")
include("../Functions_JJ/Output_JJ.jl")

operatingFile = "Klaas_Data_JJ_noise"
dataPath = "/Users/klaas/OneDrive/Documenten/Fysica/Masterproef/Code_Klaas/Data/"
end

# --- 2. Define Parameters ---
@timeit to "2. Parameters" begin
# Adler dependence system parameters are defined here
J = 1.0e-1                      # coupling parameter [meV]
γ = 1.0e-2                      # loss rate [meV]
ΔV = 0.8                        # potential difference between sites [meV]
Vl = 0*ΔV                       # left site potential [meV]
Vr = ΔV                         # right site potential [meV]
δ = 0.0                         # pump imbalance parameter
noise_strength = 1.0            # strength of the noise term (0 = no noise, 1 = full noise)

# Simulation parameters
sites = 2                       # number of sites to simulate
save_ψ_per = 1                  # eval_time steps between saving all field data
savesteps = 1.0e6               # number of samples taken during averaging
saveDuration = 1e5/J            # total duration of simulation [s]
dt = 1.0e-5/J                   # time step for integration algorithm [s]

# Simulation time
t_begin = 1e4/J                 # time to reach steady state, simulation time before sampling [s]
t_end = t_begin + saveDuration  # end time of simulation [s]
t_span = (0, t_end)             # time span of simulation [s]
save_Δt = saveDuration/savesteps    # time step between observable sampling [s]
eval_time = t_end-saveDuration:save_Δt:t_end    # sampling time vector

# System parameters
n_avg = 1.0e4                   # target number of photons per site
M = 1.0e8                       # total number of dye molecules per site
B_21 = 1.0e-7                   # Einstein coefficient
Δ = -60                         # detuning
T = 25.0                        # temperature
κ = 0.5*B_21*M*exp(Δ/T)/T       # relaxation rate
M_2_avg = M*exp(Δ/T)            # average number of excited molecules per site

"""Compute M_eff from specified parameters."""
function M_eff(M, γ, B_21, Δ, T)
    M_effective = 0.5*(M + γ*exp(-Δ/T)/B_21)/(1+cosh(Δ/T))
    return M_effective
end

M_n_rat = round(M_eff(M,γ,B_21,Δ,T)/n_avg^2, sigdigits=4)
println("Effective M/n^2 ratio: ", M_n_rat)

# Adjust potential and pump shapes
V = zeros(Float64, sites)       # potential shape
ρ = ones(Float64, sites)        # pump shape
V[1] = Vl                       # left site potential
V[2] = Vr                       # right site potential
ρ[1] = 1.0 - δ; ρ[2] = 1.0 + δ  # pump shape adjustment

# Pack parameters into struct
par_str = CFPars(J, sites, n_avg, M, B_21, Δ, γ, T, κ, V, ρ, save_Δt)
end

# --- 3. Set up and Run Simulation ---
@timeit to "3. Simulation" begin
# Calculate steady state values or critical limits
ss_result = calculate_steady_state_JJ(J, γ, Vl, Vr, B_21, M, T, Δ, κ, n_avg, δ)
# Calculate ΔV stability limit
ΔV_max_stability, plot_ΔV_range, plot_real_eigenvals, plot_imag_eigenvals, plot_max_real_eigenvals, matrix_sequence, eigenvecs= calculate_critical_ΔV_JJ(J,γ, B_21, M, T, Δ, κ, n_avg, δ)

println("ΔV stability limit: ", ΔV_max_stability, " meV")

# --- Set Initial Conditions ---
u_begin = zeros(ComplexF64, 2*sites)

# Use the calculated steady state values if locked, otherwise use the fallbacks (n_avg)
if ss_result.status == "Locked"
    u_begin[1] = sqrt(ss_result.nl_final)
    u_begin[2] = sqrt(ss_result.nr_final)*exp(im*ss_result.φ_final)
    u_begin[sites+1] = ss_result.M_2_l + ss_result.nl_final
    u_begin[sites+2] = ss_result.M_2_r + ss_result.nr_final
else
    # Fallback for desynchronized state: start with average values
    u_begin[1] = sqrt(n_avg)
    u_begin[2] = sqrt(n_avg)*exp(im*0.0)
    u_begin[sites+1] = M_2_avg + n_avg
    u_begin[sites+2] = M_2_avg + n_avg
end

# Technical parameter vector
tech_param = (u_begin, t_span, dt, eval_time, save_ψ_per, save_Δt)

# Pack all parameters into p vector
p = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength)

# Set up SDE problem
prob = SDEProblem(classicalField_JJ, σ_classicalField_JJ_add, u_begin, t_span, p)
sol = solve(prob, SOSRA(); dt=dt, saveat=eval_time[1:save_ψ_per:end], maxiters=1e10)

# Extract data from solution
o = extractSolDataX(sol, par_str)
end

# --- 4. Save data to file ---
@timeit to "4. Save Data" begin
filename = dataPath*operatingFile
jldsave(filename*".jld2"; X_data=o[o_dict["X_data"]], t_vec=o[o_dict["t_vec"]], parameters=par_str, tech_parameters=tech_param)
end

# --- 5. Load data and define parameters for plotting ---
@timeit to "5. Load and Process Data" begin
# Call the function from the external script
sim_data = load_data_JJ("Data/Klaas_Data_JJ_noise.jld2")

# Unpack the variables needed for the next sections
J = sim_data.J
γ = sim_data.γ
Vl = sim_data.Vl
Vr = sim_data.Vr
ΔV = sim_data.ΔV
δ = sim_data.δ

t_vec = sim_data.t_vec
X_data = sim_data.X_data
n = sim_data.n
M_1 = sim_data.M_1
M_2 = sim_data.M_2
theta = sim_data.theta
φ = sim_data.φ

# Process Arrays (Photon numbers, Populations, Phases)
# Pre-allocate arrays
len = length(t_vec)
n = zeros(Float64, sites, len)
M_2 = zeros(Float64, sites, len)
M_1 = zeros(Float64, sites, len)
theta = zeros(Float64, sites, len)
φ = zeros(Float64, len)

# Calculate values
n[1,:] = abs2.(X_data[1,:])
n[2,:] = abs2.(X_data[2,:])

M_2[1,:] = real.(X_data[3,:]) .- n[1,:]
M_2[2,:] = real.(X_data[4,:]) .- n[2,:]

M_1[1,:] = M .- M_2[1,:]
M_1[2,:] = M .- M_2[2,:]

theta[1,:] = angle.(X_data[1,:])
theta[2,:] = angle.(X_data[2,:])

# Phase difference (θ1 - θ2)
φ[:] = angle.(conj.(X_data[1,:]) .* X_data[2,:])
# Unwrap phase difference to avoid discontinuities
φ_unwrapped = unwrap(φ)

# Calculate Fourier transform of n and φ for plotting
fs = 1.0 / save_Δt  # Sampling frequency
N = length(φ)
freqs = fftshift(fftfreq(N, fs))
n_FT = fftshift(fft(n, 2), 2)
φ_FT = fftshift(fft(φ))
ψ_1_FT = fftshift(fft(X_data[1,:]))  # Fourier transform of the field amplitude for site 1
ψ_2_FT = fftshift(fft(X_data[2,:]))  # Fourier transform of the field amplitude for site 2

# Calculate fields in the normal mode basis
ψ_S = (X_data[1,:] .+ X_data[2,:]) ./ sqrt(2)
ψ_A = (X_data[1,:] .- X_data[2,:]) ./ sqrt(2)

# Take the FFT of the antisymmetric (excited) mode
ψ_A_FT = fftshift(fft(ψ_A))
N = length(ψ_A)
power_spectrum_A = abs2.(ψ_A_FT) .* (save_Δt / N)


# --- Calculate Excited State Population ---
# 1. Define the linear Hamiltonian for the current ΔV
H = [Vl -J; -J Vr]

# 2. Get eigenvalues and eigenvectors
vals, vecs = eigen(H)

# 3. Identify the excited state (highest eigenvalue)
idx_ex = argmax(real.(vals))
v_ex = vecs[:, idx_ex] # This is a 2-element column vector

# 4. Project the time-series field data onto the excited state vector
# Using conj() to compute the proper inner product
ψ_ex = conj(v_ex[1]) .* X_data[1,:] .+ conj(v_ex[2]) .* X_data[2,:]

# 5. Calculate photon number in the excited state and its time-average
n_ex = abs2.(ψ_ex)
n_ex_avg = mean(n_ex)

println("Average excited state population for ΔV = $ΔV : ", n_ex_avg)
end

# --- 6. Plot results ---
@timeit to "6. Plot Results" begin
# plot_results_JJ(t_vec, J, n, theta, φ, M_1, M_2, ss_result, plot_ΔV_range, plot_real_eigenvals, plot_imag_eigenvals)
CairoMakie.activate!()  # Activate CairoMakie for plotting
f1 = Figure(size=(800, 800), title="ΔV = $(ΔV) meV")

f1[0, 1:2] = Label(f1, "ΔV = $(ΔV) meV", fontsize = 22, font = :bold)

ax1 = Axis(f1[1, 1], title="Photon Number Time Series", xlabel="Time (1/meV)", ylabel="Photon Number")
lines!(ax1, t_vec*J, n[1,:], label="Site 1")
lines!(ax1, t_vec*J, n[2,:], label="Site 2")
axislegend(position = :rb)

ax2 = Axis(f1[1, 2], title="Phase Difference Time Series", xlabel="Time (1/meV)", ylabel="Phase Difference (rad)")
lines!(ax2, t_vec*J, φ, label="Phase Difference")

ax3 = Axis(f1[2, 1], title="Unwrapped Phase Difference Time Series", xlabel="Time (1/meV)", ylabel="Unwrapped Phase Difference (rad/(2π))")
lines!(ax3, t_vec*J, φ_unwrapped/(2π), label="Unwrapped Phase Difference")

ax4 = Axis(f1[2, 2], title="Fourier Transform of Phase Difference", xlabel="Frequency (meV)", ylabel="Magnitude")
lines!(ax4, freqs, abs.(φ_FT), label="|FT(Phase Difference)|")

ax5 = Axis(f1[3, 1], title="Fourier Transform of Site 1 Photon Number", xlabel="Frequency (meV)", ylabel="Magnitude")
lines!(ax5, freqs, abs.(n_FT[1,:]), label="|FT(Site 1 Photon Number)|")

ax6 = Axis(f1[3, 2], title="Fourier Transform of Site 2 Photon Number", xlabel="Frequency (meV)", ylabel="Magnitude")
lines!(ax6, freqs, abs.(n_FT[2,:]), label="|FT(Site 2 Photon Number)|")

ax7 = Axis(f1[4, 1], title="Fourier Transform of Site 1 Field Amplitude", xlabel="Frequency (meV)", ylabel="Magnitude")
lines!(ax7, freqs, abs.(ψ_1_FT), label="|FT(Site 1 Field Amplitude)|")

ax8 = Axis(f1[4, 2], title="Fourier Transform of Site 2 Field Amplitude", xlabel="Frequency (meV)", ylabel="Magnitude")
lines!(ax8, freqs, abs.(ψ_2_FT), label="|FT(Site 2 Field Amplitude)|")

ax9 = Axis(f1[5, 1], title="Fourier Transform of Antisymmetric Mode", xlabel="Frequency (meV)", ylabel="Magnitude")
lines!(ax9, freqs, abs.(ψ_A_FT), label="|FT(Antisymmetric Mode)|")

ax10 = Axis(f1[5, 2], title="Power Spectrum of Antisymmetric Mode", xlabel="Frequency (meV)", ylabel="Power Spectrum")
lines!(ax10, freqs, power_spectrum_A, label="Power Spectrum (Antisymmetric Mode)")
display(f1)
end

# ----- 7. Calculate Spectral Power (Lorentzian Fit) ---
@timeit to "7. Calculate Spectral Power" begin
# 1. Isolate the data around the peak for the fitting algorithm
peak_idx = argmax(power_spectrum_A)

# We use a window to isolate the peak for the optimizer, 
# preventing other spectrum features from ruining the fit.
window = 200 
freq_data = freqs[peak_idx-window:peak_idx+window]
power_data = power_spectrum_A[peak_idx-window:peak_idx+window]

# 2. Define the Lorentzian model function for LsqFit
# p[1] = Amplitude (A), p[2] = Center (x₀), p[3] = Width (γ)
@. model_lorentz(x, p) = p[1] * (p[3]^2 / ((x - p[2])^2 + p[3]^2))

# 3. Initial guesses for the parameters [Amplitude, Center, Width]
p0 = [maximum(power_data), freqs[peak_idx], (freq_data[end] - freq_data[1]) / 4.0]

# 4. Perform the curve fitting
fit_lorentz = curve_fit(model_lorentz, freq_data, power_data, p0)

# Extract fitted parameters
p_l = fit_lorentz.param

# 5. Calculate analytical integral (Spectral Power)
# Integral of Lorentzian = A * γ * π
integral_lorentz = p_l[1] * abs(p_l[3]) * π
println("mean(abs2.(ψ_A)) = ", mean(abs2.(ψ_A)))
println("trapz(freqs, power_spectrum_A) = ", trapz(freqs, power_spectrum_A))

println("Integrated Spectral Power (Lorentzian fit): ", integral_lorentz)
end

# --- 8. Calculate Integrated Spectral Power (Secondary Weight) ---
@timeit to "8. Calculate Spectral Power" begin
# Calculate frequency bin width (Δf) for integration
df = freqs[2] - freqs[1]

# Find the index of the main DC component (frequency ~0)
dc_idx = argmin(abs.(freqs))

# Define the Lorentzian model
lorentzian(x, p) = p[1] .* (p[3]^2 ./ ((x .- p[2]).^2 .+ p[3]^2)) .+ p[4]

# Isolate a small region around DC just for fitting
fit_window = 1000 
fit_range = max(1, dc_idx - fit_window) : min(length(freqs), dc_idx + fit_window)
f_fit = freqs[fit_range]

# Define bounds for NORMALIZED fitting: [Amplitude, CenterFreq, Width, Baseline]
# This prevents the solver from exploring unphysical parameters (like negative widths)
lb = [0.0, -0.05, 1e-6, 0.0]
ub = [2.0,  0.05, 0.1,  1.0]

# --- Process Site 1 ---
power_n1 = abs2.(n_FT[1,:])
power_n1_fit = power_n1[fit_range]

# NORMALIZE the data to prevent solver failure due to massive 10^20 values
max_p1 = maximum(power_n1_fit)
p1_norm = power_n1_fit ./ max_p1
p0_norm_1 = [1.0, 0.0, 0.005, minimum(p1_norm)] # Initial guess on normalized data

# Fit the normalized data
fit_result_1 = curve_fit(lorentzian, f_fit, p1_norm, p0_norm_1, lower=lb, upper=ub)
p_norm_best_1 = fit_result_1.param

# Un-normalize the amplitude and baseline parameters back to actual scale
p_best_1 = [p_norm_best_1[1]*max_p1, p_norm_best_1[2], p_norm_best_1[3], p_norm_best_1[4]*max_p1]

fitted_peak_1 = lorentzian(freqs, p_best_1)
power_n1_corrected = max.(power_n1 .- fitted_peak_1 .+ p_best_1[4], 0.0)
weight_1 = sum(power_n1_corrected) * df


# --- Process Site 2 ---
power_n2 = abs2.(n_FT[2,:])
power_n2_fit = power_n2[fit_range]

# NORMALIZE
max_p2 = maximum(power_n2_fit)
p2_norm = power_n2_fit ./ max_p2
p0_norm_2 = [1.0, 0.0, 0.005, minimum(p2_norm)]

# Fit the normalized data
fit_result_2 = curve_fit(lorentzian, f_fit, p2_norm, p0_norm_2, lower=lb, upper=ub)
p_norm_best_2 = fit_result_2.param

# Un-normalize parameters
p_best_2 = [p_norm_best_2[1]*max_p2, p_norm_best_2[2], p_norm_best_2[3], p_norm_best_2[4]*max_p2]

fitted_peak_2 = lorentzian(freqs, p_best_2)
power_n2_corrected = max.(power_n2 .- fitted_peak_2 .+ p_best_2[4], 0.0)
weight_2 = sum(power_n2_corrected) * df

println("--- Integrated Spectral Power (Lorentzian Subtraction) ---")
println("Site 1 Weight: ", weight_1)
println("Site 2 Weight: ", weight_2)
end


# ==============================================================================
# --- 9. New Section: Plot Fit Comparisons (Lorentzian vs Gaussian) ---
# ==============================================================================
@timeit to "9. Plot Fit Comparisons" begin
println("\n--- Generating Fit Comparison Plots ---")

# Define Gaussian Model
gaussian_model(x, p) = p[1] .* exp.(-(x .- p[2]).^2 ./ (2 .* p[3]^2)) .+ p[4]

# --- Fit Gaussian Site 1 (Using Normalized Data) ---
p0_G_norm_1 = [1.0, 0.0, 0.005, minimum(p1_norm)]
fit_result_G1 = curve_fit(gaussian_model, f_fit, p1_norm, p0_G_norm_1, lower=lb, upper=ub)
p_norm_best_G1 = fit_result_G1.param
# Un-normalize
p_best_G1 = [p_norm_best_G1[1]*max_p1, p_norm_best_G1[2], p_norm_best_G1[3], p_norm_best_G1[4]*max_p1]
fitted_peak_G1 = gaussian_model(freqs, p_best_G1)

# --- Fit Gaussian Site 2 (Using Normalized Data) ---
p0_G_norm_2 = [1.0, 0.0, 0.005, minimum(p2_norm)]
fit_result_G2 = curve_fit(gaussian_model, f_fit, p2_norm, p0_G_norm_2, lower=lb, upper=ub)
p_norm_best_G2 = fit_result_G2.param
# Un-normalize
p_best_G2 = [p_norm_best_G2[1]*max_p2, p_norm_best_G2[2], p_norm_best_G2[3], p_norm_best_G2[4]*max_p2]
fitted_peak_G2 = gaussian_model(freqs, p_best_G2)


# --- Plotting Setup ---
view_f_limit = 0.4 # Define frequency limits to view [± meV]
view_indices = findall( -view_f_limit .<= freqs .<= view_f_limit )

f_fits = Figure(size=(1000, 800))
Label(f_fits[0, 1:2], "Photon Number FT Power Spectrum: Data vs Fits (ΔV=$(ΔV))", fontsize=20, font=:bold)

titles = ["Site 1 (Linear Data)", "Site 1 (Log Data)", "Site 2 (Linear Data)", "Site 2 (Log Data)"]
xlab = "Frequency (meV)"
ylab_lin = "Power Spectral Density (abs²)"
ylab_log = "log₁₀(Power Spectral Density)"

ax1 = Axis(f_fits[1, 1], title=titles[1], xlabel=xlab, ylabel=ylab_lin)
ax2 = Axis(f_fits[1, 2], title=titles[2], xlabel=xlab, ylabel=ylab_log)
ax3 = Axis(f_fits[2, 1], title=titles[3], xlabel=xlab, ylabel=ylab_lin)
ax4 = Axis(f_fits[2, 2], title=titles[4], xlabel=xlab, ylabel=ylab_log)

linkxaxes!(ax1, ax2, ax3, ax4)
xlims!(ax1, -view_f_limit, view_f_limit) 

data_attr = (color=:black, linestyle=:solid, linewidth=1.5, label="Data")
lorentz_attr = (color=:dodgerblue3, linestyle=:dot, linewidth=2.5, label="Lorentzian Fit")
gauss_attr = (color=:firebrick1, linestyle=:dash, linewidth=2.5, label="Gaussian Fit")

function plot_comparison!(ax, data_all, l_fit_all, g_fit_all, view_idx; is_log=false)
    freq_v = freqs[view_idx]
    
    d_plot = data_all[view_idx]
    l_plot = l_fit_all[view_idx]
    g_plot = g_fit_all[view_idx]
    
    if is_log
        eps_val = 1e-15
        d_plot = log10.(d_plot .+ eps_val)
        l_plot = log10.(l_plot .+ eps_val)
        g_plot = log10.(g_plot .+ eps_val)
    end
    
    lines!(ax, freq_v, d_plot; data_attr...)
    lines!(ax, freq_v, l_plot; lorentz_attr...)
    lines!(ax, freq_v, g_plot; gauss_attr...)
end

# Add plots
plot_comparison!(ax1, power_n1, fitted_peak_1, fitted_peak_G1, view_indices, is_log=false)
axislegend(ax1, position=:rt, nbanks=1, framevisible=true)
plot_comparison!(ax2, power_n1, fitted_peak_1, fitted_peak_G1, view_indices, is_log=true)

plot_comparison!(ax3, power_n2, fitted_peak_2, fitted_peak_G2, view_indices, is_log=false)
axislegend(ax3, position=:rt, nbanks=1, framevisible=true)
plot_comparison!(ax4, power_n2, fitted_peak_2, fitted_peak_G2, view_indices, is_log=true)

display(f_fits)
println("Fit comparison plots generated and displayed.")
end

# --- Print Timer Output ---
println("\n--- Timer Outputs ---")
show(to)
println()
