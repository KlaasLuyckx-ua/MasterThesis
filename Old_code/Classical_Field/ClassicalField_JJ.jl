t0 = time()

# --- 1. Import Packages, include Functions and set dataPath ---
using DifferentialEquations, LinearAlgebra, NPZ, JLD2, Distributed, CairoMakie, GLMakie, LaTeXStrings, Integrals, Trapz, Printf

include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../Functions_JJ/Analytics_JJ.jl")
include("../Functions_JJ/Output_JJ.jl")

operatingFile = "Klaas_Data_JJ"
dataPath = "/Users/klaas/OneDrive/Documenten/Fysica/Masterproef/Code_Klaas/Data/"

# --- 2. Define Parameters ---
# Adler dependence system parameters are defined here
J = 1.0e-1                      # coupling parameter [meV]
γ = 2.0e-3                      # loss rate [meV]
ΔV = 0.406                       # potential difference between sites [meV]
Vl = 0*ΔV                       # left site potential [meV]
Vr = ΔV                         # right site potential [meV]
δ = 0.0                         # pump imbalance parameter

# Simulation parameters
sites = 2                       # number of sites to simulate
save_ψ_per = 100                # eval_time steps between saving all field data
savesteps = 1.0e6               # number of samples taken during averaging
saveDuration = 1e6/J            # total duration of simulation [s]
dt = 1.0e-4/J                   # time step for integration algorithm [s]

# Simulation time
t_begin = 0e5/J                 # time to reach steady state, simulation time before sampling [s]
t_end = t_begin + saveDuration  # end time of simulation [s]
t_span = (0, t_end)             # time span of simulation [s]
save_Δt = saveDuration/savesteps    # time step between observable sampling [s]
eval_time = t_end-saveDuration:save_Δt:t_end    # sampling time vector

# System parameters
n_avg = 3.0e3                   # target number of photons per site
M = 1.0e8                       # total number of dye molecules per site
B_21 = 1.0e-7                   # Einstein coefficient
Δ = -60                         # detuning
T = 25.0                        # temperature
κ = 0.5*B_21*M*exp(Δ/T)/T       # relaxation rate
M_2_avg = M*exp(Δ/T)            # average number of excited molecules per site

# Adjust potential and pump shapes
V = zeros(Float64, sites)       # potential shape
ρ = ones(Float64, sites)        # pump shape
V[1] = Vl                       # left site potential
V[2] = Vr                       # right site potential
ρ[1] = 1.0 - δ; ρ[2] = 1.0 + δ  # pump shape adjustment


# Pack parameters into struct
par_str = CFPars(J, sites, n_avg, M, B_21, Δ, γ, T, κ, V, ρ, save_Δt)

# --- 3. Set up and Run Simulation ---
# Calculate steady state values or critical limits
ss_result = calculate_steady_state_JJ(J, γ, Vl, Vr, B_21, M, T, Δ, κ, n_avg, δ)
# Calculate ΔV stability limit
ΔV_max_stability, plot_ΔV_range, plot_real_eigenvals, plot_imag_eigenvals, plot_max_real_eigenvals, matrix_sequence, eigenvecs= calculate_critical_ΔV_JJ(J,γ, B_21, M, T, Δ, κ, n_avg, δ)

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
p = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ)

# Set up ODE problem
prob = ODEProblem(classicalField_JJ, u_begin, t_span, p)
sol = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false)); maxiters=1e10, reltol=1e-6, abstol=1e-8, saveat=eval_time[1:save_ψ_per:end], dense=false)

# Extract data from solution
o = extractSolDataX(sol, par_str)

# --- 4. Save data to file ---
filename = dataPath*operatingFile
jldsave(filename*".jld2"; X_data=o[o_dict["X_data"]], t_vec=o[o_dict["t_vec"]], parameters=par_str, tech_parameters=tech_param)

# --- 5. Load data and define parameters for plotting ---
# Call the function from the external script
sim_data = load_data_JJ("Data/Klaas_Data_JJ.jld2")

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

# Calculate Fourier transform of n and φ for plotting
n_FT = fft(n)

φ_FT = fft(φ)

# --- 6. Plot results ---
plot_results_JJ(t_vec, X_data, J, n, theta, φ, M_1, M_2, ss_result, plot_ΔV_range, plot_real_eigenvals, plot_imag_eigenvals, n_FT, φ_FT)

# --- 7. Stability Matrix Evolution Visualization ---
# generate_stability_video(matrix_sequence)

# --- 8. Comparison Table and System Status ---
print_stats_JJ(ss_result, φ, n, M_2, J, γ, Vl, Vr, ΔV, δ, κ, ΔV_max_stability)

elapsed = time() - t0
println("Total Simulation Time: $(elapsed) seconds")