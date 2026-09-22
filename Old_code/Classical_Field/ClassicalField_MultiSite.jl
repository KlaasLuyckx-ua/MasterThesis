t0 = time()

# --- 1. Import Packages, Include Functions and set dataPath ---
using DifferentialEquations, NPZ, JLD2, Distributed, CairoMakie, LaTeXStrings, Integrals, Trapz, Printf

include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../Functions_MultiSite/Output_MultiSite.jl")
include("../Functions_MultiSite/Analytics_MultiSite.jl")

operatingFile = "Klaas_Data_multisite"
dataPath = "/Users/klaas/OneDrive/Documenten/Fysica/Masterproef/Code_Klaas/Data/"

# --- 2. Define Parameters ---
# Adler dependence system parameters are defined here
J = 1.0e-1                      # coupling parameter [meV]
γ = 2.0e-3                      # loss rate [meV]
ΔV = 0.015                     # potential difference between outer sites [meV]
sites = 10                       # number of sites to simulate
V_sites = LinRange(0, ΔV, sites)  # linear potential across sites

# Simulation parameters
save_ψ_per = 100                # eval_time steps between saving all field data
savesteps = 1.0e6               # number of samples taken during averaging
saveDuration = 1e5/J            # total duration of simulation [s]
dt = 1.0e-4/J                   # time step for integration algorithm [s]

# Simulation time
t_begin = 1.0e5/J               # time to reach steady state, simulation time before sampling [s]
t_end = t_begin + saveDuration  # end time of simulation [s]
t_span = (0, t_end)             # time span of simulation [s]
save_Δt = saveDuration/savesteps    # time step between observable sampling [s]
eval_time = t_end-saveDuration:save_Δt:t_end    # sampling time vector

# System parameters
n_avg = 3.0e3                   # target number of photons per site
M = 1.0e8                       # total number of dye molecules per site
B_21 = 1.0e-6                   # Einstein coefficient
Δ = -60                         # detuning
T = 25.0                        # temperature
κ = 0.5*B_21*M*exp(Δ/T)/T       # relaxation rate
M_2_avg = M*exp(Δ/T)            # average number of excited molecules per site

# Adjust potential and pump shapes
V = zeros(Float64, sites)       # potential shape
ρ = ones(Float64, sites)        # pump shape
V = collect(V_sites)            # set potential for all sites

# Pack parameters into struct
par_str = CFPars(J, sites, n_avg, M, B_21, Δ, γ, T, κ, V, ρ, save_Δt)

# --- 3. Set up and Run Simulation ---
# Calculate steady state values or critical limits
ss_state = calculate_steady_state_multisite(sites, J, γ, V, B_21, M, T, Δ, κ, n_avg)

# Calculate ΔV stability threshold
time_stability_start = time()
ΔV_max_stability, ΔV_range, max_real_eigenvals, crossing_matrix_A, crossing_eigenvecs_A = calculate_critical_ΔV_multisite(sites, J, γ, B_21, M, T, Δ, κ, n_avg)
time_stability_end = time()

# Calculate absolute phases from phase differences
phase = zeros(Float64, sites)
for i in 2:sites
    phase[i] = phase[i-1] + ss_state.φ[i-1]
end

# Initial conditions
u_begin = zeros(ComplexF64, 2*sites)
# Use steady state populations for initial conditions
if ss_state.converged
    for i in 1:sites
        u_begin[i] = sqrt(ss_state.n[i]) * exp(im*phase[i])
    end
    u_begin[sites+1:2*sites] .= ss_state.M_2 .+ ss_state.n
else
    println("Warning: Steady state did not converge, using default initial conditions.")
    u_begin[1:sites] .= sqrt(par_str.n_avg)
    u_begin[sites+1:2*sites] .= M_2_avg + par_str.n_avg
end

# Technical parameter vector
tech_param = (u_begin, t_span, dt, eval_time, save_ψ_per, save_Δt)

# Pack all parameters into p vector
p = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ)

# Set up ODE problem
prob = ODEProblem(classicalField_MultiSite_shift, u_begin, t_span, p)
sol = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false)); maxiters=1e10, reltol=1e-6, abstol=1e-8, saveat=eval_time[1:save_ψ_per:end], dense=false)

# Extract data from solution
o = extractSolDataX(sol, par_str)

# --- 4. Save data to file ---
filename = dataPath*operatingFile
jldsave(filename*".jld2"; X_data=o[o_dict["X_data"]], t_vec=o[o_dict["t_vec"]], parameters=par_str, tech_parameters=tech_param)

# --- 5. Load data and define parameters ---
# Load the data into a struct
sim_data = load_data_multisite("Data/Klaas_Data_multisite.jld2")

n = sim_data.n
φ = sim_data.φ

# Fourier transform of photon number and phase difference
n_FT = zeros(ComplexF64, sites, length(sim_data.t_vec))
φ_FT = zeros(ComplexF64, sites-1, length(sim_data.t_vec))
for i in 1:sites
    n_FT[i, :] = fft(sim_data.n[i, :])
end
for i in 1:sites-1
    φ_FT[i, :] = fft(sim_data.φ[i, :])
end

# --- 6. Plot results ---
plot_results_multisite(sim_data, ss_state, ΔV_range, max_real_eigenvals, crossing_matrix_A, crossing_eigenvecs_A, n_FT, φ_FT, save_path="Figures/ClassicalField_X_multisite.png")

# --- 7. Print summary statistics ---
print_stats_multisite(sim_data, ss_state, ΔV_max_stability)

elapsed = time() - t0
println("Total time: ", elapsed, " seconds.")
println("ΔV stability calculation time: ", time_stability_end - time_stability_start, " seconds.")
