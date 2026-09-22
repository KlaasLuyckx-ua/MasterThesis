t0 = time()

# --- 1. Import Packages, include Functions and set dataPath ---
using DifferentialEquations, LinearAlgebra, NPZ, JLD2, Distributed, CairoMakie, GLMakie, LaTeXStrings, Integrals, Trapz, Printf

include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../Functions_Multisite/Analytics_Multisite.jl")
include("../Functions_Multisite/Output_Multisite.jl")

operatingFile = "Klaas_Data_multisite_noise"
dataPath = "/Users/klaas/OneDrive/Documenten/Fysica/Masterproef/Code_Klaas/Data/"

# --- 2. Define Parameters ---
# Adler dependence system parameters are defined here
J = 1.0e-1                      # coupling parameter [meV]
γ = 2.0e-3                      # loss rate [meV]
ΔV = 0.1                        # potential difference between sites [meV]
sites = 3                       # number of sites to simulate
V_sites = LinRange(0, ΔV, sites)  # linear potential across sites

# Simulation parameters
save_ψ_per = 100                # eval_time steps between saving all field data
savesteps = 1.0e6               # number of samples taken during averaging
saveDuration = 1e4/J            # total duration of simulation [s]
dt = 1.0e-4/J                   # time step for integration algorithm [s]

# Simulation time
t_begin = 1e3/J                   # time to reach steady state, simulation time before sampling [s]
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
V = collect(V_sites)            # set potential for all sites

# Pack parameters into struct
par_str = CFPars(J, sites, n_avg, M, B_21, Δ, γ, T, κ, V, ρ, save_Δt)

# --- 3. Set up and Run Simulation ---
# Calculate steady state values or critical limits
ss_result = calculate_steady_state_multisite(sites, J, γ, V, B_21, M, T, Δ, κ, n_avg)
# Calculate ΔV stability limit
# ΔV_max_stability, plot_ΔV_range, plot_real_eigenvals, plot_imag_eigenvals, plot_max_real_eigenvals, matrix_sequence, eigenvecs= calculate_critical_ΔV_multisite(J,γ, B_21, M, T, Δ, κ, n_avg)

# Calculate absolute phases from phase differences
phase = zeros(Float64, sites)
for i in 2:sites
    phase[i] = phase[i-1] + ss_result.φ[i-1]
end

# --- Set Initial Conditions ---
u_begin = zeros(ComplexF64, 2*sites)

# Use steady state populations for initial conditions
if ss_result.converged
    for i in 1:sites
        u_begin[i] = sqrt(ss_result.n[i]) * exp(im*phase[i])
    end
    u_begin[sites+1:2*sites] .= ss_result.M_2 .+ ss_result.n
else
    println("Warning: Steady state did not converge, using default initial conditions.")
    u_begin[1:sites] .= sqrt(par_str.n_avg)
    u_begin[sites+1:2*sites] .= M_2_avg + par_str.n_avg
end

# Technical parameter vector
tech_param = (u_begin, t_span, dt, eval_time, save_ψ_per, save_Δt)

# Pack all parameters into p vector
p = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ)

# Set up SDE problem
prob = SDEProblem(classicalField_MultiSite_shift, σ_classicalField_MultiSite_add, u_begin, t_span, p)
sol = solve(prob, EM(); dt=dt, saveat=eval_time[1:save_ψ_per:end])

# Extract data from solution
o = extractSolDataX(sol, par_str)

# --- 4. Save data to file ---
filename = dataPath*operatingFile
jldsave(filename*".jld2"; X_data=o[o_dict["X_data"]], t_vec=o[o_dict["t_vec"]], parameters=par_str, tech_parameters=tech_param)

# --- 5. Load data and define parameters for plotting ---
# Call the function from the external script
sim_data = load_data_JJ("Data/Klaas_Data_multisite_noise.jld2")

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
φ = zeros(Float64, sites-1, len)

# Calculate values
for i in 1:sites
    n[i,:] = abs2.(X_data[i,:])
end

for i in 1:sites
    M_2[i,:] = real.(X_data[sites+i,:]) .- n[i,:]
end

for i in 1:sites
    M_1[i,:] = M .- M_2[i,:]
end

for i in 1:sites
    theta[i,:] = angle.(X_data[i,:])
end

for i in 1:sites-1
    φ[i,:] = angle.(conj.(X_data[i,:]) .* X_data[i+1,:])
end

# Calculate Fourier transform of n and φ for plotting
n_FT = fft(n)
φ_FT = fft(φ)

# --- 6. Plot results ---
CairoMakie.activate!()
f1 = Figure(size=(1200, 800))

ax1 = Axis(f1[1, 1], title="Photon Number Time Series", xlabel="Time (1/meV)", ylabel="Photon Number")
for i in 1:sites
    lines!(ax1, t_vec*J, n[i,:], label="Site $i")
end

ax2 = Axis(f1[1, 2], title="Phase Difference Time Series", xlabel="Time (1/meV)", ylabel="Phase Difference (rad)")
for i in 1:sites-1
    lines!(ax2, t_vec*J, φ[i,:], label="Phase Difference Site $i - Site $(i+1)")
end

ax3 = Axis(f1[2, 1], title="Excited Molecule Time Series", xlabel="Time (1/meV)", ylabel="Number of Excited Molecules")
for i in 1:sites
    lines!(ax3, t_vec*J, M_2[i,:], label="Site $i")
end

ax4 = Axis(f1[2, 2], title="Phase Difference FT", xlabel="Frequency (meV)", ylabel="FT of Phase Difference")
for i in 1:sites-1
    lines!(ax4, abs.(φ_FT[i, 2:end]), label="|FT(Phase Difference Site $i - Site $(i+1))|")
end

ax5 = Axis(f1[3, 1], title="Photon Number FT", xlabel="Frequency (meV)", ylabel="FT of Photon Number")
for i in 1:sites
    lines!(ax5, abs.(n_FT[i, 2:end]), label="|FT(Photon Number Site $i)|")
end

display(f1)

elapsed = time() - t0
println("Total Simulation Time: $(elapsed) seconds")