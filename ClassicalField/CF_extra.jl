# ==============================================================================
# UNIFIED CLASSICAL FIELD SIMULATION (2-sites & N-sites, Deterministic & Stochastic)
# ==============================================================================



using DifferentialEquations, LinearAlgebra, JLD2, Printf


# 1. Setup
include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../ExtraFunctions/Analytics.jl")


# --- Run Configuration ---
run_name = "sites_15_0.08"  # Unique identifier for this simulation run
dataPath = "ClassicalField/Data/"
mkpath(dataPath)

# --- Define Parameters ---
sites = 15                      # 2 for 2-site, >2 for N-Site
noise_strength = 0              # 0.0 = Deterministic, >0.0 = Stochastic, 1.0 = Full Noise

# Potential difference
ΔV = 0.08                      # total potential difference between extreme sites [meV]

# System parameters
J = 1.0e0                      # coupling parameter [meV]
γ = 1.0e-3                      # loss rate [meV]
δ = 0.0                         # pump imbalance parameter (Used for N=2)
n_avg = 1.0e4                   # target number of photons per site
M = 1.0e8                       # total number of dye molecules per site
B_21 = 1.0e-7                   # Einstein coefficient
Δ = -60.0                       # detuning
T = 25.0                        # temperature

κ = 0.5*B_21*M*exp(Δ/T)/T       # relaxation rate
M_2_avg = M*exp(Δ/T)            # average number of excited molecules per site

# Simulation parameters
save_ψ_per = 100                # steps between saving all field data
savesteps = 1.0e6               # number of samples taken during averaging
saveDuration = 1e5/J            # total duration of simulation [1/meV]
dt = 1.0e-4/J                   # time step for integration algorithm [1/meV]

# Simulation time
t_begin = 1e5/J                 # time to reach steady state before sampling [1/meV]
t_end = t_begin + saveDuration  # total simulation time [1/meV]
t_span = (0, t_end)             # time span for integration
save_Δt = saveDuration/savesteps    # time step between saving observables [1/meV]
eval_time = t_end-saveDuration:save_Δt:t_end    # time points at which to save data



# Adjust potential and pump shapes
V = collect(LinRange(0.0, ΔV, sites))   
ρ = ones(Float64, sites)
if sites == 2
    ρ[1] = 1.0 - δ
    ρ[2] = 1.0 + δ
end

# Pack parameters into a struct for saving and a tuple for DE solvers
par_str = CFPars(J, sites, n_avg, M, B_21, Δ, γ, T, κ, V, ρ, save_Δt)


# 2. Theoretical Calculations
println("\n--- Calculating Theoretical Properties ---")
# Calculate steady state and critical ΔV for stability using the analytics functions. 
# These will be used to compare against the simulation results and to set initial conditions for the DE solvers.
ss_result = calculate_steady_state(sites; J=J, γ=γ, V=V, B_21=B_21, M=M, T=T, Δ=Δ, κ=κ, n_avg=n_avg, δ=δ)
crit_result = calculate_critical_ΔV(sites; J=J, γ=γ, B_21=B_21, M=M, T=T, Δ=Δ, κ=κ, n_avg=n_avg, δ=δ)

stability_limit = isnothing(crit_result.critical_ΔV) ? NaN : crit_result.critical_ΔV
println("Steady State Converged: ", ss_result.converged)
println("Calculated ΔV Stability Limit: ", stability_limit, " meV")


# 3. Simulation Setup & Run
println("\n--- Setting up Initial Conditions ---")
u_begin = zeros(ComplexF64, 2*sites)

if ss_result.converged
    # Use the calculated steady state values to initialize the fields.
    phase = zeros(Float64, sites)
    for i in 2:sites
        phase[i] = phase[i-1] + ss_result.φ[i-1]
    end
    
    for i in 1:sites
        u_begin[i] = sqrt(ss_result.n[i]) * exp(im*phase[i])
        u_begin[sites+i] = ss_result.M_2[i] + ss_result.n[i]
    end
else
    # If the steady state did not converge, use a fallback homogeneous state as the initial condition.
    println("Warning: Steady state did not converge. Using fallback homogeneous state.")
    u_begin[1:sites] .= sqrt(n_avg)
    u_begin[sites+1:2*sites] .= M_2_avg + n_avg
end

# Pack physical parameters for DE
p = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength)

# Router: Deterministic vs Stochastic
# Run the appropriate solver based on the noise strength parameter. If noise_strength is zero, we run a deterministic ODE solver;
# if it's greater than zero, we run a stochastic SDE solver with the defined noise term.
if noise_strength == 0.0
    println("\n--- Running DETERMINISTIC Solver (ODE) ---")
    prob = ODEProblem(classicalField, u_begin, t_span, p)
    sol = solve(prob, AutoTsit5(Rosenbrock23(autodiff=false)); maxiters=1e10, reltol=1e-6, abstol=1e-8, saveat=eval_time[1:save_ψ_per:end], dense=false)
else
    println("\n--- Running STOCHASTIC Solver (SDE) ---")
    prob = SDEProblem(classicalField, σ_classicalField_add, u_begin, t_span, p)
    sol = solve(prob, SOSRA(); dt=dt, maxiters=1e10, saveat=eval_time[1:save_ψ_per:end])
end

# Extract the time vector and field data from the solution for saving and analysis.
o_t_vec = sol.t
o_X_mat = zeros(ComplexF64, 2*sites, length(o_t_vec))
o_X_mat[:,:] .= Array(sol)


# 4. Save Data
println("\n--- Saving Data ---")
filename = joinpath(dataPath, run_name * ".jld2")
jldsave(filename; 
        X_data = o_X_mat, 
        t_vec = o_t_vec, 
        parameters = par_str,
        ss_result = ss_result,
        crit_result = crit_result,
        noise_strength = noise_strength,
        ΔV = ΔV)
println("Data saved to: $filename")

















using JLD2, CairoMakie, GLMakie, Printf, Statistics, FFTW

include("../ExtraFunctions/Analytics.jl")
include("../ExtraFunctions/Structs.jl")
include("../ExtraFunctions/Simulation_functions.jl")

# --- Run Configuration ---
filepath = "ClassicalField/Data/$(run_name).jld2"
output_dir = "ClassicalField/Plots/sites_15"
mkpath(output_dir)

# =========================================================================
# 1. Load and Process Data
# =========================================================================
println("Loading data from: $filepath")
data = load(filepath)

# Extract Data
t_vec = data["t_vec"]::Vector{Float64}
X_data = data["X_data"]::Matrix{ComplexF64}
pars = data["parameters"]


# Extract Parameters
sites = pars.sites


len = length(t_vec)
avg_range = floor(Int, 0.5 * len):len  # Use the second half of the data for averaging to ensure we are in the steady state regime

φ = zeros(Float64, sites-1, len)
for i in 1:sites-1
    φ[i,:] = angle.(conj.(X_data[i,:]) .* X_data[i+1,:])
end


# Calculate time-averaged frequency (∂_t θ) for each site
mean_freq = zeros(Float64, sites)

for i in 1:sites
    # Using angle(X_t* * X_{t+1}) to cleanly calculate Δθ without phase wrapping issues
    phase_diffs = angle.(conj.(X_data[i, avg_range[1:end-1]]) .* X_data[i, avg_range[2:end]])
    
    # Frequency is the mean phase difference divided by the time step
    mean_freq[i] = mean(phase_diffs) / save_Δt
end

println("Data processing complete.")


# =========================================================================
# 3. Generating Plots
# =========================================================================
CairoMakie.activate!()

f1 = Figure(size = (800, 600))

# Plot: Time Series of Phase Differences (φ_j)
ax1 = Axis(f1[1, 1],
    title="Tijdreeks Faseverschil (φ_j) \nbij ΔV = $(round(ΔV, sigdigits=3)) meV", 
    xlabel="Tijd (1/meV)", 
    ylabel="Faseverschil (rad)", 
    ytickformat = values -> ["$(round(v, sigdigits=5))" for v in values],
    titlesize=30, 
    xlabelsize=25, 
    ylabelsize=25, 
    xticklabelsize=20, 
    yticklabelsize=20
    )
for i in 1:sites-1
    lines!(ax1, t_vec, φ[i,:], label="φ_$(i)", linewidth=2)
end
display(f1)
plot_file = joinpath(output_dir, "$(run_name)_phase_diffs.png")
save(plot_file, f1)

f2 = Figure(size = (800, 600))

# Plot: Time-Averaged Frequency Distribution (∂_t θ)
ax = Axis(f2[1, 1],
    title="Tijdgemiddelde Frequentie (∂_t θ_j) \nbij ΔV = $(round(ΔV, sigdigits=3)) meV", 
    xlabel="Site", 
    ylabel="Frequentie",
    # Use ytickformat to limit the significant digits (e.g., to 7 sig digits)
    ytickformat = values -> ["$(round(v, sigdigits=5))" for v in values],
    # Slightly reduced titlesize to ensure it fits well within the 800x600 resolution
    titlesize=30, 
    xlabelsize=25, 
    ylabelsize=25, 
    xticklabelsize=20, 
    yticklabelsize=20,
    limits = (nothing, nothing, -10, -2)  # Set x-limits to center the scatter points
    )
for i in 1:sites
    scatter!(ax, [i], [mean_freq[i]], label="Frequentie (Site $i)", markersize=20, color=:orange)
end
display(f2)
plot_file = joinpath(output_dir, "$(run_name)_freqs.png")
save(plot_file, f2)



f_3 = Figure(size = (800, 600))
# Plot: Time-averaged phase differences (φ_j) as a function of site index
ax3 = Axis(f_3[1, 1],
    title="Tijdgemiddelde Faseverschil (φ_j) \nbij ΔV = $(round(ΔV, sigdigits=3)) meV", 
    xlabel="Site Index j", 
    ylabel="Tijdgemiddeld Faseverschil (rad)", 
    xticks=1:sites-1, 
    ytickformat = values -> ["$(round(v, sigdigits=5))" for v in values],
    titlesize=30, 
    xlabelsize=25, 
    ylabelsize=25, 
    xticklabelsize=20, 
    yticklabelsize=20
    )
for i in 1:sites-1
    scatter!(ax3, [i], [mean(φ[i, avg_range])], label="Tijdgemiddeld φ_$(i)", markersize=20, color=:blue)
end
display(f_3)
plot_file = joinpath(output_dir, "$(run_name)_avg_phase_diffs.png")
save(plot_file, f_3)