# ==============================================================================
# UNIFIED CLASSICAL FIELD SIMULATION (2-sites & N-sites, Deterministic & Stochastic)
# ==============================================================================
using DifferentialEquations, LinearAlgebra, JLD2, Printf


# 1. Setup
include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../ExtraFunctions/Analytics.jl")


# --- Run Configuration ---
run_name = "sites_2_deterministic"  # Unique identifier for this simulation run
dataPath = "ClassicalField/Data/"
mkpath(dataPath)

# --- Define Parameters ---
sites = 2                      # 2 for 2-site, >2 for N-Site
noise_strength = 0              # 0.0 = Deterministic, >0.0 = Stochastic, 1.0 = Full Noise

# Potential difference
ΔV = 0.01                      # total potential difference between extreme sites [meV]

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
