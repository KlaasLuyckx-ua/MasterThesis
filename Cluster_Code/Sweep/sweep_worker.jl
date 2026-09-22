using DifferentialEquations, Printf, JLD2, DSP, FFTW, LinearAlgebra, Statistics

include("../Analytics/Simulation_functions.jl")
include("../Analytics/Structs.jl")
include("../Analytics/Analytics.jl") 

# 1. Parse Command Line Arguments from the Cluster

task_id = parse(Int, ARGS[1])
file_name = ARGS[2]
param_string = ARGS[3]

# --- PARSE THE BASH STRING INTO A DICTIONARY ---
params = Dict{String, Float64}()
for pair in split(param_string, ",")
    key, val = split(pair, "=")
    params[strip(key)] = parse(Float64, strip(val))
end

# --- Define Base Parameters ---
# Safely extract sites and δ. Defaults to 2 sites and 0 asymmetry if omitted.
sites = haskey(params, "sites") ? Int(params["sites"]) : 2
δ = haskey(params, "δ") ? params["δ"] : 0.0

J = params["J"]
γ = params["γ"]
noise_strength = params["noise_strength"]
N = Int(params["N"])

dt = 1.0e-5/J
t_begin = params["t_begin_factor"]/J

n_avg = params["n_avg"]
M = params["M"]
B_21 = params["B_21"]
Δ = params["Δ"]
T = params["T"]
κ = 0.5*B_21*M*exp(Δ/T)/T
M_2_avg = M*exp(Δ/T)

ρ = ones(Float64, sites)
if sites == 2
    ρ[1] = 1.0 - δ
    ρ[2] = 1.0 + δ
end

# --- Calculate deterministic critical ΔV ---
# Uses the fully unified theoretical calculator
crit_result = calculate_critical_ΔV(sites, J=J, γ=γ, B_21=B_21, M=M, T=T, Δ=Δ, κ=κ, n_avg=n_avg, δ=δ)
# Fallback scaling if critical_ΔV isn't found (e.g. no crossing exists)
ΔV_crit = isnothing(crit_result.critical_ΔV) ? 1.0 : crit_result.critical_ΔV

# --- Define sweep range ---
ΔV_range = range(0.0 * ΔV_crit, 2.0 * ΔV_crit, length=N)
current_ΔV = ΔV_range[task_id]

println("Worker $(task_id) starting: Total ΔV = $(current_ΔV) meV across $sites sites")

# Generate linear potential drop across all sites
# For 2 sites, this correctly creates: [0.0, current_ΔV]
V_current = collect(LinRange(0.0, current_ΔV, sites))

# --- Calculate Initial Conditions ---
ss_result = calculate_steady_state(sites, J=J, γ=γ, V=V_current, B_21=B_21, M=M, T=T, Δ=Δ, κ=κ, n_avg=n_avg, δ=δ)

u_begin = zeros(ComplexF64, 2*sites)

if ss_result.converged
    # Calculate absolute phases from phase differences
    phase = zeros(Float64, sites)
    for i in 2:sites
        phase[i] = phase[i-1] + ss_result.φ[i-1]
    end
    
    # Apply steady state ICs
    for i in 1:sites
        u_begin[i] = sqrt(ss_result.n[i]) * exp(im*phase[i])
        u_begin[sites+i] = ss_result.M_2[i] + ss_result.n[i]
    end
else
    # Fallback to default homogeneous state
    u_begin[1:sites] .= sqrt(n_avg)
    u_begin[sites+1:2*sites] .= M_2_avg + n_avg
end

# Pack parameters for the SDE
p_current = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V_current, ρ, noise_strength)

# ==============================================================================
# PHASE UNWRAPPER CALLBACK SETUP
# ==============================================================================
mutable struct PhaseUnwrapper
    last_phi::Float64
    unwrapped_phi::Float64
    initialized::Bool
    t_begin::Float64
end

function (pu::PhaseUnwrapper)(integrator)
    if integrator.t >= pu.t_begin
        # Measure global phase winding (Site 1 to Site N)
        # This explicitly covers the 2-site limit perfectly well!
        phi = angle(conj(integrator.u[1]) * integrator.u[sites]) 
        if !pu.initialized
            pu.last_phi = phi
            pu.unwrapped_phi = phi
            pu.initialized = true
        else
            dphi = phi - pu.last_phi
            pu.unwrapped_phi += rem2pi(dphi, RoundNearest) 
            pu.last_phi = phi
        end
    end
    u_modified!(integrator, false) 
end

# ==============================================================================
# RUN SIMULATION
# ==============================================================================
saveDuration = params["saveDuration_factor"] / J
savesteps = 1.0e6
save_Δt = saveDuration / savesteps

t_end = t_begin + saveDuration
t_span = (0, t_end)
eval_time = (t_end - saveDuration):save_Δt:t_end

println("--> Running duration: $saveDuration")

unwrapper = PhaseUnwrapper(0.0, 0.0, false, t_begin)
cb = DiscreteCallback((u, t, integrator) -> true, unwrapper, save_positions=(false, false))

# Router: Deterministic vs Stochastic
if noise_strength == 0.0
    println("\n--- Running DETERMINISTIC Solver (ODE) ---")
    prob_current = ODEProblem(classicalField, u_begin, t_span, p_current)
    sol = solve(prob_current, AutoTsit5(Rosenbrock23(autodiff=false)); 
                maxiters=1e10, reltol=1e-6, abstol=1e-8, 
                saveat=eval_time, dense=false, callback=cb)
else
    println("\n--- Running STOCHASTIC Solver (SDE) ---")
    prob_current = SDEProblem(classicalField, σ_classicalField_add, u_begin, t_span, p_current)
    sol = solve(prob_current, SOSRA(); 
                dt=dt, maxiters=1e10, 
                saveat=eval_time, callback=cb)
end

# ==============================================================================
# PROCESS DATA
# ==============================================================================
X_data = Array(sol)

# Extract first and last site metrics for boundary comparisons
ψ_1 = X_data[1, :]
ψ_L = X_data[sites, :]
n1 = abs2.(ψ_1)
nL = abs2.(ψ_L)

# 1. Phase Velocity (Global drop)
v_φ = unwrapper.unwrapped_phi / (sol.t[end] - unwrapper.t_begin)

# 2. Excited State Population (n_exc)
# Construct the NxN tight-binding Hamiltonian
H = zeros(Float64, sites, sites)
for i in 1:sites
    H[i, i] = V_current[i]
end
for i in 1:sites-1
    H[i, i+1] = -J
    H[i+1, i] = -J
end

vals, vecs = eigen(H)
idx_ex = argmax(real.(vals))
v_ex = vecs[:, idx_ex]

# Project the full chain state onto the highest energy eigenvector
ψ_ex = sum(conj(v_ex[i]) .* X_data[i, :] for i in 1:sites)
n_exc = mean(abs2.(ψ_ex))

# 3. Average Population Imbalance (Site 1 vs Last Site)
avg_dn = mean(n1 .- nL)

# 4. Dominant Frequency (Site 1)
X1_AC = ψ_1 .- mean(ψ_1)
X1_FT = abs.(fft(X1_AC))
fs = 1.0 / save_Δt
freqs = fftfreq(length(X1_AC), fs)
pos_idx = freqs .> 0
dom_freq = freqs[pos_idx][argmax(X1_FT[pos_idx])]

# 5. Equal-Time Coherence (Across the entire chain: Site 1 to Site N)
cross_term = mean(conj.(ψ_1) .* ψ_L)
I1 = mean(abs2.(ψ_1))
IL = mean(abs2.(ψ_L))
equal_time_coherence = abs(cross_term) / sqrt(I1 * IL)

# ==============================================================================
# SAVE DATA
# ==============================================================================
output_dir = "Sweep/Data/$file_name"
mkpath(output_dir)
filename = joinpath(output_dir, "result_$(task_id).jld2")

jldsave(filename; 
        parameters=(sites=sites, J=J, γ=γ, B_21=B_21, M=M, T=T, Δ=Δ, n_avg=n_avg, κ=κ, δ=δ),
        dV=current_ΔV, 
        n_exc=n_exc, 
        avg_dn=avg_dn, 
        dom_freq=dom_freq, 
        v_phi=v_φ,
        equal_time_coh=equal_time_coherence)

println("    Successfully saved all metrics for Total ΔV = $current_ΔV")
println("Worker $(task_id) completely finished.")