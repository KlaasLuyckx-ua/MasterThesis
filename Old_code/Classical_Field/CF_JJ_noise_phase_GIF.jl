# --- 1. Import Packages and Include Functions ---
using DifferentialEquations, CairoMakie, DSP, LaTeXStrings

# Include required dependencies for SDE solver and struct handling
include("../ExtraFunctions/Simulation_functions.jl")
include("../ExtraFunctions/Structs.jl")
include("../Functions_JJ/Analytics_JJ.jl")
include("../Functions_JJ/Output_JJ.jl")

# Directory for saving the GIF
dataPath = "Old_code/Classical_Field/Data/"
mkpath(dataPath)

# --- 2. Define Parameters ---
# System parameters
J = 1.0e-1                      # coupling parameter [meV]
γ = 1.0e-3                      # loss rate [meV]
ΔV = 2                        # potential difference between sites [meV]
Vl = 0*ΔV                       # left site potential [meV]
Vr = ΔV                         # right site potential [meV]
δ = 0.0                         # pump imbalance parameter
noise_strength = 1.0            # strength of the noise term

# Simulation timing parameters
sites = 2                       
save_ψ_per = 1                
savesteps = 200                 # Changed to 200 steps (gives 201 frames)

saveDuration = (200 / 30) / J   # Still ≈ 6.666 / J
dt = 1.0e-6/J                   
t_begin = 0# 1e4/J                 
t_end = t_begin + saveDuration  
t_span = (0, t_end)          

save_Δt = saveDuration/savesteps    
eval_time = t_end-saveDuration:save_Δt:t_end

# Dye/Photon parameters
n_avg = 1.0e4                   
M = 1.0e8                       
B_21 = 1.0e-7                   
Δ = -60                         
T = 25.0                        
κ = 0.5*B_21*M*exp(Δ/T)/T       
M_2_avg = M*exp(Δ/T)            

# Shape arrays
V = zeros(Float64, sites)       
ρ = ones(Float64, sites)        
V[1] = Vl                       
V[2] = Vr                       
ρ[1] = 1.0 - δ; ρ[2] = 1.0 + δ  

# Pack parameters into struct (needed for extraction function)
par_str = CFPars(J, sites, n_avg, M, B_21, Δ, γ, T, κ, V, ρ, save_Δt)
p = (sites, n_avg, M, B_21, κ, J, Δ, γ, T, V, ρ, noise_strength)

# --- 3. Set up Initial Conditions ---
ss_result = calculate_steady_state_JJ(J, γ, Vl, Vr, B_21, M, T, Δ, κ, n_avg, δ)

u_begin = zeros(ComplexF64, 2*sites)
if ss_result.status == "Locked"
    u_begin[1] = sqrt(ss_result.nl_final)
    u_begin[2] = sqrt(ss_result.nr_final)*exp(im*ss_result.φ_final)
    u_begin[sites+1] = ss_result.M_2_l + ss_result.nl_final
    u_begin[sites+2] = ss_result.M_2_r + ss_result.nr_final
else
    u_begin[1] = sqrt(n_avg)
    u_begin[2] = sqrt(n_avg)*exp(im*0.0)
    u_begin[sites+1] = M_2_avg + n_avg
    u_begin[sites+2] = M_2_avg + n_avg
end

# --- 4. Run SDE Simulation ---
println("\n--- Running SDE Simulation (ΔV = $(ΔV) meV) ---")
prob = SDEProblem(classicalField_JJ, σ_classicalField_JJ_add, u_begin, t_span, p)
sol = solve(prob, SOSRA(); dt=dt, saveat=eval_time[1:save_ψ_per:end], maxiters=1e10)

# Extract generated data
o = extractSolDataX(sol, par_str)
t_vec = o[o_dict["t_vec"]]
X_data = o[o_dict["X_data"]]

# --- 5. Extract and Process Phase Difference ---
φ = angle.(conj.(X_data[1,:]) .* X_data[2,:])
φ_unwrapped = unwrap(φ) 

# --- 6. Generate Phase Difference GIF ---
println("\n--- Generating Phase Difference GIF ---")

# Define the window you want to animate to match the 200/30 duration
slice_end_time = t_vec[end] * J
slice_start_time = slice_end_time - (200 / 30)

# Find corresponding indices
start_idx = argmin(abs.(t_vec .* J .- slice_start_time))
end_idx   = argmin(abs.(t_vec .* J .- slice_end_time))
frame_indices = start_idx:1:end_idx

# Setup the Figure and Axis
CairoMakie.activate!()
f_gif = Figure(size = (400, 400))
ax_gif = Axis(f_gif[1, 1], aspect = DataAspect(), 
            title = "Phase Difference Trajectory\nΔV = $(ΔV) meV",
            xlabel = "Re(exp(iϕ))", ylabel = "Im(exp(iϕ))")

xlims!(ax_gif, -1.2, 1.2)
ylims!(ax_gif, -1.2, 1.2)

# Draw static Unit Circle
θ_circ = range(0, 2π, length=100)
lines!(ax_gif, cos.(θ_circ), sin.(θ_circ), color = :gray, linestyle = :dash)

# Create Observables for Animation
time_index = Observable(start_idx)
trail_length = 200 

phasor_x = @lift([0.0, cos(φ_unwrapped[$time_index])])
phasor_y = @lift([0.0, sin(φ_unwrapped[$time_index])])

trail_x = @lift(cos.(φ_unwrapped[max(1, $time_index - trail_length):$time_index]))
trail_y = @lift(sin.(φ_unwrapped[max(1, $time_index - trail_length):$time_index]))

curr_x = @lift(cos(φ_unwrapped[$time_index]))
curr_y = @lift(sin(φ_unwrapped[$time_index]))

# Plot animated elements
lines!(ax_gif, trail_x, trail_y, color = (:blue, 0.4), linewidth = 3) 
lines!(ax_gif, phasor_x, phasor_y, color = :red, linewidth = 2)       
scatter!(ax_gif, curr_x, curr_y, color = :red, markersize = 12)       

time_text = @lift("t = $(round(t_vec[$time_index]*J-t_begin*J, digits=2))")
text!(ax_gif, 0.6, 1.05, text = time_text, align = (:left, :center))

# Record the GIF
gif_filename = dataPath * "Phase_Slice_dV_$(ΔV).gif"
playback_fps = 30 # Changed to 30 FPS

println("Rendering $(length(frame_indices)) frames at $(playback_fps) FPS...")

record(f_gif, gif_filename, frame_indices; framerate = playback_fps) do i
    time_index[] = i
end
