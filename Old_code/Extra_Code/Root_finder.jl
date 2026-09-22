using Roots, Optim

# Parameters
J = 1.0e-1                   # coupling parameter [meV]
γ = 2.0e-2 * J              # loss rate [meV]
ΔV = -1.0 * J              # potential difference between sites [meV]

B_21 = 1e-7                 # stimulated emission coefficient [meV]
M = 1e7                     # number of molecules per site
T = 25.0                    # temperature [meV]
Δ = -60.0                   # detuning [meV]
κ = 0.5 * B_21 * M * exp(Δ/T) / T  # relaxation rate [meV]
n_avg = 3.0e3               # average photon number per site


# Define the function for phi_dot (scalar version)
function dynamics(φ)
    # Coherence parameter
    α = 2*J/γ*sin(φ)
    
    # Calculate S factor for convenience
    S = sqrt(α^2/(1+α^2))
    
    # Site photon numbers
    nl = (1-S)*n_avg
    nr = (1+S)*n_avg
    
    # Coefficients A and B
    A = sqrt(nl/nr) - sqrt(nr/nl)
    B = sqrt(nl/nr) + sqrt(nr/nl)
    
    # Equation of motion for phase
    # φ_dot = A*cos(φ) - κ*B*sin(φ) - ΔV/J
    return A*cos(φ)-κ*B*sin(φ)
end

# Find the smallest root in the range [0, 0.1]
# We use a bracketing method since we know the root is small and positive
root = find_zero(φ -> dynamics(φ)-ΔV/J, 0.0)

println("The smallest root of φ_dot at ΔV=" , ΔV, " is: ", root)

φ = root

α = 2*J/γ*sin(φ)

S = sqrt(α^2/(1+α^2))

nl = (1-S)*n_avg
nr = (1+S)*n_avg

println("Left site photon number at root: ", nl)
println("Right site photon number at root: ", nr)

# Find the Maximum of φ_dot
# Since it finds the minimum, we minimize the NEGATIVE of the function.
res = optimize(φ -> -dynamics(φ), 0.0, 2π)

# --- 4. Extract and Print Results ---
max_val = -Optim.minimum(res)  # Negate back to get the maximum
max_loc = Optim.minimizer(res)

println("Maximum φ_dot value at ΔV=0: ", max_val)

ΔV_max = -max_val*J

root_max = find_zero(φ -> dynamics(φ)-ΔV_max/J, 0.0)

println("The smallest root of φ_dot at ΔV_max=" , ΔV_max, " is: ", root_max)

φ_max = root_max

α_max = 2*J/γ*sin(φ_max)

S_max = sqrt(α_max^2/(1+α_max^2))

nl_max = (1-S_max)*n_avg
nr_max = (1+S_max)*n_avg

println("Left site photon number at root for ΔV_max: ", nl_max)
println("Right site photon number at root for ΔV_max: ", nr_max)


φ = range(0, 10π, length=10000)
φ_dot = dynamics.(φ)
pot = ΔV / J

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = L"\phi", ylabel = L"\dot{\phi} \ \mathrm{(meV)}", title = "Phase Space Dynamics")
lines!(ax, φ, φ_dot, color = :blue)
lines!(ax, φ, ones(length(φ)) * pot, color = :red)
display(fig)