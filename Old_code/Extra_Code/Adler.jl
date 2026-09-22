using CairoMakie, Roots
J = 1.0e-1                   # coupling parameter [meV]
γ = 2.0e-2*J                # loss rate [meV]
B_21 = 1e-7               # stimulated emission coefficient [meV]
M = 1e8                    # number of molecules per site
T = 25.0                   # temperature [meV]
Δ = -60                     # detuning [meV]
κ = 0.5*B_21*M*exp(Δ/T)/T       # relaxation rate [meV]
n_avg = 3.0e3              # average photon number per site
φ = range(0, 10π, length=10000) # phase vector
Vl = 0.405                      # potential at left site [meV]
Vr = 0.0                        # potential at right site [meV]
ΔV = Vr - Vl              # potential difference between sites [meV]

α = 2*J/γ*sin.(φ)                  # coherence parameter
nl = (1 .-sqrt.(α.^2 ./(1 .+α.^2))).*n_avg # left site photon number
nr = (1 .+sqrt.(α.^2 ./(1 .+α.^2))).*n_avg # right site photon number
A = sqrt.(nl./nr).-sqrt.(nr./nl)
B = sqrt.(nl./nr).+sqrt.(nr./nl)
φ_dot = A.*cos.(φ).-κ*B.*sin.(φ).-ΔV/(J)


fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = L"\phi", ylabel = L"\dot{\phi} \ \mathrm{(meV)}", title = "Phase Space Dynamics")
lines!(ax, φ, φ_dot, color = :blue)
lines!(ax, φ, zeros(length(φ)), color = :red)
display(fig)



φ = range(0, 2π, length=10000)
