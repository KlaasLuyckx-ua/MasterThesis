using CairoMakie
using LinearAlgebra
include("../ExtraFunctions/Analytics_functions_JJ.jl")

# --- 1. Define Parameters ---
J = 1e0                        # coupling parameter [meV]
γ = 2.0e-2*J                    # loss rate [meV]
# Vl and Vr will be defined in the loop
B_21 = 1e-7                     # stimulated emission coefficient [meV]
M = 1e8                         # number of molecules per site
T = 25.0                        # temperature [meV]
Δ = -60.0                       # detuning [meV]
n_avg = 3.0e3                   # average photon number per site
# Calculate derived parameters
κ  = 0.5 * B_21 * M * exp(Δ/T) / T  # relaxation rate [meV]

# Loop over Vl values
Vl_range = 0.0:0.0001:10
real_eigenvalues = Vector{Vector{Float64}}()
max_real_eigenvalues = Vector{Float64}()

for V_val in Vl_range
     # Update Potentials
     Vl_loc = V_val*J
     Vr_loc = 0.0 * J

     # --- 2. Define Steady State parameters ---
     _, φ_loc, nl_loc, nr_loc, M_2_l_loc, M_2_r_loc = calculate_steady_state(J, γ, Vl_loc, Vr_loc, B_21, M, T, Δ, n_avg)

     # --- 3. Define derivative matrix ---
     # We define the matrix that gives the matrix equation dX/dt = A*X
     # where X = [δφ_loc, δnl_loc, δnr_loc, δM_2_l_loc, δM_2_r_loc]
     # The elements of A are derived from linearizing the system around the steady state
     # We define each element of the matrix A seperately by
     # A = [a,b,c,d,e;
     #      f,g,h,i,j;
     #      k,l,m,n,o;
     #      p,q,r,s,t;
     #      u,v,w,x,y]
     a_loc = -(J*(sqrt(nl_loc/nr_loc) - sqrt(nr_loc/nl_loc))*sin(φ_loc)+J*κ*(sqrt(nl_loc/nr_loc) + sqrt(nr_loc/nl_loc))*cos(φ_loc))
     b_loc = (J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*cos(φ_loc) - J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nr_loc/nl_loc)*1/(2*nl_loc))*sin(φ_loc))
     c_loc = -(J*(1/(2*sqrt(nl_loc*nr_loc)) + sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*cos(φ_loc) + J*κ*(1/(2*sqrt(nl_loc*nr_loc)) - sqrt(nl_loc/nr_loc)*1/(2*nr_loc))*sin(φ_loc))
     d_loc = 0
     e_loc = 0

     f_loc = -(2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)+cos(φ_loc)))
     g_loc = (J*sqrt(nr_loc/nl_loc)*(κ*cos(φ_loc)-sin(φ_loc)) + B_21*((1+exp(Δ/T))*M_2_l_loc - exp(Δ/T)*M)-γ-2*κ*Vl_loc)
     h_loc = J*sqrt(nl_loc/nr_loc)*(κ*cos(φ_loc)-sin(φ_loc))
     i_loc = B_21*(1+exp(Δ/T))*nl_loc
     j_loc = 0

     k_loc = -2*J*sqrt(nl_loc*nr_loc)*(κ*sin(φ_loc)-cos(φ_loc))
     l_loc = J*sqrt(nr_loc/nl_loc)*(sin(φ_loc)+κ*cos(φ_loc))
     m_loc = (J*sqrt(nl_loc/nr_loc)*(sin(φ_loc)+κ*cos(φ_loc)) + B_21*((1+exp(Δ/T))*M_2_r_loc - exp(Δ/T)*M)-γ-2*κ*Vr_loc)
     n_loc = 0
     o_loc = B_21*(1+exp(Δ/T))*nr_loc

     p_loc = 2*J*κ*sqrt(nl_loc*nr_loc)*sin(φ_loc)
     q_loc = -(B_21*((1+exp(Δ/T))*M_2_l_loc-exp(Δ/T)*M) + J*κ*sqrt(nr_loc/nl_loc)*cos(φ_loc) - 2*κ*Vl_loc)
     r_loc = -J*κ*sqrt(nl_loc/nr_loc)*cos(φ_loc)
     s_loc = -B_21*(1+exp(Δ/T))*nl_loc
     t_loc = 0

     u_loc = 2*J*κ*sqrt(nl_loc*nr_loc)*sin(φ_loc)
     v_loc = -J*κ*sqrt(nr_loc/nl_loc)*cos(φ_loc)
     w_loc = -(B_21*((1+exp(Δ/T))*M_2_r_loc-exp(Δ/T)*M) + J*κ*sqrt(nl_loc/nr_loc)*cos(φ_loc) - 2*κ*Vr_loc)
     x_loc = 0
     y_loc = -B_21*(1+exp(Δ/T))*nr_loc

     A_loc = [a_loc b_loc c_loc d_loc e_loc;
          f_loc g_loc h_loc i_loc j_loc;
          k_loc l_loc m_loc n_loc o_loc;
          p_loc q_loc r_loc s_loc t_loc;
          u_loc v_loc w_loc x_loc y_loc]

     # --- 4. Calculate Eigenvalues ---
     eigenvals = eigvals(A_loc)
     push!(real_eigenvalues, real.(eigenvals))
     push!(max_real_eigenvalues, maximum(real.(eigenvals)))
end

# --- 5. Plot Results ---
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1],
     title="Stability Analysis vs Left Site Potential Vl",
     xlabel="Left Site Potential Vl [meV]",
     ylabel="Real Eigenvalues",
     limits = (nothing, nothing, -0.0005, 0.0001))
for i in 1:5
     lines!(
          ax,
          Vl_range,
          getindex.(real_eigenvalues, i),
          label="Real Eigenvalue $i")
end
lines!(
     ax,
     Vl_range,
     zeros(length(Vl_range)),
     color = :red,
     linestyle = :dash,
     label="Stability Threshold")
axislegend(position = :rb)
display(fig)

crossing_idx = findfirst(x -> x > 0, max_real_eigenvalues)
println("Critical Vl at instability onset: ", Vl_range[crossing_idx], "*J")