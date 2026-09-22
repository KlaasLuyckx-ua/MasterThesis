using NLsolve
using LinearAlgebra
using Printf

# --- 1. Parameters ---
# Defined exactly as before
J_val = 1.0e-1
params = (
    J = J_val,
    γ = 2.0e-2 * J_val,
    n_avg = 3.0e3,
    
    # Potentials: V1=0.1, V2=0, V3=0.1 (Symmetric)
    # ΔV12 = -0.1, ΔV23 = +0.1
    ΔV_12 = -1.0 * J_val, 
    
    # We only need one potential difference for the symmetric half
    κ = 0.5 * 1.0e-7 * 1.0e8 * exp(-60.0/25.0) / 25.0
)

# --- 2. System of Equations (Reduced) ---
# We solve for vector x with only 2 elements: [n2, φ]
function symmetric_equations!(F, x)
    n2 = x[1]
    φ  = x[2] # This is φ_12. We assume φ_23 = -φ_12

    p = params

    # --- A. Enforce Constraints ---
    
    # 1. Soft clamp for positive n2
    n2_eff = n2
    penalty = 0.0
    if n2 < 1e-3
        n2_eff = 1e-3
        penalty = (1e-3 - n2) * 1e6
    end

    # 2. Enforce Geometric Symmetry (n1 = n3)
    # Conservation: 2*n1 + n2 = 3*n_avg  =>  n1 = (3*n_avg - n2)/2
    n1_geometric = (3 * p.n_avg - n2_eff) / 2
    n3_geometric = n1_geometric # By definition

    # --- B. Calculate Physics from Variables ---
    
    # We define φ_12 = φ and φ_23 = -φ
    α12 = (2 * p.J / p.γ) * sin(φ)
    
    # We assume α23 follows the symmetry α23 = -α12 (since sin(-φ) = -sin(φ))
    # This matches your equations: n1 = n_avg - α12..., n3 = n_avg + α23...
    # If α23 = -α12, then n3 equation becomes: n3 = n_avg - α12...
    # Which perfectly matches n1 = n3.

    # Solve the quadratic for n1 physically
    # n1 = n_avg - α12 * sqrt(n1 * n2)
    # x^2 + (α12*sqrt(n2))*x - n_avg = 0
    disc = sqrt((α12 * sqrt(n2_eff))^2 + 4 * p.n_avg)
    sqrt_n1_phys = (-(α12 * sqrt(n2_eff)) + disc) / 2
    n1_phys = sqrt_n1_phys^2

    # --- C. Residuals ---

    # Residual 1: Consistency of n1
    # Does the n1 required by conservation (n1_geometric) match 
    # the n1 dictated by the update rule (n1_phys)?
    F[1] = (n1_geometric - n1_phys) / p.n_avg + penalty

    # Residual 2: Phase Equation (Site 1-2 Link)
    # If the solution is symmetric, solving Eq 1 solves Eq 2 automatically.
    
    # Ratios
    s12 = sqrt(n1_geometric / n2_eff)
    s21 = sqrt(n2_eff / n1_geometric)
    s32 = sqrt(n3_geometric / n2_eff) # Same as s12 in value
    
    # Note: φ_23 = -φ
    cos_phi = cos(φ)
    sin_phi = sin(φ)
    cos_phi23 = cos(-φ) # = cos(φ)
    sin_phi23 = sin(-φ) # = -sin(φ)

    term_A = p.J * (s12 - s21) * cos_phi
    term_B = -p.J * p.κ * (s12 + s21) * sin_phi
    term_C = p.J * s32 * (cos_phi23 + p.κ * sin_phi23)
    
    F[2] = (term_A + term_B + term_C - p.ΔV_12) / p.J
end

# --- 3. Initial Guess ---
# [n2, φ]
# We guess n2 is high (accumulation in well), φ is small positive.
x0 = [5000.0, 0.01] 

# --- 4. Run Solver ---
println("Solving symmetric system...")
result = nlsolve(symmetric_equations!, x0, ftol=1e-10)

if converged(result)
    sol = result.zero
    n2_sol = sol[1]
    φ_sol = sol[2]
    
    # Calculate dependent variables
    p = params
    n1_sol = (3 * p.n_avg - n2_sol) / 2
    n3_sol = n1_sol
    
    println("\n--- Symmetric Solution Found ---")
    @printf("n1 (= n3) : %.6f\n", n1_sol)
    @printf("n2        : %.6f\n", n2_sol)
    @printf("Sum       : %.6f\n", 2*n1_sol + n2_sol)
    println("------------------------------")
    @printf("φ_12      : %.6e rad\n", φ_sol)
    @printf("φ_23      : %.6e rad\n", -φ_sol)
    println("------------------------------")
    @printf("ΔV_12     : %.4f\n", p.ΔV_12)
else
    println("Solver failed to converge.")
    println(result)
end