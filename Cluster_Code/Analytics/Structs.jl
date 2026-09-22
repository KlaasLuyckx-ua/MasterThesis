abstract type GPars end
"""Struct which contains system parameters:
J: tunneling | sites: number of sites | n_avg: target number of photons | 
M: total number of dye molecules | B_21: Einstein coefficient | Δ: detuning | 
γ: loss rate | T: temperature | κ: relaxation rate | V: potential shape | 
ρ: pump shape | save_Δt: time step between observable sampling | 
Δt_steps_t_coh: number of eval time steps between coherence time sampling"""
struct CFPars <:GPars
    J::Float64                                      # tunneling [meV]
    sites::Int64                                    # number of sites
    n_avg::Float64                                  # target number photons
    M::Float64                                      # number of molecules
    B_21::Float64                                   # Einstein coefficient [meV]
    Δ::Float64                                      # detuning [meV]
    γ::Float64                                      # loss rate [meV]
    T::Float64                                      # temperature [meV]
    κ::Float64                                      # relaxation rate [meV]
    V::Vector{Float64}                              # potential shape [meV]
    ρ::Vector{Float64}                              # pump shape ∈ [0,sites]
    save_Δt::Float64                                # sample time step [1/J]
end