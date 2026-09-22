using Plots

# 1. Parameters voor de fysica
L_1 = 5.0             # Lengte van de touwen (zorgt voor de synchrone beweging)
L_2 = 3.5             # Lengte van de touwen (zorgt voor de synchrone beweging)
L=(L_1 + L_2) / 2.0  # Gemiddelde lengte van de touwen
g = 9.81            # Valversnelling
θ_max = pi / 4      # Maximale uitwijking (45 graden)
ω = sqrt(g / L)   # Hoekfrequentie

# 2. Ophangpunten van de twee slingers
x1, y1 = -2.0, 0.0  # Linker slinger
x2, y2 = 2.0, 0.0   # Rechter slinger

# 3. Tijdstappen voor de animatie
fps = 60
t_end = 4 * pi / ω  # Tijd voor twee volledige periodes
t_steps = range(0, t_end, length=120)

# 4. De animatie loop
println("Genereren van de frames...")
anim = @animate for t in t_steps
    # Bereken de actuele hoek θ op tijdstip t
    # Formule: θ(t) = θ_max * cos(ω * t)
    θ = θ_max * cos(ω * t)
    
    # Bereken de (x, y) posities van de bollen
    bob1_x = x1 + L_1 * sin(θ)
    bob1_y = y1 - L_1 * cos(θ)
    
    bob2_x = x2 + L_2 * sin(θ)
    bob2_y = y2 - L_2 * cos(θ)
    
    # Maak een leeg canvas met vaste assen om bibberen te voorkomen
    plot(xlims=(-6, 6), ylims=(-6, 1), aspect_ratio=:equal, legend=false, 
         grid=false, framestyle=:none, background_color=:white)
    
    # Teken de balk waaraan ze hangen
    plot!([-4, 4], [0, 0], color=:black, linewidth=6)
    
    # Teken de touwen
    plot!([x1, bob1_x], [y1, bob1_y], color=:grey, linewidth=2)
    plot!([x2, bob2_x], [y2, bob2_y], color=:grey, linewidth=2)
    
    # Teken de bollen (bobs)
    scatter!([bob1_x, bob2_x], [bob1_y, bob2_y], color=:red, markersize=15, markerstrokewidth=0)
end

# 5. Sla het resultaat op
uitvoer_bestand = "synchrone_slingers.gif"
gif(anim, uitvoer_bestand, fps=fps)
println("Klaar! De GIF is opgeslagen als: ", uitvoer_bestand)