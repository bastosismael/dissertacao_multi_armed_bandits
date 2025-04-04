using Interact, StatsBase, ProgressMeter, Plots, Distributions, LaTeXStrings
include("algoritmos.jl")

n_simulations = 20
horizon = 10000 
n_arms = 10
φ = 0.01
distr = [Bernoulli(0.2 - φ*(i-1)) for i ∈ 1:n_arms]

algorithms = ["UCB", "ETC", "epsgreedy"] 
final_avg_reward = Dict("UCB" => [], "ETC" => [], "epsgreedy" => [])
final_avg_regret = Dict("UCB" => [], "ETC" => [], "epsgreedy" => [])
final_count = Dict("UCB" => Dict{Int64, Float64}(), "ETC" => Dict{Int64, Float64}(), "epsgreedy" => Dict{Int64, Float64}())
@showprogress @distributed for alg in algorithms
    final_avg_reward[alg], final_avg_regret[alg], final_count[alg] = evaluate( 
    horizon, alg, distr, n_simulations, false)
end

# Salvando
FILE = "results_en.json"
open(FILE, "w") do f
    JSON.print(f, Dict("reward" => final_avg_reward,
     "regret" => final_avg_regret,
     "count" => final_count))
end

# UCB
bar(final_count["UCB"], dpi=300, labels="", grid=false, xticks=xticks)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/arm_selection_ucb.png")

#ETC
bar(final_count["ETC"], dpi=300, labels="", grid=false, xticks=xticks)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/arm_selection_etc.png")

#ϵ-guloso
bar(final_count["epsgreedy"], dpi=300, labels="", grid=false, xticks=xticks)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/arm_selection_eps.png")

# Sequential Halving
selected_arms_halving, avg_reward_vector_halving, c_arms_halving = simulate_pure_exp(arms, horizon, "seq_halving", "NON_CDF", ν, n_simulations)
bar(c_arms_halving, dpi=300, label=nothing)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_seqhalving_disc.png")

# Sequential Elimination
selected_arms_elim, avg_reward_vector_elim, c_arms_elim = simulate_pure_exp(arms, horizon, "seq_elim", "NON_CDF", ν, n_simulations)
bar(c_arms_elim, dpi=300, label=nothing)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_seqelim_disc.png")

# Gráfico - Comparação da recompensa entre os algoritmos de exploraçã0
xticks = ([i for i in 0:2000:8000],[i+2000 for i in 0:2000:8000])
plot(final_avg_reward["epsgreedy"][2000:end], label="ε-guloso", xticks=xticks, dpi=300)
plot!(final_avg_reward["UCB"][2000:end], label="UCB")
plot!(final_avg_reward["ETC"][2000:end], label="ETC")
hline!([maximum(x -> maximum([distr.p[a] *a for a ∈ arms]),0:60)], label="max")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/reward_disc.png")

# Gráfico - Arrependimento 
plot((final_avg_regret["epsgreedy"][2000:end]), label="ε-greedy", dpi=300, xticks=xticks)
plot!((final_avg_regret["UCB"][2000:end]), label="UCB")
plot!((final_avg_regret["ETC"][2000:end]), label="ETC")
xlabel!("t")
ylabel!(L"\mathrm{Arr}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/regret_disc.png")

arms_wp = Dict(d =>  d*ν[d] for d in arms)
k = length(arms)
arms_mean = collect(values(arms_wp))
best_arm_mean = findmax(arms_mean)[1]
Δ = best_arm_mean .- arms_mean
R = 3 * sum(Δ) + 16*20^2*log(10000)/sum(Δ)