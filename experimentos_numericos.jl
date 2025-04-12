using Interact, StatsBase, ProgressMeter, Plots, Distributions, LaTeXStrings, JSON
include("algoritmos.jl")
include("cotas.jl")
n_simulations = 20000
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

FILE = "results_en.json"

# Salvando
open(FILE, "w") do f
    JSON.print(f, Dict("reward" => final_avg_reward,
     "regret" => final_avg_regret,
     "count" => final_count))
end

# Lendo
open(FILE, "r") do f
    global data = JSON.parse(f)
end
for alg in algorithms
    println(alg)
    final_avg_reward[alg], final_avg_regret[alg], final_count[alg] = data["reward"][alg],  data["regret"][alg], Dict(parse(Int, k) => v for (k,v) ∈ pairs(data["count"][alg]))
end


# UCB
bar(final_count["UCB"], dpi=300, labels="", grid=false, xticks=xticks)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/arm_selection_ucb_$φ.png")

#ETC
bar(final_count["ETC"], dpi=300, labels="", grid=false, xticks=xticks)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/arm_selection_etc_$φ.png")

#ϵ-guloso
bar(final_count["epsgreedy"], dpi=300, labels="", grid=false, xticks=xticks)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/arm_selection_eps_$φ.png")

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
hline!([maximum(x -> maximum([d.p for d ∈ distr]),0:60)], label="max")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/reward_$φ.png")

# Gráfico - Arrependimento 
plot((final_avg_regret["epsgreedy"][2000:end]), label="ε-greedy", dpi=300, xticks=xticks)
plot!((final_avg_regret["UCB"][2000:end]), label="UCB")
plot!((final_avg_regret["ETC"][2000:end]), label="ETC")
xlabel!("t")
ylabel!(L"\mathrm{Arr}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/experimentos_numericos/regret_$φ.png")

# Cota superior do ETC
function upper_bound_etc(arms, m, n, σ)
    arms_wp = Dict(d =>  d.p for d in arms)
	k = length(arms)
	arms_mean = collect(values(arms_wp))
	best_arm_mean = findmax(arms_mean)[1]
	R = m * sum(best_arm_mean .- arms_mean) + (n - m*k)*sum((best_arm_mean .- arms_mean) .* exp.(-(m*  (best_arm_mean .- arms_mean).^2)./(4*σ^2) ) ) 
	return R
end

# Cota superior do UCB 
function upper_bound_ucb(arms, n, σ)
    arms_wp = Dict(d =>  d.p for d in arms)
	k = length(arms)
	arms_mean = collect(values(arms_wp))
	best_arm_mean = findmax(arms_mean)[1]
	Δ = best_arm_mean .- arms_mean
	R = 3 * sum(Δ) + 16*\sigma^2*log(n)*sum([1/i for i in Δ if i != 0])
    return R
end

upper_bound_etc(distr, 100, horizon, 1/2)
upper_bound_ucb(distr, horizon, 1/2)

arms_wp = Dict(d =>  d.p for d in distr)
k = length(distr)
arms_mean = collect(values(arms_wp))
best_arm_mean = findmax(arms_mean)[1]
Δ = best_arm_mean .- arms_mean
