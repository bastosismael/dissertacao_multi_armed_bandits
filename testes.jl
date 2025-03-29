using Interact, StatsBase, ProgressMeter, Plots, Distributions, LaTeXStrings, JSON

include("algoritmos.jl")
using Distributed

# Definições 
n_simulations = 20000
horizon = 10000 
n_arms = 20
arms = [i*2 for i in 1:n_arms]
dict_arms = sort(Dict(k[1]=>k[2] for k ∈ enumerate(arms)), byvalue=true, rev=true)
ν = Exponential(20)

# Gráfico do valor esperado da recompensa
plot(x -> (1-cdf(ν, x))*x, minimum(support(ν)):100, labels="", grid=false)
xlabel!("preço")
ylabel!("Valor esperado da recompesa")
# Média máxima da recompensa
print("Max: $(maximum(x -> (1-cdf(ν, x))*x, minimum(support(ν)):60)) 
        \nArgmax: $(argmax(x -> (1-cdf(ν, x))*x, minimum(support(ν)):60))")


algorithms = ["UCB", "ETC", "epsgreedy"] 
final_avg_reward = Dict("UCB" => [], "ETC" => [], "epsgreedy" => [])
final_avg_regret = Dict("UCB" => [], "ETC" => [], "epsgreedy" => [])
final_count = Dict("UCB" => Dict{Int64, Float64}(), "ETC" => Dict{Int64, Float64}(), "epsgreedy" => Dict{Int64, Float64}())

@showprogress @distributed for alg in algorithms
    final_avg_reward[alg], final_avg_regret[alg], final_count[alg] = evaluate(arms, 
    horizon, alg, "CDF", ν, n_simulations)
end

#Salvando
FILE = "results_dp_exp.json"
open(FILE, "w") do f
    JSON.print(f, Dict("reward" => final_avg_reward,
     "regret" => final_avg_regret,
     "count" => final_count))
end
#Lendo o arquivo JSON 
open(FILE, "r") do f
    global data = JSON.parse(f)
end
final_count = Dict("UCB" => Dict{String, Any}(), "ETC" => Dict{String, Any}(), "epsgreedy" => Dict{String, Any}())
for alg in algorithms
    println(alg)
    final_avg_reward[alg], final_avg_regret[alg], final_count[alg] = data["reward"][alg],  data["regret"][alg], sort(data["count"][alg], byvalue=false, rev=true)
end
final
# UCB
bar(final_count["UCB"], dpi=300, labels="", grid=false)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_ucb_exp.png")

#ETC
bar(final_count["ETC"], dpi=300, labels="", grid=false)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_etc_exp.png")

#ϵ-guloso
bar(final_count["epsgreedy"], dpi=300, labels="", grid=false)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_eps_exp.png")

# Sequential Halving
selected_arms_halving, avg_reward_vector_halving, c_arms_halving = simulate_pure_exp(arms, horizon, "seq_halving","CDF", ν, n_simulations)
bar(c_arms_halving, dpi=300, label=nothing)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_seqhalving_.png")

# Sequential Elimination
selected_arms_elim, avg_reward_vector_elim, c_arms_elim = simulate_pure_exp(arms, horizon, "seq_elim", "CDF", ν, n_simulations)
bar(c_arms_elim, dpi=300, label=nothing)
xlabel!("preço")
ylabel!("proporção")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/arm_selection_seqelim_.png")

# Gráfico - Comparação da recompensa entre os algoritmos de exploraçã0
xticks = ([i for i in 0:2000:8000],[i+2000 for i in 0:2000:8000])
yticks = ([i for i in 6.25:.25:8],[replace(string(i), "." => ",") for i in 6.25:.25:8])
plot(final_avg_reward["epsgreedy"][2000:end], label="ε-guloso", xticks=xticks, yticks=yticks, dpi=300)
plot!(final_avg_reward["UCB"][2000:end], label="UCB")
plot!(final_avg_reward["ETC"][2000:end], label="ETC")
hline!([maximum(x -> (1-cdf(ν, x))*x, minimum(support(ν)):60)], label="max")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/reward_exp.png")

# Gráfico - Arrependimento 
plot((final_avg_regret["epsgreedy"][2000:end]), label="ε-greedy", dpi=300, xticks=xticks)
plot!((final_avg_regret["UCB"][2000:end]), label="UCB")
plot!((final_avg_regret["ETC"][2000:end]), label="ETC")
xlabel!("t")
ylabel!(L"\mathrm{Arr}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/regret_exp.png")

#Gráfico - Comparação Reward Pure Exploration
plot(avg_reward_vector_halving[Int(ceil(0.2*horizon)):end], label="metades", xticks=xticks, yticks=yticks, dpi=300)
plot!(avg_reward_vector_elim[Int(ceil(0.2*horizon)):end], label="sequencial")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/mean_reward_pure_exp.png")

# Sequential Halving ThEn Commit
reward_shtc, _ = SHTC(arms, horizon, ν, n_simulations, 0.5)

# Gráfico - Comparaçlão final da recompensa entre os algoritmos
plot(final_avg_reward["epsgreedy"][2000:end], label="ε-guloso", xticks=xticks, yticks=yticks, dpi=300)
plot!(final_avg_reward["UCB"][2000:end], label="UCB")
plot!(final_avg_reward["ETC"][2000:end], label="ETC")
plot!(avg_reward_vector_halving[Int(ceil(0.2*horizon)):end], label="metades")
plot!(reward_shtc[2000:end], label="SHTC")
hline!([maximum(x -> (1-cdf(ν, x))*x, minimum(support(ν)):60)], label="max")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/shtc_comp.png")


