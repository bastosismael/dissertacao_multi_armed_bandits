using Distributions, StatsBase, Plots
include("algoritmos.jl")

H1 = 150
ν = Exponential(20)
n_simulations = 1000
horizon = 10000 
n_arms = 20
arms = [i*2 for i in 1:n_arms]
dict_arms = sort(Dict(k[1]=>k[2] for k ∈ enumerate(arms)), byvalue=true, rev=true)

a = simulate_pure_exp(arms, H1, "seq_halving", ν, 1000)

function freq_error(arms, horizon, strategy, distr,  n_simulations)
	_,_,c = simulate_pure_exp(arms, horizon, strategy, distr,  n_simulations)
	return c[20]
end

plot(n -> 1-freq_error(arms, n, "seq_halving", ν, n_simulations), 1000:10:10000, dpi=3000)
plot!(n -> 1-freq_error(arms, n, "seq_elim", ν, n_simulations), 1000:10:10000)