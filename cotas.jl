include("algoritmos.jl")

# Cota superior do ETC
function upper_bound_etc(arms, m, n, σ)
    arms_wp = Dict(d =>  d*get_reward(d, ν) for d in arms)
	k = length(arms)
	arms_mean = collect(values(arms_wp))
	best_arm_mean = findmax(arms_mean)[1]
	R = m * sum(best_arm_mean .- arms_mean) + (n - m*k)*sum((best_arm_mean .- arms_mean) .* exp.(-(m*  (best_arm_mean .- arms_mean).^2)./(4*σ^2) ) ) 
	return R
end

# Cota superior do UCB 
function upper_bound_ucb(arms, n, σ)
    arms_wp = Dict(d =>  d*get_reward(d, ν) for d in arms)
	k = length(arms)
	arms_mean = collect(values(arms_wp))
	best_arm_mean = findmax(arms_mean)[1]
	Δ = best_arm_mean .- arms_mean
	R = 3 * sum(Δ) + 16*σ^2*log(n)/sum(Δ)
    return R
end

# Cota superior sequential halving
function upper_bound_seq_halving(arms, n, σ)
    arms_wp = Dict(d =>  d*get_reward(d, ν) for d in arms)
    k = length(arms)
    arms_mean = collect(values(arms_wp))
    best_arm_mean = findmax(arms_mean)[1]
    Δ = best_arm_mean .- arms_mean
    H₂ = findmax(i/Δ[i] for i in 1:k if Δ[i] != 0)
    fs = 3 *log2(k) * exp(n/16*σ^2*H₂*log2(k))
    return fs
end

# Cota superior sequential elimination
function upper_bound_seq_elim(arms, n, σ)
    arms_wp = Dict(d =>  d*get_reward(d, ν) for d in arms)
    k = length(arms)
    arms_mean = collect(values(arms_wp))
    best_arm_mean = findmax(arms_mean)[1]
    Δ = sort(best_arm_mean .- arms_mean)
    bar_log(x) = 1/2 + sum([1/i for i in 2:x])
    H₂ = findmax(x -> i/Δ[i]  for i in 1:k if Δ[i] != 0)
    fs = k*(k-1)/2 * exp((n-k)/(2*σ^2*bar_log(k)*H₂))
    return fs
end
