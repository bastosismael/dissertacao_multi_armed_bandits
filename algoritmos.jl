using ProgressMeter

function UCB(arm_avg_reward, arm_counter, δ=0.05)
	if iszero(arm_avg_reward)
		arm = rand(1:length(arm_avg_reward))
	elseif 0 ∈ arm_counter
		arm = findmin(arm_counter)[2]
	end
	ucb = arm_avg_reward .+ sqrt.(2*log(1/δ)./arm_counter)
	return findmax(ucb)[2]
end

function ETC(m, arm_avg_reward, t)
	k = length(arm_avg_reward)
	if t ≤ m * k
		arm =t%k + 1
	else
		arm =findmax(arm_avg_reward)[2]
	end
	return arm
end


function greedy(arm_avg_reward)
	if iszero(arm_avg_reward)
		arm = rand(1:length(arm_avg_reward))
	else
		arm =findmax(arm_avg_reward)[2]
	end
	return arm
end

function epsilon_greedy(arm_avg_reward, eps=0.1)
	U = Uniform(0,1)
	u = rand(U,1)[1]
	if u < eps
		arm = rand(1:length(arm_avg_reward))
	elseif iszero(arm_avg_reward)
		arm = rand(1:length(arm_avg_reward))
	else
		arm = findmax(arm_avg_reward)[2]
	end
	return arm
end


function seq_halving(n, distr, regret=false)
	best_arm_mean = findmax([d.p for d ∈ distr])[1]
	avg_reward_vector = []
	cum_regret_vector = []
	avg_reward = 0
	cum_regret = 0
	K = [i for i in 1:length(distr)]
	means = Dict(i => 0. for i = 1:length(K))
	A = K
	L = Int(ceil(log2(length(K))))
	it=0
	for 𝓁=1:L
		T_l = floor(n/(L * length(A)))
		for j=1:T_l
			for 𝒶 ∈ A
				it += 1
				X = rand(distr[𝒶], 1)[1]
				avg_reward += (1/(it+1))*(X - avg_reward)
				cum_regret += best_arm_mean - X
				push!(avg_reward_vector, avg_reward)
				push!(cum_regret_vector, cum_regret)
				means[𝒶] += X
			end
		end
		means = Dict(i => v/T_l for (i,v) ∈ means)
		top = collect(keys(sort(means; byvalue=true, rev=true)))
		top = top[1:(Int(ceil(length(A)/2)))]
		A = top
		means = Dict(i => 0. for i ∈ keys(A))
	end
	if regret
    	return A, avg_reward_vector, cum_regret_vector 
	else
		return A, avg_reward_vector
	end
end


function n_k(k, n, K)
    """
    This is an auxiliar function to generate the number of
    samples in each round in the sequential elimination
    """
    if k==0
        return 0
    end
    p_1 = 1/2 + sum([1/i for i in 2:K])
    return ceil((1/p_1) * (n-K)/(K+1-k))
end

function seq_elim(n, distr)
    """
    This function was made specifically for the dynamic pricing case
    """
	K = [i for i in 1:length(distr)]
	means = Dict(i => 0. for i = 1:length(K))
	A = K
    L = length(K) -1
	it=0
	avg_reward_vector = []
	avg_reward = 0
    for 𝓁=1:L
        T_l = n_k(𝓁, n, length(K)) - n_k(𝓁-1, n, length(K))
        for j=1:T_l
            for 𝒶 ∈ A
				it += 1
				X = rand(distr[𝒶], 1)[1]
				avg_reward = avg_reward + (1/(it+1))*(X - avg_reward)
				push!(avg_reward_vector, avg_reward)
                means[𝒶] += X
            end
        end
        means = Dict(i => v/T_l for (i,v) ∈ means)
        min_mean = findmin(means)[2]
        A = [k  for k in A if k != min_mean]
        means = Dict(i => 0. for i ∈ A)
    end
    return A, avg_reward_vector
end

function simulate(n, best_arm_mean, distr , strategy, c = nothing, avg_reward=0, cum_regret=0, m=100)
	selected_arms = []
	avg_reward_vector = []
	arm_avg_reward = zeros(length(distr))
	arm_counter = zeros(length(distr))
	cum_regret_vector = []
	for iteration in 0:n-1
		if strategy == "epsgreedy"
			𝒶 = epsilon_greedy(arm_avg_reward, 0.1)
		elseif strategy == "greedy"
			𝒶 = greedy(arm_avg_reward)
		elseif strategy == "UCB"
			𝒶 = UCB(arm_avg_reward, arm_counter)
		elseif strategy == "ETC"
			𝒶 = ETC(m, arm_avg_reward, iteration)
		elseif strategy == "CONST"
			𝒶 = c
		end
		push!(selected_arms, 𝒶)
		reward = rand(distr[𝒶], 1)[1]
		regret = best_arm_mean - reward
		push!(avg_reward_vector, avg_reward)
		if isnothing(c)
			avg_reward = avg_reward + (1/(iteration+1))*(reward - avg_reward)
		else
			avg_reward =  avg_reward + (1/(iteration+1+(10000 - n)))*(reward - avg_reward)
		end
		push!(cum_regret_vector, cum_regret)
		cum_regret = cum_regret + regret
		arm_counter[𝒶] += 1
		if strategy != "ETC" || iteration < m*length(distr)
			arm_avg_reward[𝒶] = ((arm_counter[𝒶] -1) * arm_avg_reward[𝒶] + reward)/arm_counter[𝒶]
		end
	end
	return selected_arms, avg_reward_vector, cum_regret_vector
end

function simulate_pure_exp(horizon, strategy, distr,  n_simulations=1000)
	selected_arms = []
	@showprogress for i in 1:n_simulations
		if strategy=="seq_halving"
			𝒶, avg_reward = seq_halving(horizon, distr)
		elseif strategy=="seq_elim"
			𝒶, avg_reward = seq_elim(horizon, distr)
		end
		push!(selected_arms, float(collect(keys(𝒶))[1]))
		i ==1 ? final_avg_reward =  avg_reward : final_avg_reward += avg_reward
	end
	final_avg_reward = final_avg_reward/n_simulations
	count_arms = countmap(selected_arms)
	count_arms = sort(Dict(k=>v/sum(values(count_arms)) for (k,v) ∈ count_arms), byvalue=true, rev=true)
	count_arms = Dict(k => v for (k,v) ∈ count_arms)
	return selected_arms, final_avg_reward, count_arms
end

function evaluate(horizon, strategy, distr, n_simulations=1000, c=nothing, p_avg_reward=0, p_cum_regret=0)
	best_arm_mean = findmax([d.μ for d ∈ distr])[1]
    final_arms = []
    final_avg_reward = zeros(horizon)
    final_avg_regret = zeros(horizon)
    @showprogress for i in 1:n_simulations
        selected_arms, avg_reward,  cum_regret = simulate(horizon, best_arm_mean, distr, strategy, c, p_avg_reward, p_cum_regret)
        final_arms = [final_arms ; selected_arms]
        final_avg_reward =  final_avg_reward .+ avg_reward
        final_avg_regret = final_avg_regret .+ cum_regret
    end
    final_avg_reward = final_avg_reward/n_simulations
    final_avg_regret = final_avg_regret/n_simulations
    count_arms = countmap(final_arms)
	count_arms = sort(Dict(k=>v/sum(values(count_arms)) for (k,v) ∈ count_arms), byvalue=true, rev=true)
	count_arms = Dict(k => v for (k,v) ∈ count_arms)
    return final_avg_reward, final_avg_regret, count_arms
end

function SHTC(arms, horizon, distr, method, n_simulations, p_expl = 0.5)
	final_reward = []
	@showprogress for i in 1:n_simulations
		𝒶, avg_reward_htc_seq, cum_regret_htc_seq = seq_halving(arms, trunc(Int, p_expl*horizon), distr, method, true)
		avg_reward_htc_explt, cum_regret_htc, _ = evaluate(arms, trunc(Int, horizon * (1-p_expl)), "CONST", method, ν, 1, collect(keys(𝒶))[1], avg_reward_htc_seq[end], cum_regret_htc_seq[end])
		avg_reward = [avg_reward_htc_seq; avg_reward_htc_explt]
		cum_regret = [cum_regret_htc_seq; cum_regret_htc]
		if i ==1
			final_reward =  avg_reward
			final_regret = cum_regret
		else
			final_reward += avg_reward
			final_regret += cum_regret
		end
	end
	final_reward = final_reward./n_simulations
	final_regret =final_regret./n_simulations
	return (final_reward, final_regret)
end