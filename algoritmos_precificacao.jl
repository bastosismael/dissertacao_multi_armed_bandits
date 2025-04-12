using ProgressMeter

function UCB(arm_avg_reward, arm_counter, δ=1e-8)
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


function seq_halving(arms, n, distr=nothing, method="CDF", regret=false)
    """
    This function was made specifically for the dynamic pricing case
    """
	if method == "CDF"
		prices_wp = Dict(d =>  d*(1 - cdf(distr, d)) for d in arms)
	else
		prices_wp = Dict(d =>  d*distr[d] for d in arms)
	end
	best_arm_mean = findmax(collect(values(prices_wp)))[1]
	avg_reward_vector = []
	cum_regret_vector = []
	avg_reward = 0
	cum_regret = 0
	K = Dict(k => v for (k,v) ∈ enumerate(arms))
	means = Dict(i => 0. for i = 1:length(K))
	A = K
	L = Int(ceil(log2(length(K))))
	it=0
	for 𝓁=1:L
		T_l = floor(n/(L * length(A)))
		for j=1:T_l
			for 𝒶 ∈ keys(A)
				it += 1
				if method == "CDF"
					X = get_reward(A[𝒶], distr) * arms[𝒶]
				else
					X = get_reward_2(A[𝒶], distr) * arms[𝒶]
				end
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
		A = Dict(k => v for (k,v) in A if k ∈ top)
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

function seq_elim(arms, n, distr=nothing, method="CDF")
    """
    This function was made specifically for the dynamic pricing case
    """
    K = Dict(k => v for (k,v) ∈ enumerate(arms))
    means = Dict(i => 0. for i = 1:length(K))
    A = K
    L = length(K) -1
	it=0
	avg_reward_vector = []
	avg_reward = 0
    for 𝓁=1:L
        T_l = n_k(𝓁, n, length(K)) - n_k(𝓁-1, n, length(K))
        for j=1:T_l
            for 𝒶 ∈ keys(A)
				it += 1
				if method == "CDF"
					X = get_reward(A[𝒶], distr) * arms[𝒶]
				else
					X = get_reward_2(A[𝒶], distr) * arms[𝒶]
				end
				avg_reward = avg_reward + (1/(it+1))*(X - avg_reward)
				push!(avg_reward_vector, avg_reward)
                means[𝒶] += X
            end
        end
        means = Dict(i => v/T_l for (i,v) ∈ means)
        min_mean = findmin(means)[2]
        A = Dict(k => v for (k,v) in A if k != min_mean)
        means = Dict(i => 0. for i ∈ keys(A))
    end
    return A, avg_reward_vector
end

function get_reward(price, distribution)
    # Reward according to the definition made on the text.
	prob = 1 - cdf(distribution, price)
	b = Bernoulli(prob)
	return rand(b, 1)[1]
end

function get_reward_2(price, distribution)
	prob = distribution[price]
	b = Bernoulli(prob)
	return rand(b, 1)[1]
end

function simulate(arms, n, best_arm_mean, distribution =  nothing, strategy="epsgreedy", method="CDF", c = nothing, avg_reward=0, cum_regret=0, m=100,)
    """
	This function was made specifically for the dynamic pricing case
    By default, the simulation chooses the ε-greedy algorithm and set m=100 in ETC. 
    
    See the pluto notebooks to understand the arguments of the function. 
    """
	selected_arms = []
	avg_reward_vector = []
	arm_avg_reward = zeros(length(arms))
	arm_counter = zeros(length(arms))
	cum_regret_vector = []
	for iteration in 0:n-1
		if strategy == "epsgreedy"
			arm = epsilon_greedy(arm_avg_reward, 0.1)
		elseif strategy == "greedy"
			arm = greedy(arm_avg_reward)
		elseif strategy == "UCB"
			arm = UCB(arm_avg_reward, arm_counter)
		elseif strategy == "ETC"
			arm = ETC(m, arm_avg_reward, iteration)
		elseif strategy == "KL-UCB"
			arm = KL_UCB(arm_avg_reward, arm_counter, iteration+1)
		elseif strategy == "CONST"
			arm = c
		end
		push!(selected_arms, arm)
		if method == "CDF"
			reward = arms[arm]*get_reward(arms[arm], distribution)
		else
			reward = arms[arm]*get_reward_2(arms[arm], distribution)
		end
		regret = best_arm_mean - reward
		push!(avg_reward_vector, avg_reward)
		if isnothing(c)
			avg_reward = avg_reward + (1/(iteration+1))*(reward - avg_reward)
		else
			avg_reward =  avg_reward + (1/(iteration+1+(10000-n)))*(reward - avg_reward)
		end
		push!(cum_regret_vector, cum_regret)
		cum_regret = cum_regret + regret
		arm_counter[arm] += 1
		if strategy != "ETC" || iteration < m*length(arms)
			arm_avg_reward[arm] = ((arm_counter[arm] -1) * arm_avg_reward[arm] + reward)/arm_counter[arm]
		end
	end
	return selected_arms, avg_reward_vector, cum_regret_vector
end

function simulate_pure_exp(arms, horizon, strategy, method="CDF", distr=nothing,  n_simulations=1000)
	selected_arms = []
	@showprogress for i in 1:n_simulations
		if strategy=="seq_halving"
			𝒶, avg_reward = seq_halving(arms, horizon, distr, method)
		elseif strategy=="seq_elim"
			𝒶, avg_reward = seq_elim(arms, horizon, distr, method)
		end
		push!(selected_arms, float(collect(keys(𝒶))[1]))
		i ==1 ? final_avg_reward =  avg_reward : final_avg_reward += avg_reward
	end
	final_avg_reward = final_avg_reward/n_simulations
	count_arms = countmap(selected_arms)
	count_arms = sort(Dict(k=>v/sum(values(count_arms)) for (k,v) ∈ count_arms), byvalue=true, rev=true)
	count_arms = Dict(arms[Int(k)] => v for (k,v) ∈ count_arms)
	return selected_arms, final_avg_reward, count_arms
end

function evaluate(arms, horizon, strategy="epsgreedy", method="CDF", distr = nothing, n_simulations=1000, c=nothing, p_avg_reward=0, p_cum_regret=0)
	if method == "CDF"
		prices_wp = Dict(d =>  d*(1 - cdf(distr, d)) for d in arms)
	else
		prices_wp = Dict(d =>  d*distr[d] for d in arms)
	end
	best_arm_mean = findmax(collect(values(prices_wp)))[1]
    final_arms = []
    final_avg_reward = zeros(horizon)
    final_avg_regret = zeros(horizon)
    @showprogress for i in 1:n_simulations
        selected_arms, avg_reward,  cum_regret = simulate(arms, horizon, best_arm_mean, distr, strategy, method, c, p_avg_reward, p_cum_regret)
        final_arms = [final_arms ; selected_arms]
        final_avg_reward =  final_avg_reward .+ avg_reward
        final_avg_regret = final_avg_regret .+ cum_regret
    end
    final_avg_reward = final_avg_reward/n_simulations
    final_avg_regret = final_avg_regret/n_simulations
    count_arms = countmap(final_arms)
	count_arms = sort(Dict(k=>v/sum(values(count_arms)) for (k,v) ∈ count_arms), byvalue=true, rev=true)
	count_arms = Dict(arms[k] => v for (k,v) ∈ count_arms)
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