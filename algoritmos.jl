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

function seq_halving(prices, n, distr)
    """
    This function was made specifically for the dynamic pricing case
    """
	K = Dict(k => v for (k,v) ∈ enumerate(prices))
	global means = Dict(i => 0. for i = 1:length(K))
	global A = K
	L = ceil(log2(length(K)))
	for 𝓁=1:L
		T_l = floor(n/(L * length(A)))
		for j=1:T_l
			for 𝒶 ∈ keys(A)
				X = get_reward(A[𝒶], distr) * prices[𝒶]
				means[𝒶] += X
			end
		end
		means = Dict(i => v/T_l for (i,v) ∈ means)
		top = collect(keys(sort(means; byvalue=true, rev=true)))
		top = top[1:(Int(ceil(length(A)/2)))]
		global A = Dict(k => v for (k,v) in A if k ∈ top)
		global means = Dict(i => 0. for i ∈ keys(A))
	end
    return A
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

function seq_elim(prices, n, distr)
    """
    This function was made specifically for the dynamic pricing case
    """
    K = Dict(k => v for (k,v) ∈ enumerate(prices))
    means = Dict(i => 0. for i = 1:length(K))
    A = K
    L = length(K) -1
    for 𝓁=1:L
        T_l = n_k(𝓁, n, length(K)) - n_k(𝓁-1, n, length(K))
        for j=1:T_l
            for 𝒶 ∈ keys(A)
                X = get_reward(A[𝒶], distr) * prices[𝒶]
                means[𝒶] += X
            end
        end
        means = Dict(i => v/T_l for (i,v) ∈ means)
        min_mean = findmin(means)[2]
        A = Dict(k => v for (k,v) in A if k != min_mean)
        means = Dict(i => 0. for i ∈ keys(A))
    end
    return A
end

function get_reward(price, distribution)
    # Reward according to the definition made on the text.
	prob = 1 - cdf(distribution, price)
	b = Bernoulli(prob)
	return rand(b, 1)[1]
end

function simulate(prices, n, distribution, strategy="epsgreedy")
    """
	This function was made specifically for the dynamic pricing case
    By default, the simulation chooses the ε-greedy algorithm and set m=100 in ETC. 
    
    See the pluto notebooks to understand the arguments of the function. 
    """
	prices_wp = Dict(d =>  1 - cdf(distribution, d) for d in prices)
	best_arm_mean = findmax(collect(values(prices_wp)))[1]
	selected_arms = []
	avg_reward = 0
	avg_reward_vector = []
	arm_avg_reward = zeros(length(prices))
	arm_counter = zeros(length(prices))
	cum_regret = 0
	cum_regret_vector = []
	for iteration in 0:n-1
		if strategy == "epsgreedy"
			arm = epsilon_greedy(arm_avg_reward, 0.1)
		elseif strategy == "UCB"
			arm = UCB(arm_avg_reward, arm_counter)
		elseif strategy == "ETC"
			arm = ETC(100, arm_avg_reward, iteration)
		elseif strategy == "KL-UCB"
			arm = KL_UCB(arm_avg_reward, arm_counter, iteration+1)
		end
		push!(selected_arms, arm)
		reward = prices[arm]*get_reward(prices[arm], distribution)
		push!(avg_reward_vector, avg_reward)
		avg_reward = avg_reward + (1/(iteration+1))*(reward - avg_reward)
		regret = best_arm_mean - prices_wp[prices[arm]]
		push!(cum_regret_vector, cum_regret)
		cum_regret = cum_regret + regret
		arm_counter[arm] += 1
		arm_avg_reward[arm] = ((arm_counter[arm] -1) * arm_avg_reward[arm] + reward)/arm_counter[arm]
	end
	return selected_arms, avg_reward_vector, cum_regret_vector
end

function evaluate(arms, horizon, strategy, distr)
    final_arms = []
    final_avg_reward = zeros(horizon)
    final_avg_regret = zeros(horizon)
    for i in 1:n_simulations
        selected_arms, avg_reward,  cum_regret = simulate(arms, horizon, distr, strategy)
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