include("algoritmos_precificacao.jl")
using LaTeXStrings

n_simulations = 20000
horizon = 10000 
n_arms = 20
arms = [i*2 for i in 1:n_arms]
dict_arms = sort(Dict(k[1]=>k[2] for k ∈ enumerate(arms)), byvalue=true, rev=true)
ν = Exponential(20)

reward_shtc = Dict(k => [] for k in .1:.2:.9)
regret_shtc = Dict(k => [] for k in .1:.2:.9)
@showprogress for p in keys(reward_shtc)
    reward_shtc[p], regret_shtc[p] = SHTC(arms, horizon, ν, "CDF", n_simulations, p)
end

# Gráfico comparando as diferentes escolhas de horizonte
acc = 1
xticks_f = ([i for i in 0:1000:8000],[i+1000 for i in 0:1000:8000])
yticks = ([i for i in 6.25:.25:8],[replace(string(i), "." => ",") for i in 6.25:.25:8])
for (k,v) in sort(reward_shtc, byvalue=false, rev=true)
    if acc == 1
        plot(reward_shtc[k][1000:end], label="$(k*100)%", dpi=300, xticks=xticks_f, yticks=yticks)
    else
        plot!(v[1000:end], label = "$(k*100)%")
    end
    acc += 1
end
hline!([maximum(x -> (1-cdf(ν, x))*x, minimum(support(ν)):60)], label="max")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/shtc_comp_2.png")

# Gráfico anterior com zoom no intervalo 8000:10000
acc = 1
yticks_z = ([i for i in 6.9:.1:7.3],[replace(string(i), "." => ",") for i in 6.9:.1:7.3])
xticks_z = ([i for i in 0:500:2000],[i+8000 for i in 0:500:2000])
for (k,v) in sort(reward_shtc, byvalue=false, rev=true)
    if acc == 1
        plot(reward_shtc[k][8000:end], label="$(k*100)%", dpi=300, xticks=xticks_z, yticks=yticks_z)
    else
        plot!(v[8000:end], label = "$(k*100)%")
    end
    acc += 1
end
hline!([maximum(x -> (1-cdf(ν, x))*x, minimum(support(ν)):60)], label="max")
xlabel!("t")
ylabel!(L"\mathrm{RecM}(20.000,t)")
savefig("/home/ismael/Documents/Disserta-o/imagens_experimentos_numericos/precificaca_dinamica/shtc_comp_3.png")

# Arrependimento
acc = 1
for (k,v) in sort(regret_shtc, byvalue=false, rev=true)
    if acc == 1
        plot(v, label = "$(k*100)%", dpi=300)
    else
        plot!(v, label = "$(k*100)%")
    end
    acc += 1
end
xlabel!("t")
ylabel!(L"\mathrm{Arr}(20.000,t)")