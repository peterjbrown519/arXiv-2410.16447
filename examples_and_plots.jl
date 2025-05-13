using LinearAlgebra
using LaTeXStrings
using Plots

function renyi_entropy(p::AbstractVector{<:Real}, α::Float64)
    if abs(α - 1.0) < 1e-8
        return -sum(pi * log2(pi) for pi in p if pi > 0)
    else
        q_alpha = sum(pi^α for pi in p if pi > 0)
        return (1 / (1 - α)) * log2(q_alpha)
    end
end

function R_up_PVM(ρ::AbstractMatrix{<:Number}, α::Float64)
    # Intrinsic PVM randomness of ρ measured by sandwiched ↑ entropy
    λ = eigvals(ρ)
    if isapprox(imag(λ), zeros(length(λ)), atol=1e-8)
        return log2(size(ρ)[1]) - renyi_entropy(real(λ), α/(2*α - 1))
    else
        error("Large complex eigenvalues found")
    end
end

function R_down_PVM(ρ::AbstractMatrix{<:Number}, α::Float64)
    # Intrinsic PVM randomness of ρ measured by sandwiched ↓ entropy
    λ = eigvals(ρ)
    if isapprox(imag(λ), zeros(length(λ)), atol=1e-8)
        return log2(size(ρ)[1]) - renyi_entropy(real(λ), 1/α)
    else
        error("Large complex eigenvalues found")
    end
end

function R_up_POVM(ρ::AbstractMatrix{<:Number}, α::Float64)
    # Intrinsic PVM randomness of ρ measured by sandwiched ↑ entropy
    λ = eigvals(ρ)
    if isapprox(imag(λ), zeros(length(λ)), atol=1e-8)
        return 2*log2(size(ρ)[1]) - renyi_entropy(real(λ), α/(2*α - 1))
    else
        error("Large complex eigenvalues found")
    end
end

function R_down_POVM(ρ::AbstractMatrix{<:Number}, α::Float64)
    # Intrinsic PVM randomness of ρ measured by sandwiched ↓ entropy
    λ = eigvals(ρ)
    if isapprox(imag(λ), zeros(length(λ)), atol=1e-8)
        return 2*log2(size(ρ)[1]) - renyi_entropy(real(λ), 1/α)
    else
        error("Large complex eigenvalues found")
    end
end

function PA_rates(ρ::AbstractMatrix{<:Number}, ϵ::Real, n::Int64, α::Real)
    return R_up_POVM(ρ, α) + α * log2(ϵ)/( n * (α - 1))
end

function PA_rates_lb(ρ::AbstractMatrix{<:Number}, ϵ::Real, n::Int64, α::Real)
    return R_down_POVM(ρ, α) + α * log2(ϵ)/( n * (α - 1))
end

function binary_maximize(f::Function, xmin::Real, xmax::Real)
    tol = 1e-7
    while (xmax - xmin) > tol
        x1, x2 = (2*xmin + xmax)/3, (xmin + 2*xmax)/3
        if f(x1) > f(x2)
            xmax = x2
        else
            xmin = x1
        end
    end
    return (xmin + xmax)/2, f((xmin + xmax)/2)
end

####################
# Figure 1 example #
####################


# State 
ρ = Matrix([3 1; 1 1]) / 4

# Define plot range
eps = 1e-6
α_min = 1
α_max = 4.0
α_vals = α_min:0.01:α_max

# Compute values 
R_up_vals = [R_up_PVM(ρ, α) for α in α_vals]
R_down_vals = [R_down_PVM(ρ, α) for α in α_vals]
limit_val = [R_down_PVM(ρ, 2.0) for α in α_vals]

plot(α_vals, R_up_vals, xlabel="α", ylim=(0,1), label=L"\mathcal{R}_{H_{\alpha}^{\uparrow}}^{\mathrm{PVM}}(\rho_A)",ylabel=L"Maximal Intrinsic $\mathbb{H}$ randomness")
plot!(α_vals, R_down_vals, label=L"\mathcal{R}_{H_{\alpha}^{\downarrow}}^{\mathrm{PVM}}(\rho_A)")
plot!(α_vals, limit_val, label=L"\lim_{\alpha\to\infty} \mathcal{R}_{H_{\alpha}^{\uparrow}}^{\mathrm{PVM}}(\rho_A)")
savefig("figure_1.png")


####################
# Figure 3 example #
####################

ρ = Matrix([3 1/sqrt(2) 1/sqrt(2);
            1/sqrt(2) 2 1;
            1/sqrt(2) 1 2]) ./ 7
num_points = 50
ϵ = 1e-4
eps_vals = [1e-4 1e-12 1e-20]
n_vals = 10 .^ (range(2,stop=6,length=50))
asym_rate = [R_up_POVM(ρ, 1.0) for n in n_vals]
plot(n_vals, asym_rate, xlabel=L"$n$", xaxis=:log, label="Asymptotic rate", ylim=(1.5, 1.85))
savefig("figure_3.png")
for ϵ in eps_vals

    rates = zeros(num_points)
    rates_lb = zeros(num_points)
    for k = 1:num_points
        n = Int64(round(n_vals[k]))
        function obj(α::Real)
            return PA_rates(ρ, ϵ, n, α)
        end

        a0, f0 = binary_maximize(obj, 1, 2)

        rates[k] = f0

        function obj2(α::Real)
            return PA_rates_lb(ρ, ϵ, n, α)
        end

        a0, f0 = binary_maximize(obj2, 1, 2)

        rates_lb[k] = f0
        println([a0,rates[k], rates_lb[k]])
    end
    plot!(n_vals, rates, label = "ϵ=$ϵ")
    plot!(n_vals, rates_lb, label = "ϵ=$ϵ, lower bound")
end
savefig("figure_3.png")