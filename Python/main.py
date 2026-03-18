import sys
# Point Python to your compiled .so file
sys.path.append('./build/PSO_engine') 
import pso_engine
from src.load_data import get_market_statistics
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

def get_model_scenarios(mu, returns, tickers):
    obs_returns = returns.values
    with pm.Model() as portfolio_model:
        # use exponential prior so the data can decide how 'fat' the tails are
        # 1. Register the base variable
        nu_minus_2 = pm.Exponential("nu_minus_2", 1/10)
        # 2. Add 2 deterministically
        nu = pm.Deterministic("nu", nu_minus_2 + 2)
        # use a normal prior centered around historical average (mu)
        expected_rtns = pm.Normal("expected_rtns", mu=mu.values, sigma=0.01, shape=len(tickers))
        # tell PyMC: "The real returns we saw follow a Student-T distribution"
        likelihood = pm.StudentT("likelihood", nu=nu, mu=expected_rtns, sigma=returns.std().values, observed=obs_returns)
        # generate 1000 'possible versions' of the expected returns
        trace = pm.sample(1000, target_accept=0.9, chains=2)
    
    # Extract the actual numbers (2 chains * 1000 samples = 2000 scenarios)
    # Reshapes it into a neat 2000 x 4 matrix
    scenarios = trace.posterior["expected_rtns"].values.reshape(-1, len(tickers))
    return scenarios, trace

def run_ensemble_optimization(scenarios, sigma, tickers):
    data = pso_engine.MarketData()
    data.covariance_matrix = sigma.values.tolist()
    data.risk_free_rate = 0.03

    all_weights = []

    # We loop through the 'universes' created by PyMC
    for scenario in scenarios:
        data.expected_returns = scenario.tolist() # Convert to C++ vector
        swarm = pso_engine.Swarm(30, len(tickers))
        swarm.optimize(50, data)
        all_weights.append(swarm.global_best_position)

    return np.array(all_weights)

def plot_dashboard(all_weights, trace, tickers, returns):
    # Set up a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # 1. Weight Uncertainty (Boxplot)
    sns.boxplot(data=all_weights, ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_xticks(range(len(tickers)))
    axes[0, 0].set_xticklabels(tickers)
    axes[0, 0].set_title("Bayesian Weight Distribution (2,000 Scenarios)")
    axes[0, 0].set_ylabel("Portfolio Weight")

    # 2. PyMC Posterior Expected Returns (Density Plot)
    scenarios = trace.posterior["expected_rtns"].values.reshape(-1, len(tickers))
    for i in range(len(tickers)):
        sns.kdeplot(scenarios[:, i], ax=axes[0, 1], label=tickers[i], fill=True)
    axes[0, 1].set_title("Posterior Expected Returns (The 'What-Ifs')")
    axes[0, 1].legend()

    # 3. Asset Correlation (Heatmap)
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=axes[1, 0], vmin=-1, vmax=1)
    axes[1, 0].set_title("Asset Correlation Matrix")

    # 4. Final Robust Allocation (Pie Chart)
    avg_weights = all_weights.mean(axis=0)
    axes[1, 1].pie(avg_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    axes[1, 1].set_title("Final Robust Allocation")

    plt.tight_layout()
    plt.show()
    
def main():
    tickers = [
        # US Large Cap
        'AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ',
        # Different sectors
        'XOM',   # Energy
        'GLD',   # Gold ETF
        'TLT',   # Long-term bonds ETF
        'VNQ',   # Real estate
        # International
        'EEM',   # Emerging markets
    ]
    mu, sigma, returns = get_market_statistics(tickers)

    print("Annualized Expected Returns:\n", mu)
    
    # 1. Generate 2000 Bayesian Scenarios
    print("\nRunning Bayesian Simulation...")
    scenarios, trace = get_model_scenarios(mu, returns, tickers)

    # 2. Run C++ PSO on all 2000 Scenarios
    print("\nRunning Ensemble PSO...")
    all_weights = run_ensemble_optimization(scenarios, sigma, tickers)

    # 3. Average the results to find the most "Robust" portfolio
    final_weights = all_weights.mean(axis=0)

    print("\n" + "="*30)
    print("BAYESIAN ROBUST OPTIMIZATION RESULTS")
    print("="*30)
    for ticker, weight in zip(tickers, final_weights):
        print(f"{ticker}: {weight:.2%}")
    
    plot_dashboard(all_weights, trace, tickers, returns)

if __name__ == "__main__":
    main()