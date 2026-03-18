# Bayesian PSO Portfolio Optimizer

A high-performance portfolio optimization engine built with a C++ backend and a Python frontend. This project uses Particle Swarm Optimization (PSO) combined with Bayesian inference to find robust, risk-adjusted asset allocations.

## 🧠 1. Theoretical Foundation

The engine maximizes the **Sharpe Ratio** while strictly penalizing over-concentration (Idiosyncratic Risk).

* **Expected Portfolio Return:**
    $$R_p = \sum_{i=1}^{n} w_i \mu_i$$
* **Portfolio Volatility (Risk):**
    $$\sigma_p = \sqrt{\sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j Cov(R_i, R_j)}$$
* **Constrained Sharpe Ratio (Fitness Function):**
    To prevent 100% allocations ("corner solutions"), the engine applies a quadratic penalty if any asset weight $w_i$ exceeds a 40% threshold:
    $$S = \frac{R_p - R_f}{\sigma_p} - \sum_{w_i > 0.4} 10.0 (w_i - 0.4)^2$$

## 🐬 2. Particle Swarm Optimization (PSO)

The C++ backend simulates a swarm of particles navigating the multi-dimensional weight space. Each particle adjusts its trajectory based on its own memory and the swarm's collective intelligence.

* **Velocity Update:**
    $$V_{i}^{t+1} = wV_{i}^t + c_1 r_1 (P_{best} - X_i^t) + c_2 r_2 (G_{best} - X_i^t)$$
    * $w = 0.7$: Inertia (Momentum)
    * $c_1 = 1.5$: Cognitive Coefficient (Personal Best)
    * $c_2 = 1.5$: Social Coefficient (Global Best)
* **Position Update:**
    $$X_{i}^{t+1} = X_{i}^t + V_{i}^{t+1}$$

## 📊 3. Bayesian Inference Layer

Standard optimizers fail because they treat historical averages as guaranteed facts. This engine uses **PyMC** to model market uncertainty.

* **Fat-Tailed Likelihood:** Market returns are modeled using a Student-T distribution to account for "Black Swan" events.
* **Ensemble Optimization:** Markov Chain Monte Carlo (MCMC) generates thousands of possible future market scenarios. The C++ engine solves the optimal portfolio for *every single scenario*, averaging the results into one highly robust, final allocation.

## 🛠️ 4. Tech Stack & Architecture

* **Backend:** C++14 (High-speed vector math and swarm mechanics).
* **Bridge:** PyBind11 & CMake (Compiles C++ into a native Python `.so` module).
* **Frontend:** Python 3.x, PyMC (Bayesian sampling), `yfinance` (Data), Pandas/NumPy, Matplotlib/Seaborn (Dashboard generation).