"""
Convergence of Monte Carlo estimates to the Black-Scholes analytic price.

Runs the Monte Carlo pricer at increasing numbers of simulations and
plots how the estimate converges, illustrating the characteristic
1/sqrt(n) error decay and random scatter.
"""

import numpy as np
import matplotlib.pyplot as plt
from pricinglab.analytic import bs_call
from pricinglab.montecarlo import mc_european

# Option parameters
S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

# Black-Scholes analytic price - the ground truth
bs_price = bs_call(S, K, r, sigma, T)

# Run Monte Carlo at increasing simulation counts
sim_counts = np.logspace(2, 6, 40).astype(int) #np.logspace(2, 6, 40) - creates 40 simulation counts spaced logarithmically from 10² (100) to 10⁶ (1,000,000). We use log spacing because we want to see behaviour across a huge range - from 100 to a million sims. .astype(int) converts them to whole numbers since you can't run half a simulation.
mc_prices = [mc_european(S, K, r, sigma, T, n, "call", seed=1) for n in sim_counts]

# Plot convergence
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sim_counts, mc_prices, "o-", color="seagreen", markersize=4, linewidth=0.8, label="Monte Carlo estimate")
ax.axhline(bs_price, color="crimson", linewidth=1.5, linestyle="--", label=f"Black-Scholes: {bs_price:.4f}")

ax.set_xscale("log")
ax.set_xlabel("Number of simulations (log scale)")
ax.set_ylabel("Call option price (£)")
ax.set_title("Convergence of Monte Carlo Estimate to Black-Scholes Price")
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig("results/figures/convergence_montecarlo.png", dpi=150, bbox_inches="tight")
print("Plot saved to results/figures/convergence_montecarlo.png")
plt.show()