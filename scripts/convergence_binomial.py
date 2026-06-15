"""
Convergence of the CRR binomial tree to the Black-Scholes analytic price.

Runs the binomial tree for N = 1 to 500 steps and plots the price
at each N against the true Black-Scholes price.
"""

import numpy as np
import matplotlib.pyplot as plt
from pricinglab.analytic import bs_call
from pricinglab.lattice import crr_european

# Option parameters
S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

# Black-Scholes analytic price - the ground truth
bs_price = bs_call(S, K, r, sigma, T)

# Run binomial tree for N = 1 to 500 and store each price
steps = np.arange(1, 501)
prices = [crr_european(S, K, r, sigma, T, n, "call") for n in steps]

# Plot the convergence
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(steps, prices, color="steelblue", linewidth=0.8, label="CRR binomial tree")
ax.axhline(bs_price, color="crimson", linewidth=1.5, linestyle="--", label=f"Black-Scholes: {bs_price:.4f}")

ax.set_xlabel("Number of steps N")
ax.set_ylabel("Call option price (£)")
ax.set_title("Convergence of CRR Binomial Tree to Black-Scholes Price")
ax.legend()
ax.grid(True, alpha=0.3)

# Save to results/figures/
fig.savefig("results/figures/convergence_binomial.png", dpi=150, bbox_inches="tight")
print("Plot saved to results/figures/convergence_binomial.png")
plt.show()