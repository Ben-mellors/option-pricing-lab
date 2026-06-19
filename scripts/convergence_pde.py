"""
Convergence of the explicit finite difference scheme as M (price steps)
increases, with N (time steps) scaled at each M to satisfy the CFL
stability condition N >= sigma^2 * M^2 * T. Compares against the
Black-Scholes analytic price.
"""
import numpy as np
import matplotlib.pyplot as plt
from pricinglab.analytic import bs_call
from pricinglab.pde import explicit_fd_european

S, K, r, sigma, T, S_max = 100, 100, 0.05, 0.2, 1, 300
true_price = bs_call(S, K, r, sigma, T)

M_values = [10, 20, 40, 80, 160, 320]
errors = []

for M in M_values:
    # stay safely inside the stable region: scale N to clear the threshold
    # with margin, since the exact boundary is sensitive to rounding
    N = int(np.ceil(1.5 * sigma**2 * M**2 * T))
    price = explicit_fd_european(
        S=S, K=K, r=r, sigma=sigma, T=T, M=M, N=N, S_max=S_max,
        option_type="call",
    )
    error = abs(price - true_price)
    errors.append(error)
    print(f"M={M:4d}  N={N:7d}  price={price:.6f}  error={error:.2e}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(M_values, errors, marker="o", color="seagreen")
ax.set_xlabel("M (price steps)")
ax.set_ylabel("Absolute error vs Black-Scholes")
ax.set_title("Explicit FD convergence (stable region, N scaled to M)")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/pde_convergence.png", dpi=150)
print("Saved to results/figures/pde_convergence.png")