"""
Demonstrates the CFL stability condition for the explicit finite difference
scheme. Same M (price steps), two different N (time steps) - one violates
the stability condition N >= sigma^2 * M^2 * T, one satisfies it comfortably.
"""
import matplotlib.pyplot as plt
from pricinglab.pde import explicit_fd_european

S, K, r, sigma, T, M, S_max = 100, 100, 0.05, 0.2, 1, 200, 300

# unstable: N=100 is far below the threshold (~1600 for these parameters)
_, S_unstable, V_unstable = explicit_fd_european(
    S=S, K=K, r=r, sigma=sigma, T=T, M=M, N=100, S_max=S_max,
    option_type="call", return_grid=True,
)

# stable: N=2000 comfortably clears the threshold
_, S_stable, V_stable = explicit_fd_european(
    S=S, K=K, r=r, sigma=sigma, T=T, M=M, N=2000, S_max=S_max,
    option_type="call", return_grid=True,
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(S_unstable, V_unstable, color="crimson", linewidth=0.8)
axes[0].set_title("N = 100 (unstable)")
axes[0].set_xlabel("Stock price S")
axes[0].set_ylabel("Option value V(S, 0)")

axes[1].plot(S_stable, V_stable, color="seagreen", linewidth=1.2)
axes[1].set_title("N = 2000 (stable)")
axes[1].set_xlabel("Stock price S")

fig.suptitle("Explicit FD scheme: CFL stability condition (M = 200 fixed)")
fig.tight_layout()
fig.savefig("results/figures/pde_stability.png", dpi=150)
print("Saved to results/figures/pde_stability.png")