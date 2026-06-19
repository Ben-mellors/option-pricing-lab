# Numerical Methods for Option Pricing
*Stability, convergence and early exercise*

This project prices European call and put options four ways: the closed-form
Black-Scholes formula, a Cox-Ross-Rubinstein binomial tree, Monte Carlo
simulation of the underlying price process, and an explicit finite-difference
solver for the Black-Scholes partial differential equation. The closed-form
price is exact, so it serves as a benchmark against which the three numerical
methods are tested for accuracy, convergence and stability. The binomial method
is then extended to American options, where the holder may exercise before
expiry - a feature with no closed-form solution that has to be priced
numerically.

## What's implemented

- `analytic.py` - Black-Scholes closed-form prices for European calls and puts;
  put-call parity is checked in the test suite.
- `lattice.py` - Cox-Ross-Rubinstein binomial tree for European and American
  options; early exercise is handled by a single comparison at each node.
- `montecarlo.py` - Monte Carlo pricing of European options by simulating
  geometric Brownian motion under the risk-neutral measure, with a fixed seed
  for reproducibility.
- `pde.py` - explicit finite-difference solver for the Black-Scholes PDE, used
  to examine numerical stability and convergence on a grid.

## Key results

All four methods agree on the price of a one-year at-the-money European call
(`S = K = 100`, `r = 5%`, `σ = 20%`):

| Method | Price | Difference from Black-Scholes |
| --- | ---: | ---: |
| Black-Scholes (closed form) | 10.4506 | - |
| Binomial tree (CRR, N = 5000) | 10.4502 | 0.0004 |
| Monte Carlo (10⁶ paths) | 10.4532 ± 0.0147 | 0.0026 |
| Finite difference (M = 200, N = 2000) | 10.4550 | 0.0044 |

Two results from the finite-difference solver stand out:

- **Stability is conditional.** The explicit scheme stays stable only when the
  number of time steps exceeds roughly σ²M²T - about 1600 for M = 200. The
  stable run above (N = 2000) gives 10.4550. Drop N below the threshold and the
  central coefficient `b = 1 − Δt(σ²i² + r)` turns negative; the scheme then
  amplifies error at every step and the price diverges to around −3.1×10⁵³
  within 100 time steps. (`results/figures/pde_stability.png`)
- **Convergence is second order.** With the stability condition met, the error
  against the exact price falls as roughly 1/M² - halving the space step
  quarters the error - which is the order expected from the central-difference
  stencil. (`results/figures/pde_convergence.png`)

The binomial and Monte Carlo methods have their own convergence figures in
`results/figures/`: the tree shows the characteristic odd-even oscillation
settling onto the Black-Scholes price, and Monte Carlo shows the 1/√n error
decay of a simulation estimate.

## Repository structure

```
option-pricing-lab/
├── src/
│   └── pricinglab/
│       ├── __init__.py
│       ├── analytic.py        # Black-Scholes closed form
│       ├── lattice.py         # CRR binomial tree (European + American)
│       ├── montecarlo.py      # Monte Carlo via GBM
│       ├── pde.py             # explicit finite-difference solver
│       └── utils.py           # shared helpers
├── scripts/
│   ├── convergence_binomial.py
│   ├── convergence_montecarlo.py
│   ├── convergence_pde.py
│   └── stability_pde.py
├── tests/
│   ├── test_analytic.py
│   └── test_numerical.py
├── results/
│   └── figures/
│       ├── convergence_binomial.png
│       ├── convergence_montecarlo.png
│       ├── pde_stability.png
│       └── pde_convergence.png
├── report/
│   ├── report.tex
│   └── report.pdf
├── pyproject.toml
└── README.md
```

## Installation and use

Clone the repository and install in editable mode with the development
dependencies:

```bash
git clone https://github.com/Ben-mellors/option-pricing-lab.git
cd option-pricing-lab
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash); macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Regenerate the figures:

```bash
python scripts/convergence_binomial.py
python scripts/convergence_montecarlo.py
python scripts/stability_pde.py
python scripts/convergence_pde.py
```

## Limitations and further work

The scope is deliberately narrow: vanilla European and American options under
constant volatility and interest rates, priced by four standard methods on a
single benchmark problem. The aim is to study numerical behaviour - convergence
and stability - against a known answer, not to model market frictions.

Some natural extensions:

- **American options beyond the lattice.** Early exercise is handled here only
  in the binomial tree, where the decision is a single comparison at each node.
  Extending it to Monte Carlo needs a regression-based method such as
  Longstaff-Schwartz, because a simulated path knows only its own future, not
  the continuation value at an interior point - that value has to be estimated
  by regressing discounted future payoffs on the current state. Extending it to
  the PDE needs the exercise constraint enforced at every grid point, which
  turns the problem into a linear complementarity problem solved by projected
  SOR or a penalty method.
- **Unconditional stability.** The explicit scheme's conditional stability
  forces a large number of time steps as the grid is refined. An implicit or
  Crank-Nicolson scheme is unconditionally stable and removes that constraint,
  at the cost of solving a linear system at each step.
- **Constant coefficients.** Volatility and the interest rate are fixed. Local
  or stochastic volatility, or a term structure of rates, would be the next
  step towards a more realistic model.