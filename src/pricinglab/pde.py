"""
Explicit finite difference method for solving the Black-Scholes PDE.

Assumes:
- European exercise (no early exercise)
- No dividends
- Constant volatility and risk-free rate
- Frictionless market (no transaction costs)

Convergence to the Black-Scholes analytic price improves
as the grid gets finer.

The explicit scheme is only stable when the time step is small enough
relative to the spatial step (the CFL stability condition).
"""

import numpy as np


def explicit_fd_european(S: float, K: float, r: float, sigma: float, T: float, M: int, N: int, S_max: float, option_type: str) -> float:
    """
    Explicit finite difference method for solving the Black-Scholes PDE.

    Parameters
    ----------
    S           : current spot price
    K           : strike price
    r           : continuously-compounded risk-free rate (annual)
    sigma       : annualised volatility
    T           : time to maturity in years
    M           : number of stock price steps (points on the price axis)
    N           : number of time steps (points on the time axis)
    S_max       : maximum stock price on the grid - capped because we can't
                  have an infinite grid, usually 3-4x the strike
    option_type : "call" or "put"

    Returns
    -------
    float : estimated option price
    """
    dt = T / N          # length of each time step (T split into N steps)
    dS = S_max / M      # size of each price step (S_max split into M steps)

    # grid spacing is dt going down (time) and dS going across (price)

    # all the stock prices on the grid: 0, dS, 2dS, ... up to S_max
    S_values = np.linspace(0, S_max, M + 1)

    # the value grid itself - rows are stock prices, columns are time points.
    # start it as all zeros and fill it in
    grid = np.zeros((M + 1, N + 1))

    # expiry is the last column - here the optio n is just worth its payoff.
    # this is what we step backwards from
    if option_type == "call":
        grid[:, -1] = np.maximum(S_values - K, 0)   # [:, -1] = every row, last column
    else:
        grid[:, -1] = np.maximum(K - S_values, 0)

    # now the top and bottom edges - what the option is worth at the extreme
    # stock prices (S=0 and S=S_max) for every point in time
    time_steps = np.arange(N + 1)
    if option_type == "call":
        grid[0, :] = 0      # at S=0 a call is worthless, you'd never exercise
        # at S_max the call is deep in the money - roughly S_max minus the
        # discounted strike, discount depends on time left
        grid[-1, :] = S_max - K * np.exp(-r * dt * (N - time_steps))
    else:
        grid[0, :] = K * np.exp(-r * dt * (N - time_steps))
        grid[-1, :] = 0

    # the a, b, c weights come from approximating the derivatives in the
    # Black-Scholes PDE. each interior point is built from 3 points in the
    # next column: the one below (a), level (b) and above (c)
    j = np.arange(1, M)
    a = 0.5 * dt * (sigma**2 * j**2 - r * j)
    b = 1 - dt * (sigma**2 * j**2 + r)
    c = 0.5 * dt * (sigma**2 * j**2 + r * j)

    # walk backwards through time, expiry -> today, filling each column from
    # the one to its right. same idea as the binomial tree but using these
    # PDE weights instead of risk-neutral probabilities
    for i in range(N - 1, -1, -1):
        grid[1:M, i] = (
            a * grid[0:M-1, i+1]
            + b * grid[1:M, i+1]
            + c * grid[2:M+1, i+1]
        )

    # grid is full now. read off today's column (column 0) at the actual spot S.
    # S probably isn't exactly on a grid line so interpolate between neighbours
    price = np.interp(S, S_values, grid[:, 0])
    return float(price)