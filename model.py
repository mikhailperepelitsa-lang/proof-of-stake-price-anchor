"""Dual-Consumption Model of a Proof-of-Stake network.

Reference implementation of the dynamical system analysed in:

    M. Perepelitsa, "Proof-of-Stake Dynamics: The Elusive Price Anchor and
    Endogenous Volatility Harvesting."

This module contains the model only and produces no output.  ``make_figures.py``
uses it to draw the two figures; ``verify.py`` uses it to reproduce the
numerical claims made in the manuscript.

Model summary
-------------
Two agent classes share one token.  The *Consumer* is a utility user that spends
a fixed fraction ``nu`` of its fiat-denominated wealth each period and burns
tokens for transactions; the *Investor* holds stake purely for yield and injects
or withdraws exogenous fiat ``Lambda^t``.  Per period::

    y^t       = c / sqrt(S^t)                                     native yield
    p^{t+1}   = (Lambda^{t+1} + xi I_c) / (nu S_c^t (1 + y^t))     market clearing
    L_c^{t+1} = gamma / (p^{t+1} (1 + y^{t+1}))                    transactional burn
    S^{t+1}   = S^t (1 + y^t) - L_c^{t+1}                          total staked supply
    S_c^{t+1} = xi S_c^t (1 + y^t) + (xi I_c - gamma/(1+y^{t+1})) / p^{t+1}
    S_i^{t+1} = S_i^t (1 + y^t) + Lambda^{t+1} / p^{t+1}
    W_c^{t+1} = p^{t+1} S_c^t (1 + y^t) + I_c^{t+1}

Setting ``Lambda == 0`` and ``S_i == 0`` recovers the Consumer-only economy.

The supply update is *implicit*: ``S^{t+1}`` appears on both sides, because the
burn depends on ``y^{t+1} = c / sqrt(S^{t+1})``.  It is the only step solved
numerically (Brent's method, :func:`solve_next_S`); every other step is a
closed-form assignment.

:func:`simulate_dual` advances ``S``, ``S_c`` and ``S_i`` by three separate
equations and does *not* impose ``S = S_c + S_i``.  That identity is a
consequence of the model rather than an input to it, so the agreement of the
three paths is a check on this implementation; ``verify.py`` tests it.

Parameter provenance
--------------------
======== ========== =========================================== ========
Symbol   Value      Meaning                                     Source
======== ========== =========================================== ========
NU       0.01       Consumer fiat consumption propensity        Table 2
XI       0.99       Crypto retention rate, xi = 1 - nu          Table 2
GAMMA    8.6625e7   Gas/utility demand parameter (USD)          Table 2
C        15.81      Network issuance parameter                  Table 2
IC0      3.50e8     Baseline monthly fiat inflow (USD)          Table 2
S0       4.00e7     Initial physical staked supply (ETH)        Table 2
======== ========== =========================================== ========

``C`` follows from the targeted 3% annualised staking yield: with a monthly
``y* = 0.0025`` and ``S* = 4e7``, ``c = y* sqrt(S*) = 15.81``.  ``GAMMA`` follows
from ``y* = nu gamma / (xi I_c)``.  One period is one month.

Usage
-----
    >>> import model as m
    >>> m.steady_state(m.IC0)            # (S*, p*, y*) at the baseline inflow
    >>> m.simulate_dual(80, Lam_path)    # 80-month two-class path

Run ``python3 model.py`` for a self-check.  Requires numpy and scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "NU", "XI", "GAMMA", "C", "IC0", "S0",
    "yield_of", "solve_next_S", "steady_state",
    "manifold_price", "clearing_locus_price",
    "simulate_consumer_only", "simulate_dual",
]

# --------------------------------------------------------------------------
# Calibration (Table 2).  One period = one month.
# --------------------------------------------------------------------------
NU = 0.01             # nu    : Consumer fiat consumption propensity
XI = 1.0 - NU         # xi    : crypto retention rate
GAMMA = 86.625e6      # gamma : gas/utility demand parameter, USD
C = 15.81             # c     : network issuance parameter
IC0 = 350e6           # I_c   : baseline monthly fiat inflow, USD
S0 = 40e6             # S^0   : initial physical staked supply, ETH


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def yield_of(S):
    """Native staking yield ``y(S) = c / sqrt(S)``.

    Parameters
    ----------
    S : float or ndarray
        Physical staked supply, in tokens.
    """
    return C / np.sqrt(S)


def solve_next_S(S_now, p_next, gamma=GAMMA, c=C):
    """Solve the total-supply update for ``S^{t+1}``.

    Solves for ``S_next`` in::

        S_next = S_now (1 + y(S_now)) - gamma / (p_next (1 + c / sqrt(S_next)))

    which is implicit because the burn depends on the next-period yield.  The
    residual is strictly increasing in ``S_next``, so the root is unique and is
    bracketed by ``(0, S_now (1 + y(S_now)))``.

    If the burn would exceed the gross supply the gross supply is returned
    unchanged, which keeps the path defined at prices low enough to leave the
    model's domain.
    """
    gross = S_now * (1.0 + yield_of(S_now))

    def residual(S_next):
        return S_next - gross + gamma / (p_next * (1.0 + c / np.sqrt(S_next)))

    if residual(gross) < 0:
        return gross
    return brentq(residual, 1.0, gross, xtol=1e-8, rtol=1e-14, maxiter=500)


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------
def steady_state(Ic):
    """Consumer-only steady state for a constant fiat inflow ``Ic``.

    Returns ``(S, p, y)``, the closed forms of Eq. (23)::

        S* = (c xi I_c / (gamma nu))^2
        p* = gamma^2 nu / (c^2 (xi I_c + gamma nu))
        y* = gamma nu / (xi I_c)
    """
    S = (C * XI * Ic / (GAMMA * NU)) ** 2
    p = GAMMA ** 2 * NU / (C ** 2 * (XI * Ic + GAMMA * NU))
    y = GAMMA * NU / (XI * Ic)
    return S, p, y


def manifold_price(S):
    """Steady-state manifold in the ``(S, p)`` plane.

    Eliminating ``I_c`` from Eq. (23) via ``xi I_c = gamma nu sqrt(S) / c``
    gives ``p = gamma / (c (sqrt(S) + c))``.  Solid black curve of Fig. 1.
    """
    return GAMMA / (C * (np.sqrt(S) + C))


def clearing_locus_price(S, Ic):
    """Asset-market clearing locus, Eq. (25).  Dashed curve of Fig. 1."""
    return (XI * Ic / NU - GAMMA / (1.0 + yield_of(S))) / S


# --------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------
def simulate_consumer_only(T, Ic_path, S_init=S0):
    """Consumer-only economy over ``T`` periods.

    Parameters
    ----------
    T : int
        Number of periods to simulate.
    Ic_path : sequence of float, length ``T + 1``
        Fiat inflow per period; ``Ic_path[t]`` arrives at period ``t``.
    S_init : float
        Physical staked supply at ``t = 0``.

    Returns
    -------
    S : ndarray, shape (T + 1,)
        Physical staked supply.
    p : ndarray, shape (T + 1,)
        Clearing price.  ``p[0]`` is ``nan``: the price at ``t = 0`` is not
        determined by the recursion.
    """
    S = np.empty(T + 1)
    p = np.full(T + 1, np.nan)
    S[0] = S_init
    for t in range(T):
        p[t + 1] = XI * Ic_path[t + 1] / (NU * S[t] * (1.0 + yield_of(S[t])))
        S[t + 1] = solve_next_S(S[t], p[t + 1])
    return S, p


def simulate_dual(T, Lam_path, Ic=IC0, S_c0=10e6, S_i0=30e6):
    """Consumer/Investor economy over ``T`` periods.

    Parameters
    ----------
    T : int
        Number of periods to simulate.
    Lam_path : sequence of float, length ``T + 1``
        Investor capital flow; positive is an injection, negative a withdrawal.
        Requires ``Lam_path[t] > -XI * Ic`` for the price to stay positive.
    Ic : float
        Constant Consumer fiat inflow per period.
    S_c0, S_i0 : float
        Consumer and Investor staked balances at ``t = 0``.

    Returns
    -------
    dict
        Keys ``S_c``, ``S_i``, ``S``, ``p``, ``W_c``, ``L_c``, ``y`` and
        ``acc_wealth``, each an ndarray of shape ``(T + 1,)``.  ``acc_wealth``
        is the Consumer's accumulated fiat-denominated wealth,
        ``sum_{k<=t} (nu W_c^k - I_c) + W_c^t``, plotted in Fig. 2(b).
    """
    S_c = np.empty(T + 1)
    S_i = np.empty(T + 1)
    S = np.empty(T + 1)
    p = np.empty(T + 1)
    W_c = np.empty(T + 1)
    L_c = np.empty(T + 1)

    S_c[0], S_i[0] = S_c0, S_i0
    S[0] = S_c0 + S_i0
    W_c[0] = Ic / NU
    # p^0 follows from S_c^0 = xi W_c^0 / p^0 - L_c^0.
    y0 = yield_of(S[0])
    p[0] = (XI * W_c[0] - GAMMA / (1.0 + y0)) / S_c[0]
    L_c[0] = GAMMA / (p[0] * (1.0 + y0))

    for t in range(T):
        y_t = yield_of(S[t])
        Lam = Lam_path[t + 1]

        p[t + 1] = (Lam + XI * Ic) / (NU * S_c[t] * (1.0 + y_t))
        S[t + 1] = solve_next_S(S[t], p[t + 1])
        y_next = yield_of(S[t + 1])

        L_c[t + 1] = GAMMA / (p[t + 1] * (1.0 + y_next))
        S_c[t + 1] = (XI * S_c[t] * (1.0 + y_t)
                      + (XI * Ic - GAMMA / (1.0 + y_next)) / p[t + 1])
        S_i[t + 1] = S_i[t] * (1.0 + y_t) + Lam / p[t + 1]
        W_c[t + 1] = p[t + 1] * S_c[t] * (1.0 + y_t) + Ic

    net_fiat = np.concatenate(([0.0], np.cumsum(NU * W_c[1:] - Ic)))

    return {
        "S_c": S_c,
        "S_i": S_i,
        "S": S,
        "p": p,
        "W_c": W_c,
        "L_c": L_c,
        "y": yield_of(S),
        "acc_wealth": net_fiat + W_c,
    }


# --------------------------------------------------------------------------
if __name__ == "__main__":
    S_eq, p_eq, y_eq = steady_state(IC0)
    lam = (1 + y_eq / 2) / (1 + y_eq + y_eq * y_eq / 2)
    half_life = np.log(0.5) / np.log(lam)

    print("Consumer-only steady state at the baseline calibration")
    print(f"  I_c    = {IC0:,.0f} USD/month")
    print(f"  S*     = {S_eq / 1e6:.3f} million ETH")
    print(f"  p*     = {p_eq:,.2f} USD")
    print(f"  y*     = {y_eq:.6f} per month ({12 * y_eq:.2%} annualised, simple)")
    print(f"  lambda = {lam:.8f}")
    print(f"  half-life = {half_life:,.2f} months = {half_life / 12:.2f} years")
