r"""Check every numerical claim in the manuscript against the model.

    M. Perepelitsa, "Proof-of-Stake Dynamics: The Elusive Price Anchor and
    Endogenous Volatility Harvesting."

Run ``python3 verify.py``.  Each row prints the value stated in the paper, the
value computed here, and PASS or FAIL.  The script exits with status 1 if any
check fails, so it can be used in CI.

Checks are grouped by where the claim appears:

    Table 2      calibration of the Consumer-only economy
    Theorem 1    relaxation rate and half-life
    Figure 1     the three states and the trajectory endpoint
    Proposition 1  impact vs. long-run response to a permanent demand shock
    Table 3      calibration of the Consumer/Investor economy
    Figure 2     plotted paths, and Sec. 4.1 of the text
    Theorem 2    hypothesis and conclusion of the harvesting result
    Identities   accounting relations the implementation must satisfy

Requires numpy and scipy.
"""

from __future__ import annotations

import sys

import numpy as np

import model as m

TOL_REL = 5e-3          # 0.5%: the paper quotes rounded figures
_results: list[bool] = []


def check(label, stated, computed, unit="", tol=TOL_REL, fmt="{:,.2f}"):
    """Compare a stated value with a computed one and record the outcome."""
    if stated == 0:
        ok = abs(computed) < 1e-9
    else:
        ok = abs(computed - stated) <= tol * abs(stated)
    _results.append(ok)
    su = f"{fmt.format(stated)} {unit}".strip()
    cu = f"{fmt.format(computed)} {unit}".strip()
    print(f"  {label:<44} paper {su:>18}   model {cu:>18}   "
          f"{'PASS' if ok else 'FAIL'}")


def check_true(label, condition, detail=""):
    """Record a qualitative claim."""
    _results.append(bool(condition))
    print(f"  {label:<44} {detail:>40}   {'PASS' if condition else 'FAIL'}")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------
def main():
    xIc = m.XI * m.IC0

    # ---------------------------------------------------------- Table 2
    section("Table 2  baseline calibration")
    S_eq, p_eq, y_eq = m.steady_state(m.IC0)
    check("monthly staking yield y*", 0.0025, y_eq, fmt="{:.6f}")
    check("annualised yield (simple)", 0.03, 12 * y_eq, fmt="{:.4f}")
    check("steady-state supply S*", 40e6, S_eq, "ETH", fmt="{:,.0f}")

    # -------------------------------------------------------- Theorem 1
    section("Theorem 1  relaxation")
    lam = (1 + y_eq / 2) / (1 + y_eq + y_eq ** 2 / 2)
    half_life = np.log(0.5) / np.log(lam)
    check("relaxation rate lambda", 0.99875, lam, fmt="{:.6f}")
    check("half-life", 554.17, half_life, "months")
    check("half-life", 46.0, half_life / 12, "years", tol=1e-2)

    # --------------------------------------------------------- Figure 1
    section("Figure 1  response to a permanent 50% inflow increase")
    Ic_new = 1.5 * m.IC0
    T1 = 1200
    Ic_path = np.full(T1 + 1, Ic_new)
    Ic_path[0] = m.IC0
    S_path, p_path = m.simulate_consumer_only(T1, Ic_path, S_init=m.S0)
    S_new, p_new, _ = m.steady_state(Ic_new)

    check("State 1  supply", 40e6, S_eq, "ETH", fmt="{:,.0f}")
    check("State 1  price", 864.0, p_eq, "USD")
    check("State 2  price (impact)", 1296.0, p_path[1], "USD")
    check("State 3  supply", 90e6, S_new, "ETH", fmt="{:,.0f}")
    check("State 3  price", 577.0, p_new, "USD")
    check("trajectory end t=1200  supply", 69.3e6, S_path[T1], "ETH", fmt="{:,.0f}")
    check("trajectory end t=1200  price", 749.0, p_path[T1], "USD")

    # --------------------------------------------------- Proposition 1
    section("Proposition 1  impact vs. long-run response")
    kappa = 1.5
    impact = m.XI * kappa * m.IC0 / (m.NU * S_eq * (1 + m.yield_of(S_eq)))
    check("impact ratio  p^1/p*", kappa, impact / p_eq, fmt="{:.4f}")
    check("long-run ratio  p**/p*", (1 + y_eq) / (kappa + y_eq), p_new / p_eq,
          fmt="{:.4f}")
    check("overshoot  p^1/p**", kappa * (kappa + y_eq) / (1 + y_eq),
          impact / p_new, fmt="{:.4f}")
    check_true("long-run price below pre-shock price", p_new < p_eq,
               f"{p_new:,.2f} < {p_eq:,.2f} USD")

    # ---------------------------------------------------------- Table 3
    section("Table 3  Consumer/Investor calibration")
    T2 = 80
    Lam = np.zeros(T2 + 1)
    Lam[20:31] = 300e6
    Lam[40:51] = -300e6
    r = m.simulate_dual(T2, Lam)
    check("initial price p^0", 3456.0, r["p"][0], "USD")
    check("initial Consumer wealth W_c^0", 35e9, r["W_c"][0], "USD", fmt="{:,.0f}")

    # --------------------------------------------------- Figure 2 / Sec 4.1
    section("Figure 2 and Sec. 4.1  Buy-Sell cycle")
    check("price peak", 6700.0, r["p"].max(), "USD", tol=1e-2)
    check("price trough", 300.0, r["p"].min(), "USD", tol=5e-2)
    check("price at t=80", 2100.0, r["p"][T2], "USD", tol=5e-2)
    check("S_c at t=80", 16e6, r["S_c"][T2], "ETH", tol=1e-2, fmt="{:,.0f}")
    check("S_i at t=80", 27.5e6, r["S_i"][T2], "ETH", tol=1e-2, fmt="{:,.0f}")
    check("S at t=80", 43.6e6, r["S"][T2], "ETH", tol=1e-2, fmt="{:,.0f}")
    check("Consumer share at t=0", 0.25, r["S_c"][0] / r["S"][0], fmt="{:.3f}")
    check("Consumer share at t=80", 0.37, r["S_c"][T2] / r["S"][T2], fmt="{:.3f}")
    check("accumulated wealth returns to anchor", 35e9, r["acc_wealth"][T2],
          "USD", fmt="{:,.0f}")

    # -------------------------------------------------------- Theorem 2
    section("Theorem 2  volatility harvesting")
    D = np.array([m.NU * r["S_c"][t - 1] * (1 + r["y"][t - 1])
                  for t in range(1, T2 + 1)])
    Lam_t = Lam[1:]
    g = Lam_t / (Lam_t + xIc)
    phi = m.XI * (Lam_t + m.IC0) / (Lam_t + xIc)

    check_true("phi = 1 - nu g  (dilution map identity)",
               np.allclose(phi, 1 - m.NU * g),
               f"max err {np.max(np.abs(phi - (1 - m.NU * g))):.2e}")
    lam_bar = float(np.sum(D * Lam_t) / np.sum(D))
    check("weighted mean injection  Lambda_bar_D", -8.17e6, lam_bar, "USD",
          tol=1e-2, fmt="{:,.0f}")
    check_true("hypothesis  Lambda_bar_D <= 0", lam_bar <= 0,
               f"{lam_bar / 1e6:,.2f} M USD")
    net = float(-np.sum(D * g))
    check_true("conclusion  Delta S_c,net > 0", net > 0,
               f"{net / 1e6:,.3f} M ETH")
    check("net trading transfer", 8.25e6, net, "ETH", tol=1e-2, fmt="{:,.0f}")

    # frictionless closed form of the Remark
    prod = float(np.prod(phi))
    check_true("frictionless  prod phi > 1", prod > 1, f"{prod:.6f}")
    check("frictionless  (D^1/nu)(prod phi - 1)", 8.92e6,
          D[0] / m.NU * (prod - 1), "ETH", tol=1e-2, fmt="{:,.0f}")

    # ------------------------------------------------------- Identities
    section("Identities  the implementation must satisfy these")
    check_true("S = S_c + S_i at every period",
               np.allclose(r["S"], r["S_c"] + r["S_i"], rtol=1e-9),
               f"max rel err {np.max(np.abs(r['S'] - r['S_c'] - r['S_i']) / r['S']):.2e}")
    D_check = np.array([(Lam[t] + xIc) / r["p"][t] for t in range(1, T2 + 1)])
    check_true("D^t = (Lambda^t + xi I_c)/p^t",
               np.allclose(D, D_check, rtol=1e-9),
               f"max rel err {np.max(np.abs(D - D_check) / D):.2e}")
    check_true("nu W_c^t = Lambda^t + I_c  (fiat balance)",
               np.allclose(m.NU * r["W_c"][1:], Lam[1:] + m.IC0, rtol=1e-9),
               f"max rel err "
               f"{np.max(np.abs(m.NU * r['W_c'][1:] - Lam[1:] - m.IC0) / (Lam[1:] + m.IC0)):.2e}")
    check_true("cumulative fiat neutrality  sum(I_c - nu W_c) = 0",
               abs(np.sum(m.IC0 - m.NU * r["W_c"][1:])) < 1e-3,
               f"{np.sum(m.IC0 - m.NU * r['W_c'][1:]):.3e} USD")

    # ----------------------------------------------------------- summary
    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed.")
    return 0 if n == total else 1


if __name__ == "__main__":
    sys.exit(main())
