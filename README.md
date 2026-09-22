# Proof-of-Stake Dynamics — replication code

Reference implementation of the Dual-Consumption Model, together with the scripts
that generate the figures and check every numerical claim in:

> M. Perepelitsa, *Proof-of-Stake Dynamics: The Elusive Price Anchor and Endogenous
> Volatility Harvesting.* Submitted to **Ledger**.

Three short Python files, no build step, no configuration. The whole repository runs
in a few seconds.

## Quick start

```bash
pip install -r requirements.txt
python3 verify.py         # check the paper's numbers against the model
python3 make_figures.py   # regenerate Figures 1 and 2
```

`verify.py` exits non-zero if any check fails, so it can be wired into CI.

## Contents

| File | Purpose |
| --- | --- |
| `model.py` | The model. Parameters, closed forms, and the two simulators. No output. |
| `make_figures.py` | Draws Figures 1 and 2 to the Ledger figure specification. |
| `verify.py` | Reproduces every number quoted in the manuscript; prints PASS/FAIL. |
| `requirements.txt` | Dependencies. |

## The model

Two agent classes share one token. The **Consumer** is a utility user that spends a
fixed fraction `nu` of its fiat wealth each period and burns tokens for transactions.
The **Investor** holds stake purely for yield and injects or withdraws exogenous fiat
`Lambda^t`. Each period:

```
y^t       = c / sqrt(S^t)                                     native staking yield
p^{t+1}   = (Lambda^{t+1} + xi I_c) / (nu S_c^t (1 + y^t))    market clearing
L_c^{t+1} = gamma / (p^{t+1} (1 + y^{t+1}))                   transactional burn
S^{t+1}   = S^t (1 + y^t) - L_c^{t+1}                         total staked supply
S_c^{t+1} = xi S_c^t (1 + y^t) + (xi I_c - gamma/(1+y^{t+1})) / p^{t+1}
S_i^{t+1} = S_i^t (1 + y^t) + Lambda^{t+1} / p^{t+1}
W_c^{t+1} = p^{t+1} S_c^t (1 + y^t) + I_c^{t+1}
```

Setting `Lambda == 0` and `S_i == 0` recovers the Consumer-only economy of Section 3.

The supply update is the only step solved numerically. It is implicit in `S^{t+1}`,
because the burn depends on `y^{t+1} = c / sqrt(S^{t+1})`; `solve_next_S` brackets the
root on `(0, S^t(1 + y^t))` and calls Brent's method. Every other step is a closed-form
assignment.

`simulate_dual` advances `S`, `S_c` and `S_i` by three separate equations and does
**not** impose `S = S_c + S_i`. That identity is a consequence of the model, not an
input to it, so the agreement of the three paths is a genuine check on the
implementation. `verify.py` tests it (it holds to `4e-15` relative error).

### Calibration

| Symbol | Value | Meaning | Source |
| --- | --- | --- | --- |
| `NU` | 0.01 | Consumer fiat consumption propensity | Table 2 |
| `XI` | 0.99 | Crypto retention rate, `xi = 1 - nu` | Table 2 |
| `GAMMA` | 8.6625e7 | Gas/utility demand parameter (USD) | Table 2 |
| `C` | 15.81 | Network issuance parameter | Table 2 |
| `IC0` | 3.50e8 | Baseline monthly fiat inflow (USD) | Table 2 |
| `S0` | 4.00e7 | Initial physical staked supply (ETH) | Table 2 |

One period is one month. `C` follows from the targeted 3% annualised yield: with
monthly `y* = 0.0025` and `S* = 4e7`, `c = y* sqrt(S*) = 15.81`. `GAMMA` follows from
`y* = nu gamma / (xi I_c)`.

## Reproducing the figures

`make_figures.py` writes four files into the working directory:

```
Consumer_Attractor_ETH.png    Consumer_Attractor_ETH.pdf     Figure 1
Buy-Sell_Shock_ETH.png        Buy-Sell_Shock_ETH.pdf         Figure 2
```

PNGs are 600 DPI; PDFs are vector. Both follow the Ledger Author Guide, Sec. 5:
8 pt Arial labels, ½ pt black axes and frames, ¼ pt gray gridlines, data lines at
1 pt or heavier, grayscale with differentiation by line style and marker shape, and
lower-case `(a)`–`(d)` panel labels on the multipart figure.

**Fonts.** Arial is substituted by Liberation Sans, which is metric-compatible with
it, so glyph widths and line breaks are identical. On Debian/Ubuntu:

```bash
sudo apt-get install fonts-liberation
```

Without it matplotlib falls back to DejaVu Sans and labels come out slightly wider
than specified.

**Sizing — important.** The figures are generated at their *final printed size*
(4.6 × 3.3 in and 5.8 × 4.8 in), so the 8 pt labels print as genuine 8 pt only if the
graphic is not rescaled. The manuscript includes them with explicit widths:

```latex
\includegraphics[width=4.6in]{Consumer_Attractor_ETH.png}
\includegraphics[width=5.8in]{Buy-Sell_Shock_ETH.png}
```

5.8 in fits inside Ledger's text block on both US Letter and A4 at the journal's
3.1 cm side margins. Using `width=\textwidth` would rescale the graphic and push the
label sizes off specification.

## Verifying the manuscript's numbers

`verify.py` re-derives every figure quoted in the paper and compares it against the
stated value, grouped by where the claim appears:

| Group | What is checked |
| --- | --- |
| Table 2 | Baseline calibration: `y*`, annualised yield, `S*` |
| Theorem 1 | Relaxation rate `lambda`, half-life in months and years |
| Figure 1 | The three states and the `t = 1200` trajectory endpoint |
| Proposition 1 | Impact ratio, long-run ratio, overshoot, and the sign reversal |
| Table 3 | Consumer/Investor calibration: `p^0`, `W_c^0` |
| Figure 2 / Sec. 4.1 | Price peak, trough and final level; `S_c`, `S_i`, `S`; stake shares |
| Theorem 2 | Dilution-map identity, the hypothesis `Lambda_bar_D <= 0`, the conclusion, and the frictionless closed form |
| Identities | `S = S_c + S_i`, the clearing identity for `D^t`, the fiat balance, and cumulative fiat neutrality |

Tolerance is 0.5% by default, since the paper quotes rounded figures; tighter
tolerances are set per check where the paper is more precise. All 39 checks pass.

## Notes and limitations

- These are replication scripts, not a library. Parameters are module-level constants
  in `model.py`; change them there, or pass overrides to `steady_state`,
  `simulate_consumer_only` and `simulate_dual`, which accept them as arguments.
- The model itself is stylized in the ways set out in Section 5 of the paper: aggregate
  agent classes, myopic price expectations, and no transaction costs, slippage, taxes,
  staking delays, leverage or strategic Investor behaviour.
- `simulate_dual` will run past the model's domain if the Investor withdraws more than
  it holds (`S_i` goes negative). This happens for cycles substantially longer than the
  one in the paper. The function does not guard against it; check `S_i` if you extend
  the horizon.
- Tested on Python 3.12 with numpy 2.4, scipy 1.17 and matplotlib 3.10. The pinned
  minimums in `requirements.txt` are lower and expected to work, but are not tested.

## License

Code is released under the MIT License (see `LICENSE`). Ledger publishes under
Creative Commons Attribution, so the manuscript and this code carry compatible terms.

## Citation

```bibtex
@article{Perepelitsa_PoSDynamics,
  author  = {Perepelitsa, Mikhail},
  title   = {Proof-of-Stake Dynamics: The Elusive Price Anchor and
             Endogenous Volatility Harvesting},
  journal = {Ledger},
  note    = {Submitted},
  year    = {2026}
}
```
