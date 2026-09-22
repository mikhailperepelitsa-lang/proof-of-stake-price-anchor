r"""Regenerate Figures 1 and 2 of the manuscript to the Ledger figure specification.

    M. Perepelitsa, "Proof-of-Stake Dynamics: The Elusive Price Anchor and
    Endogenous Volatility Harvesting."

Run ``python3 make_figures.py`` to write, into the current directory:

    Consumer_Attractor_ETH.png / .pdf    Figure 1
    Buy-Sell_Shock_ETH.png     / .pdf    Figure 2

and print the key plotted values to stdout for checking against the text.

Ledger figure specification (Author Guide, Sec. 5 "Figures")
-----------------------------------------------------------
* text labels in 8 pt Arial
* axes, frames and tick marks black, 1/2 pt
* gridlines gray, 1/4 pt
* plotted functions and continuous data 1 pt or heavier
* black and white, with differentiation achieved by gray scaling
* multipart figures denoted (a), (b), (c) ..., lower case, near the bottom
* raster output at 600 DPI

Arial is substituted by Liberation Sans, which is metric-compatible with it, so
glyph widths and therefore line breaks are identical.  If neither font is
installed matplotlib falls back to DejaVu Sans and the labels will be slightly
wider than specified; install ``fonts-liberation`` to avoid this.

Figure sizing
-------------
Both figures are generated at their *final printed size* (4.6 x 3.3 in and
5.8 x 4.8 in), so the 8 pt labels print as genuine 8 pt only if the graphic is
not rescaled.  The manuscript therefore includes them with explicit widths::

    \includegraphics[width=4.6in]{Consumer_Attractor_ETH.png}
    \includegraphics[width=5.8in]{Buy-Sell_Shock_ETH.png}

5.8 in fits inside Ledger's text block on both US Letter and A4 at the
journal's 3.1 cm side margins.  Using ``width=\textwidth`` instead would
rescale the graphic and push the label sizes off specification.

Requires numpy, scipy and matplotlib.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import model as m

# ------------------------------------------------------------ Ledger styling
PT = 1 / 72.0  # points -> inches (matplotlib linewidths are already in points)

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.labelsize":     8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    # variables italic, per Ledger Sec. 5 "Mathematical equations"
    "mathtext.fontset":   "custom",
    "mathtext.rm":        "Liberation Sans",
    "mathtext.it":        "Liberation Sans:italic",
    "mathtext.bf":        "Liberation Sans:bold",
    # 1/2 pt black frame and ticks
    "axes.linewidth":     0.5,
    "axes.edgecolor":     "black",
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.color":        "black",
    "ytick.color":        "black",
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    # 1/4 pt gray gridlines
    "grid.color":         "0.75",
    "grid.linewidth":     0.25,
    "grid.linestyle":     "-",
    "axes.grid":          True,
    "legend.frameon":     True,
    "legend.edgecolor":   "black",
    "legend.framealpha":  1.0,
    "legend.borderpad":   0.4,
    "legend.handlelength": 2.6,
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

BLACK = "black"
GRAY_MID = "0.45"
GRAY_LT = "0.62"
SHADE = "0.91"
LW_DATA = 1.0          # >= 1 pt, per the Ledger specification
LW_DATA_HEAVY = 1.3

# ---------------------------------------------------------- experiments
# Figure 1: permanent 50% rise in the fiat inflow at t = 1, tracked for a century.
FIG1_KAPPA = 1.5       # inflow multiplier
FIG1_T = 1200          # months (100 years)

# Figure 2: closed Buy-Sell capital cycle, equal and opposite 11-month blocks.
FIG2_T = 80            # months
FIG2_LAMBDA = 300e6    # USD per month injected, then withdrawn
FIG2_BUY = (20, 31)    # buy window, [start, stop)
FIG2_SELL = (40, 51)   # sell window, [start, stop)


def finish(ax):
    """Apply the 1/2 pt black frame and put gridlines behind the data."""
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("black")
    ax.set_axisbelow(True)
    ax.tick_params(which="both", top=False, right=False)


# =============================================================== FIGURE 1
def figure_1(path_png, path_pdf):
    T = FIG1_T
    Ic_new = FIG1_KAPPA * m.IC0
    Ic_path = np.full(T + 1, Ic_new)
    Ic_path[0] = m.IC0

    S, p = m.simulate_consumer_only(T, Ic_path, S_init=m.S0)

    S1, p1, _ = m.steady_state(m.IC0)
    S3, p3, _ = m.steady_state(Ic_new)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))

    Sgrid = np.linspace(30e6, 112e6, 600)

    # steady-state manifold (thin solid black)
    ax.plot(Sgrid / 1e6, m.manifold_price(Sgrid),
            color=BLACK, lw=LW_DATA, ls="-", zorder=3,
            label="Steady-state manifold")

    # asset-market clearing locus for the post-shock inflow (thin dashed black)
    ax.plot(Sgrid / 1e6, m.clearing_locus_price(Sgrid, Ic_new),
            color=BLACK, lw=LW_DATA, ls=(0, (4, 2.2)), zorder=3,
            label="Asset-market clearing locus")

    # realised trajectory (thick gray band underneath)
    ax.plot(S[1:] / 1e6, p[1:],
            color=GRAY_LT, lw=2.4, ls="-", solid_capstyle="round", zorder=2,
            label="Market trajectory $(S^{\\,t},p^{\\,t})$")

    # instantaneous price jump at t = 1
    ax.annotate("", xy=(m.S0 / 1e6, p[1]), xytext=(m.S0 / 1e6, p1),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.9,
                                shrinkA=2.5, shrinkB=3.5,
                                mutation_scale=7), zorder=5)

    # states: distinct marker shapes, since colour is unavailable
    ax.plot(S1 / 1e6, p1, marker="o", ms=4.5, mfc=BLACK, mec=BLACK,
            ls="none", zorder=6, label="State 1: initial equilibrium")
    ax.plot(m.S0 / 1e6, p[1], marker="s", ms=4.5, mfc="white", mec=BLACK,
            mew=0.8, ls="none", zorder=6, label="State 2: shock overshoot")
    ax.plot(S3 / 1e6, p3, marker="^", ms=5.0, mfc=BLACK, mec=BLACK,
            ls="none", zorder=6, label="State 3: new steady state")
    ax.plot(S[T] / 1e6, p[T], marker="D", ms=4.0, mfc=GRAY_MID, mec=BLACK,
            mew=0.6, ls="none", zorder=6,
            label="Trajectory end ($t=1{,}200$ months)")

    ax.annotate("State 1", xy=(S1 / 1e6, p1), xytext=(-2, -11),
                textcoords="offset points", ha="right", va="top", fontsize=8)
    ax.annotate("State 2", xy=(m.S0 / 1e6, p[1]), xytext=(6, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=8)
    ax.annotate("State 3", xy=(S3 / 1e6, p3), xytext=(7, 6),
                textcoords="offset points", ha="left", va="bottom", fontsize=8)

    ax.set_xlim(30, 112)
    ax.set_ylim(400, 1750)
    ax.set_xlabel("Physical staked supply $S^{\\,t}$ (millions of ETH)")
    ax.set_ylabel("Nominal token price $p^{\\,t}$ (USD)")

    leg = ax.legend(loc="upper right", fontsize=8, handletextpad=0.6,
                    borderaxespad=0.5, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.5)

    finish(ax)
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
    return dict(S1=S1, p1=p1, p2=p[1], S3=S3, p3=p3, Send=S[T], pend=p[T])


# =============================================================== FIGURE 2
def figure_2(path_png, path_pdf):
    T = FIG2_T
    Lam = np.zeros(T + 1)
    Lam[slice(*FIG2_BUY)] = FIG2_LAMBDA
    Lam[slice(*FIG2_SELL)] = -FIG2_LAMBDA

    r = m.simulate_dual(T, Lam)
    t = np.arange(T + 1)

    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.8))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    def shade(ax):
        ax.axvspan(20, 30, color=SHADE, lw=0, zorder=0)
        ax.axvspan(40, 50, color=SHADE, lw=0, zorder=0)

    # ---- (a) nominal token price
    shade(ax_a)
    ax_a.plot(t, r["p"], color=BLACK, lw=LW_DATA_HEAVY, zorder=3)
    ax_a.set_ylabel("Nominal token price (USD)")
    ax_a.set_xlabel("Months")
    ax_a.set_ylim(0, 7600)
    ax_a.set_yticks([0, 2000, 4000, 6000])
    ax_a.text(25, 7150, "buy", ha="center", va="top", fontsize=8)
    ax_a.text(45, 7150, "sell", ha="center", va="top", fontsize=8)
    finish(ax_a)

    # ---- (b) consumer accumulated wealth
    shade(ax_b)
    ax_b.axhline(r["acc_wealth"][0] / 1e9, color=GRAY_MID, lw=0.8,
                 ls=(0, (4, 2.2)), zorder=3,
                 label="Initial wealth anchor, $I_c/\\nu$")
    ax_b.plot(t, r["acc_wealth"] / 1e9, color=BLACK, lw=LW_DATA_HEAVY, zorder=4,
              label="Accumulated wealth")
    ax_b.set_ylabel("Accumulated wealth\n(billions of USD)")
    ax_b.set_xlabel("Months")
    ax_b.set_ylim(0, 85)
    leg = ax_b.legend(loc="upper left", fontsize=7, handletextpad=0.6,
                      borderaxespad=0.4, labelspacing=0.3, handlelength=2.4)
    leg.get_frame().set_linewidth(0.5)
    finish(ax_b)

    # ---- (c) physical token distribution
    shade(ax_c)
    ax_c.plot(t, r["S"] / 1e6, color=GRAY_MID, lw=LW_DATA_HEAVY,
              ls=(0, (5, 1.6, 1, 1.6)), zorder=3,
              label="Total staked, $S^{\\,t}$")
    ax_c.plot(t, r["S_i"] / 1e6, color=BLACK, lw=LW_DATA_HEAVY,
              ls=(0, (4, 2.2)), zorder=4,
              label="Investor, $S_i^{\\,t}$")
    ax_c.plot(t, r["S_c"] / 1e6, color=BLACK, lw=LW_DATA_HEAVY, ls="-", zorder=5,
              label="Consumer, $S_c^{\\,t}$")
    ax_c.set_ylabel("Staked supply\n(millions of ETH)")
    ax_c.set_xlabel("Months")
    ax_c.set_ylim(5, 58)
    leg = ax_c.legend(loc="upper left", fontsize=7, handletextpad=0.6,
                      borderaxespad=0.4, labelspacing=0.3, handlelength=2.4,
                      ncol=1)
    leg.get_frame().set_linewidth(0.5)
    finish(ax_c)

    # ---- (d) burn vs yield (twin axis)
    shade(ax_d)
    ax_d.plot(t, r["L_c"] / 1e3, color=BLACK, lw=LW_DATA_HEAVY, ls="-",
                    zorder=4, label="Tokens burned, $L_c^{\\,t}$ (left)")
    ax_d.set_ylabel("Tokens burned\n(thousands of ETH)")
    ax_d.set_xlabel("Months")
    ax_d.set_ylim(0, 340)
    finish(ax_d)

    ax_d2 = ax_d.twinx()
    ax_d2.plot(t, 100 * r["y"], color=GRAY_MID, lw=LW_DATA_HEAVY,
                     ls=(0, (4, 2.2)), zorder=3,
                     label="Staking yield, $y^{\\,t}$ (right)")
    ax_d2.set_ylabel("Monthly staking yield (%)")
    ax_d2.set_ylim(0.2380, 0.2515)
    ax_d2.set_yticks([0.239, 0.242, 0.245, 0.248, 0.251])
    ax_d2.grid(False)
    for s in ax_d2.spines.values():
        s.set_linewidth(0.5)
        s.set_color("black")
    ax_d2.tick_params(which="both", width=0.5, color="black", labelsize=8)

    ax_d.annotate("$L_c^{\\,t}$, left axis", xy=(64, r["L_c"][64] / 1e3),
                  xytext=(0, 10), textcoords="offset points",
                  ha="center", va="bottom", fontsize=8, zorder=6)
    ax_d2.annotate("$y^{\\,t}$, right axis", xy=(12, 100 * r["y"][12]),
                   xytext=(7, 7), textcoords="offset points",
                   ha="left", va="bottom", fontsize=8, zorder=6)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlim(0, 80)
        ax.set_xticks([0, 20, 40, 60, 80])

    fig.tight_layout(pad=0.4, w_pad=1.6, h_pad=1.9)

    # multipart labels, lower case in brackets, near the bottom of each panel
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        tb = ax.get_tightbbox(rend).transformed(inv)
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, tb.y0 - 0.012, f"({lab})",
                 ha="center", va="top", fontsize=8)

    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)

    return dict(p_peak=r["p"].max(), p_trough=r["p"].min(), p_end=r["p"][T],
                Sc0=r["S_c"][0], ScT=r["S_c"][T],
                Si0=r["S_i"][0], SiT=r["S_i"][T],
                S0=r["S"][0], ST=r["S"][T],
                share0=r["S_c"][0] / r["S"][0], shareT=r["S_c"][T] / r["S"][T])


if __name__ == "__main__":
    f1 = figure_1("Consumer_Attractor_ETH.png", "Consumer_Attractor_ETH.pdf")
    f2 = figure_2("Buy-Sell_Shock_ETH.png", "Buy-Sell_Shock_ETH.pdf")

    print("Figure 1  Consumer_Attractor_ETH.{png,pdf}")
    print(f"  State 1  initial equilibrium   S = {f1['S1']/1e6:7.3f} M   p = {f1['p1']:8.2f} USD")
    print(f"  State 2  shock overshoot       S = {m.S0/1e6:7.3f} M   p = {f1['p2']:8.2f} USD")
    print(f"  State 3  new steady state      S = {f1['S3']/1e6:7.3f} M   p = {f1['p3']:8.2f} USD")
    print(f"  trajectory end (t = {FIG1_T})   S = {f1['Send']/1e6:7.3f} M   p = {f1['pend']:8.2f} USD")

    print("\nFigure 2  Buy-Sell_Shock_ETH.{png,pdf}")
    print(f"  price   peak {f2['p_peak']:8.2f}   trough {f2['p_trough']:7.2f}"
          f"   final {f2['p_end']:8.2f} USD")
    print(f"  S_c     {f2['Sc0']/1e6:6.2f} -> {f2['ScT']/1e6:6.2f} M ETH")
    print(f"  S_i     {f2['Si0']/1e6:6.2f} -> {f2['SiT']/1e6:6.2f} M ETH")
    print(f"  S       {f2['S0']/1e6:6.2f} -> {f2['ST']/1e6:6.2f} M ETH")
    print(f"  Consumer share of stake  {f2['share0']:.1%} -> {f2['shareT']:.1%}")
