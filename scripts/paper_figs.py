#!/usr/bin/env python
"""Generate the manuscript figures into paper/figs/.

Data provenance: measured numbers are transcribed from the tagged evidence
logs (results/w3c_corner_run.log, results/m8_corner_run.log at
v1.2-publication-base); exact curves are recomputed here from the model
operators. No fits, no smoothing.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca3d.models.coincube import annealed_u, annealed_u8

OUT = pathlib.Path("paper/figs")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})


def fig_cone():
    """Quenched slope ratios vs the diamond prediction (w3c_corner_run.log)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rows = {
        "annealed gate\n$p=0.08$": (1.000, 1.001, 0.027, 0.035),
        "quenched\n$q=0.08$": (1.000, 1.000, 0.007, 0.014),
        "quenched\n$q=0.15$": (1.001, 1.002, 0.208, 0.209),
    }
    xs = np.arange(len(rows))
    r110 = [v[0] for v in rows.values()]
    r111 = [v[1] for v in rows.values()]
    e110 = [v[2] for v in rows.values()]
    e111 = [v[3] for v in rows.values()]
    ax.errorbar(xs - 0.08, r110, yerr=e110, fmt="o", capsize=3,
                label=r"$v_{110}/v_{100}$")
    ax.errorbar(xs + 0.08, r111, yerr=e111, fmt="s", capsize=3,
                label=r"$v_{111}/v_{100}$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(np.sqrt(2), color="r", ls="--", lw=0.8)
    ax.axhline(np.sqrt(3), color="r", ls=":", lw=0.8)
    ax.text(2.02, np.sqrt(2) + 0.02, "diamond $\\sqrt{2}$", color="r",
            fontsize=7)
    ax.text(2.02, np.sqrt(3) + 0.02, "diamond $\\sqrt{3}$", color="r",
            fontsize=7)
    ax.set_xticks(xs, rows.keys())
    ax.set_ylabel("slope ratio at the node")
    ax.set_ylim(0.7, 1.9)
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cone_ratios.pdf")


def fig_mass():
    """Measured massive dispersion vs the exact operator branches, and the
    arctan mass law (m8_corner_run.log; m8_mass_exact.py)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    Q, QM = 0.08, 0.05
    K0 = np.array([np.pi, 0, 0])
    lam0 = np.linalg.eigvals(annealed_u8(K0, Q, QM))
    p0 = sorted(np.angle(l) for l in lam0 if l.imag > 0.02)
    om_c = 0.5 * (p0[0] + p0[-1])
    for name, dv, mk in (("100", (1, 0, 0), "o"), ("110", (1, 1, 0), "s"),
                         ("111", (1, 1, 1), "^")):
        u = np.array(dv, float)
        u /= np.linalg.norm(u)
        ds = np.linspace(0, 0.12, 40)
        ex = []
        for d in ds:
            lams = np.linalg.eigvals(annealed_u8(K0 + d * u, Q, QM))
            phs = sorted(np.angle(l) for l in lams if l.imag > 0.02)
            ex.append(phs[-1] - om_c)
        ax1.plot(ds, ex, lw=1)
        meas = {"100": (0.0782, 0.1173, 0.1625),
                "110": (0.0800, 0.1233, 0.1756),
                "111": (0.0820, 0.1335, 0.1996)}[name]
        ax1.plot((0.04, 0.07, 0.10), meas, mk, ms=4, label=name)
    m = np.arctan(QM / (1 - QM))
    ax1.axhline(m, color="k", lw=0.6, ls=":")
    ax1.text(0.001, m + 0.004, r"$m=\arctan\frac{q_m}{1-q_m}$", fontsize=7)
    ax1.set_xlabel(r"$|\delta k|$")
    ax1.set_ylabel(r"$\omega_+-\omega_c$")
    ax1.legend(fontsize=7, title="quenched, meas.", title_fontsize=7)
    qms = np.linspace(0, 0.25, 60)
    ax2.plot(qms, np.arctan(qms / (1 - qms)), "k-", lw=1,
             label=r"$\arctan[q_m/(1-q_m)]$ (exact)")
    meas_m = [(0.01, 0.0101006666), (0.02, 0.0204053307),
              (0.05, 0.0525830616), (0.1, 0.1106572212),
              (0.2, 0.2449786631)]
    ax2.plot(*zip(*meas_m), "o", ms=4, label="operator gap (machine prec.)")
    ax2.set_xlabel(r"$q_m$")
    ax2.set_ylabel(r"$m$")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "mass.pdf")


def fig_chord():
    """Chord-vs-arc geometry of the Q-pinning theorem and the two-boundary
    completion."""
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    th = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1)
    ax.plot([0, 1], [0, 0], "k:", lw=0.5)
    ax.plot([0, 0], [0, 1], "k:", lw=0.5)
    qs = np.linspace(0, 1, 100)
    ax.plot(1 - qs, qs, "C3-", lw=1.5,
            label=r"in-in mixture $(1-q)\,1+q\,C$")
    ax.plot(np.sqrt(1 - qs), np.sqrt(qs), "C0-", lw=1.5,
            label=r"in-out $\sqrt{1-q}\,1+\sqrt{q}\,C$")
    q = 0.25
    ax.plot([1 - q], [q], "C3o", ms=5)
    ax.plot([np.sqrt(1 - q)], [np.sqrt(q)], "C0o", ms=5)
    ax.annotate("", xy=(np.sqrt(1 - q), np.sqrt(q)), xytext=(1 - q, q),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.05, 0.02, r"$1$", fontsize=9)
    ax.text(0.02, 0.93, r"$C$ ($e^{i\pi/2}$)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "chord_arc.pdf")


if __name__ == "__main__":
    fig_cone()
    fig_mass()
    fig_chord()
    print("figures written to paper/figs/")
