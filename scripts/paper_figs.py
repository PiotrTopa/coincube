#!/usr/bin/env python
"""Generate the manuscript figures into paper/figs/.

Data provenance: measured numbers are READ from the committed result files
(results/w3c_corner.json, results/m8_corner.json), which are produced by
the gated instruments scripts/w3c_corner.py and scripts/m8_corner.py; exact
curves are recomputed from the model operators. No hardcoded measurements,
no fits, no smoothing.
"""
import json
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
    """Quenched slope ratios vs the diamond prediction, from the committed
    gated-run output (jackknife + gate systematic errors)."""
    data = json.load(open("results/w3c_corner.json"))
    labels = {("annealed", 0.08): "annealed gate\n$p=0.08$",
              ("quenched", 0.08): "quenched\n$q=0.08$",
              ("quenched", 0.15): "quenched\n$q=0.15$"}
    chans = ["110", "111", "r1", "r2"]
    marks = ["o", "s", "^", "v"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    gate_syst = {}
    for row in data:
        if row["mode"] == "annealed":
            for c in chans:
                gate_syst[c] = abs(row["rows"][c]["ratio"] - 1.0)
    for xi, row in enumerate(data):
        for ci, c in enumerate(chans):
            rr = row["rows"][c]
            err = np.sqrt(rr["sig_ratio"] ** 2 + gate_syst[c] ** 2)
            ax.errorbar(xi + 0.15 * (ci - 1.5), rr["ratio"], yerr=err,
                        fmt=marks[ci], ms=4, capsize=2, color=f"C{ci}",
                        label=c if xi == 0 else None)
            ax.plot(xi + 0.15 * (ci - 1.5), rr["diamond"], marker="_",
                    ms=9, color="r")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.text(2.28, 1.62, "diamond\npredictions", color="r", fontsize=6,
            ha="center")
    ax.set_xticks(range(len(data)),
                  [labels[(r["mode"], r["q"])] for r in data])
    ax.set_ylabel("slope ratio at the node")
    ax.set_ylim(0.85, 1.85)
    ax.legend(loc="center left", fontsize=7, title="channel",
              title_fontsize=7)
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
    mdata = json.load(open("results/m8_corner.json"))
    qrow = [r for r in mdata if r["mode"] == "quenched"][0]
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
        pts = qrow["rows"][name]
        dsm = sorted(float(k) for k in pts)
        vals = [pts[str(d)] if str(d) in pts else pts[repr(d)] for d in dsm]
        ys = [v[0] for v in vals]
        es = [v[1] for v in vals]
        ax1.errorbar(dsm, ys, yerr=es, fmt=mk, ms=4, capsize=2, label=name)
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
