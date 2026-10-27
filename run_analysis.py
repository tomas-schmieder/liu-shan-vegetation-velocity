"""Run the Liu & Shan (2019) model: Table 1 cases, lambda calibration and sensitivity analysis (cf. Figs. 5, 12)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cases import CASES, MEASURED_UD0_OVER_U0
from liu_shan_solver import Ud0_over_U0_fit, calibrate_lambda, solve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def xgrid(p):
    L = p.L if p.L is not None else 1.5 * p.li
    return np.linspace(-3 * p.b, L, 1500)


def summary_table():
    print(f"{'case':6s} {'Cdab':>5s} {'Lu':>4s} {'LI(Eq1)':>8s} {'lam':>4s} {'Ud0/U0 model':>13s} "
          f"{'measured':>9s} {'Eq.17':>6s} {'lam*(meas)':>11s} {'Ud4/U0':>7s}")
    rows = []
    for k, p in CASES.items():
        s = solve(p)
        u0 = s.Ud(0.0)[0] / p.U0
        lam_star = calibrate_lambda(p, MEASURED_UD0_OVER_U0[k])
        print(f"{k:6s} {p.Cdab:5.2f} {p.lu:4.0f} {p.li:8.0f} {p.lam:4.0f} {u0:13.3f} "
              f"{MEASURED_UD0_OVER_U0[k]:9.2f} {Ud0_over_U0_fit(p.Cdab):6.2f} {lam_star:11.1f} {s.Ud4 / p.U0:7.3f}")
        rows.append((k, p.Cdab, u0, s.Ud4 / p.U0))
    return rows


def plot_cases():
    groups = [("C$_d$ab = 1.9", ["Case1", "Case2", "Case3"]),
              ("C$_d$ab = 2.2", ["Case4", "Case5", "Case6"]),
              ("C$_d$ab = 4.3", ["Case7", "Case8", "Case9"])]
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for ax, (title, names) in zip(axes, groups):
        for k in names:
            p = CASES[k]
            x = xgrid(p)
            u = solve(p).Ud(x)
            ax.plot(x / p.b, u / p.U0, label=f"{k} (L={p.L:.0f} cm)")
            ax.axvline(p.L / p.b, color="grey", lw=0.5, ls=":")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("U$_d$/U$_0$")
        ax.text(0.97, 0.9, title, transform=ax.transAxes, ha="right")
        ax.legend(loc="lower left", fontsize=8)
    axes[-1].set_xlabel("x/b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profiles_table1_cases.png"), dpi=150)
    plt.close(fig)


def sensitivity():
    """Case 3, one parameter changed at a time (others as in Table 1), cf. Fig. 12 of the paper."""
    base = CASES["Case3"]
    sweeps = [
        ("lam", r"$\lambda$", [1, 14, 40], "{:g}"),
        ("Cd", "C$_d$", [0.8, 1.0, 1.2], "{:g}"),
        ("Cf", "C$_f$", [0.003, 0.006, 0.012], "{:g}"),
        ("S", "S", [0.5e-4, 1e-4, 2e-4], "{:.1e}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    x = np.linspace(-base.lu * 3, base.L, 2000)
    print("\nSensitivity (Case 3): Ud(0)/U0, Ud at x=1 m /U0, Ud(L)/U0")
    for ax, (name, label, vals, fmt) in zip(axes.flat, sweeps):
        for v in vals:
            p = base.with_(**{name: v})
            s = solve(p)
            u = s.Ud(x)
            ax.plot(x / base.b, u / base.U0, label=f"{label} = {fmt.format(v)}")
            print(f"  {name:3s}={fmt.format(v):>8s}  Ud(0)/U0={s.Ud(0.0)[0] / p.U0:.3f}  "
                  f"Ud(100)/U0={s.Ud(100.0)[0] / p.U0:.3f}  Ud(L)/U0={s.Ud(p.L)[0] / p.U0:.3f}  "
                  f"(r1,r2,r3,r4)=({s.r1:.2e},{s.r2:.2e},{s.r3:.2e},{s.r4:.2e})")
        for xv in (0, base.L / base.b):
            ax.axvline(xv, color="grey", ls="--", lw=0.7)
        ax.set_ylim(0, 1.3)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("x/b")
    for ax in axes[:, 0]:
        ax.set_ylabel("U$_d$/U$_0$")
    fig.suptitle("Sensitivity analysis, Case 3 (C$_d$ab = 1.9)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sensitivity_case3.png"), dpi=150)
    plt.close(fig)


def sensitivity_density():
    """Extra: effect of patch density (frontal area a) and flow blockage on the profile."""
    base = CASES["Case3"].with_(LI=None)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in (0.02, 0.048, 0.1, 0.144, 0.3):
        p = base.with_(a=a, L=None)
        x = np.linspace(-3 * p.b, 1.4 * p.li, 1500)
        ax.plot(x / p.b, solve(p).Ud(x) / p.U0, label=f"a = {a} cm$^{{-1}}$, C$_d$ab = {p.Cdab:.1f}")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("x/b"); ax.set_ylabel("U$_d$/U$_0$"); ax.set_ylim(0, 1.2); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sensitivity_density.png"), dpi=150)
    plt.close(fig)


def plot_case3_regions():
    """Case 3 profile with the four regions marked (cf. Figs. 2, 3)."""
    p = CASES["Case3"]
    s = solve(p)
    x = np.linspace(-150, p.L, 2000)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, s.Ud(x) / p.U0, "k", lw=2)
    for xv, lab in ((-p.lu, "x = −L$_u$"), (0, "x = 0"), (p.li, "x = L$_I$")):
        ax.axvline(xv, color="grey", ls="--", lw=0.8)
        ax.text(xv, 1.17, lab, ha="center", fontsize=9)
    for xm, lab in ((-p.lu - 55, "Region 1"), (-p.lu / 2, "2"), (p.li / 2, "Region 3"), ((p.li + p.L) / 2, "Region 4")):
        ax.text(xm, 0.55 if lab != "2" else 0.3, lab, ha="center", fontsize=10)
    ax.set_ylim(0, 1.25); ax.set_xlabel("x (cm)"); ax.set_ylabel("U$_d$/U$_0$")
    ax.set_title("Case 3 (C$_d$ab = 1.9, L = 5 m): predicted velocity profile")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profile_case3_regions.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_case3_regions()
    summary_table()
    plot_cases()
    sensitivity()
    sensitivity_density()
    print(f"\nFigures written to {OUT}")
