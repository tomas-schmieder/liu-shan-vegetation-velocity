"""Compare the solver with the measured profiles of Liu & Shan (2019).

The paper gives no tabulated velocities, so the markers of Figs. 5, 6 and 8 were digitised from the published
raster figures (digitize/run_dig.py, digitize/run_dig2.py -> data/*.json; template matching of the symbols).
The solid model curves of Figs. 5-8 are also extracted from the images and compared with this solver.
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "digitize"))
from dig import load, mask_color  # noqa: E402

from cases import CASES, CASES2, PAPER_RMSE_CMS  # noqa: E402
from liu_shan_solver import rmse, solve  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
D5 = json.load(open(os.path.join(HERE, "data", "digitized_fig5.json")))
D678 = json.load(open(os.path.join(HERE, "data", "digitized_fig678.json")))

# the paper's figures show Ud/U0 = 1 upstream, i.e. S = Cf U0^2/(g h): use S=None for every case
C1 = {k: p.with_(S=None) for k, p in CASES.items()}


def model(p, xb):
    return solve(p).Ud(np.asarray(xb) * p.b) / p.U0


def rmse_cms(p, xb, y):
    """Eq. (4) in cm/s, over digitised points (x <= L when the patch length is finite)."""
    xb, y = np.asarray(xb), np.asarray(y)
    if p.L is not None:
        k = xb <= p.L / p.b + 1e-9
        xb, y = xb[k], y[k]
    return rmse(y * p.U0, model(p, xb) * p.U0), len(xb)


def paper_curve(img, frame, xr, yr, kind, ref, xs, win=0.12):
    """Extract the solid model curve printed in the figure: mean row of line-coloured pixels in each column,
    searched within +-win (in Ud/U0) of the reference curve so that text/markers are ignored."""
    im = load(os.path.join(HERE, "digitize", img))
    m = mask_color(im, kind) > 0
    x0, x1, y0, y1 = frame
    out = np.full(len(xs), np.nan)
    for i, (x, r) in enumerate(zip(xs, ref)):
        col = int(round(x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)))
        lo = int(round(y1 - (min(r + win, yr[1]) - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)))
        hi = int(round(y1 - (max(r - win, yr[0]) - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)))
        rows = np.where(m[lo:hi + 1, col - 1:col + 2].any(1))[0]
        if len(rows):
            out[i] = yr[0] + (y1 - (lo + rows.mean())) / (y1 - y0) * (yr[1] - yr[0])
    return out


rows = []

# ------------------------------------------------------------------ Fig. 5 (this study, Cases 1-9)
F5 = [(1, 2, 3, (146, 1422, 22, 393.5)), (4, 5, 6, (146, 1422, 576, 947)), (7, 8, 9, (146, 1422, 1129.5, 1500.5))]
fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
sty = [("tab:blue", "^"), ("tab:red", "o"), ("0.45", "s")]
for ax, (c1, c2, c3, fr), lab in zip(axes, F5, ("C$_d$ab = 1.9", "C$_d$ab = 2.2", "C$_d$ab = 4.3")):
    for c, (col, mk) in zip((c1, c2, c3), sty):
        k = f"Case{c}"; p = C1[k]; d = D5[k]
        xb, y = np.array(d["x_over_b"]), np.array(d["Ud_over_U0"])
        ax.plot(xb, y, mk, mfc="none", color=col, ms=6, label=f"{k} measured (L={p.L / 100:g} m)")
        xm = np.linspace(-10, p.L / p.b, 600)
        ax.plot(xm, model(p, xm), color=col, lw=1.6)
        r, n = rmse_cms(p, xb, y)
        rows.append((k, p.U0, n, r, r / p.U0 * 100, PAPER_RMSE_CMS[k]))
    ax.axvline(0, color="k", ls="--", lw=0.7); ax.set_ylim(0, 1.2); ax.set_ylabel("U$_d$/U$_0$")
    ax.text(0.97, 0.9, lab, transform=ax.transAxes, ha="right"); ax.legend(fontsize=8, loc="lower left")
axes[-1].set_xlabel("x/b"); axes[0].set_title("This solver (lines) vs measured velocities digitised from Fig. 5")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_fig5_this_study.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig. 6 (Rominger & Nepf) and Fig. 8 (Chen)
F6 = {"RN1": (21.5, 386.5), "RN2": (567.5, 932.5), "RN3": (1113, 1478)}
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, k in zip(axes, F6):
    p = CASES2[k]; d = D678[k]
    xb, y = np.array(d["x_over_b"]), np.array(d["Ud_over_U0"])
    ax.plot(xb, y, "^", mfc="none", color="tab:red", ms=5, label="measured (digitised)")
    xm = np.linspace(-6, 10, 600); ax.plot(xm, model(p, xm), "k", lw=1.6, label="this solver")
    fr = (165, 1421, *F6[k]); pc = paper_curve("fig-001.png", fr, (-6, 10), (-0.2, 1.2), "black", model(p, xm), xm)
    ax.plot(xm, pc, "c--", lw=1.2, label="paper's curve (extracted)")
    r, n = rmse_cms(p, xb, y); rows.append((k, p.U0, n, r, r / p.U0 * 100, PAPER_RMSE_CMS[k]))
    ok = ~np.isnan(pc); dev = np.max(np.abs(pc[ok] - model(p, xm)[ok]))
    rows.append((k + " curve", p.U0, int(ok.sum()), rmse(pc[ok], model(p, xm)[ok]) * p.U0, dev, np.nan))
    ax.set_title(f"{k}: C$_d$ab = {p.Cdab:.1f}, $\\lambda$ = {p.lam:g}"); ax.set_xlabel("x/b")
    ax.axvline(0, color="grey", ls="--", lw=0.7); ax.set_ylim(-0.2, 1.2)
axes[0].set_ylabel("U$_d$/U$_0$"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_fig6_rominger_nepf.png"), dpi=150); plt.close(fig)

p = CASES2["Chen"]; d = D678["Chen"]; xb, y = np.array(d["x_over_b"]), np.array(d["Ud_over_U0"])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xb, y, "D", mfc="none", color="tab:red", label="measured (digitised)")
xm = np.linspace(-3, 2, 400); ax.plot(xm, model(p, xm), "k", lw=1.6, label="this solver")
pc = paper_curve("fig-003.png", (129.5, 1431.5, 13.5, 393), (-3, 3), (0, 1.2), "black", model(p, xm), xm)
ax.plot(xm, pc, "c--", label="paper's curve (extracted)")
r, n = rmse_cms(p, xb[xb <= 2.0 + 1e-9], y[xb <= 2.0 + 1e-9]); r_all = rmse(y * p.U0, model(p, xb) * p.U0)
rows.append(("Chen", p.U0, n, r, r / p.U0 * 100, PAPER_RMSE_CMS["Chen"]))
ax.set_xlabel("x/b"); ax.set_ylabel("U$_d$/U$_0$"); ax.set_ylim(0, 1.2); ax.legend(fontsize=8)
ax.set_title("Chen et al. (2012), C$_d$ab = 1.3"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_fig8_chen.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig. 7 (Zong & Nepf): paper's curves only
F7 = [(("ZN1", "ZN2"), (156.5, 1420.5, 22.5, 390)), (("ZN3", "ZN4"), (156.5, 1420.5, 572.5, 939.5)),
      (("ZN5", "ZN6"), (156.5, 1420.5, 1121.5, 1488.5))]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, ((ka, kb), fr) in zip(axes, F7):
    for k, kind, col in ((ka, "red", "tab:red"), (kb, "blue", "tab:blue")):
        p = CASES2[k]; xm = np.linspace(-4.5, 20, 500); mod = model(p, xm)
        ax.plot(xm, mod, color=col, lw=1.6, label=f"{k} this solver")
        pc = paper_curve("fig-002.png", fr, (-5, 20), (-0.2, 1.2), kind, mod, xm, win=0.06)
        ax.plot(xm, pc, "--", color="k", lw=0.9)
        ok = ~np.isnan(pc); dev = np.max(np.abs(pc[ok] - mod[ok]))
        rows.append((k + " curve", p.U0, int(ok.sum()), rmse(pc[ok], mod[ok]) * p.U0, dev, np.nan))
    ax.set_title(f"U$_0$ = {p.U0:g} cm/s, h = {p.h:g} cm"); ax.set_xlabel("x/b"); ax.set_ylim(-0.2, 1.2)
    ax.axvline(0, color="grey", ls="--", lw=0.7); ax.legend(fontsize=8)
axes[0].set_ylabel("U$_d$/U$_0$"); axes[1].set_title(axes[1].get_title() + "   (dashed = paper's curve)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_fig7_zong_nepf.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ Fig. 5 model curves from the paper vs ours
for c, (cs, fr) in enumerate(zip(F5, [f[3] for f in F5])):
    pass

# ------------------------------------------------------------------ tables
print(f"{'case':14s} {'U0':>5s} {'N':>4s} {'RMSE cm/s':>10s} {'RMSE/U0 %':>10s} {'paper RMSE':>11s}")
for k, U0_, n, r, rp, pr in rows:
    if "curve" in k:
        print(f"{k:14s} {U0_:5.1f} {n:4d}  rms diff to paper curve {r:.2f} cm/s, max |diff| in Ud/U0 = {rp:.3f}")
    else:
        print(f"{k:14s} {U0_:5.1f} {n:4d} {r:10.2f} {rp:10.1f} {pr:11.1f}")
