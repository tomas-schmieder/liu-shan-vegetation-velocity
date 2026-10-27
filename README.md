# Streamwise velocity in a partially vegetated channel

Python implementation of Liu & Shan (2019), *Analytical model for predicting the longitudinal profiles of
velocities in a channel with a model vegetation patch*, J. Hydrol. 576, 561–574
(the research guide cites it as "Liu & Chang"; the authors are Chao Liu and Yuqi Shan).

| File | Purpose |
|---|---|
| `liu_shan_solver.py` | Model: Eqs. (1)–(3), (11)–(17), roots r1–r4, boundary conditions → A1–A4, profile U_d(x) |
| `cases.py` | Experimental parameters of Table 1 (Cases 1–9) |
| `test_solver.py` | Verifies Eq. (11) residual, the four BCs, derivative continuity, monotonic profile |
| `compare_with_paper.py` | Compares the solver with measured profiles digitised from the paper's Figs. 5, 6, 8 (and its curves in Figs. 6–7) |
| `digitize/`, `data/` | Marker digitising scripts (template matching on the figure images) and the extracted points |
| `run_analysis.py` | Table-1 cases, λ calibration, sensitivity analysis → `outputs/` |

Run: `python3 test_solver.py && python3 run_analysis.py`

## Model assumptions (Task 1)
1. Emergent, rigid cylinders; steady, subcritical, turbulent flow; time-averaged, depth-averaged velocity U_d(x).
2. Only the **central region** of the patch (−b+δ ≤ y ≤ b−δ) is modelled, where Kelvin–Helmholtz vortices from the
   patch edges do not penetrate. There ∂u/∂y = 0 and ∂τ_yx/∂y = 0 (Eq. 7), so the problem reduces to a 1-D
   (longitudinal) equation.
3. Vertical velocity ≈ 0 at the bed and surface; shear stress ≈ 0 at the free surface, equal to the bed stress
   τ_b = ρ C_f U_d² at the bed.
4. Drag force F_x = ½ C_d a/(1−φ) u² (Eq. 8); free-surface slope S is uniform and equal to the bed slope.
5. Longitudinal Reynolds stress closed with an eddy viscosity ε_d = λ h u*, u* = √C_f U_d, λ = O(1–10) constant per patch density.
6. Flow adjustment is independent of patch length (a patch shorter than L_I simply uses the first part of the profile).
7. Four regions: 1 (x ≤ −L_u, U_0), 2 (upstream adjustment), 3 (interior adjustment, 0 ≤ x ≤ L_I), 4 (fully developed, x ≥ L_I).
   L_u ≈ b (measured), L_I from Eq. (1).
8. Bed friction C_f = 0.006, C_d ≈ 1.0 (Eq. 3). Validated for C_d·a·b = 1.3–8.4, φ = 0.015–0.38; rigid cylinders only.

## Mathematics (Task 2)
With W = U_d², Eq. (11) is the linear constant-coefficient ODE
`λh²√C_f W'' − h W' − K W + ghS = 0` (K = C_f in Region 2, K = C_f + ½C_d a h/(1−φ) in Region 3),
so W = c₁e^{r₁x} + c₂e^{r₂x} + ghS/K, i.e. U_d = (A₁e^{r₁x} + A₂e^{r₂x} + ω)^{1/2} (Eqs. 14, 15).

The four boundary conditions are linear in (A₁…A₄). Because U_d is continuous at x = 0 (BC 2), the derivative
condition dU_d/dx = W'/(2U_d) (BC 3) reduces to W' continuity, so no derivative calculator is needed:

```
A1 e^{-r1 Lu} + A2 e^{-r2 Lu}        = Ud1² − ω1 = 0     (BC1)
A1 + A2 + ω1 − A3 − A4 − ω2          = 0                 (BC2)
r1 A1 + r2 A2 − r3 A3 − r4 A4        = 0                 (BC3)
A3 e^{r3 LI} + A4 e^{r4 LI}          = Ud4² − ω2 = 0     (BC4)
```
The code solves this 4×4 system with `numpy.linalg.solve`, in a rescaled exponential basis (every exponential ≤ 1) so it
does not overflow for small λ. `test_solver.py` additionally checks the BCs and Eq. (11) numerically.

### Deviation from the printed paper
The paper prints `r2 = 2h − r1` (and `r4 = 2h − r3`). The second root of the characteristic equation is
`r2 = (h − √(h²+4λh²√C_f K)) / (2λh²√C_f)`, which is what is used (the literal "2h − r1" is dimensionally inconsistent
and does not satisfy Eq. 11; the ODE residual check confirms the corrected roots).

## Results
Case 3 (C_d·a·b = 1.9, S = 1×10⁻⁴, λ = 14): U_d(0)/U_0 = 0.83 (measured 0.86), U_d4/U_0 = 0.11 (Fig. 2 of the paper ≈ 0.1).
Predicted U_d(0)/U_0 for all nine cases is 0.75–0.84 vs measured 0.77–0.94.
The profiles of patches of different length collapse onto one curve (`outputs/profiles_table1_cases.png`), as in Fig. 5.

Sensitivity (Case 3, `outputs/sensitivity_case3.png`, cf. Fig. 12):
- **λ**: only Regions 2–3 change. Small λ → sharper drop concentrated at the patch edge; large λ → smoother, longer adjustment. U_d(0)/U_0 is not monotonic in λ (minimum near λ≈40), so λ calibration against U_d(0) uses the branch λ < λ_min.
- **C_d**: lowers Region 3/4 velocity (U_d4/U_0 0.124 → 0.101 for C_d 0.8 → 1.2); Region 1–2 unaffected.
- **C_f**: strongly changes Regions 1–2 (U_d1 = √(ghS/C_f)); Region 4 nearly unaffected because bed drag ≪ vegetation drag (C_f/vegetation drag ≈ 0.014).
- **S**: scales all velocities (U_d ∝ √S) except via the shape, which is unchanged.
- Extra: patch density sweep in `outputs/sensitivity_density.png`.

## Caveats
- With the Table-1 slope S = 1×10⁻⁴ the upstream velocity √(ghS/C_f) = 17.1 cm/s is 5% below U_0 = 18 cm/s; use `S=None` to
  get S = C_f U_0²/(gh) (Sec. 8.2) so Region 1 equals U_0 exactly.
- Raw measured velocity profiles are not in the paper, so RMSE (Eq. 4, implemented as `rmse`) is not reproduced here; only
  U_d(0)/U_0 from Tables 1–2 is compared. For the Table 2 studies (Rominger & Nepf, Zong & Nepf, …) add entries to `cases.py`.

## Comparison with the measured profiles
The paper tabulates no velocities, so the markers in its Figs. 5, 6 and 8 were digitised from the figure images
(`digitize/run_dig.py`, `run_dig2.py` → `data/*.json`; roughly ±0.01–0.02 in U_d/U_0). Run `python3 compare_with_paper.py`.
All cases use S = C_f U_0²/(gh), the paper's λ, C_d = 1, C_f = 0.006, L_u = b.

| Case | RMSE here (cm/s) | RMSE in paper (cm/s) |
|---|---|---|
| 1–9 (Table 1) | 1.5, 1.7, 1.2, 2.4, 2.3, 1.1, 1.0, 1.5, 1.4 | 1.3, 1.7, 1.3, 2.5, 2.4, 1.2, 0.9, 1.5, 1.2 |
| RN1, RN2, RN3 | 0.62, 0.91, 0.55 | 0.6, 0.9, 0.6 |
| Chen et al. | 0.66 | 0.7 |

- Extracting the solid model curves printed in Figs. 6–7 and comparing them with this solver gives max |ΔU_d/U_0| ≤ 0.06
  for Zong & Nepf (ZN1–6, L_u = b), ≤ 0.04 for Rominger & Nepf and ≈ 0.05 for Chen — the solver reproduces the paper's model.
- **Rominger & Nepf cases need a longer upstream adjustment region than L_u = b.** The paper's curves for RN1–3 are consistent with
  L_u/b ≈ 2.1, 3.0, 3.2 (fitted; the paper states L_u ≈ b only for its own experiments). With L_u = b the RMSE would be 0.69, 1.15, 0.94 cm/s.
- Gray squares (Cases 3, 6, 9) are only partly detected; Fig. 7 markers were not digitised (the model curves overlap them).

## Reproducing the digitised comparison
The paper (Elsevier, not redistributed here) is needed only to regenerate the digitised data; `data/*.json` is included.
To rerun the digitising and the curve extraction, put the PDF in the repository root and extract its figures:
`pdfimages -png -f 7 -l 10 <paper>.pdf digitize/fig`, then `cd digitize && python3 run_dig.py && python3 run_dig2.py`.
Install dependencies with `pip install -r requirements.txt`.
