"""Experimental cases of Liu & Shan (2019), Table 1 (this study) and Table 2 (published studies).

Units: cm, cm/s. Cd = 1.0 and Cf = 0.006 for this study (Sec. 2); S = 1e-4 (Table 1 footnote).
Lu and LI are the measured values where reported; otherwise Lu = b and LI from Eq. (1).
"""
import numpy as np
from liu_shan_solver import Params

H, U0, S_THIS = 17.8, 18.0, 1e-4

# case: (b, L, a, Lu, LI measured or None, lambda, Ud(0)/U0 measured)
_T1 = {
    "Case1": (40, 100, 0.048, 50, None, 14, 0.79),
    "Case2": (40, 300, 0.048, 50, None, 14, 0.94),
    "Case3": (40, 500, 0.048, 40, 350, 14, 0.86),
    "Case4": (30, 100, 0.072, 20, None, 10, 0.93),
    "Case5": (30, 250, 0.072, 40, None, 10, 0.90),
    "Case6": (30, 400, 0.072, 30, 300, 10, 0.85),
    "Case7": (30, 100, 0.144, 35, None, 10, 0.77),
    "Case8": (30, 200, 0.144, 30, None, 10, 0.84),
    "Case9": (30, 300, 0.144, 30, 210, 10, 0.78),
}

CASES = {
    k: Params(U0=U0, h=H, b=b, a=a, d=0.4, Cd=1.0, Cf=0.006, S=S_THIS, lam=lam, L=L, Lu=Lu, LI=LI)
    for k, (b, L, a, Lu, LI, lam, _) in _T1.items()
}
MEASURED_UD0_OVER_U0 = {k: v[-1] for k, v in _T1.items()}


# ---- Table 2: published studies (S in per-mille in the paper -> converted to absolute slope) -------------
# name: (U0, b, a, phi, h, S_permille, LI, lam, Ud0/U0, RMSE cm/s)
_T2 = {
    "RN1": (10, 15, 0.2, 0.09, 14, 0.044, 99, 12, 0.75, 0.6),
    "RN2": (10, 6, 0.8, 0.38, 14, 0.044, 36, 8, 0.58, 0.9),
    "RN3": (10, 10, 0.8, 0.38, 14, 0.044, 57, 15, 0.53, 0.6),
    "ZN1": (5, 40, 0.04, 0.02, 12, 0.013, 352, 48, 0.80, 0.3),
    "ZN2": (5, 40, 0.21, 0.10, 12, 0.013, 226, 60, 0.58, 0.3),
    "ZN3": (9, 40, 0.04, 0.02, 13, 0.038, 352, 26, 0.85, 0.2),
    "ZN4": (9, 40, 0.21, 0.10, 13, 0.038, 226, 39, 0.62, 0.5),
    "ZN5": (11.6, 40, 0.04, 0.02, 14, 0.059, 352, 27, 0.84, 0.4),
    "ZN6": (11.6, 40, 0.21, 0.10, 14, 0.059, 226, 40, 0.61, 0.5),
    "Chen": (9.8, 21, 0.06, 0.03, 13.3, 0.042, 214, 19, 0.84, 0.7),
}
# L_u/b fitted to the model curves printed in Fig. 6 (the paper only states L_u/b ~ 1 for its own experiments)
LU_OVER_B_FITTED = {"RN1": 2.14, "RN2": 2.98, "RN3": 3.19}
CASES2 = {}
for _k, (_U0, _b, _a, _phi, _h, _S, _LI, _lam, _u0, _r) in _T2.items():
    # cylinder diameter recovered from phi = pi/4 a d; Cd = 1, Cf = 0.006 (Sec. 5), S = Cf U0^2/(g h) ~ Table 2
    CASES2[_k] = Params(U0=_U0, h=_h, b=_b, a=_a, d=4 * _phi / (np.pi * _a), Cd=1.0, Cf=0.006,
                        S=None, lam=_lam, L=(2 * _b if _k == "Chen" else None), Lu=LU_OVER_B_FITTED.get(_k, 1.0) * _b, LI=_LI)
PAPER_RMSE_CMS = {  # RMSE (cm/s) reported in Tables 1 and 2
    "Case1": 1.3, "Case2": 1.7, "Case3": 1.3, "Case4": 2.5, "Case5": 2.4, "Case6": 1.2,
    "Case7": 0.9, "Case8": 1.5, "Case9": 1.2,
    **{k: v[-1] for k, v in _T2.items()},
}
