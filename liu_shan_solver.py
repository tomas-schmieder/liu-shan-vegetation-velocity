"""
Analytical model for the longitudinal profile of the depth-averaged streamwise
velocity U_d(x) in the central region of a channel with an emergent model
vegetation patch.

Implements Eqs. (1)-(3), (11)-(17) and the four boundary conditions of
Sec. 4.2 of

    Liu, C. & Shan, Y. (2019). Analytical model for predicting the longitudinal
    profiles of velocities in a channel with a model vegetation patch.
    Journal of Hydrology 576, 561-574. https://doi.org/10.1016/j.jhydrol.2019.06.076

Units: cm, s (g = 981 cm/s^2), matching the paper's tables.

Governing equation (Eq. 11), written for W(x) = U_d(x)^2 in Regions 2 and 3:

    lam*h^2*sqrt(Cf) * W'' - h * W' - K * W + g*h*S = 0,
    K = Cf                          (Region 2, no vegetation drag)
    K = Cf + 0.5*Cd*a*h/(1 - phi)   (Region 3, inside the patch)

Because the ODE is linear in W, the general solution is W = c1 e^{r+ x} + c2 e^{r- x} + g h S / K,
i.e. U_d = (A e^{r x} + B e^{r' x} + omega)^(1/2)  (Eqs. 14, 15).
Regions 1 and 4 are the constant solutions (Eqs. 13, 16).

Note on the printed roots: the paper gives r2 = 2h - r1 (and r4 = 2h - r3). Those
are the second root of the characteristic equation with the *same denominator*
2*lam*h^2*sqrt(Cf), i.e. r2 = (h - sqrt(D)) / (2*lam*h^2*sqrt(Cf)). Taken literally as
"2h minus r1" the expression is dimensionally inconsistent and does not satisfy Eq. (11),
so the root of the characteristic polynomial is used here (verified numerically in test_solver.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

G = 981.0  # cm/s^2


@dataclass(frozen=True)
class Params:
    """Inputs of the model (cm, cm/s)."""
    U0: float            # mean channel velocity far upstream (cm/s)
    h: float             # flow depth (cm)
    b: float             # half patch width (cm)
    a: float             # frontal area per patch volume, a = n d (1/cm)
    d: float = 0.4       # cylinder diameter (cm)
    Cd: float = 1.0      # drag coefficient
    Cf: float = 0.006    # bed friction coefficient
    S: Optional[float] = None   # free-surface slope; None -> S = Cf U0^2 /(g h) (Sec. 8.2)
    lam: float = 14.0    # dimensionless eddy-viscosity parameter, eps_d = lam * h * u*
    L: Optional[float] = None   # patch length (cm); None -> long patch (> L_I)
    Lu: Optional[float] = None  # upstream adjustment length (cm); None -> Lu = b (Lu/b ~ 1)
    LI: Optional[float] = None  # interior adjustment length (cm); None -> Eq. (1)

    @property
    def phi(self) -> float:
        """Solid volume fraction, phi = pi/4 n d^2 = pi/4 a d."""
        return np.pi / 4.0 * self.a * self.d

    @property
    def Cdab(self) -> float:
        return self.Cd * self.a * self.b

    @property
    def slope(self) -> float:
        return self.S if self.S is not None else self.Cf * self.U0**2 / (G * self.h)

    @property
    def lu(self) -> float:
        return self.Lu if self.Lu is not None else self.b

    @property
    def li(self) -> float:
        return self.LI if self.LI is not None else interior_adjustment_length(self.Cd, self.a, self.b)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


# --------------------------------------------------------------------------- empirical relations
def interior_adjustment_length(Cd: float, a: float, b: float) -> float:
    """Eq. (1) (Rominger & Nepf 2011): L_I = 5.5 * sqrt((2/(Cd a))^2 + b^2)."""
    return 5.5 * np.sqrt((2.0 / (Cd * a)) ** 2 + b**2)


def penetration_distance(Cd: float, a: float, d: float) -> float:
    """Eq. (2): delta = max[0.5 (Cd a)^-1, 1.8 d]."""
    return max(0.5 / (Cd * a), 1.8 * d)


def drag_coefficient(Re_d0: float) -> float:
    """Eq. (3) (White 1991): Cd = 1 + Re_d(0)^(-2/3)."""
    return 1.0 + Re_d0 ** (-2.0 / 3.0)


def Ud0_over_U0_fit(Cdab: float) -> float:
    """Eq. (17): U_d(0)/U_0 = 1 - 0.15 Cd a b (valid for 0 <= Cd a b <= 9)."""
    return 1.0 - 0.15 * Cdab


# --------------------------------------------------------------------------- model
def _roots(p: Params, K: float):
    """Roots of lam h^2 sqrt(Cf) r^2 - h r - K = 0 (Eqs. 14, 15)."""
    c = p.lam * p.h**2 * np.sqrt(p.Cf)
    disc = np.sqrt(p.h**2 + 4.0 * c * K)
    return (p.h + disc) / (2.0 * c), (p.h - disc) / (2.0 * c)


@dataclass
class Solution:
    params: Params
    r1: float; r2: float; r3: float; r4: float
    w1: float; w2: float
    A1: float; A2: float; A3: float; A4: float   # constants of Eqs. (14), (15)
    C2: float; C3: float                         # A2 e^{-r2 Lu}, A3 e^{r3 LI}: same terms in a bounded basis
    Ud1: float  # Region 1 velocity, Eq. (13)
    Ud4: float  # Region 4 velocity, Eq. (16)

    def W(self, x):
        """U_d^2 profile in the four regions; x measured from the upstream patch edge (cm)."""
        p = self.params
        x = np.atleast_1d(np.asarray(x, dtype=float))
        LI, Lu = p.li, p.lu
        # growing exponentials are evaluated relative to the far end of their region so they never overflow
        out = np.empty_like(x)
        r1 = x < -Lu
        r2 = (x >= -Lu) & (x < 0)
        r3 = (x >= 0) & (x < LI)
        r4 = x >= LI
        out[r1] = self.w1
        xs = x[r2]
        out[r2] = self.A1 * np.exp(self.r1 * xs) + self.C2 * np.exp(self.r2 * (xs + Lu)) + self.w1
        xs = x[r3]
        out[r3] = self.C3 * np.exp(self.r3 * (xs - LI)) + self.A4 * np.exp(self.r4 * xs) + self.w2
        out[r4] = self.w2
        return out

    def Ud(self, x):
        """Depth-averaged streamwise velocity U_d(x) (cm/s); patch ends at L if L is set."""
        W = self.W(x)
        return np.sqrt(np.clip(W, 0.0, None))

    def dUd_dx(self, x):
        """Analytical derivative dU_d/dx = W'/(2 U_d) in Regions 2 and 3 (used in BC 3)."""
        p = self.params
        x = np.atleast_1d(np.asarray(x, dtype=float))
        LI, Lu = p.li, p.lu
        dW = np.zeros_like(x)
        m2 = (x >= -Lu) & (x < 0)
        m3 = (x >= 0) & (x < LI)
        dW[m2] = (self.A1 * self.r1 * np.exp(self.r1 * x[m2])
                  + self.C2 * self.r2 * np.exp(self.r2 * (x[m2] + Lu)))
        dW[m3] = (self.C3 * self.r3 * np.exp(self.r3 * (x[m3] - LI))
                  + self.A4 * self.r4 * np.exp(self.r4 * x[m3]))
        return dW / (2.0 * self.Ud(x))


def solve(p: Params) -> Solution:
    """Solve for A1..A4 from the four boundary conditions of Sec. 4.2."""
    h, Cf, phi = p.h, p.Cf, p.phi
    S = p.slope
    Lu, LI = p.lu, p.li

    K2 = Cf
    K3 = Cf + 0.5 * p.Cd * p.a * h / (1.0 - phi)
    r1, r2 = _roots(p, K2)
    r3, r4 = _roots(p, K3)
    w1 = G * h * S / K2          # Eq. (14): omega_1
    w2 = G * h * S / K3          # Eq. (15): omega_2
    Ud1_sq = G * h * S / Cf      # Eq. (13)
    Ud4_sq = G * h * S / K3      # Eq. (16)

    # Unknowns [A1, C2, C3, A4] with C2 = A2 e^{-r2 Lu}, C3 = A3 e^{r3 LI}. This is the same 4x4 linear
    # system as for [A1..A4] but with every exponential <= 1 (the plain basis overflows for small lambda).
    # BC1: Ud(1) = Ud(2) at x = -Lu    -> A1 e^{-r1 Lu} + A2 e^{-r2 Lu} + w1 = Ud1^2
    # BC2: Ud(2) = Ud(3) at x = 0      -> A1 + A2 + w1 = A3 + A4 + w2
    # BC3: dUd(2)/dx = dUd(3)/dx at 0  -> (r1 A1 + r2 A2)/(2 Ud) = (r3 A3 + r4 A4)/(2 Ud); Ud equal by BC2
    # BC4: Ud(3) = Ud(4) at x = LI     -> A3 e^{r3 LI} + A4 e^{r4 LI} + w2 = Ud4^2
    f2 = np.exp(r2 * Lu)    # A2 = C2 e^{r2 Lu}
    f3 = np.exp(-r3 * LI)   # A3 = C3 e^{-r3 LI}
    M = np.array([
        [np.exp(-r1 * Lu), 1.0, 0.0, 0.0],
        [1.0, f2, -f3, -1.0],
        [r1, r2 * f2, -r3 * f3, -r4],
        [0.0, 0.0, 1.0, np.exp(r4 * LI)],
    ])
    rhs = np.array([Ud1_sq - w1, w2 - w1, 0.0, Ud4_sq - w2])
    A1, C2, C3, A4 = np.linalg.solve(M, rhs)
    with np.errstate(over="ignore"):
        A2, A3 = C2 * f2, C3 * f3
    return Solution(p, r1, r2, r3, r4, w1, w2, A1, A2, A3, A4, C2, C3,
                    np.sqrt(Ud1_sq), np.sqrt(Ud4_sq))


def profile(p: Params, x):
    """Convenience: U_d(x) for the patch described by p. Points beyond x = L (if L set) are NaN."""
    sol = solve(p)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    u = sol.Ud(x)
    if p.L is not None:
        u = np.where(x > p.L, np.nan, u)
    return u


def calibrate_lambda(p: Params, Ud0_over_U0: float, lo: float = 0.05, hi: float = 200.0) -> float:
    """Find lam so that the predicted U_d(0)/U_0 equals the measured/Eq.(17) value (Sec. 3 / Sec. 6).

    U_d(0)/U_0 decreases with lam up to a minimum (about lam = 40 for Case 3) and then rises slowly, so the
    search is restricted to the monotonic branch lam < lam_min. Returns NaN if the target is not reachable.
    """
    from scipy.optimize import brentq, minimize_scalar
    u0 = lambda lam: solve(p.with_(lam=lam)).Ud(0.0)[0] / p.U0
    lam_min = float(np.exp(minimize_scalar(lambda t: u0(np.exp(t)), bounds=(np.log(lo), np.log(hi)),
                                           method="bounded").x))
    if not (u0(lam_min) <= Ud0_over_U0 <= u0(lo)):
        return float("nan")
    return brentq(lambda lam: u0(lam) - Ud0_over_U0, lo, lam_min)


def rmse(measured, predicted) -> float:
    """Eq. (4)."""
    m, q = np.asarray(measured, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((m - q) ** 2)))
