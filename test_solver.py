"""Checks that the solver satisfies Eq. (11) and the four boundary conditions of Sec. 4.2."""
import numpy as np

from liu_shan_solver import G, Params, solve
from cases import CASES


def check(name, p):
    s = solve(p)
    h, Cf, S, phi = p.h, p.Cf, p.slope, p.phi
    Lu, LI = p.lu, p.li
    K = {2: Cf, 3: Cf + 0.5 * p.Cd * p.a * h / (1 - phi)}
    c = p.lam * h**2 * np.sqrt(Cf)

    # Eq. (11) residual by central finite differences on W = Ud^2 inside Regions 2 and 3
    for reg, xs in ((2, np.linspace(-Lu * 0.9, -Lu * 0.1, 5)), (3, np.linspace(0.1 * LI, 0.9 * LI, 5))):
        dx = 1e-2
        W = lambda x: s.W(x)
        W1 = (W(xs + dx) - W(xs - dx)) / (2 * dx)
        W2 = (W(xs + dx) - 2 * W(xs) + W(xs - dx)) / dx**2
        res = G * h * S + c * W2 - h * W1 - K[reg] * W(xs)
        scale = G * h * S
        assert np.max(np.abs(res)) / scale < 1e-4, (name, reg, res / scale)

    eps = 1e-6
    u = lambda x: s.Ud(x)[0]
    # BC1..BC4
    assert abs(u(-Lu + eps) - s.Ud1) / s.Ud1 < 1e-6, name
    assert abs(u(-eps) - u(eps)) / u(0.0) < 1e-5, name
    d2 = s.dUd_dx(-eps)[0]
    d3 = s.dUd_dx(eps)[0]
    assert abs(d2 - d3) < 1e-6 * max(1.0, abs(d2)), (name, d2, d3)
    # analytical derivative agrees with a numerical derivative of Ud
    num = (u(0.01) - u(-0.01)) / 0.02
    assert abs(num - 0.5 * (d2 + d3)) < 5e-3 * max(1e-3, abs(d2)), (name, num, d2)
    assert abs(u(LI - eps) - s.Ud4) / s.Ud4 < 1e-5, name
    # profile is monotonically non-increasing and bounded between Ud4 and Ud1
    x = np.linspace(-2 * Lu, 1.2 * LI, 600)
    v = s.Ud(x)
    assert np.all(np.diff(v) <= 1e-9), name
    print(f"{name}: OK  (Ud(0)/U0 = {u(0.0) / p.U0:.3f}, Ud4/U0 = {s.Ud4 / p.U0:.3f})")


if __name__ == "__main__":
    for name, p in CASES.items():
        check(name, p)
    print("all checks passed")
