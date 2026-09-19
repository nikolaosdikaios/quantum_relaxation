#!/usr/bin/env python3

import numpy as np
import itertools

np.set_printoptions(precision=4, suppress=True)
PASS = lambda ok: "PASS" if ok else "FAIL"

# ============================================================================
# [A] Kubo oscillator benchmark (transverse dephasing by OU noise)
# ============================================================================
print("=" * 76)
print("[A] SINGLE SPIN, OU DEPHASING: pair equation vs exact Kubo vs Markov")

def kubo_exact(t, D, tc):
    return np.exp(-(D * tc) ** 2 * (np.exp(-t / tc) - 1 + t / tc))

def pair_fid(t, D, tc):
    disc = 1 / (4 * tc ** 2) - D ** 2
    if disc > 0:
        r1 = -1 / (2 * tc) + np.sqrt(disc)
        r2 = -1 / (2 * tc) - np.sqrt(disc)
        return (-r2 * np.exp(r1 * t) + r1 * np.exp(r2 * t)) / (r1 - r2)
    nu = np.sqrt(-disc)
    return np.exp(-t / (2 * tc)) * (np.cos(nu * t) + np.sin(nu * t) / (2 * tc * nu))

def markov_fid(t, D, tc):
    return np.exp(-D ** 2 * tc * t)

def hierarchy_fid(t, D, tc, N):
    """Exact SLE hierarchy for Gaussian OU noise, truncated at tier N.
       Tier N=1 is the pair equation. g_n coupled by the ladder
       dg_n/dt = -(n/tc) g_n - i D (sqrt(n+1) g_{n+1} + sqrt(n) g_{n-1})."""
    M = np.zeros((N + 1, N + 1), complex)
    for n in range(N + 1):
        M[n, n] = -n / tc
        if n + 1 <= N:
            M[n, n + 1] = -1j * D * np.sqrt(n + 1)
            M[n + 1, n] = -1j * D * np.sqrt(n + 1)
    w, V = np.linalg.eig(M)
    Vi = np.linalg.inv(V)
    g0 = np.zeros(N + 1, complex); g0[0] = 1
    c = Vi @ g0
    return np.real(np.array([(V @ (np.exp(w * tt) * c))[0] for tt in t]))

for Dtc, label in ((0.2, "motional narrowing"), (5.0, "rigid / slow bath")):
    D = 1.0; tc = Dtc / D
    rate_slow = min(D ** 2 * tc, D)
    t = np.linspace(0, 5 / rate_slow, 1200)
    ex = kubo_exact(t, D, tc)
    pr = pair_fid(t, D, tc)
    mk = markov_fid(t, D, tc)
    h1 = hierarchy_fid(t, D, tc, 1)
    h8 = hierarchy_fid(t, D, tc, 8)
    print(f"  Delta*tau_c = {Dtc}  ({label})")
    print(f"    max|pair - exact|        = {np.abs(pr - ex).max():.3f}")
    print(f"    max|Markov/Bloch - exact|= {np.abs(mk - ex).max():.3f}")
    print(f"    max|tier-1 - pair|       = {np.abs(h1 - pr).max():.2e}"
          f"   [{PASS(np.abs(h1 - pr).max() < 1e-8)}]  (pair = first Mori tier)")
    print(f"    max|tier-8 - exact|      = {np.abs(h8 - ex).max():.4f}"
          f"   (hierarchy converges to exact)")
    # short-time second moment: G ~ 1 - (D t)^2/2
    ts = t[t < 0.2 / D]
    sm_pair = np.abs(pair_fid(ts, D, tc) - (1 - (D * ts) ** 2 / 2)).max()
    sm_mark = np.abs(markov_fid(ts, D, tc) - (1 - (D * ts) ** 2 / 2)).max()
    print(f"    short-time 2nd moment: pair err {sm_pair:.1e}, "
          f"Markov err {sm_mark:.1e}   [{PASS(sm_pair < 5 * sm_mark or sm_pair < 1e-3)}]")

# ============================================================================
# [B] Three-spin complex lift
# ============================================================================
print("=" * 76)
print("[B] THREE-SPIN COMPLEX LIFT: stability, vacuum, parity, BPP emergence")

N = 3
verts = list(itertools.product((0, 1), repeat=N))
vidx = {v: i for i, v in enumerate(verts)}
EDGE = []          # (tail, head, axis), oriented bit 0 -> 1
for v in verts:
    for a in range(N):
        if v[a] == 0:
            u_ = list(v); u_[a] = 1
            EDGE.append((v, tuple(u_), a))
B1 = np.zeros((8, 12))
for e, (t_, h_, a) in enumerate(EDGE):
    B1[vidx[h_], e] = +1
    B1[vidx[t_], e] = -1

Dsq  = np.array([0.7, 1.3, 2.1])        # bare second moments per spin axis
omg  = np.array([4.0, 5.0, 6.0])        # Bohr frequencies per axis
Dax  = np.array([np.sqrt(Dsq[a]) for (_, _, a) in EDGE])
Oax  = np.array([omg[a] for (_, _, a) in EDGE])
Q    = np.diag(Dax) @ B1.T              # bare supercharge (high-T weights)

def lift_generator(tc, oscale=1.0):
    G = np.zeros((32, 32))
    G[0:8, 8:20]    = -Q.T
    G[8:20, 0:8]    = Q
    G[8:20, 8:20]   = -np.eye(12) / tc
    G[8:20, 20:32]  = np.diag(oscale * Oax)
    G[20:32, 8:20]  = -np.diag(oscale * Oax)
    G[20:32, 20:32] = -np.eye(12) / tc
    return G

def H_eff(tc):
    return Q.T @ np.diag(tc / (1 + (Oax * tc) ** 2)) @ Q

def H_BPP(tc):
    w_ax = np.array([Dsq[a] * tc / (1 + (omg[a] * tc) ** 2)
                     for (_, _, a) in EDGE])
    return B1 @ np.diag(w_ax) @ B1.T

tc = 1.5
G = lift_generator(tc)

# (i) stability: Lyapunov argument says max Re(eig) <= 0
ev = np.linalg.eigvals(G)
print(f"  stability      max Re eig(G) = {ev.real.max():+.2e}"
      f"   [{PASS(ev.real.max() < 1e-10)}]")
print(f"  oscillation    max Im eig(G) = {np.abs(ev.imag).max():.2f}"
      f"   (lift supports ringing; Markov generator is purely real)")

# (ii) Gibbs/Witten vacuum of the lift
vac = np.zeros(32); vac[0:8] = 1 / np.sqrt(8)
print(f"  vacuum         ||G (e^-f, 0)|| = {np.linalg.norm(G @ vac):.2e}"
      f"   [{PASS(np.linalg.norm(G @ vac) < 1e-13)}]")

# (iii) generalized detailed balance with parity (u:+, Jc:-, Js:+)
E = np.diag([1] * 8 + [-1] * 12 + [1] * 12).astype(float)
res_par = np.abs(E @ G @ E - G.T).max()
print(f"  parity DB      ||E G E - G^T|| = {res_par:.2e}"
      f"   [{PASS(res_par < 1e-13)}]   (Onsager-Casimir reversibility)")

# (iv) adiabatic elimination -> Markov generator with emergent BPP rates
res_bpp = np.abs(H_eff(tc) - H_BPP(tc)).max()
print(f"  BPP emergence  ||H_eff - H_BPP|| = {res_bpp:.2e}"
      f"   [{PASS(res_bpp < 1e-12)}]")
print("    emergent rates w_a = D_a^2 tau_c/(1+w_a^2 tau_c^2):",
      np.round(Dsq * tc / (1 + (omg * tc) ** 2), 5))

# (v) dynamics: fast bath agrees with Bloch, slow bath departs and rings
def rk4(G, x0, T, dt):
    x = x0.copy(); traj = [x0.copy()]; ts = [0.0]; t = 0.0
    while t < T:
        k1 = G @ x
        k2 = G @ (x + dt / 2 * k1)
        k3 = G @ (x + dt / 2 * k2)
        k4 = G @ (x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt; traj.append(x.copy()); ts.append(t)
    return np.array(ts), np.array(traj)

Z1 = np.array([(-1) ** v[0] for v in verts], float) / np.sqrt(8)
for tc_, osc_, T_, dt_, label in (
        (0.02, 1.0, 120.0, 0.004, "fast bath, high field (narrowing)"),
        (2.0, 0.08, 25.0, 0.002, "slow bath, low field (rigid, w0*tc<1)")):
    Gl = lift_generator(tc_, osc_)
    x0 = np.zeros(32); x0[0:8] = 0.6 * Z1          # longitudinal perturbation
    ts, tr = rk4(Gl, x0, T_, dt_)
    m_lift = tr[:, 0:8] @ Z1
    lam1 = 2 * Dsq[0] * tc_ / (1 + (osc_ * omg[0] * tc_) ** 2)
    m_markov = 0.6 * np.exp(-lam1 * ts)
    dev = np.abs(m_lift - m_markov).max() / 0.6
    n_cross = int(np.sum(np.diff(np.sign(m_lift + 1e-15)) != 0))
    print(f"  dynamics ({label}): tau_c = {tc_}")
    print(f"    max relative deviation lift vs Bloch = {dev:.3f}"
          f"   zero crossings of m1(t): {n_cross}")
    if tc_ == 0.02:
        print(f"    [{PASS(dev < 0.05)}]  lift reproduces Bloch when "
              f"Delta*tau_c, omega*tau_c are small")
    else:
        print(f"    [{PASS(dev > 0.3 and n_cross >= 1)}]  oscillatory "
              f"(nutation-like) recovery: no single T1 can fit this")

print("=" * 76)
print("SUMMARY: the pair/lift equations are stable, thermodynamically exact")
print("(Gibbs vacuum, parity detailed balance), reduce to Bloch with emergent")
print("BPP rates when the superpartner is fast, reproduce exact Kubo physics")
print("at tier 1 of the Mori hierarchy, and predict inertial ringing that the")
print("Bloch parametrization cannot express.")
