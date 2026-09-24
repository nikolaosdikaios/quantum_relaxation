#!/usr/bin/env python3
# ============================================================================
# lift_validation.py
#
# Tests of the candidate "superpartner-pair" (lifted / inertial) relaxation
# equations on the state complex:
#
#     du/dt   = - Q^T Jc
#     dJc/dt  = - Jc/tau_c + Omega Js + Q u          (per edge, Bohr freq w_e)
#     dJs/dt  = - Js/tau_c - Omega Jc
#
# with Q built from BARE second moments (no spectral density input).
#
# Claims tested:
#  [A] Single-spin dephasing: the pair equation reproduces the EXACT Kubo
#      lineshape in the motional-narrowing regime, captures the correct
#      short-time (second-moment) behavior in the rigid regime where the
#      Markov/Bloch prediction fails catastrophically, and converges to the
#      exact Kubo result as the memory hierarchy (Mori/HEOM tower) is deepened.
#      Tier N=1 of the exact hierarchy IS the pair equation.
#  [B] Three-spin complex: the lift is (i) unconditionally stable
#      (Lyapunov: dissipation lives only in the J sector), (ii) has the
#      Gibbs/Witten vacuum (u,J)=(e^{-f},0) as exact stationary state,
#      (iii) satisfies generalized detailed balance with parity
#      (u even, Jc odd, Js even): E G E = G^T  (Onsager-Casimir),
#      (iv) reduces by adiabatic elimination of J to the Markov generator
#      with EMERGENT Bloembergen-Purcell-Pound rates
#      w_e = Delta_e^2 * tau_c/(1+w_e^2 tau_c^2), and (v) departs from
#      Bloch in the slow-bath regime, where the pair (tier 1) rings.
#  [C] Low-field benchmark against the Gaussian stochastic Liouville
#      hierarchy: for Gaussian fluctuations the ringing of the pair is a
#      truncation artifact (the pair is exact for one two-state fluctuator,
#      see hierarchy_checks.py); the converged recovery is non-exponential.
# ============================================================================
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
        print("    the tier-1 pair departs from Bloch and rings here;"
              " section [C] compares it with higher tiers")

# ============================================================================
# [C] Low-field benchmark: pair (tier 1) vs the stochastic Liouville hierarchy
# ============================================================================
print("=" * 76)
print("[C] LOW-FIELD RECOVERY: pair (tier 1) vs stochastic Liouville hierarchy")
# With infinite-temperature weights and per-axis parameters the spin-flip
# symmetry decouples the three-spin lift into Walsh sectors. The sector of
# spin 1 is one spin with Delta^2 = Dsq[0], Bohr frequency osc*omg[0] and
# memory tc. Its microscopic model is a spin in a static field along z and an
# Ornstein-Uhlenbeck field along x with second moment 2*Dsq[0]. The SLE ladder
# below is exact for that model as N -> infinity, and its tier 1 is the pair.
Rz = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
Rx = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])

def sle_mz(Delta, tau, w0, N, t):
    d = 3 * (N + 1); A = np.zeros((d, d))
    for n in range(N + 1):
        A[3*n:3*n+3, 3*n:3*n+3] = w0 * Rz - (n / tau) * np.eye(3)
        if n < N:
            A[3*n:3*n+3, 3*(n+1):3*(n+1)+3] = Delta * np.sqrt(n + 1) * Rx
            A[3*(n+1):3*(n+1)+3, 3*n:3*n+3] = Delta * np.sqrt(n + 1) * Rx
    w, V = np.linalg.eig(A); c = np.linalg.solve(V, np.eye(d)[:, 2])
    return np.real((V[2, :] * c) @ np.exp(np.outer(w, t)))

def zero_crossings(y):
    return int(np.sum(np.diff(np.sign(y + 1e-15)) != 0))

tcL, oscL = 2.0, 0.08
tL = np.linspace(0, 25.0, 5001)
GL = lift_generator(tcL, oscL)
wL, VL = np.linalg.eig(GL)
x0 = np.zeros(32); x0[0:8] = Z1
cL = np.linalg.solve(VL, x0)
m_pair = np.real(((Z1 @ VL[0:8, :]) * cL) @ np.exp(np.outer(wL, tL)))
w0L, DeltaL = oscL * omg[0], np.sqrt(2 * Dsq[0])
sle = {N: sle_mz(DeltaL, tcL, w0L, N, tL) for N in (1, 2, 4, 8, 16, 32)}
d1 = np.abs(sle[1] - m_pair).max()
print(f"  benchmark: kappa = {np.sqrt(Dsq[0]) * tcL:.2f}, mu = {w0L * tcL:.2f},"
      f" infinite-temperature weights")
print(f"  max|SLE tier 1 - three-spin pair, sector of spin 1| = {d1:.1e}"
      f"   [{PASS(d1 < 1e-8)}]")
print(f"  tier 1 (= pair): zero crossings = {zero_crossings(sle[1])}")
zc = [zero_crossings(sle[N]) for N in (2, 4, 8, 16, 32)]
print(f"  zero crossings at tiers 2, 4, 8, 16, 32: {zc}")
print(f"  tiers 2-32: max zero crossings = {max(zc)}   [{PASS(max(zc) == 0)}]")
conv = np.abs(sle[32] - sle[16]).max()
mono = bool(np.all(np.diff(sle[32]) <= 1e-12))
print(f"  convergence max|tier 32 - tier 16| = {conv:.1e}, converged recovery"
      f" monotonic: {mono}   [{PASS(conv < 1e-3 and mono)}]")
rates = np.linspace(0.1, 2.0, 1901)
r_best = rates[int(np.argmin([np.sum((sle[32] - np.exp(-r * tL)) ** 2)
                              for r in rates]))]
dev_best = np.abs(sle[32] - np.exp(-r_best * tL)).max()
r_bpp = DeltaL ** 2 * tcL / (1 + (w0L * tcL) ** 2)
dev_bpp = np.abs(sle[32] - np.exp(-r_bpp * tL)).max()
dev_pair = np.abs(sle[32] - sle[1]).max()
print(f"  best single exponential (rate {r_best:.3f}): max deviation {dev_best:.3f}")
print(f"  Markovian BPP exponential (rate {r_bpp:.3f}): max deviation {dev_bpp:.3f}")
print(f"  pair (tier 1): max deviation from converged {dev_pair:.3f}")
print("  -> for Gaussian fluctuations the ringing of the pair is a truncation")
print("     artifact; for one two-state fluctuator the pair is exact and the")
print("     ringing physical (hierarchy_checks.py). No single T1 fits either.")

print("=" * 76)
print("SUMMARY: the pair equations are stable, keep the Gibbs state stationary,")
print("satisfy the parity identity, and reduce to Bloch with emergent BPP rates")
print("when the currents are fast. For OU dephasing the pair is tier 1 of the")
print("hierarchy, which converges to the exact Kubo decay. In the slow-bath,")
print("low-field regime the pair rings, exactly so for one two-state")
print("fluctuator; for Gaussian fluctuations the recovery is non-exponential.")
