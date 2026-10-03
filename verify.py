#!/usr/bin/env python3
"""
verify_jcp.py -- independent checks of the analytical statements in
"Spin relaxation with bath memory beyond the correlation function".

Each check prints PASS or FAIL. Requires numpy and scipy. Runtime ~10 s.

 [1] Eqs. (3)-(4) for a spin-1/2 equal the exact dynamics for one two-state
     jump process (M = 1) of the transverse field.
 [2] Short memory: Eqs. (3)-(4) give the BPP rate, Eq. (9), and the Redfield
     transverse rate and dynamic frequency shift follow from the exact model.
 [3] Any temperature: the thermal state is stationary, probability is
     conserved and F decreases at the rate of Eq. (7) (random five-level system).
 [4] Eq. (10): slow decay rate of the coherence and its series; Eq. (13): the
     long-time rate for M jump processes and its Gaussian limit.
 [5] Ladder of Eq. (12) against direct simulation of the 2^M joint states;
     symmetric two-site exchange (Bloch-McConnell) against Eq. (5);
     factorization G_M = G_1^M; kurtosis 3 - 2/M, Eq. (14).
 [6] Two spins: Eqs. (3)-(4) equal the joint ladder (B1) cut at one excitation;
     short-time result, Eq. (15); Solomon limit sigma = W2 - W0.
 [7] Positivity: the bound |p_a - peq_a| <= sqrt(2 F(0) peq_a), and the negative
     population quoted in the limitations (two spins, zero field, pure state).
"""
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

rng = np.random.default_rng(7)
ok_all = True


def report(label, value, tol):
    global ok_all
    ok = bool(value < tol)
    ok_all &= ok
    print(f"  {label:<62} {value:9.2e}  [{'PASS' if ok else 'FAIL'}]")


def transition_current_rhs(edges, w, d2, tau, peq):
    """Right-hand side of Eqs. (3)-(4). State vector: populations, then currents."""
    n = len(peq)
    c = d2 * np.sqrt(peq[[a for a, _ in edges]] * peq[[b for _, b in edges]])

    def rhs(_t, y):
        p, v = y[:n], y[n:]
        dp = np.zeros(n, complex)
        dv = np.zeros(len(edges), complex)
        for e, (a, b) in enumerate(edges):
            dp[a] -= v[e].real
            dp[b] += v[e].real
            dv[e] = -(1 / tau[e] + 1j * w[e]) * v[e] + c[e] * (p[a] / peq[a] - p[b] / peq[b])
        return np.concatenate([dp, dv])
    return rhs, c


def ladder(delta, tau, n_max, M=None):
    L = n_max if M is None else min(n_max, M)
    A = np.zeros((L + 1, L + 1), complex)
    for n in range(L + 1):
        A[n, n] = -n / tau
        if n < L:
            b = np.sqrt(n + 1) if M is None else np.sqrt((n + 1) * (1 - n / M))
            A[n, n + 1] = A[n + 1, n] = -1j * delta * b
    return A


def propagate(A, t, row=0, start=0):
    w, V = np.linalg.eig(A)
    c = np.linalg.solve(V, np.eye(len(A))[:, start])
    return (V[row, :] * c) @ np.exp(np.outer(w, t))


RZ = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
RX = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])

# ---------------------------------------------------------------------- [1]
print("[1] spin-1/2 in a transverse field: Eqs. (3)-(4) vs exact M = 1 dynamics")
b_rms, tau, w0 = np.sqrt(1.4), 2.0, 0.32
t = np.linspace(0, 25, 2001)
A = np.zeros((6, 6))
A[:3, :3] = w0 * RZ
A[3:, 3:] = w0 * RZ - np.eye(3) / tau
A[:3, 3:] = A[3:, :3] = b_rms * RX
mz_exact = np.real(propagate(A, t, row=2, start=2))
rhs, _ = transition_current_rhs([(0, 1)], np.array([w0]), np.array([b_rms**2 / 2]),
                                np.array([tau]), np.array([0.5, 0.5]))
sol = solve_ivp(rhs, (0, 25), np.array([1, 0, 0], complex), t_eval=t, rtol=1e-11, atol=1e-13)
report("max |m_z(Eqs. 3-4) - m_z(exact two-state)|",
       np.abs((sol.y[0] - sol.y[1]).real - mz_exact).max(), 1e-8)

# ---------------------------------------------------------------------- [2]
print("[2] short memory (kappa = 0.05): BPP rate, Redfield T2 and dynamic shift")
tau, w0 = 0.5, 3.0
bx2, bz2 = 2 * 0.1**2, 0.1**2                    # <b_x^2>, <b_z^2>; d2_perp = bx2 / 2
J = lambda w: tau / (1 + (w * tau)**2)
rhs, _ = transition_current_rhs([(0, 1)], np.array([w0]), np.array([bx2 / 2]),
                                np.array([tau]), np.array([0.5, 0.5]))
G = np.array([rhs(0, np.eye(3, dtype=complex)[k]) for k in range(3)]).T
# real form of the generator on (p0 - p1, Re v, Im v)
Gr = np.array([[0, -2, 0], [bx2 / 2, -1 / tau, w0], [0, -w0, -1 / tau]])
slow = np.sort(np.abs(np.linalg.eigvals(Gr).real))[0]
T1_inv = 2 * (bx2 / 2) * J(w0)
report("relative error of the slow rate vs Eq. (9)", abs(slow - T1_inv) / T1_inv, 5e-3)
# exact model: independent two-state jump processes for b_x and b_z (4 blocks of 3)
I3 = np.eye(3)
blocks = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
A = np.zeros((12, 12))
for (nx, nz), k in blocks.items():
    A[3*k:3*k+3, 3*k:3*k+3] = w0 * RZ - (nx + nz) / tau * I3
    for (mx, mz), l in blocks.items():
        if mz == nz and abs(mx - nx) == 1:
            A[3*k:3*k+3, 3*l:3*l+3] = np.sqrt(bx2) * RX
        if mx == nx and abs(mz - nz) == 1:
            A[3*k:3*k+3, 3*l:3*l+3] = np.sqrt(bz2) * RZ
ev = np.linalg.eigvals(A)
ev_t = ev[np.argmin(np.abs(ev - 1j * w0))]       # transverse mode near +i w0
T2_inv = 0.5 * T1_inv + bz2 * J(0)
report("relative error of the transverse rate vs T1/2 + d2_z J(0)",
       abs(-ev_t.real - T2_inv) / T2_inv, 2e-2)
shift = (bx2 / 2) * w0 * tau**2 / (1 + (w0 * tau)**2)
report("relative error of the frequency shift vs d2 w tau^2/(1+w^2 tau^2)",
       abs(abs(ev_t.imag - w0) - shift) / shift, 2e-2)
print(f"      (shift of the Larmor frequency: {ev_t.imag - w0:+.3e}; formula magnitude {shift:.3e})")

# ---------------------------------------------------------------------- [3]
print("[3] thermal equilibrium, conservation and F at finite temperature (five levels)")
n = 5
E = np.sort(rng.uniform(0, 3, n))
peq = np.exp(-E / 0.8); peq /= peq.sum()
edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.8]
edges += [(k, k + 1) for k in range(n - 1) if (k, k + 1) not in edges]     # connected
m = len(edges)
w = np.array([E[a] - E[b] for a, b in edges])
d2, tau_e = rng.uniform(0.3, 2.0, m), rng.uniform(0.3, 3.0, m)
rhs, c = transition_current_rhs(edges, w, d2, tau_e, peq)
y_eq = np.concatenate([peq, np.zeros(m)]).astype(complex)
report("|d/dt| of the thermal state with zero currents", np.abs(rhs(0, y_eq)).max(), 1e-14)
p0 = rng.random(n); p0 /= p0.sum()
tt = np.linspace(0, 60, 6001)
sol = solve_ivp(rhs, (0, 60), np.concatenate([p0, np.zeros(m)]).astype(complex),
                t_eval=tt, rtol=1e-11, atol=1e-13)
p, v = sol.y[:n].real, sol.y[n:]
report("max deviation of total probability from 1", np.abs(p.sum(axis=0) - 1).max(), 1e-9)
F = 0.5 * ((p - peq[:, None])**2 / peq[:, None]).sum(axis=0) \
    + 0.5 * (np.abs(v)**2 / c[:, None]).sum(axis=0)
dF_formula = -(np.abs(v)**2 / (tau_e * c)[:, None]).sum(axis=0)
dy = np.array([rhs(0, sol.y[:, j]) for j in range(0, len(tt), 50)]).T     # exact dy/dt
dF_exact = (((p[:, ::50] - peq[:, None]) / peq[:, None]) * dy[:n].real).sum(axis=0) \
    + ((np.conj(v[:, ::50]) * dy[n:]).real / c[:, None]).sum(axis=0)
report("max |dF/dt - Eq. (7)| along the trajectory",
       np.abs(dF_exact - dF_formula[::50]).max(), 1e-13)
report("largest increase of F between samples", max(0.0, np.diff(F).max()), 1e-12)
report("distance from thermal populations at t = 60", np.abs(p[:, -1] - peq).max(), 1e-6)
pop_part = 0.5 * ((p - peq[:, None])**2 / peq[:, None]).sum(axis=0)
print(f"      (population part of F alone increases somewhere: {bool(np.diff(pop_part).max() > 1e-9)})")

# ---------------------------------------------------------------------- [4]
print("[4] Eq. (10): slow decay rate of a coherence and its series")
worst = 0.0
for kappa in (0.05, 0.1, 0.2, 0.3):
    ev = np.linalg.eigvals(np.array([[0, -1.0], [kappa**2, -1.0]]))     # tau_c = 1
    slow = np.sort(-ev.real)[0]
    worst = max(worst, abs(slow - (1 - np.sqrt(1 - 4 * kappa**2)) / 2))
report("max |slow rate - closed form|", worst, 1e-13)
k = 0.1
series = k**2 * (1 + k**2 + 2 * k**4 + 5 * k**6)
report("series vs closed form at kappa = 0.1 (next term 14 kappa^10)",
       abs(series - (1 - np.sqrt(1 - 4 * k**2)) / 2), 2e-9)
print(f"      (kappa = 0.3: rate theory {0.09:.3f}, Eq. (10) {(1-np.sqrt(1-0.36))/2:.3f}, "
      f"i.e. {100*(1-0.09/((1-np.sqrt(1-0.36))/2)):.0f}% too low)")
worst = 0.0
for kappa in (0.1, 0.3):
    for M in (1, 2, 3, 4, 8):
        slow = np.sort(-np.linalg.eigvals(ladder(kappa, 1.0, 64, M)).real)[0]
        worst = max(worst, abs(slow - M * (1 - np.sqrt(1 - 4 * kappa**2 / M)) / 2))
report("max |slow rate of the M ladder - Eq. (13)|, M = 1..8", worst, 1e-12)
slow_g = np.sort(-np.linalg.eigvals(ladder(0.3, 1.0, 80, None)).real)[0]
report("|Gaussian long-time rate - rate theory| at kappa = 0.3", abs(slow_g - 0.09), 1e-12)

# ---------------------------------------------------------------------- [5]
print("[5] ladder of Eq. (12) vs direct simulation of the 2^M joint states")


def direct_fid(delta, tau, M, t):
    k = 1 / (2 * tau)
    S = list(itertools.product((-1, 1), repeat=M))
    pos = {s: a for a, s in enumerate(S)}
    W = np.zeros((len(S),) * 2)
    for a, s in enumerate(S):
        for i in range(M):
            s2 = list(s); s2[i] = -s2[i]
            W[pos[tuple(s2)], a] += k; W[a, a] -= k
    xi = delta / np.sqrt(M) * np.array([sum(s) for s in S])
    w, V = np.linalg.eig(W - 1j * np.diag(xi))
    c = np.linalg.solve(V, np.full(len(S), 1 / len(S)))
    return (V.sum(axis=0) * c) @ np.exp(np.outer(w, t)), xi


t = np.linspace(0, 10, 1001)
worst, worst_k = 0.0, 0.0
for M in (1, 2, 3, 4, 5):
    f_dir, xi = direct_fid(1.0, 5.0, M, t)
    worst = max(worst, np.abs(propagate(ladder(1.0, 5.0, 40, M), t) - f_dir).max())
    worst_k = max(worst_k, abs(np.mean(xi**4) / np.mean(xi**2)**2 - (3 - 2 / M)))
report("max |ladder - direct simulation|, M = 1..5", worst, 1e-12)
report("max |kurtosis - (3 - 2/M)|, M = 1..5", worst_k, 1e-12)
# symmetric two-site exchange (Bloch-McConnell) against Eq. (5), tau_c = 1/(2 k_ex)
worst_bm = 0.0
for k_ex in (0.2, 1.0, 5.0):          # k_ex = 1 is the point where the eigenvalues merge
    BM = np.array([[-1j - k_ex, k_ex], [k_ex, 1j - k_ex]])       # sites at -+ Delta_omega = 1
    PAIR = np.array([[0, -1.0], [1.0, -2 * k_ex]])               # Eq. (5) for (rho, v), tau_c = 1/(2 k_ex)
    for tk in t[::50]:
        m_tot = (expm(BM * tk) @ np.array([0.5, 0.5])).sum()
        worst_bm = max(worst_bm, abs(m_tot - expm(PAIR * tk)[0, 0]))
report("max |Bloch-McConnell total magnetization - Eq. (5)|", worst_bm, 1e-12)
f1 = propagate(ladder(1 / np.sqrt(2), 5.0, 40, 1), t)
report("max |G_2 - G_1(delta/sqrt 2)^2|", np.abs(propagate(ladder(1.0, 5.0, 40, 2), t) - f1**2).max(), 1e-12)

# ---------------------------------------------------------------------- [6]
print("[6] two spins: Eqs. (3)-(4), short times and the Solomon limit")
E = np.array([0., 5.6, 5.0, 10.6])                       # |aa>, |ab>, |ba>, |bb> (I, S)
edges = [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2), (0, 3)]
d2 = np.array([1.5, 1.5, 1.5, 1.5, 1.0, 6.0])
tauN = 1.5
w = np.array([E[a] - E[b] for a, b in edges])
ZI = np.array([1., 1, -1, -1]); ZS = np.array([1., -1, 1, -1])
p0 = 0.25 - 0.25 * ZI
rhs, _ = transition_current_rhs(edges, w, d2, np.full(6, tauN), np.full(4, 0.25))
tt = np.linspace(0, 12, 1201)
sol = solve_ivp(rhs, (0, 12), np.concatenate([p0, np.zeros(6)]).astype(complex),
                t_eval=tt, rtol=1e-11, atol=1e-13)
mS_tc = sol.y[:4].real.T @ ZS
# joint ladder (B1) cut at one excitation: 7 blocks of 4x4 density matrices
I4 = np.eye(4)
L0 = np.diag([-1j * (E[a] - E[b]) for a in range(4) for b in range(4)])
G = np.zeros((16 * 7, 16 * 7), complex)
G[:16, :16] = L0
for e, (a, b) in enumerate(edges):
    X = np.zeros((4, 4)); X[a, b] = X[b, a] = 1
    C = -1j * np.sqrt(d2[e] / 2) * (np.kron(X, I4) - np.kron(I4, X.T))
    k = 16 * (e + 1)
    G[k:k+16, k:k+16] = L0 - np.eye(16) / tauN
    G[:16, k:k+16] = C; G[k:k+16, :16] = C
x0 = np.zeros(16 * 7, complex); x0[[0, 5, 10, 15]] = p0
P = expm(G * (tt[1] - tt[0]))
x, mS_lad = x0.copy(), []
for _ in tt:
    mS_lad.append(x[[0, 5, 10, 15]].real @ ZS)
    x = P @ x
report("max |m_S(Eqs. 3-4) - m_S(ladder cut at one excitation)|",
       np.abs(mS_tc - np.array(mS_lad)).max(), 1e-8)
report("|slope of m_S at t = 0|", abs((G @ x0)[[0, 5, 10, 15]].real @ ZS), 1e-14)
curv = (G @ (G @ x0))[[0, 5, 10, 15]].real @ ZS
report("|curvature - (-(d2_DQ - d2_ZQ) m_I(0))|, Eq. (15)", abs(curv - (-(6.0 - 1.0) * (-1.0))), 1e-12)
# same curvature from the exact two-state dynamics (64 blocks) -- generic statistics
idx = list(itertools.product((0, 1), repeat=6)); pos = {nn: k for k, nn in enumerate(idx)}
x0b = np.zeros(16 * 64, complex); x0b[[0, 5, 10, 15]] = p0


def apply(xv):
    out = np.zeros_like(xv)
    for k, nn in enumerate(idx):
        blk = xv[16*k:16*k+16]
        out[16*k:16*k+16] += (np.diag(L0) - sum(nn) / tauN) * blk
        for e, (a, b) in enumerate(edges):
            mm = list(nn); mm[e] = 1 - mm[e]; l = pos[tuple(mm)]
            X = np.zeros((4, 4)); X[a, b] = X[b, a] = 1
            R = blk.reshape(4, 4)
            out[16*l:16*l+16] += (-1j * np.sqrt(d2[e] / 2) * (X @ R - R @ X)).ravel()
    return out


curv64 = apply(apply(x0b))[[0, 5, 10, 15]].real @ ZS
report("|curvature of the exact two-state dynamics - 5|", abs(curv64 - 5.0), 1e-12)
Wr = d2 * tauN / (1 + (w * tauN)**2)
B = np.zeros((4, 6))
for e, (a, b) in enumerate(edges):
    B[a, e], B[b, e] = -1, 1
Hm = B @ np.diag(Wr) @ B.T                              # dp/dt = -Hm p
slope_S = -(Hm @ p0) @ ZS
report("|Solomon slope - (-(W2 - W0) m_I(0))|", abs(slope_S - (-(Wr[5] - Wr[4]) * (-1.0))), 1e-13)
print(f"      (sigma = W2 - W0 = {Wr[5] - Wr[4]:+.3f}; W0 : W1 : W2 second moments = 2 : 3 : 12)")

# ---------------------------------------------------------------------- [7]
print("[7] positivity: bound from F, and its failure for a pure state at zero field")
peq4 = np.full(4, 0.25)
tt = np.linspace(0, 20, 4001)
worst_ratio = 0.0
for E_test, tau_test in ((E, tauN), (np.zeros(4), tauN), (np.zeros(4), 5.0)):
    w_test = np.array([E_test[a] - E_test[b] for a, b in edges])
    rhs, _ = transition_current_rhs(edges, w_test, d2, np.full(6, tau_test), peq4)
    for p_init in (np.array([1., 0, 0, 0]), 0.25 - 0.25e-5 * ZI, rng.dirichlet(np.ones(4))):
        sol = solve_ivp(rhs, (0, 20), np.concatenate([p_init, np.zeros(6)]).astype(complex),
                        t_eval=tt, rtol=1e-10, atol=1e-13)
        dev = np.abs(sol.y[:4].real - 0.25).max()
        bound = np.sqrt(((p_init - 0.25)**2 / 0.25).sum() * 0.25)
        worst_ratio = max(worst_ratio, dev / bound)
report("max |p_a - peq_a| / sqrt(2 F(0) peq_a) - 1 (must not be positive)",
       max(0.0, worst_ratio - 1.0), 1e-9)
rhs, _ = transition_current_rhs(edges, np.zeros(6), d2, np.full(6, tauN), peq4)
sol = solve_ivp(rhs, (0, 20), np.concatenate([[1., 0, 0, 0], np.zeros(6)]).astype(complex),
                t_eval=tt, rtol=1e-10, atol=1e-13)
p_min = sol.y[:4].real.min()
report("|minimum population (zero field, from |aa>) - (-0.20)|", abs(p_min + 0.20), 5e-3)
print(f"      (minimum population {p_min:+.3f}; weakly polarized states stay positive)")

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
