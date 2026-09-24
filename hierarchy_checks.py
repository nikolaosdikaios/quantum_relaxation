#!/usr/bin/env python3

import itertools, numpy as np, scipy.sparse as sps
PASS = lambda ok: "PASS" if ok else "FAIL"
zc = lambda y: int(np.sum(np.diff(np.sign(np.real(y) + 1e-15)) != 0))

def ladder(Delta, tau, N, M=None):
    L = N if M is None else min(N, M)
    A = np.zeros((L + 1, L + 1), complex)
    for n in range(L + 1):
        A[n, n] = -n / tau
        if n < L:
            c = np.sqrt(n + 1) if M is None else np.sqrt((n + 1) * (1 - n / M))
            A[n, n + 1] = A[n + 1, n] = -1j * Delta * c
    return A

def evolve0(A, t, obs_row=0, start=0):
    w, V = np.linalg.eig(A); c = np.linalg.solve(V, np.eye(len(A))[:, start])
    return (V[obs_row, :] * c) @ np.exp(np.outer(w, t))

def telegraph_fid(Delta, tau, M, t):
    k = 1 / (2 * tau); S = list(itertools.product((-1, 1), repeat=M))
    pos = {s: a for a, s in enumerate(S)}; n = len(S); W = np.zeros((n, n))
    for a, s in enumerate(S):
        for i in range(M):
            s2 = list(s); s2[i] = -s2[i]; W[pos[tuple(s2)], a] += k; W[a, a] -= k
    xi = Delta / np.sqrt(M) * np.array([sum(s) for s in S])
    w, V = np.linalg.eig(W - 1j * np.diag(xi)); c = np.linalg.solve(V, np.full(n, 1 / n))
    return (V.sum(axis=0) * c) @ np.exp(np.outer(w, t))

print("=" * 76); print("[H1] M telegraph fluctuators <-> Krawtchouk ladder, terminating at tier M")
t = np.linspace(0, 10, 1001)
for M in (1, 2, 3, 5):
    d = np.abs(evolve0(ladder(1.0, 5.0, 40, M), t) - telegraph_fid(1.0, 5.0, M, t)).max()
    print(f"  M = {M}: max|ladder - exact joint dynamics| = {d:.1e}  [{PASS(d < 1e-10)}]")
print(f"  M = 1 ladder is the pair (tier-1 Gaussian ladder): "
      f"{np.abs(ladder(1,5,1,1) - ladder(1,5,1)).max():.1e}")

print("=" * 76); print("[H2] rigid lattice, kappa = 5: free-induction decay for M fluctuators")
t = np.linspace(0, 20, 4001)
kubo = np.exp(-25 * (np.exp(-t / 5) - 1 + t / 5))
for M in (1, 2, 4, 16, None):
    F = kubo if M is None else np.real(evolve0(ladder(1.0, 5.0, 64, M), t))
    lab = "Gaussian (Kubo)" if M is None else f"M = {M:>2}"
    print(f"  {lab:>16}: min F = {F.min():+.3f}, zero crossings on [0, 20] = {zc(F)}")

print("=" * 76); print("[H3] low-field recovery (kappa = 1.67, mu = 0.64) for M fluctuators")
Rz = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]]); Rx = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])
def sle_z(Delta, tau, w0, N, M, t):
    L = N if M is None else min(N, M); d = 3 * (L + 1); A = np.zeros((d, d))
    for n in range(L + 1):
        A[3*n:3*n+3, 3*n:3*n+3] = w0 * Rz - (n / tau) * np.eye(3)
        if n < L:
            c = np.sqrt(n + 1) if M is None else np.sqrt((n + 1) * (1 - n / M))
            A[3*n:3*n+3, 3*(n+1):3*(n+1)+3] = A[3*(n+1):3*(n+1)+3, 3*n:3*n+3] = Delta * c * Rx
    return np.real(evolve0(A, t, obs_row=2, start=2))
t = np.linspace(0, 25, 5001)
for M in (1, 2, 3, 4, 8, 16, None):
    m = sle_z(np.sqrt(1.4), 2.0, 0.32, 48, M, t)
    lab = "Gaussian" if M is None else f"M = {M:>2}"
    print(f"  {lab:>9}: zero crossings = {zc(m)}, min m_z = {m.min():+.3f}")

print("=" * 76); print("[H4] two-spin Overhauser transient, kappa_ZQ = 1.5, beta -> 0")
E = np.array([0., 5.6, 5.0, 10.6])              # states (00),(01),(10),(11)
EDGES = [(0, 2, 1.0), (1, 3, 1.0), (0, 1, 1.0), (2, 3, 1.0), (1, 2, 1.0), (0, 3, 2.0)]
tauN, I4 = 1.5, np.eye(4)
L0d = np.array([-1j * (E[a] - E[b]) for a in range(4) for b in range(4)])
Acoo, sig = [], []
for (i, j, dq) in EDGES:
    X = np.zeros((4, 4)); X[i, j] = X[j, i] = 1
    Acoo.append(sps.coo_matrix(-1j * (np.kron(X, I4) - np.kron(I4, X.T)))); sig.append(np.sqrt(dq / 2))
def indices(N, cap):
    return [n for n in itertools.product(range(min(N, cap) + 1), repeat=6) if sum(n) <= N]
def build(N, cap):
    idx = indices(N, cap); pos = {n: k for k, n in enumerate(idx)}
    r, c, v = [np.arange(16 * len(idx))], [np.arange(16 * len(idx))], \
        [np.concatenate([L0d - sum(n) / tauN for n in idx])]
    for k, n in enumerate(idx):
        for e in range(6):
            m = list(n); m[e] += 1; m = tuple(m)
            if m in pos:
                f = sig[e] * np.sqrt(n[e] + 1); A = Acoo[e]; l = pos[m]
                r += [16 * k + A.row, 16 * l + A.row]; c += [16 * l + A.col, 16 * k + A.col]
                v += [f * A.data, f * A.data]
    G = sps.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                       shape=(16 * len(idx),) * 2)
    return G, len(idx)
ZI = np.array([1., 1, -1, -1]); ZS = np.array([1., -1, 1, -1]); p0 = 0.25 - 0.2 * ZI
dt, T = 0.004, 12.0; ts = np.arange(0, T + dt / 2, dt)
def run(G):
    x = np.zeros(G.shape[0], complex); x[[0, 5, 10, 15]] = p0; out = []
    for _ in ts:
        out.append(np.real(x[[0, 5, 10, 15]]) @ ZS)
        k1 = G @ x; k2 = G @ (x + dt/2*k1); k3 = G @ (x + dt/2*k2); k4 = G @ (x + dt*k3)
        x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    x0 = np.zeros(G.shape[0], complex); x0[[0, 5, 10, 15]] = p0
    curv = np.real((G @ (G @ x0))[[0, 5, 10, 15]]) @ ZS
    return np.array(out), curv
# Solomon (Markov) and the pair, as in paper3_milestones.py
B = np.zeros((4, 6)); Om = np.zeros(6); Dq = np.zeros(6)
for e, (i, j, dq) in enumerate(EDGES): B[j, e] = 1; B[i, e] = -1; Om[e] = abs(E[j] - E[i]); Dq[e] = dq
H = B @ np.diag(Dq * tauN / (1 + (Om * tauN) ** 2)) @ B.T
wm, Vm = np.linalg.eigh(H); mSol = np.array([Vm @ (np.exp(-wm * tt) * (Vm.T @ (-0.4 * ZI))) for tt in ts]) @ ZS / 2
def summary(lab, m, curv=None):
    i = int(np.argmin(m)); later = m[i:]; ext = int(np.sum(np.diff(np.sign(np.diff(later))) != 0))
    enh = 100 * (abs(m[i]) - abs(mSol.min())) / abs(mSol.min())
    extra = "" if curv is None else f", curvature(0) = {curv:+.3f}"
    print(f"  {lab:<34} extremum {m[i]:+.4f} at t = {ts[i]:.2f} ({enh:+.0f}% vs Solomon),"
          f" later turning points {ext}{extra}")
    return m
summary("Solomon (Markov)", mSol)
res = {}
for N in (1, 2, 3, 4, 6, 8):
    G, nb = build(N, N); res[N], curv = run(G)
    summary(f"Gaussian hierarchy, tier {N} ({16*nb} dim)", res[N], curv)
print(f"  convergence max|tier 8 - tier 6| = {np.abs(res[8] - res[6]).max():.1e}")
G, nb = build(6, 1); mdich, curv = run(G)
summary("dichotomous, exact (64 blocks)", mdich, curv)
np.savez("noe_hierarchy.npz", t=ts, solomon=mSol, pair=res[1], gauss=res[8], dich=mdich)
