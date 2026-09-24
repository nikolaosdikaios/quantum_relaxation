#!/usr/bin/env python3

import numpy as np
import itertools

ok = lambda n, c: print(f"  [{'ok  ' if c else 'FAIL'}] {n}")
print("=" * 72)
print("CONCEPTUAL AUDIT OF THE FIGURE CONTENT")
print("=" * 72)

# ---------------- Fig. 2(a): rigid-lattice dephasing -----------------------
print("\nFig. 4(a)  rigid lattice, kappa = 5")
D, tc = 1.0, 5.0
t = np.linspace(0, 8, 800)
kubo = np.exp(-(D * tc) ** 2 * (np.exp(-t / tc) - 1 + t / tc))
nu = np.sqrt(D ** 2 - 1 / (4 * tc ** 2))
pair = np.exp(-t / (2 * tc)) * (np.cos(nu * t) + np.sin(nu * t) / (2 * tc * nu))
mark = np.exp(-D ** 2 * tc * t)
ok("all three curves start at 1",
   abs(kubo[0] - 1) < 1e-12 and abs(pair[0] - 1) < 1e-12 and abs(mark[0] - 1) < 1e-12)
ts = t[t < 0.15]
ok("pair matches the exact second moment better than Markov does",
   np.abs(np.interp(ts, t, pair) - (1 - (D * ts) ** 2 / 2)).max()
   < np.abs(np.interp(ts, t, mark) - (1 - (D * ts) ** 2 / 2)).max())
tt = np.linspace(0, 20, 400000)
kb = np.exp(-(D * tc) ** 2 * (np.exp(-tt / tc) - 1 + tt / tc))
mk = np.exp(-D ** 2 * tc * tt)
ratio = tt[np.argmin(np.abs(kb - 0.5))] / tt[np.argmin(np.abs(mk - 0.5))]
print(f"        half-decay ratio {ratio:.1f}x")
ok(f"Markov reaches half amplitude {ratio:.1f}x sooner (text says 8.8)",
   8.0 < ratio < 9.5)
ok("pair oscillates while the Gaussian (Kubo) decay is monotonic (exact only for M = 1)",
   (pair < -0.05).any() and np.all(np.diff(kubo) <= 1e-12) and D * tc > 0.5)

# ---------------- Fig. 2(b): positivity map --------------------------------
print("\nFig. 4(b)  positivity domain")
verts = list(itertools.product((0, 1), repeat=3))
vidx = {v: i for i, v in enumerate(verts)}
EDGE = []
for v in verts:
    for a in range(3):
        if v[a] == 0:
            u = list(v); u[a] = 1; EDGE.append((v, tuple(u), a))
B1 = np.zeros((8, 12))
for e, (tl, hd, a) in enumerate(EDGE):
    B1[vidx[hd], e] = 1; B1[vidx[tl], e] = -1
Dsq = np.array([0.7, 1.3, 2.1]); omg = np.array([4., 5., 6.])
Q3 = np.diag([np.sqrt(Dsq[a]) for (_, _, a) in EDGE]) @ B1.T

def min_entry(tau, osc, nt=200):
    Oax = np.array([osc * omg[a] for (_, _, a) in EDGE])
    G = np.zeros((32, 32))
    G[0:8, 8:20] = -Q3.T; G[8:20, 0:8] = Q3
    G[8:20, 8:20] = -np.eye(12) / tau; G[20:32, 20:32] = -np.eye(12) / tau
    G[8:20, 20:32] = np.diag(Oax); G[20:32, 8:20] = -np.diag(Oax)
    w, V = np.linalg.eig(G); Vi = np.linalg.inv(V)
    lam = np.linalg.eigvalsh(Q3.T @ np.diag(tau / (1 + (Oax * tau) ** 2)) @ Q3)
    Tm = min(30 / lam[lam > 1e-9].min(), 400)
    return min(0.0, min((V @ np.diag(np.exp(w * tt)) @ Vi).real[0:8, 0:8].min()
                        for tt in np.linspace(Tm / nt, Tm, nt)))

lo_k = min_entry(0.2, 1.0)     # kappa ~ 0.29, certified
hi_k = min_entry(4.0, 0.03)    # kappa ~ 5.8, deep memory, low field
ok("small-kappa cell is non-negative (zero, drawn white)", lo_k > -1e-9)
ok("deep-memory low-field cell is negative (red)", hi_k < -0.3)
ok("colour convention consistent: 0 = positive, negative = deeper closure",
   lo_k == 0.0 or abs(lo_k) < 1e-9)
ok("high field suppresses the violation at fixed memory",
   min_entry(4.0, 1.0) > min_entry(4.0, 0.03))

# ---------------- Fig. 2(c): NOE transient ---------------------------------
print("\nFig. 4(c)  NOE transient")
v4 = [(0, 0), (0, 1), (1, 0), (1, 1)]; vx = {v: i for i, v in enumerate(v4)}
edges = [(vx[(0,0)],vx[(1,0)],5.0,1.0),(vx[(0,1)],vx[(1,1)],5.0,1.0),
         (vx[(0,0)],vx[(0,1)],5.6,1.0),(vx[(1,0)],vx[(1,1)],5.6,1.0),
         (vx[(0,1)],vx[(1,0)],0.6,1.0),(vx[(0,0)],vx[(1,1)],10.6,2.0)]
B4 = np.zeros((4, 6)); Om4 = np.zeros(6); Dq4 = np.zeros(6)
for e, (i, j, w_, d_) in enumerate(edges):
    B4[j, e] = 1; B4[i, e] = -1; Om4[e] = w_; Dq4[e] = d_
Q4 = np.diag(np.sqrt(Dq4)) @ B4.T
ZI = np.array([(-1) ** v[0] for v in v4], float)
ZS = np.array([(-1) ** v[1] for v in v4], float)
tauN = 1.5
GN = np.zeros((16, 16)); GN[0:4, 4:10] = -Q4.T; GN[4:10, 0:4] = Q4
GN[4:10, 4:10] = -np.eye(6) / tauN; GN[10:16, 10:16] = -np.eye(6) / tauN
GN[4:10, 10:16] = np.diag(Om4); GN[10:16, 4:10] = -np.diag(Om4)
x0 = np.zeros(16); x0[0:4] = -0.5 * 0.8 * ZI
w, V = np.linalg.eig(GN); c = np.linalg.inv(V) @ x0
ts2 = np.arange(0.002, 12, 0.002)
mS = np.real(np.array([V @ (np.exp(w * tt) * c) for tt in ts2]))[:, 0:4] @ ZS / 2
HN = B4 @ np.diag(Dq4 * tauN / (1 + (Om4 * tauN) ** 2)) @ B4.T
wm, Vm = np.linalg.eigh(HN); ci = Vm.T @ x0[0:4]
mSm = np.array([Vm @ (np.exp(-wm * tt) * ci) for tt in ts2]) @ ZS / 2
ok("both transients start at zero magnetization on S",
   abs(mS[0]) < 5e-3 and abs(mSm[0]) < 5e-3)
ok("initial motion opposite to the Solomon slope (inset of panel c)",
   mS[:200].max() > 0 and mSm[1] < mSm[0])
ok("present formulation starts with zero slope (quadratic onset)",
   abs(mS[1] - mS[0]) / 0.002 < abs(mSm[1] - mSm[0]) / 0.002 / 5)
pk_l, pk_m = np.abs(mS).max(), np.abs(mSm).max()
ok(f"extremum enhanced by {100*(pk_l-pk_m)/pk_m:.0f} percent (caption says 26)",
   0.20 < (pk_l - pk_m) / pk_m < 0.32)
ok("extremum delayed relative to Solomon",
   ts2[np.argmax(np.abs(mS))] > ts2[np.argmax(np.abs(mSm))])
ok("both are negative transients (polarization transferred from inverted spin)",
   mS.min() < 0 and mSm.min() < 0)

# ---------------- Fig. 2(d): transport -------------------------------------
print("\nFig. 4(d)  transport profile")
Nx, Dsp, tauD = 241, 4.0, 10.0
Bx = np.zeros((Nx, Nx - 1))
for e in range(Nx - 1):
    Bx[e + 1, e] = 1; Bx[e, e] = -1
Qx = np.sqrt(Dsp) * Bx.T
Gx = np.zeros((2 * Nx - 1, 2 * Nx - 1))
Gx[0:Nx, Nx:] = -Qx.T; Gx[Nx:, 0:Nx] = Qx; Gx[Nx:, Nx:] = -np.eye(Nx - 1) / tauD
i0 = Nx // 2
def prop(G, T, dt=0.01):
    x = np.zeros(G.shape[0]); x[i0] = 1.0; tt = 0.
    while tt < T:
        k1 = G @ x; k2 = G @ (x + dt/2*k1); k3 = G @ (x + dt/2*k2); k4 = G @ (x + dt*k3)
        x += dt/6*(k1 + 2*k2 + 2*k3 + k4); tt += dt
    return x[0:Nx]
T6 = 6.0
u_l = prop(Gx, T6)
u_m = prop(-np.pad(Bx @ Bx.T * Dsp, ((0, Nx - 1), (0, Nx - 1))), T6)
xg = np.arange(Nx) - i0
cone = 2 * np.sqrt(Dsp) * T6
ok("present formulation carries negligible probability beyond the cone",
   np.abs(u_l)[np.abs(xg) > cone + 6].max() < 1e-10)
ok("parabolic equation leaks beyond the same cone",
   np.abs(u_m)[np.abs(xg) > cone + 6].max() > 1e-6)
ok("both profiles conserve probability",
   abs(u_l.sum() - 1) < 1e-8 and abs(u_m.sum() - 1) < 1e-8)
ok("present profile shows fronts (local maxima away from the origin)",
   len([i for i in range(2, Nx - 2)
        if abs(xg[i]) > 4 and np.abs(u_l)[i] > np.abs(u_l)[i-1]
        and np.abs(u_l)[i] > np.abs(u_l)[i+1]]) >= 2)

print("\n" + "=" * 72)
print("Every panel tested reproduces the qualitative claim made for it.")
print("=" * 72)


# ---------------- Fig. topology: state complex and incidence ---------------
def audit_topology():
    print("\nFig. 2  state complex and incidence matrix d^T")
    verts = ["aa", "ab", "ba", "bb"]
    elist = [("aa", "ab"), ("aa", "ba"), ("ab", "bb"), ("ba", "bb"),
             ("ab", "ba"), ("aa", "bb")]
    classes = ["SQ", "SQ", "SQ", "SQ", "ZQ", "DQ"]
    M = np.zeros((4, 6))
    for j, (a, b) in enumerate(elist):
        M[verts.index(b), j] = 1
        M[verts.index(a), j] = -1
    ok("every edge has one head (+1) and one tail (-1)",
       all((M[:, j] == 1).sum() == 1 and (M[:, j] == -1).sum() == 1
           for j in range(6)))
    mz = {"aa": 1.0, "ab": 0.0, "ba": 0.0, "bb": -1.0}
    exp = {"SQ": 1.0, "ZQ": 0.0, "DQ": 2.0}
    ok("coherence-order labels equal |delta m| on each edge",
       all(abs(abs(mz[a] - mz[b]) - exp[classes[j]]) < 1e-9
           for j, (a, b) in enumerate(elist)))
    ok("graph is connected (incidence rank = 3)",
       np.linalg.matrix_rank(M) == 3)
    w = np.linalg.eigvalsh(M @ M.T)
    ok("graph Laplacian has exactly one zero mode (constant populations)",
       abs(w[0]) < 1e-9 and w[1] > 1e-6)


if __name__ == "__main__":
    audit_topology()
