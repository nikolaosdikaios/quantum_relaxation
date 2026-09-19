#!/usr/bin/env python3
# bloch_dictionary_check.py -- the agreement appendix.
#
# QUESTION
#   Do the kinetic primitives numerically give the Bloch constants where
#   Bloch is valid? Extract T1, T2, T2* and D from the kinetic DYNAMICS the
#   way a spectrometer would (exponential fits to inversion recovery, to the
#   free induction with and without static offsets, and the slope of the
#   mean-square displacement), and compare with the closed-form dictionary
#     1/T1  = 2 Dp^2 Jbar(w0),   1/T2 = Dz^2 Jbar(0) + Dp^2 Jbar(w0),
#     1/T2* = 1/T2 + w_s,        D = D_in (long times),
#   at two Kubo numbers. The residuals are the next-order adiabatic
#   corrections and must shrink as kappa^2. Noise-free by design: this is a
#   consistency check of the reduction, not an inference demonstration.
import numpy as np

PASS = lambda ok: "PASS" if ok else "FAIL"
I2 = np.eye(2)
sx = np.array([[0, 1], [1, 0]], complex) / 2
sy = np.array([[0, -1j], [1j, 0]], complex) / 2
sz = np.array([[1, 0], [0, -1]], complex) / 2

def adop(A):
    d = A.shape[0]
    return np.kron(np.eye(d), A) - np.kron(A.T, np.eye(d))

def build_lift(H0, channels, taus, Deltas):
    L0 = adop(H0)
    d2 = L0.shape[0]
    dim = d2 * (1 + len(channels))
    G = np.zeros((dim, dim), complex)
    G[0:d2, 0:d2] = -1j * L0
    for c, (A, tau, De) in enumerate(zip(channels, taus, Deltas)):
        Ac = adop(A); s = d2 * (1 + c)
        G[0:d2, s:s+d2] = -De * Ac
        G[s:s+d2, 0:d2] = +De * Ac
        G[s:s+d2, s:s+d2] = -1j * L0 - np.eye(d2) / tau
    return G

def traj(G, rho0, obs, tgrid):
    d2 = obs[0].size
    x0 = np.zeros(G.shape[0], complex); x0[0:d2] = rho0.reshape(-1)
    w, V = np.linalg.eig(G); c = np.linalg.inv(V) @ x0
    out = []
    for t in tgrid:
        r = (V @ (np.exp(w * t) * c))[0:d2].reshape(rho0.shape)
        out.append([np.real(np.trace(r @ O.conj().T)) for O in obs])
    return np.array(out)

def fit_rate(t, y):
    """log-linear least squares on a positive decay."""
    m = y > 1e-8
    p = np.polyfit(t[m], np.log(y[m]), 1)
    return -p[0]

Jb = lambda w, tau: tau / (1 + (w * tau) ** 2)
w0, tau, ws = 3.0, 0.4, 0.4
rows = []
for kappa in (0.20, 0.05):
    Dp = Dz = kappa / tau
    H0 = w0 * sz
    G = build_lift(H0, [sx, sy, sz], [tau] * 3, [Dp, Dp, Dz])
    R1_B = 2 * Dp**2 * Jb(w0, tau)
    R2_B = Dz**2 * Jb(0, tau) + Dp**2 * Jb(w0, tau)
    # inversion recovery
    t1 = np.linspace(max(5 * tau, 0.02 / R1_B), 3.0 / R1_B, 220)
    mz = traj(G, 2 * sz, [2 * sz], t1)[:, 0]
    R1_K = fit_rate(t1, np.abs(mz))
    # free induction, no offsets -> homogeneous 1/T2
    t2 = np.linspace(max(5 * tau, 0.02 / R2_B), 3.0 / R2_B, 220)
    mxy = traj(G, 2 * sx, [2 * sx, 2 * sy], t2)
    amp = np.sqrt(mxy[:, 0] ** 2 + mxy[:, 1] ** 2)
    R2_K = fit_rate(t2, amp)
    # free induction with Lorentzian offsets -> 1/T2*
    R2s_K = fit_rate(t2, amp * np.exp(-ws * t2))
    rows.append((kappa, R1_B, R1_K, R2_B, R2_K, R2s_K))
    print(f"kappa={kappa:.2f} : 1/T1 kinetic {R1_K:.6f} vs Bloch {R1_B:.6f}"
          f"  rel dev {abs(R1_K/R1_B-1):.2e}")
    print(f"kappa={kappa:.2f} : 1/T2 kinetic {R2_K:.6f} vs Bloch {R2_B:.6f}"
          f"  rel dev {abs(R2_K/R2_B-1):.2e}")
    print(f"kappa={kappa:.2f} : 1/T2* minus 1/T2 = {R2s_K-R2_K:.6f}"
          f" vs w_s {ws:.4f}  dev {abs(R2s_K-R2_K-ws):.1e}")

r_big, r_small = rows[0], rows[1]
dev = lambda r, i, j: abs(r[i] / r[j] - 1)
rat1 = dev(r_big, 2, 1) / dev(r_small, 2, 1)
rat2 = dev(r_big, 4, 3) / dev(r_small, 4, 3)
print(f"scaling of residuals (kappa 0.20 vs 0.05, kappa^2 predicts 16):"
      f"  1/T1 ratio {rat1:.1f}   1/T2 ratio {rat2:.1f}")
ok_small = max(dev(r_small, 2, 1), dev(r_small, 4, 3)) < 1e-2
ok_scale = 6 < rat1 < 40 and 6 < rat2 < 40
print(f"  small-kappa agreement below 1e-2  [{PASS(ok_small)}]"
      f"   kappa^2 scaling  [{PASS(ok_scale)}]")

# ---------------- diffusion: MSD slope of the spatial lift -------------------
Nx, Din, tauD = 201, 1.0, 1.0
Bx = np.zeros((Nx, Nx - 1))
for e in range(Nx - 1):
    Bx[e + 1, e] = 1; Bx[e, e] = -1
Qx = np.sqrt(Din) * Bx.T
Gx = np.zeros((2 * Nx - 1, 2 * Nx - 1))
Gx[0:Nx, Nx:] = -Qx.T; Gx[Nx:, 0:Nx] = Qx
Gx[Nx:, Nx:] = -np.eye(Nx - 1) / tauD
i0 = Nx // 2
x = np.zeros(2 * Nx - 1); x[i0] = 1.0
dt, T = 0.01, 60.0
xs = (np.arange(Nx) - i0).astype(float)
t_rec, msd_rec = [], []
t = 0.0
while t < T:
    k1 = Gx @ x; k2 = Gx @ (x + dt/2*k1); k3 = Gx @ (x + dt/2*k2)
    k4 = Gx @ (x + dt*k3)
    x += dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt
    step = 0.25 if t < 4.0 else 2.0
    if abs(t / step - round(t / step)) < dt / 2:
        u = np.abs(x[0:Nx]); u = u / u.sum()
        t_rec.append(t); msd_rec.append(float((xs**2 * u).sum()))
t_rec, msd_rec = np.array(t_rec), np.array(msd_rec)
late = t_rec > 30
slope = np.polyfit(t_rec[late], msd_rec[late], 1)[0]
D_ext = slope / 2
i_tau = int(np.argmin(np.abs(t_rec - tauD)))
Dapp = msd_rec[i_tau] / (2 * t_rec[i_tau])
Dapp_ref = 1 - (tauD / t_rec[i_tau]) * (1 - np.exp(-t_rec[i_tau] / tauD))
dev_D = abs(D_ext / Din - 1)
print(f"D: long-time MSD slope/2 = {D_ext:.4f} vs input {Din:.4f}"
      f"  rel dev {dev_D:.1e}  [{PASS(dev_D < 1e-2)}]")
print(f"   apparent D(t=tau_D)/D = {Dapp:.3f} vs analytic MSD formula"
      f" {Dapp_ref:.3f}  dev {abs(Dapp-Dapp_ref):.1e}"
      f"  [{PASS(abs(Dapp-Dapp_ref) < 5e-3)}]")
