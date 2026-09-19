#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14,
                     "axes.labelsize": 13.5, "xtick.labelsize": 12,
                     "ytick.labelsize": 12,
                     "mathtext.fontset": "dejavusans"})
Rz = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
Rx = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])

# ---------------- two-noise tower: exact generator ---------------------------
def gen2(Dx, Dz, tau, w0, N):
    T = N + 1
    d = 3 * T * T
    A = np.zeros((d, d))
    idx = lambda n, m: 3 * (n * T + m)
    for n in range(T):
        for m in range(T):
            i = idx(n, m)
            A[i:i+3, i:i+3] = w0 * Rz - ((n + m) / tau) * np.eye(3)
            if n + 1 < T:
                j = idx(n + 1, m)
                A[i:i+3, j:j+3] += Dx * np.sqrt(n + 1) * Rx
                A[j:j+3, i:i+3] += Dx * np.sqrt(n + 1) * Rx
            if m + 1 < T:
                j = idx(n, m + 1)
                A[i:i+3, j:j+3] += Dz * np.sqrt(m + 1) * Rz
                A[j:j+3, i:i+3] += Dz * np.sqrt(m + 1) * Rz
    return A

def battery_curves(Dx, Dz, tau, w_s, w0, t_ir, t_fid, te, N):
    """Return IR M_z(t), FID |M_xy|(t) (with static factor), echo E(2 te)."""
    A = gen2(Dx, Dz, tau, w0, N)
    w, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    d = A.shape[0]
    # IR: start inverted, z-channel only interacts with Dx; full tower fine
    x0 = np.zeros(d); x0[2] = 1.0
    c = Vi @ x0
    ir = np.real((V[2, :] * c) @ np.exp(np.outer(w, t_ir)))   # vectorized over t
    # FID: start on x (components 0, 1 give M_x, M_y at every time at once)
    x0 = np.zeros(d); x0[0] = 1.0
    c = Vi @ x0
    Ef = np.exp(np.outer(w, t_fid))
    fid = np.abs((V[0, :] * c) @ Ef + 1j * ((V[1, :] * c) @ Ef))
    fid = fid * np.exp(-w_s * t_fid)          # Lorentzian static offsets
    # Hahn echo: evolve te, pi_x pulse (Mx, -My, -Mz) on every tier, evolve te
    P = np.kron(np.eye((N + 1) ** 2), np.diag([1., -1., -1.]))
    ech = []
    for t_ in te:
        U = np.real(V @ np.diag(np.exp(w * t_)) @ Vi)
        x1 = U @ (P @ (U @ x0))
        ech.append(abs(x1[0] + 1j * x1[1]))   # offsets refocus exactly
    return ir, fid, np.array(ech)

# ---------------- synthetic truth ---------------------------------------------
Dx_t, Dz_t, tau_t, ws_t = 0.8, 1.0, 1.2, 0.5
fields = [1.5, 4.0]
grids = {1.5: (np.linspace(0.1, 12, 160), np.linspace(0.05, 4, 140),
               np.linspace(0.15, 1.5, 10)),
         4.0: (np.linspace(0.2, 40, 160), np.linspace(0.05, 4, 140),
               np.linspace(0.15, 1.5, 10))}
truth, data = {}, {}
for w0 in fields:
    t_ir, t_fid, te = grids[w0]
    ir, fid, ech = battery_curves(Dx_t, Dz_t, tau_t, ws_t, w0,
                                  t_ir, t_fid, te, N=8)
    truth[w0] = (ir, fid, ech)
    data[w0] = (ir + 0.01 * rng.standard_normal(len(ir)),
                fid + 0.01 * rng.standard_normal(len(fid)),
                ech + 0.01 * rng.standard_normal(len(ech)))

# ---------------- Model A: Bloch, one constant per curve ----------------------
def fit_rate(t, y, model):
    Rg = np.logspace(-2.5, 1.0, 500)
    rss = [np.sum((y - model(R, t)) ** 2) for R in Rg]
    i = int(np.argmin(rss))
    return Rg[i], rss[i]

bloch = {}
for w0 in fields:
    t_ir, t_fid, te = grids[w0]
    R1, r1 = fit_rate(t_ir, data[w0][0], lambda R, t: np.exp(-R * t))
    R2s, r2 = fit_rate(t_fid, data[w0][1], lambda R, t: np.exp(-R * t))
    bloch[w0] = dict(R1=R1, R2star=R2s, rss=r1 + r2)
print("BATTERY FIT  (truth: D_perp=0.8, D_z=1.0, tau=1.2, w_s=0.5)")
print("  Bloch per-field constants (4 fitted): "
      + "  ".join(f"w0={w0}: 1/T1={bloch[w0]['R1']:.3f}, "
                  f"1/T2*={bloch[w0]['R2star']:.3f}" for w0 in fields))

# ---------------- Model B: kinetic, 4 global primitives from IR+FID only ------
NM = 6    # model closure depth
def kin_rss(p):
    Dx, Dz, tau, ws = p
    rss = 0.0
    for w0 in fields:
        t_ir, t_fid, _ = grids[w0]
        ir, fid, _ = battery_curves(Dx, Dz, tau, ws, w0, t_ir, t_fid,
                                    np.array([0.3]), N=NM)
        rss += np.sum((data[w0][0] - ir) ** 2)
        rss += np.sum((data[w0][1] - fid) ** 2)
    return rss

p = np.array([0.7, 1.1, 1.0, 0.4])
rng_scan = [(0.4, 1.2), (0.6, 1.4), (0.7, 1.8), (0.2, 0.8)]
best = kin_rss(p)
for sweep in range(3):
    for k in range(4):
        lo, hi = rng_scan[k]
        for val in np.linspace(lo, hi, 17):
            q = p.copy(); q[k] = val
            r = kin_rss(q)
            if r < best:
                best, p = r, q
    rng_scan = [(max(l, p[k] - 0.12), min(h, p[k] + 0.12))
                for k, (l, h) in enumerate(rng_scan)]
Dx_f, Dz_f, tau_f, ws_f = p
print(f"  kinetic global primitives (4 fitted, IR+FID only): "
      f"D_perp={Dx_f:.3f}  D_z={Dz_f:.3f}  tau={tau_f:.3f}  w_s={ws_f:.3f}")

# ---------------- held-out echo prediction -------------------------------------
rms_pred, rms_h1, rms_h2 = [], [], []
for w0 in fields:
    t_ir, t_fid, te = grids[w0]
    _, _, ech_pred = battery_curves(Dx_f, Dz_f, tau_f, ws_f, w0,
                                    t_ir[:2], t_fid[:2], te, N=NM)
    y = data[w0][2]
    rms_pred.append(np.sqrt(np.mean((y - ech_pred) ** 2)))
    h1 = np.exp(-bloch[w0]["R2star"] * 2 * te)          # T2 = T2*
    h2 = np.exp(-0.5 * bloch[w0]["R1"] * 2 * te)        # T2 = 2 T1
    rms_h1.append(np.sqrt(np.mean((y - h1) ** 2)))
    rms_h2.append(np.sqrt(np.mean((y - h2) ** 2)))
rms_pred, rms_h1, rms_h2 = map(lambda a: float(np.mean(a)),
                               (rms_pred, rms_h1, rms_h2))
print(f"  held-out echo RMS: kinetic prediction {rms_pred:.4f}"
      f"  vs Bloch heuristics T2=T2*: {rms_h1:.4f}  T2=2T1: {rms_h2:.4f}")
print(f"  echo heuristic/prediction ratios: {rms_h1/rms_pred:.1f}"
      f" and {rms_h2/rms_pred:.1f}")
print(f"  constants for the full protocol: Bloch 6 (all fitted, no relations)"
      f"  vs kinetic 4 (echoes predicted)")

# ---------------- diffusion: persistent vs Torrey ------------------------------
Dsp, tauD = 1.0, 1.0
msd = lambda t: 2 * Dsp * (t - tauD * (1 - np.exp(-t / tauD)))
tD = np.array([0.5, 1.0, 2.0, 4.0]) * tauD
ratio = msd(tD) / (2 * Dsp * tD)
print("  diffusion (second cumulant): MSD/(2Dt) at t/tau_D = 0.5,1,2,4 :",
      np.round(ratio, 3), " -> Torrey overestimates attenuation at short times")

# ---------------- figure --------------------------------------------------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.4, 5.5))
fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.13, wspace=0.24)
cols = {1.5: "#2E5F8A", 4.0: "#B5502A"}
for w0 in fields:
    t_ir, t_fid, te = grids[w0]
    ir_f, fid_f, _ = battery_curves(Dx_f, Dz_f, tau_f, ws_f, w0, t_ir, t_fid,
                                    np.array([0.3]), N=NM)
    axa.plot(t_fid, data[w0][1], color=cols[w0], lw=1.0, alpha=0.5)
    axa.plot(t_fid, fid_f, color=cols[w0], lw=2.4, ls="--",
             label=rf"FID, $\omega_0={w0}$")
    axa.plot(t_ir * 0.32, data[w0][0], color=cols[w0], lw=1.0, alpha=0.25)
    axa.plot(t_ir * 0.32, ir_f, color=cols[w0], lw=1.8, ls=":",
             label=rf"IR (compressed), $\omega_0={w0}$")
axa.set_title("(a)  fitted set: IR and FID, two fields,"
              " four global primitives", fontsize=14, pad=9)
axa.set_xlabel(r"$t\,\Delta\omega$  (IR axis compressed $\times0.32$)")
axa.set_ylabel("signal"); axa.legend(fontsize=10.5, frameon=False)
for w0 in fields:
    te = grids[w0][2]
    _, _, ech_pred = battery_curves(Dx_f, Dz_f, tau_f, ws_f, w0,
                                    grids[w0][0][:2], grids[w0][1][:2], te, N=NM)
    axb.plot(2 * te, data[w0][2], "o", ms=7, color=cols[w0],
             label=rf"echo data, $\omega_0={w0}$")
    axb.plot(2 * te, ech_pred, color=cols[w0], lw=2.6)
    axb.plot(2 * te, np.exp(-bloch[w0]["R2star"] * 2 * te), color=cols[w0],
             lw=1.4, ls="--", alpha=0.7)
    axb.plot(2 * te, np.exp(-0.5 * bloch[w0]["R1"] * 2 * te), color=cols[w0],
             lw=1.4, ls=":", alpha=0.7)
axb.set_title("(b)  held out: echoes predicted with zero new parameters",
              fontsize=14, pad=9)
axb.set_xlabel(r"echo time $2\tau_e\,\Delta\omega$")
axb.set_ylabel("echo amplitude")
axb.legend(fontsize=10.5, frameon=False, loc="lower left",
           borderaxespad=0.7)
fig.savefig("fig_p3_protocol.pdf"); fig.savefig("fig_p3_protocol.png", dpi=300)
print("figure written: fig_p3_protocol")
