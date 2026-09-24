#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14,
                     "axes.labelsize": 13.5, "xtick.labelsize": 12,
                     "ytick.labelsize": 12,
                     "mathtext.fontset": "dejavusans"})

# ---------- exact truth: SLE tower, tier 24 ---------------------------------
Rz = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
Rx = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])

def tower_z(Delta, tau, w0, tgrid, N=24):
    d = 3 * (N + 1); A = np.zeros((d, d))
    for n in range(N + 1):
        A[3*n:3*n+3, 3*n:3*n+3] = w0 * Rz - (n / tau) * np.eye(3)
        if n + 1 <= N:
            A[3*n:3*n+3, 3*(n+1):3*(n+1)+3] = Delta * np.sqrt(n + 1) * Rx
            A[3*(n+1):3*(n+1)+3, 3*n:3*n+3] = Delta * np.sqrt(n + 1) * Rx
    w, V = np.linalg.eig(A); c = np.linalg.inv(V) @ np.eye(d)[:, 2]
    # vectorized over tgrid: row 2 of V @ (exp(w t) c) for all t at once
    return np.real((V[2, :] * c) @ np.exp(np.outer(w, tgrid)))

# ---------- kinetic model at closure depth N (same 2 physical parameters) ----
def pair_z(Delta, tau, w0, tgrid, N=3):
    return tower_z(Delta, tau, w0, tgrid, N=N)

# ---------- synthetic data ----------------------------------------------------
D_true, tau_true, sigma = 1.0, 2.0, 0.01
fields = [0.10, 0.25, 0.50, 1.00, 2.00]
tgrid = np.linspace(0.05, 20.0, 400)
data = {w0: tower_z(D_true, tau_true, w0, tgrid)
        + sigma * rng.standard_normal(len(tgrid)) for w0 in fields}

# ---------- model A: per-field Bloch exponential ------------------------------
def fit_bloch(y):
    Rgrid = np.logspace(-2.6, 0.7, 400)
    rss = [np.sum((y - np.exp(-R * tgrid)) ** 2) for R in Rgrid]
    i = int(np.argmin(rss))
    return Rgrid[i], rss[i]

bloch = {w0: fit_bloch(data[w0]) for w0 in fields}
rss_B = sum(v[1] for v in bloch.values())

# ---------- model C: global Markovian (BPP) fit, same two parameters --------
# m(t) = exp(-Delta^2 J(w0) t), J(w) = tau/(1 + w^2 tau^2)
def bpp_curve(D_, T_, w0):
    return np.exp(-D_**2 * T_ / (1 + (w0 * T_)**2) * tgrid)

def bpp_rss(D_, T_):
    return sum(np.sum((data[w0] - bpp_curve(D_, T_, w0))**2) for w0 in fields)

def fit_bpp():
    best = (None, None, np.inf)
    for D_ in np.linspace(0.3, 1.6, 131):
        for T_ in np.linspace(0.2, 4.0, 191):
            r = bpp_rss(D_, T_)
            if r < best[2]:
                best = (D_, T_, r)
    for _ in range(2):
        D0, T0, _r = best
        for D_ in np.linspace(D0 - 0.02, D0 + 0.02, 41):
            for T_ in np.linspace(T0 - 0.04, T0 + 0.04, 41):
                r = bpp_rss(D_, T_)
                if r < best[2]:
                    best = (D_, T_, r)
    return best

D_bpp, tau_bpp, rss_C = fit_bpp()

# ---------- model B: one global (Delta, tau) ---------------------------------
def global_rss(Delta, tau, N):
    return sum(np.sum((data[w0] - pair_z(Delta, tau, w0, tgrid, N)) ** 2)
               for w0 in fields)

def fit_depth(N):
    best = (None, None, np.inf)
    Dg = np.linspace(0.4, 1.6, 33); Tg = np.linspace(0.8, 3.5, 33)
    for D_ in Dg:
        for T_ in Tg:
            r = global_rss(D_, T_, N)
            if r < best[2]:
                best = (D_, T_, r)
    for _ in range(2):
        D0, T0, _ = best
        for D_ in np.linspace(D0 - 0.06, D0 + 0.06, 21):
            for T_ in np.linspace(T0 - 0.15, T0 + 0.15, 21):
                r = global_rss(D_, T_, N)
                if r < best[2]:
                    best = (D_, T_, r)
    return best

fit1 = fit_depth(1)
fit3 = fit_depth(3)
fit6 = fit_depth(6)
print("  convergence of the recovered physical parameters with closure depth:")
for N_, f_ in ((1, fit1), (3, fit3), (6, fit6)):
    print(f"    depth {N_}:  Delta = {f_[0]:.3f}   tau = {f_[1]:.3f}")
D_hat, tau_hat, rss_K = fit6

# ---------- metrics -----------------------------------------------------------
n = len(fields) * len(tgrid)
rms_B, rms_K = np.sqrt(rss_B / n), np.sqrt(rss_K / n)
aic_B = n * np.log(rss_B / n) + 2 * 5
aic_K = n * np.log(rss_K / n) + 2 * 2
rms_C = np.sqrt(rss_C / n)
aic_C = n * np.log(rss_C / n) + 2 * 2
print("FIELD-SERIES FIT  (truth: Delta = 1.000, tau = 2.000, kappa = 2)")
print(f"  headline (depth-6): Delta = {D_hat:.3f}  tau = {tau_hat:.3f}")
print(f"  per-field Bloch rates 1/T1:",
      [round(bloch[w0][0], 4) for w0 in fields],
      " (spread x%.1f)" % (bloch[fields[0]][0] / bloch[fields[-1]][0]))
print(f"  pooled RMS: Bloch {rms_B:.4f}  kinetic {rms_K:.4f}")
print(f"  RMS ratio Bloch/kinetic = {rms_B / rms_K:.2f}")
print(f"  Delta AIC (Bloch - kinetic) = {aic_B - aic_K:+.0f}"
      "   (positive prefers kinetic despite 5 vs 2 parameters)")
print(f"  global BPP (Markov, 2 parameters): Delta = {D_bpp:.3f}  tau = {tau_bpp:.3f}"
      f"  pooled RMS {rms_C:.4f}")
print(f"  Delta AIC (BPP - kinetic) = {aic_C - aic_K:+.0f}   (same number of parameters)")

# ---------- figure ------------------------------------------------------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.2, 5.4))
fig.subplots_adjust(left=0.065, right=0.98, top=0.86, bottom=0.135,
                    wspace=0.24)
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(fields)))
for w0, col in zip(fields, cols):
    axa.plot(tgrid, data[w0], color=col, lw=1.1, alpha=0.5)
    axa.plot(tgrid, pair_z(D_hat, tau_hat, w0, tgrid, 6), color=col, lw=2.4,
             ls="--", label=rf"$\omega_0={w0}$")
    axa.plot(tgrid, np.exp(-bloch[w0][0] * tgrid), color=col, lw=1.0,
             ls=":", alpha=0.9)
axa.set_title("(a)  five fields: global two-parameter fit (dashed,\n"
              "hierarchy at depth 6) vs per-field Bloch (dotted)", fontsize=13.5, pad=8)
axa.set_xlabel(r"$t\,\Delta\omega$"); axa.set_ylabel("longitudinal recovery")
axa.legend(fontsize=10.5, frameon=False, ncol=2, loc="upper right",
           borderaxespad=0.7, columnspacing=1.1)
# inset: residual vs time, both models, lowest field (where Bloch fails most)
axins = axa.inset_axes([0.575, 0.40, 0.40, 0.34])
w_lo = fields[0]
axins.axhline(0, color="#999999", lw=0.8)
axins.plot(tgrid, data[w_lo] - pair_z(D_hat, tau_hat, w_lo, tgrid, 6),
           color="#B5502A", lw=1.4, label="global fit, depth 6")
axins.plot(tgrid, data[w_lo] - np.exp(-bloch[w_lo][0] * tgrid),
           color="#2E5F8A", lw=1.4, label="Bloch")
axins.set_title(rf"residual, $\omega_0={w_lo}$", fontsize=9.5, pad=2)
axins.tick_params(labelsize=8); axins.set_xticks([0, 10, 20])
axins.legend(fontsize=8, frameon=False, loc="upper right")

x = np.arange(len(fields)); wdt = 0.27
rmsB_f = [np.sqrt(bloch[w0][1] / len(tgrid)) for w0 in fields]
rmsK_f = [np.sqrt(np.sum((data[w0] - pair_z(D_hat, tau_hat, w0, tgrid, 6)) ** 2)
          / len(tgrid)) for w0 in fields]
rmsC_f = [np.sqrt(np.sum((data[w0] - bpp_curve(D_bpp, tau_bpp, w0)) ** 2)
          / len(tgrid)) for w0 in fields]
axb.bar(x - wdt, rmsB_f, wdt, color="#2E5F8A",
        label="Bloch, one $T_1$ per field")
axb.bar(x, rmsC_f, wdt, color="#C9A13B",
        label=r"BPP (Markovian), global $(\Delta\omega,\tau_c)$")
axb.bar(x + wdt, rmsK_f, wdt, color="#B5502A",
        label=r"hierarchy (depth 6), global $(\Delta\omega,\tau_c)$")
axb.axhline(sigma, color="#8A6D1F", ls="--", lw=1.6,
            label="noise floor")
axb.set_xticks(x, [str(w0) for w0 in fields])
axb.set_xlabel(r"field $\omega_0$ (units of $\Delta\omega$)")
axb.set_ylabel("per-field residual RMS")
axb.set_title("(b)  per-field residuals; hierarchy fit favoured by\n"
              rf"$\Delta$AIC $= {aic_B - aic_K:+.0f}$ over Bloch, ${aic_C - aic_K:+.0f}$ over BPP",
              fontsize=13.5, pad=8)
axb.set_ylim(0, 0.068)
axb.legend(fontsize=10.5, frameon=False, ncol=2, loc="upper center",
           columnspacing=1.2)
fig.savefig("fig_p3_fit.pdf"); fig.savefig("fig_p3_fit.png", dpi=300)
print("figure written: fig_p3_fit")
