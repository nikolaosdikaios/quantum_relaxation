#!/usr/bin/env python3
"""
make_figures_jcp.py -- figures and quoted numbers of the JCP manuscript
"Spin relaxation with bath memory beyond the correlation function".

Figure 1  (a) two-spin level diagram with one current per transition
          (b) free-induction decay for one correlation function (kappa = 5)
              and different numbers M of two-state fluctuators
Figure 2  (a) low-field longitudinal recovery of a spin-1/2
          (b) Overhauser transient of a two-spin system

All numbers quoted in the manuscript are printed to stdout. Runtime ~20 s.
Requires numpy, scipy, matplotlib.
"""
import itertools
import numpy as np
import scipy.sparse as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False})

# colour follows the entity in both figures (validated palette, light mode)
C_MARKOV = "#2a78d6"   # Markovian theory (BPP / Solomon)
C_TWO = "#eb6834"      # one two-state fluctuator (M = 1)
C_M2 = "#1baf7a"       # two fluctuators (M = 2)
C_GAUSS = "#2b2b29"    # Gaussian fluctuations (M -> infinity)
INK, MUTED = "#0b0b0b", "#52514e"

zero_crossings = lambda y: int(np.sum(np.diff(np.sign(np.real(y) + 1e-15)) != 0))


# ------------------------------------------------------------------ ladders
def ladder(delta, tau, n_max, M=None):
    """Stochastic Liouville ladder for a coherence modulated by M two-state
    fluctuators (M=None: Gaussian). Couplings delta*b_n, b_n^2 = n(1-(n-1)/M)."""
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


def recovery(field_rms, tau, w0, n_max, M, t):
    """m_z(t)/m_z(0) of a spin-1/2 (Larmor w0) in a transverse field b_x(t) of
    rms amplitude field_rms built from M two-state fluctuators (None: Gaussian)."""
    L = n_max if M is None else min(n_max, M)
    d = 3 * (L + 1)
    A = np.zeros((d, d))
    for n in range(L + 1):
        A[3*n:3*n+3, 3*n:3*n+3] = w0 * RZ - (n / tau) * np.eye(3)
        if n < L:
            b = np.sqrt(n + 1) if M is None else np.sqrt((n + 1) * (1 - n / M))
            A[3*n:3*n+3, 3*(n+1):3*(n+1)+3] = field_rms * b * RX
            A[3*(n+1):3*(n+1)+3, 3*n:3*n+3] = field_rms * b * RX
    return np.real(propagate(A, t, row=2, start=2))


# ================================================================== numbers
print("=" * 72)
print("Free-induction decay, kappa = 5 (Delta_omega = 1, tau_c = 5)")
dw, tc = 1.0, 5.0
t_fid = np.linspace(0, 8, 1601)
kubo = np.exp(-(dw * tc) ** 2 * (np.exp(-t_fid / tc) - 1 + t_fid / tc))
markov = np.exp(-dw ** 2 * tc * t_fid)
fid = {M: np.real(propagate(ladder(dw, tc, 64, M), t_fid)) for M in (1, 2, 4)}
tt = np.linspace(0, 20, 400001)
kb = np.exp(-(dw * tc) ** 2 * (np.exp(-tt / tc) - 1 + tt / tc))
half_ratio = tt[np.argmin(np.abs(kb - 0.5))] / (np.log(2) / (dw ** 2 * tc))
print(f"  Markov reaches half amplitude {half_ratio:.1f} times sooner than Gaussian")
for M in (1, 2, 4):
    print(f"  M = {M}: min F = {fid[M].min():+.3f}, zero crossings = {zero_crossings(fid[M])}")
print(f"  Gaussian: min F = {kubo.min():+.3f}, monotonic = {bool(np.all(np.diff(kubo) <= 1e-12))}")
ts_ = t_fid[t_fid < 0.2]
print("  short-time error vs 1-(dw t)^2/2: two-state "
      f"{np.abs(fid[1][:len(ts_)] - (1 - ts_**2 / 2)).max():.1e}, Markov "
      f"{np.abs(markov[:len(ts_)] - (1 - ts_**2 / 2)).max():.1e}")

print("=" * 72)
print("Short-memory expansion of the slow rate (single channel, no precession)")
for k in (0.1, 0.3, 0.45):
    exact = (1 - np.sqrt(1 - 4 * k * k)) / 2       # in units of 1/tau_c
    series = k * k * (1 + k * k + 2 * k ** 4 + 5 * k ** 6)
    print(f"  kappa = {k:.2f}: rate*tau = {exact:.6f}, BPP {k*k:.6f}, series {series:.6f}")

print("=" * 72)
print("Long-time decay rate of a coherence for M processes, kappa = 0.3 (tau_c = 1)")
k = 0.3
for M in (1, 2, 4, None):
    slow = np.sort(-np.linalg.eigvals(ladder(k, 1.0, 80, M)).real)[0]
    lab = "Gaussian" if M is None else f"M = {M}"
    formula = k * k if M is None else M * (1 - np.sqrt(1 - 4 * k * k / M)) / 2
    print(f"  {lab:>9}: rate*tau = {slow:.6f}, closed form {formula:.6f}, "
          f"rate theory is {int(round(100 * (1 - k * k / slow)))}% too low")
print(f"  Eq. (5) (exact for M = 1) overestimates the Gaussian rate by "
      f"{100 * ((1 - np.sqrt(1 - 4 * k * k)) / 2 / (k * k) - 1):.0f}%")

print("=" * 72)
print("Low-field longitudinal recovery: Delta_omega_e^2 = 0.7 (field rms^2 = 1.4),")
print("tau_c = 2, omega_L = 0.32 -> kappa = %.2f, omega_L tau_c = %.2f"
      % (np.sqrt(0.7) * 2.0, 0.32 * 2.0))
b_rms, tau_r, w0 = np.sqrt(1.4), 2.0, 0.32
t_rec = np.linspace(0, 25, 5001)
rec = {M: recovery(b_rms, tau_r, w0, 48, M, t_rec) for M in (1, 2, 3, 4, 8, 16, None)}
print(f"  rms fluctuating field / static field = {b_rms / w0:.1f}")
r_bpp = b_rms ** 2 * tau_r / (1 + (w0 * tau_r) ** 2)
bpp = np.exp(-r_bpp * t_rec)
for M in (1, 2, 3, 4, 8, 16, None):
    lab = "Gaussian" if M is None else f"M = {M}"
    print(f"  {lab:>9}: zero crossings = {zero_crossings(rec[M])}, min = {rec[M].min():+.4f}")
print(f"  Gaussian hierarchy: max|level 16 - level 48| = "
      f"{np.abs(recovery(b_rms, tau_r, w0, 16, None, t_rec) - rec[None]).max():.1e}")
for w_test, lab in ((0.0, "zero static field"), (b_rms, "static field = rms fluctuating field")):
    print(f"  M = 1, {lab}: min = {recovery(b_rms, tau_r, w_test, 48, 1, t_rec).min():+.3f}")
gauss = rec[None]
print(f"  Gaussian recovery monotonic: {bool(np.all(np.diff(gauss) <= 1e-12))}")
rates = np.linspace(0.1, 2.0, 1901)
r_best = rates[int(np.argmin([np.sum((gauss - np.exp(-r * t_rec)) ** 2) for r in rates]))]
print(f"  BPP rate {r_bpp:.3f}: max deviation from Gaussian {np.abs(gauss - bpp).max():.2f}")
print(f"  best single exponential (rate {r_best:.3f}): max deviation "
      f"{np.abs(gauss - np.exp(-r_best * t_rec)).max():.2f}")
print(f"  two-state (= transition-current equations) vs Gaussian: max deviation "
      f"{np.abs(rec[1] - gauss).max():.2f}")

print("=" * 72)
print("Overhauser transient, two spins, one fluctuator per transition, tau_c = 1.5")
E = np.array([0., 5.6, 5.0, 10.6])              # |aa>, |ab>, |ba>, |bb> (I, S)
# (state, state, Delta_omega_e^2): SQ(I) x2, SQ(S) x2, ZQ, DQ. The second moments
# have the dipolar ratios ZQ : SQ : DQ = 2 : 3 : 12, so that the Markovian limit
# is Solomon's dipolar equations; Delta_omega_ZQ = 1 sets the frequency unit.
EDGES = [(0, 2, 1.5), (1, 3, 1.5), (0, 1, 1.5), (2, 3, 1.5), (1, 2, 1.0), (0, 3, 6.0)]
tauN, I4 = 1.5, np.eye(4)
ZI = np.array([1., 1, -1, -1]); ZS = np.array([1., -1, 1, -1])
L0d = np.array([-1j * (E[a] - E[b]) for a in range(4) for b in range(4)])
A_ops, amp = [], []
for (i, j, d2) in EDGES:
    X = np.zeros((4, 4)); X[i, j] = X[j, i] = 1
    A_ops.append(sps.coo_matrix(-1j * (np.kron(X, I4) - np.kron(I4, X.T))))
    amp.append(np.sqrt(d2 / 2))                  # |V_ab|^2 = Delta_omega_e^2 / 2


def build(n_tot, cap):
    """Joint ladder over six fluctuators: occupation n_e <= cap, sum <= n_tot.
    cap = 1 with n_tot = 6 is the exact dynamics for one two-state fluctuator per
    transition; cap = n_tot = N is the Gaussian hierarchy truncated at tier N."""
    idx = [n for n in itertools.product(range(min(n_tot, cap) + 1), repeat=6)
           if sum(n) <= n_tot]
    pos = {n: k for k, n in enumerate(idx)}
    rows, cols, vals = [np.arange(16 * len(idx))], [np.arange(16 * len(idx))], \
        [np.concatenate([L0d - sum(n) / tauN for n in idx])]
    for k, n in enumerate(idx):
        for e in range(6):
            m = list(n); m[e] += 1; m = tuple(m)
            if m in pos:
                f = amp[e] * np.sqrt(n[e] + 1); A = A_ops[e]; l = pos[m]
                rows += [16 * k + A.row, 16 * l + A.row]
                cols += [16 * l + A.col, 16 * k + A.col]
                vals += [f * A.data, f * A.data]
    return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(16 * len(idx),) * 2)


p0 = 0.25 - 0.25 * ZI                            # spin I fully inverted: m_I(0) = -1
dt, T = 0.004, 12.0
t_noe = np.arange(0, T + dt / 2, dt)


def run(G):
    x = np.zeros(G.shape[0], complex); x[[0, 5, 10, 15]] = p0
    out = []
    for _ in t_noe:
        out.append(np.real(x[[0, 5, 10, 15]]) @ ZS)
        k1 = G @ x; k2 = G @ (x + dt/2*k1); k3 = G @ (x + dt/2*k2); k4 = G @ (x + dt*k3)
        x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(out)


B = np.zeros((4, 6)); Om = np.zeros(6); D2 = np.zeros(6)
for e, (i, j, d2) in enumerate(EDGES):
    B[j, e] = 1; B[i, e] = -1; Om[e] = abs(E[j] - E[i]); D2[e] = d2
w_bpp = D2 * tauN / (1 + (Om * tauN) ** 2)
H = B @ np.diag(w_bpp) @ B.T
wm, Vm = np.linalg.eigh(H)
solomon = np.array([Vm @ (np.exp(-wm * s) * (Vm.T @ (p0 - 0.25))) for s in t_noe]) @ ZS
pair = run(build(1, 1))
two_state = run(build(6, 1))
G8 = build(8, 8)
g6, g8 = run(build(6, 6)), run(G8)
print(f"  Gaussian hierarchy at total level 8: {G8.shape[0]} variables")
sigma = w_bpp[5] - w_bpp[4]
print(f"  kappa_ZQ = {np.sqrt(D2[4]) * tauN:.2f}, omega_ZQ tau_c = {Om[4]*tauN:.2f}, "
      f"omega_DQ tau_c = {Om[5]*tauN:.1f}")
print(f"  Solomon cross-relaxation sigma = w_DQ - w_ZQ = {sigma:+.3f} (initial slope of m_S "
      f"= -sigma m_I(0) = {sigma:+.3f})")
G1 = build(1, 1); x0 = np.zeros(G1.shape[0], complex); x0[[0, 5, 10, 15]] = p0
print(f"  initial slope of m_S (transition-current eqs) = "
      f"{np.real((G1 @ x0)[[0, 5, 10, 15]]) @ ZS:+.2f}")
print(f"  initial curvature -(D2_DQ - D2_ZQ) m_I(0) = {D2[5] - D2[4]:+.2f}; from the generator "
      f"{np.real((G1 @ (G1 @ x0))[[0, 5, 10, 15]]) @ ZS:+.2f}")
ref = solomon.min()
for lab, y in (("Solomon", solomon), ("transition-current eqs", pair),
               ("two-state, exact", two_state), ("Gaussian, tier 8", g8)):
    i = int(np.argmin(y))
    print(f"  {lab:<24} extremum {y[i]:+.3f} at t/tau_c = {t_noe[i]/tauN:.2f} "
          f"(magnitude {100 * (abs(y[i]) - abs(ref)) / abs(ref):+.0f}% vs Solomon)")
print(f"  Gaussian convergence max|tier 8 - tier 6| = {np.abs(g8 - g6).max():.1e}")
print(f"  transition-current eqs vs exact two-state: max deviation {np.abs(pair - two_state).max():.3f}")
for lab, y in (("two-state, exact", two_state), ("Gaussian, tier 8", g8)):
    i_pos = int(np.argmax(y[:400]))
    i_sc = np.where(np.diff(np.sign(y[1:])) != 0)[0][0] + 1
    print(f"  early positive excursion of m_S ({lab}): max {y[i_pos]:+.3f} at t/tau_c = "
          f"{t_noe[i_pos]/tauN:.2f}; sign change at t/tau_c = {t_noe[i_sc]/tauN:.2f}")

# ================================================================== figure 1
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.75),
                                 gridspec_kw={"width_ratios": [1.0, 1.25]})
fig.subplots_adjust(left=0.02, right=0.985, bottom=0.17, top=0.93, wspace=0.22)

# (a) level diagram
ax_a.set_xlim(0, 1); ax_a.set_ylim(-0.22, 1.1); ax_a.axis("off")
lev = {r"$|\alpha\alpha\rangle$": (0.5, 0.0), r"$|\alpha\beta\rangle$": (0.18, 0.53),
       r"$|\beta\alpha\rangle$": (0.82, 0.47), r"$|\beta\beta\rangle$": (0.5, 1.0)}
# |ab> above |ba> because omega_S (5.6) > omega_I (5.0) in the Overhauser example
pts = list(lev.values())
aa, ab, ba, bb = pts


def link(p, q, label, lx, ly, ls="-"):
    ax_a.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=1.1, ls=ls, zorder=1)
    ax_a.text(lx, ly, label, ha="center", va="center", fontsize=8.5, color=INK,
              bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3)


link(aa, ab, r"$v_{1}^{S}$", 0.26, 0.20)
link(aa, ba, r"$v_{1}^{I}$", 0.74, 0.22)
link(ab, bb, r"$v_{1}^{I}$", 0.27, 0.78)
link(ba, bb, r"$v_{1}^{S}$", 0.73, 0.80)
link(ab, ba, r"$v_{0}$", 0.40, 0.555, ls="--")
link(aa, bb, r"$v_{2}$", 0.535, 0.30, ls=":")
for name, (x, y) in lev.items():
    ax_a.plot([x - 0.085, x + 0.085], [y, y], color=INK, lw=2.2, zorder=4,
              solid_capstyle="butt")
    below = y < 0.01
    ax_a.text(x, y - 0.045 if below else y + 0.055, name, ha="center",
              va="top" if below else "bottom", fontsize=8.5, color=INK, zorder=5)
ax_a.text(0.5, -0.2, "one damped, precessing current $v_e$ per transition",
          ha="center", va="center", fontsize=7.5, color=MUTED)

# (b) free-induction decay
ax_b.axhline(0, color="#cfcdc7", lw=0.6, zorder=0)
ax_b.plot(t_fid, markov, color=C_MARKOV, lw=1.3, label="Markovian (Redfield)")
ax_b.plot(t_fid, fid[1], color=C_TWO, lw=1.5, ls="--", label="$M=1$ (two-state)")
ax_b.plot(t_fid, fid[2], color=C_M2, lw=1.5, ls="-.", label="$M=2$")
ax_b.plot(t_fid, kubo, color=C_GAUSS, lw=1.6, label=r"Gaussian ($M\to\infty$)")
ax_b.set_xlim(0, 8); ax_b.set_ylim(-0.95, 1.05)
ax_b.set_xlabel(r"time $\Delta\omega\, t$")
ax_b.set_ylabel("free-induction decay")
ax_b.text(0.97, 0.97, r"same correlation function, $\kappa=\Delta\omega\,\tau_c=5$",
          transform=ax_b.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
ax_b.legend(frameon=False, loc="lower right", handlelength=2.4, borderaxespad=0.1)
fig.text(0.01, 0.97, "(a)", fontsize=9, fontweight="bold", va="top")
fig.text(0.40, 0.97, "(b)", fontsize=9, fontweight="bold", va="top")
fig.savefig("fig1_jcp.pdf"); fig.savefig("fig1_jcp.png", dpi=300)
plt.close(fig)

# ================================================================== figure 2
fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(7.0, 2.75))
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.93, wspace=0.3)

x_rec = t_rec / tau_r
ax_c.axhline(0, color="#cfcdc7", lw=0.6, zorder=0)
ax_c.plot(x_rec, bpp, color=C_MARKOV, lw=1.3, label="Markovian (BPP)")
ax_c.plot(x_rec, rec[1], color=C_TWO, lw=1.5, ls="--", label="$M=1$ (two-state)")
ax_c.plot(x_rec, rec[2], color=C_M2, lw=1.5, ls="-.", label="$M=2$")
ax_c.plot(x_rec, gauss, color=C_GAUSS, lw=1.6, label=r"Gaussian ($M\to\infty$)")
ax_c.set_xlim(0, 8); ax_c.set_ylim(-0.5, 1.05)
ax_c.set_xlabel(r"time $t/\tau_c$")
ax_c.set_ylabel(r"$m_z(t)/m_z(0)$")
ax_c.text(0.97, 0.97, r"$\kappa=1.7$, $\omega_L\tau_c=0.64$", transform=ax_c.transAxes,
          ha="right", va="top", fontsize=7.5, color=MUTED)
ax_c.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9), handlelength=2.4)

x_noe = t_noe / tauN
ax_d.axhline(0, color="#cfcdc7", lw=0.6, zorder=0)
ax_d.plot(x_noe, solomon, color=C_MARKOV, lw=1.3, label="Markovian (Solomon)")
ax_d.plot(x_noe, two_state, color=C_TWO, lw=1.5, ls="--", label="$M=1$ (two-state)")
ax_d.plot(x_noe, g8, color=C_GAUSS, lw=1.6, label=r"Gaussian ($M\to\infty$)")
ax_d.set_xlim(0, 8); ax_d.set_ylim(-0.5, 0.2)
ax_d.set_xlabel(r"time $t/\tau_c$")
ax_d.set_ylabel(r"$m_S(t)\,/\,|m_I(0)|$")
ax_d.legend(frameon=False, loc="lower right", handlelength=2.4)
ins = ax_d.inset_axes([0.5, 0.7, 0.47, 0.26])
ins.axhline(0, color="#cfcdc7", lw=0.6)
ins.plot(x_noe, solomon, color=C_MARKOV, lw=1.1)
ins.plot(x_noe, two_state, color=C_TWO, lw=1.2, ls="--")
ins.plot(x_noe, g8, color=C_GAUSS, lw=1.2)
ins.set_xlim(0, 0.4); ins.set_ylim(-0.1, 0.075)
ins.set_yticks([-0.1, -0.05, 0, 0.05])
ins.set_title("onset", fontsize=7, pad=1.5, color=MUTED)
ins.tick_params(labelsize=6.5, length=2)
for s in ("top", "right"):
    ins.spines[s].set_visible(True)
fig.text(0.01, 0.97, "(a)", fontsize=9, fontweight="bold", va="top")
fig.text(0.515, 0.97, "(b)", fontsize=9, fontweight="bold", va="top")
fig.savefig("fig2_jcp.pdf"); fig.savefig("fig2_jcp.png", dpi=300)
plt.close(fig)
print("=" * 72)
print("figures written: fig1_jcp.pdf/.png, fig2_jcp.pdf/.png")
