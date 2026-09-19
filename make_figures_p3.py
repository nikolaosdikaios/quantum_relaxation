#!/usr/bin/env python3
# make_figures_p3.py -- figures for the transition-current paper. All plotted
# quantities are recomputed here from first principles. Companion scripts:
# paper3_milestones.py, lift_validation_v2.py, positivity_domain.py.
import numpy as np, itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14.5,
    "axes.labelsize": 13.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "mathtext.fontset": "dejavusans"})
CB, CR, CG, CP, CD = "#2E5F8A", "#B5502A", "#4A7A4C", "#6B4F92", "#1E2A38"

# ===================== shared builders ======================================
def cube():
    verts = list(itertools.product((0, 1), repeat=3))
    vidx = {v: i for i, v in enumerate(verts)}
    EDGE = []
    for v in verts:
        for a in range(3):
            if v[a] == 0:
                u = list(v); u[a] = 1; EDGE.append((v, tuple(u), a))
    B1 = np.zeros((8, 12))
    for e, (t, h, a) in enumerate(EDGE):
        B1[vidx[h], e] = 1; B1[vidx[t], e] = -1
    return verts, vidx, EDGE, B1

verts, vidx, EDGE, B1 = cube()
Dsq = np.array([0.7, 1.3, 2.1]); omg = np.array([4., 5., 6.])
Q3 = np.diag([np.sqrt(Dsq[a]) for (_, _, a) in EDGE]) @ B1.T

def lift3(tau, osc):
    Oax = np.array([osc * omg[a] for (_, _, a) in EDGE])
    G = np.zeros((32, 32))
    G[0:8, 8:20] = -Q3.T; G[8:20, 0:8] = Q3
    G[8:20, 8:20] = -np.eye(12) / tau; G[20:32, 20:32] = -np.eye(12) / tau
    G[8:20, 20:32] = np.diag(Oax); G[20:32, 8:20] = -np.diag(Oax)
    return G

# ===================== FIGURE 1: the framework ==============================
# Single summary figure: microscopic inputs, the transition-current equations
# and their parameters, and the two descriptions reached by eliminating or
# retaining the current sector.
fig = plt.figure(figsize=(13.4, 11.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x0, y0, x1, y1, fc, ec, lw=2.2, r=0.010):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))

def arrow(p, q, col="#555555", lw=2.2, rad=0.0, ms=17):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                lw=lw, color=col, connectionstyle=f"arc3,rad={rad}"))

ax.text(0.5, 0.982, "from the microscopic coupling to the measured observables",
        ha="center", fontsize=15, style="italic", color="#333333")

# ---- microscopic inputs -----------------------------------------------------
inputs = [(0.030, "microscopic coupling",
           r"$H_{SB}$: dipolar, CSA," "\n" "quadrupolar, scalar", CP),
          (0.365, "bath correlation function",
           r"$\langle\delta B(t)\,\delta B(0)\rangle\propto$"
           "\n" r"$\Delta\omega_e^{2}\,e^{-|t|/\tau_e}$", CR),
          (0.700, "transition complex",
           "states, allowed transitions,\n" r"frequencies $\omega_e$", CB)]
for x, title, bodytxt, col in inputs:
    box(x, 0.845, x + 0.270, 0.958, "white", col)
    ax.text(x + 0.135, 0.932, title, ha="center", fontsize=13.2,
            color=col, fontweight="bold")
    ax.text(x + 0.135, 0.882, bodytxt, ha="center", fontsize=11.6, color="#222222")
    arrow((x + 0.135, 0.840), (0.5, 0.792), col=col,
          rad=0.0 if abs(x - 0.365) < 1e-9 else (0.10 if x < 0.3 else -0.10))

# ---- the equations ----------------------------------------------------------
box(0.140, 0.655, 0.860, 0.788, CD, CD)
ax.text(0.5, 0.762, "TRANSITION-CURRENT EQUATIONS", ha="center",
        fontsize=11.2, color="#9FB4CC")
ax.text(0.5, 0.712,
        r"$\partial_t u=-\,Q^{\top}\mathrm{Re}\,v\,,\qquad$"
        r"$\partial_t v_e=-\left(\frac{1}{\tau_e}+i\,\omega_e\right)v_e+(Q\,u)_e$",
        ha="center", va="center", fontsize=17.5, color="white")
ax.text(0.5, 0.673,
        r"$Q=\mathrm{diag}(\bar c)^{1/2}\,\mathrm{d}\,e^{f}$,"
        r"   one current per allowed transition, no spectral density input",
        ha="center", fontsize=11.6, color="#D8E2EE")

# ---- microscopic parameters -------------------------------------------------
ax.text(0.5, 0.628, "microscopic parameters (relaxation times are not among them)",
        ha="center", fontsize=12.8, style="italic", color="#333333")
pars = [(r"$X$", "topology", "which transitions\nexist", CB),
        (r"$\omega_e$", "statics", "transition frequencies,\noffsets", CG),
        (r"$\Delta\omega_e^{2}$", "interaction", "coupling second\nmoments", CR),
        (r"$\tau_e$", "bath memory", "correlation time\nper process", CP),
        (r"$\beta$", "bias", "temperature,\ndetailed balance", "#8A6D1F")]
px = np.linspace(0.030, 0.782, 5)
for (sym, role, desc, col), x in zip(pars, px):
    box(x, 0.468, x + 0.188, 0.612, "white", col, lw=2.0)
    ax.text(x + 0.094, 0.578, sym, ha="center", fontsize=19, color=col)
    ax.text(x + 0.094, 0.535, role, ha="center", fontsize=12.6,
            color=col, fontweight="bold")
    ax.text(x + 0.094, 0.495, desc, ha="center", fontsize=10.8, color="#333333")

# ---- control numbers --------------------------------------------------------
box(0.140, 0.375, 0.860, 0.445, "#F8F1DE", "#8A6D1F", lw=2.0)
ax.text(0.5, 0.424, "dimensionless control numbers", ha="center",
        fontsize=11.0, color="#8A6D1F", fontweight="bold")
ax.text(0.5, 0.394,
        r"$\kappa_e=\Delta\omega_e\tau_e$  (Kubo number)$\qquad$"
        r"$\mu_e=\omega_e\tau_e$  (adiabaticity)$\qquad$"
        r"$\kappa_e\ll1$: Markovian corner",
        ha="center", fontsize=13, color="#3B2F0B")

# ---- the fork ---------------------------------------------------------------
ax.text(0.5, 0.344, "the current sector is either eliminated or retained",
        ha="center", fontsize=12.2, style="italic", color="#444444")
arrow((0.42, 0.372), (0.245, 0.300), col="#8A6D1F", rad=0.10)
arrow((0.58, 0.372), (0.755, 0.300), col=CG, rad=-0.10)
ax.text(0.222, 0.322, "adiabatic elimination", ha="center", fontsize=13.4,
        color="#8A6D1F", fontweight="bold")
ax.text(0.778, 0.322, "currents retained", ha="center", fontsize=11.6,
        color=CG, fontweight="bold")

# ---- left branch: Markovian theory ------------------------------------------
box(0.030, 0.158, 0.470, 0.295, "#F8F1DE", "#8A6D1F")
ax.text(0.250, 0.268, "MARKOVIAN DESCRIPTIONS", ha="center", fontsize=11.2,
        color="#8A6D1F", fontweight="bold")
ax.text(0.250, 0.222, "Bloch  $\\cdot$  Redfield  $\\cdot$  BPP\n"
        "Solomon  $\\cdot$  Bloch\u2013Torrey", ha="center", fontsize=12.4,
        color="#3B2F0B")
ax.text(0.250, 0.177,
        r"$w_e=\Delta\omega_e^{2}\,\tau_e/(1+\omega_e^{2}\tau_e^{2})$",
        ha="center", fontsize=12, color="#3B2F0B")
arrow((0.250, 0.154), (0.250, 0.118), col="#8A6D1F")
box(0.030, 0.014, 0.470, 0.112, "white", "#8A6D1F")
ax.text(0.250, 0.082, r"$T_1$, $T_2$, $T_2^{*}$, $D$", ha="center",
        va="center", fontsize=15.5, color="#3B2F0B")
ax.text(0.250, 0.040, "leading term of a $\\kappa^{2}$ expansion",
        ha="center", va="center", fontsize=10.6, color="#6B5A1E",
        style="italic")

# ---- right branch: retained currents ----------------------------------------
box(0.530, 0.158, 0.970, 0.295, "#EFF6EF", CG)
ax.text(0.750, 0.268, "FINITE-MEMORY DYNAMICS", ha="center", fontsize=11.2,
        color=CG, fontweight="bold")
ax.text(0.750, 0.215, "currents evolve with their own\n"
        "memory time and precession", ha="center", fontsize=12.4,
        color="#1F3A20")
ax.text(0.750, 0.177, "equilibrium exact at every memory depth",
        ha="center", fontsize=11.4, color="#1F3A20")
arrow((0.750, 0.154), (0.750, 0.118), col=CG)
box(0.530, 0.014, 0.970, 0.112, "white", CG)
ax.text(0.750, 0.090, "quadratic NOE onset", ha="center", va="center",
        fontsize=11.2, color="#1F3A20")
ax.text(0.750, 0.066, "low-field oscillation", ha="center", va="center",
        fontsize=11.2, color="#1F3A20")
ax.text(0.750, 0.042, "causal transport, rigid second moment", ha="center",
        va="center", fontsize=11.2, color="#1F3A20")
ax.text(0.750, 0.022, "not expressible by any $T_1$, $T_2$, $D$",
        ha="center", va="center", fontsize=10.2, color="#3F6B41",
        style="italic")

fig.savefig("fig_p3_framework.pdf"); fig.savefig("fig_p3_framework.png", dpi=300)
plt.close(fig)

# ===================== FIGURE 2: validation panels ==========================
fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.4))
fig.subplots_adjust(left=0.070, right=0.955, top=0.915, bottom=0.075,
                    wspace=0.26, hspace=0.42)
fig.suptitle("Transition-current relaxation: verified reductions"
             " and predictions", fontsize=16, y=0.966)

# ---- (a) rigid-lattice FID: exact Kubo vs closures vs Markov ---------------
axa = axes[0, 0]
D, tc = 1.0, 5.0
t = np.linspace(0, 8, 800)
kubo = np.exp(-(D * tc) ** 2 * (np.exp(-t / tc) - 1 + t / tc))
nu = np.sqrt(D ** 2 - 1 / (4 * tc ** 2))
pair = np.exp(-t / (2 * tc)) * (np.cos(nu * t) + np.sin(nu * t) / (2 * tc * nu))
mark = np.exp(-D ** 2 * tc * t)
def tier(Nt):
    M = np.zeros((Nt + 1, Nt + 1), complex)
    for n in range(Nt + 1):
        M[n, n] = -n / tc
        if n + 1 <= Nt:
            M[n, n + 1] = M[n + 1, n] = -1j * D * np.sqrt(n + 1)
    w, V = np.linalg.eig(M); c = np.linalg.inv(V)[:, 0]
    return np.real(np.array([(V @ (np.exp(w * tt) * c))[0] for tt in t]))
axa.plot(t, kubo, color="#111111", lw=3.2, label="exact (Kubo)")
axa.plot(t, tier(8), color=CG, lw=2.0, ls=":", label="tier 8")
axa.plot(t, pair, color=CR, lw=2.2, ls="--", label="pair (tier 1)")
axa.plot(t, mark, color=CB, lw=1.8, label="Markov / Bloch")
axa.set_title(r"(a)  rigid lattice ($\kappa=\Delta\omega\,\tau_c=5$):"
              " dephasing", fontsize=14, pad=9)
axa.set_xlabel(r"$t\,\Delta\omega$"); axa.set_ylabel("free-induction decay")
axa.legend(fontsize=11.5, frameon=False, loc="upper right")

# ---- (b) positivity domain heatmap ------------------------------------------
axb = axes[0, 1]
taus = [0.05, 0.2, 0.5, 1.0, 2.0, 4.0]; oscs = [1.0, 0.5, 0.2, 0.08, 0.03]
Zm = np.zeros((len(taus), len(oscs)))
for it, tau in enumerate(taus):
    G = None
    for io, osc in enumerate(oscs):
        G = lift3(tau, osc); w, V = np.linalg.eig(G); Vi = np.linalg.inv(V)
        lam = np.linalg.eigvalsh(Q3.T @ np.diag(tau / (1 + (np.array(
            [osc * omg[a] for (_, _, a) in EDGE]) * tau) ** 2)) @ Q3)
        Tmax = min(30 / lam[lam > 1e-9].min(), 400)
        mn = 0.0
        for tt in np.linspace(Tmax / 300, Tmax, 300):
            M = (V @ np.diag(np.exp(w * tt)) @ Vi).real[0:8, 0:8]
            mn = min(mn, M.min())
        Zm[it, io] = mn
im = axb.imshow(Zm, cmap="RdBu", vmin=-0.45, vmax=0.45, aspect="auto",
                origin="lower")
axb.set_xticks(range(len(oscs)), [str(o) for o in oscs])
axb.set_yticks(range(len(taus)), [str(tq) for tq in taus])
axb.set_xlabel(r"field scale (multiplier of $\omega_e$)  $\longrightarrow$"
               " lower field")
axb.set_ylabel(r"bath memory $\tau_c$  $\longrightarrow$ longer memory")
for it in range(len(taus)):
    for io in range(len(oscs)):
        axb.text(io, it, f"{Zm[it, io]:.2f}", ha="center", va="center",
                 fontsize=10.5,
                 color="white" if Zm[it, io] < -0.2 else "#222222")
# outline the certified (nonnegative) region and label the two zones
axb.plot([-0.5, 4.5, 4.5, -0.5, -0.5], [-0.5, -0.5, 2.5, 2.5, -0.5],
         color="#1B7A3A", lw=2.6)
axb.set_title("(b)  positivity domain of the population closure",
              fontsize=14, pad=9)
cb = fig.colorbar(im, ax=axb, shrink=0.85)
cb.set_label("minimum entry of reduced propagator\n"
             "($0$ = positive for all preparations)", fontsize=10.5)

# ---- (c) NOE transient with a dynamical flip-flop current -------------------
axc = axes[1, 0]
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
ts = np.arange(0.002, 12, 0.002)
mS = np.real(np.array([V @ (np.exp(w * tt) * c) for tt in ts]))[:, 0:4] @ ZS / 2
HN = B4 @ np.diag(Dq4 * tauN / (1 + (Om4 * tauN) ** 2)) @ B4.T
wm, Vm = np.linalg.eigh(HN); ci = Vm.T @ x0[0:4]
mSm = np.array([Vm @ (np.exp(-wm * tt) * ci) for tt in ts]) @ ZS / 2
axc.plot(ts, mSm, color=CB, lw=2.4, label="Markov (Solomon)")
axc.plot(ts, mS, color=CR, lw=2.4, label="transition-current")
axc.set_title(r"(c)  NOE transient, slow flip-flop"
              r" ($\kappa_{\mathrm{ZQ}}=1.5$)", fontsize=14, pad=9)
axc.set_xlabel(r"$t$"); axc.set_ylabel(r"$m_S(t)$ (un-inverted spin)")
axc.legend(fontsize=11.5, frameon=False, loc="lower right",
           borderaxespad=0.6)
axc.annotate("quadratic launch", xy=(0.42, -0.022), xytext=(3.9, -0.075),
             fontsize=11.5, color=CR,
             arrowprops=dict(arrowstyle="->", color=CR, lw=1.6))

# ---- (d) spatial sector: horns and the causal cone --------------------------
axd = axes[1, 1]
Nx, Dsp, tauD = 241, 4.0, 10.0
Bx = np.zeros((Nx, Nx - 1))
for e in range(Nx - 1):
    Bx[e + 1, e] = 1; Bx[e, e] = -1
Qx = np.sqrt(Dsp) * Bx.T
Gx = np.zeros((2 * Nx - 1, 2 * Nx - 1))
Gx[0:Nx, Nx:] = -Qx.T; Gx[Nx:, 0:Nx] = Qx; Gx[Nx:, Nx:] = -np.eye(Nx - 1) / tauD
i0 = Nx // 2
def prop(G, T, dt=0.01):
    x = np.zeros(G.shape[0]); x[i0] = 1.0; t = 0.
    while t < T:
        k1 = G @ x; k2 = G @ (x + dt/2*k1); k3 = G @ (x + dt/2*k2); k4 = G @ (x + dt*k3)
        x += dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt
    return x[0:Nx]
T6 = 6.0
u_l = prop(Gx, T6)
u_m = prop(-np.pad(Bx @ Bx.T * Dsp, ((0, Nx - 1), (0, Nx - 1))), T6)
xg = np.arange(Nx) - i0
axd.semilogy(xg, np.abs(u_m) + 1e-18, color=CB, lw=2.2, label="Markov (Torrey)")
axd.semilogy(xg, np.abs(u_l) + 1e-18, color=CR, lw=2.2, label="transition-current")
cft = np.sqrt(Dsp / tauD) * T6
for s in (+1, -1):
    axd.axvline(s * cft, color="#8A6D1F", ls="--", lw=1.6)
    axd.axvline(s * 2 * np.sqrt(Dsp) * T6, color="#555555", ls=":", lw=1.6)
axd.set_ylim(1e-17, 1.2); axd.set_xlim(-70, 70)
axd.set_title("(d)  transport profile at $t=6$: horns and causal cone",
              fontsize=14, pad=9)
axd.set_xlabel("position (lattice sites)"); axd.set_ylabel(r"$|u(x)|$")
axd.legend(fontsize=11.5, frameon=False, loc="upper right")
# linear-scale inset: the causal cone is obvious without reading a log axis
axins = axd.inset_axes([0.09, 0.10, 0.42, 0.38])
axins.plot(xg, np.abs(u_m), color=CB, lw=1.8)
axins.plot(xg, np.abs(u_l), color=CR, lw=1.8)
for s in (+1, -1):
    axins.axvline(s * cft, color="#8A6D1F", ls="--", lw=1.3)
axins.set_xlim(-26, 26); axins.set_ylim(0, None)
axins.set_title("linear scale", fontsize=9.5, pad=2)
axins.tick_params(labelsize=8)
axins.set_yticks([])

fig.savefig("fig_p3_validation.pdf"); fig.savefig("fig_p3_validation.png", dpi=300)
plt.close(fig)
print('figures written: fig_p3_framework, fig_p3_validation')
