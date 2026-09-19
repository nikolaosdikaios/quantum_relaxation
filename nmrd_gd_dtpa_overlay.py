#!/usr/bin/env python3

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- hand-digitized Gd-DTPA (4 mM) points from the pasted figure ------------
# f in MHz (proton Larmor), r1 in s^-1 mM^-1
f_MHz = np.array([
    0.010, 0.015, 0.022, 0.033, 0.050, 0.080, 0.130, 0.20, 0.30, 0.50,
    0.70, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 13.0, 20.0,
    24.0, 30.0, 40.0, 55.0, 65.0, 130.0, 300.0, 400.0, 600.0, 800.0])
r1 = np.array([
    6.40, 6.42, 6.45, 6.42, 6.48, 6.42, 6.40, 6.48, 6.50, 6.52,
    6.58, 6.60, 6.68, 6.70, 6.60, 6.42, 6.30, 5.80, 4.80, 4.00,
    3.80, 3.70, 3.60, 3.60, 3.60, 3.30, 3.00, 2.95, 2.92, 2.90])
# high-field anchors (55-800 MHz) set to the values reported in the paper:
# Gd-DTPA r1 = 3.6, 3.3, 3.0 s^-1 mM^-1 at 1.5, 3, 7 T (63.9, 127.7, 298 MHz).

f = f_MHz * 1e6  # Hz

def lor1(f, c0, a1, tau1):
    return c0 + a1 / (1.0 + (2*np.pi*f*tau1)**2)

def lor2(f, c0, a1, tau1, a2, tau2):
    return (c0 + a1 / (1.0 + (2*np.pi*f*tau1)**2)
               + a2 / (1.0 + (2*np.pi*f*tau2)**2))

# ---- single-channel fit -----------------------------------------------------
p1, _ = curve_fit(lor1, f, r1, p0=[3.0, 3.5, 12e-9],
                  bounds=([0, 0, 1e-11], [10, 20, 1e-6]), maxfev=20000)
res1 = r1 - lor1(f, *p1)
rms1 = np.sqrt(np.mean(res1**2))

# ---- two-channel fit --------------------------------------------------------
p2, _ = curve_fit(lor2, f, r1, p0=[2.6, 3.0, 12e-9, 1.0, 1.0e-9],
                  bounds=([0, 0, 1e-10, 0, 1e-11], [10, 20, 1e-6, 20, 1e-7]),
                  maxfev=40000)
res2 = r1 - lor2(f, *p2)
rms2 = np.sqrt(np.mean(res2**2))

print("=== single channel (3 params: c0, a1, tau1) ===")
print(f"  c0   = {p1[0]:.2f} s^-1 mM^-1")
print(f"  a1   = {p1[1]:.2f} s^-1 mM^-1   (amplitude ~ Delta_omega^2)")
print(f"  tau1 = {p1[2]*1e9:.2f} ns       (effective correlation time)")
print(f"  RMS residual = {rms1:.3f} s^-1 mM^-1  over {len(f)} fields")
print("=== two channels (5 params) ===")
print(f"  c0   = {p2[0]:.2f}")
print(f"  a1,tau1 = {p2[1]:.2f}, {p2[2]*1e9:.2f} ns")
print(f"  a2,tau2 = {p2[3]:.2f}, {p2[4]*1e9:.2f} ns")
print(f"  RMS residual = {rms2:.3f} s^-1 mM^-1  over {len(f)} fields")
print(f"  (a per-field description would use {len(f)} independent rates.)")

# ---- overlay figure ---------------------------------------------------------
fg = np.logspace(np.log10(0.008), np.log10(1000), 600) * 1e6
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.errorbar(f_MHz, r1, yerr=0.15, fmt='o', ms=6, mfc='white',
            mec='#1E2A38', ecolor='#8899AA', capsize=2,
            label='Gd-DTPA 4 mM (digitized, PLOS ONE 2016)', zorder=3)
ax.plot(fg/1e6, lor1(fg, *p1), '-', color='#B5502A', lw=2.4,
        label=(fr'transition-current fit: one channel, $\tau_c={p1[2]*1e9:.0f}$ ns'
               '\n(3 parameters, RMS '
               fr'${rms1:.2f}$ over {len(f)} fields)'))
ax.set_xscale('log')
ax.set_xlabel('Proton Larmor frequency (MHz)')
ax.set_ylabel(r'Relaxivity $r_1$ (s$^{-1}$mM$^{-1}$)')
ax.set_title('Transition-current spectral density vs a measured NMRD profile')
ax.set_xlim(0.008, 1000)
ax.set_ylim(2, 7.5)
ax.legend(frameon=False, fontsize=9.5, loc='upper right')
ax.grid(True, which='both', alpha=0.15)
fig.tight_layout()
fig.savefig('nmrd_gd_dtpa_overlay.pdf')
fig.savefig('nmrd_gd_dtpa_overlay.png', dpi=200)
print("\nwritten: nmrd_gd_dtpa_overlay.pdf / .png")
