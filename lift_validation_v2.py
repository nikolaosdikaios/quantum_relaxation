#!/usr/bin/env python3

import numpy as np, itertools
PASS = lambda ok: "PASS" if ok else "FAIL"
N = 3
verts = list(itertools.product((0,1), repeat=N)); vidx = {v:i for i,v in enumerate(verts)}
EDGE = []
for v in verts:
    for a in range(N):
        if v[a]==0:
            u_=list(v); u_[a]=1; EDGE.append((v,tuple(u_),a))
B1 = np.zeros((8,12))
for e,(t_,h_,a) in enumerate(EDGE): B1[vidx[h_],e]=+1; B1[vidx[t_],e]=-1
Dsq = np.array([0.7,1.3,2.1]); omg = np.array([4.,5.,6.])
Dax = np.array([np.sqrt(Dsq[a]) for (_,_,a) in EDGE])
Q   = np.diag(Dax) @ B1.T

def lift(tau_e, osc=1.0):
    Oax = np.array([osc*omg[a] for (_,_,a) in EDGE])
    G = np.zeros((32,32))
    G[0:8,8:20]=-Q.T; G[8:20,0:8]=Q
    G[8:20,8:20]=-np.diag(1/tau_e); G[20:32,20:32]=-np.diag(1/tau_e)
    G[8:20,20:32]=np.diag(Oax); G[20:32,8:20]=-np.diag(Oax)
    return G, Oax

# per-edge memory times, distinct per channel
rng = np.random.default_rng(3)
tau_e = 0.5 + 1.5*rng.random(12)
G, Oax = lift(tau_e)
ev = np.linalg.eigvals(G)
vac = np.zeros(32); vac[0:8]=1/np.sqrt(8)
E = np.diag([1.]*8 + [-1.]*12 + [1.]*12)
Heff = Q.T @ np.diag(tau_e/(1+(Oax*tau_e)**2)) @ Q
w_pred = np.array([Dsq[a]*tau_e[e]/(1+(omg[a]*tau_e[e])**2) for e,(_,_,a) in enumerate(EDGE)])
Hbpp = B1 @ np.diag(w_pred) @ B1.T
print("PER-EDGE MEMORY TIMES tau_e (distinct per channel):")
print("  stability  max Re eig = %+.1e  [%s]" % (ev.real.max(), PASS(ev.real.max()<1e-10)))
print("  vacuum     ||G(e^-f,0)|| = %.1e  [%s]" % (np.linalg.norm(G@vac), PASS(np.linalg.norm(G@vac)<1e-13)))
print("  parity DB  ||EGE - G^T|| = %.1e  [%s]" % (np.abs(E@G@E-G.T).max(), PASS(np.abs(E@G@E-G.T).max()<1e-13)))
print("  per-edge BPP emergence ||Heff-Hbpp|| = %.1e  [%s]" % (np.abs(Heff-Hbpp).max(), PASS(np.abs(Heff-Hbpp).max()<1e-12)))

# positivity audit in the ringing regime (slow bath, low field, strong polarization)
def rk4(G,x0,T,dt):
    x=x0.copy(); out=[x.copy()]; t=0.
    while t<T:
        k1=G@x; k2=G@(x+dt/2*k1); k3=G@(x+dt/2*k2); k4=G@(x+dt*k3)
        x=x+dt/6*(k1+2*k2+2*k3+k4); t+=dt; out.append(x.copy())
    return np.array(out)
for osc,tau0,m,label in ((1.0,0.02,0.9,"fast bath, high field"),
                          (0.08,2.0,0.9,"slow bath, low field (ringing)")):
    Gl,_ = lift(np.full(12,tau0), osc)
    z1 = np.array([(-1)**v[0] for v in verts],float)
    p0 = (1 + m*z1)/8.0                      # valid initial distribution
    x0 = np.zeros(32); x0[0:8] = p0*np.sqrt(8)
    tr = rk4(Gl,x0, 30.0 if osc<1 else 150.0, 0.002)
    p_t = tr[:,0:8]/np.sqrt(8)
    print("POSITIVITY (%s): min_i,t p_i(t) = %+.4f ; total prob drift = %.1e"
          % (label, p_t.min(), np.abs(p_t.sum(axis=1)-1).max()))
