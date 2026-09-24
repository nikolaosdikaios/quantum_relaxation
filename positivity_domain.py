#!/usr/bin/env python3

import numpy as np, itertools
N=3
verts=list(itertools.product((0,1),repeat=N)); vidx={v:i for i,v in enumerate(verts)}
EDGE=[]
for v in verts:
    for a in range(N):
        if v[a]==0:
            u_=list(v); u_[a]=1; EDGE.append((v,tuple(u_),a))
B1=np.zeros((8,12))
for e,(t_,h_,a) in enumerate(EDGE): B1[vidx[h_],e]=+1; B1[vidx[t_],e]=-1
Dsq=np.array([0.7,1.3,2.1]); omg=np.array([4.,5.,6.])
Dax=np.array([np.sqrt(Dsq[a]) for (_,_,a) in EDGE]); Q=np.diag(Dax)@B1.T

def lift(tau,osc):
    Oax=np.array([osc*omg[a] for (_,_,a) in EDGE])
    G=np.zeros((32,32))
    G[0:8,8:20]=-Q.T; G[8:20,0:8]=Q
    G[8:20,8:20]=-np.eye(12)/tau; G[20:32,20:32]=-np.eye(12)/tau
    G[8:20,20:32]=np.diag(Oax); G[20:32,8:20]=-np.diag(Oax)
    return G

def min_entry(tau,osc,nt=400):
    G=lift(tau,osc); w,V=np.linalg.eig(G); Vi=np.linalg.inv(V)
    lam=np.linalg.eigvalsh(Q.T@np.diag(tau/(1+(np.array([osc*omg[a] for (_,_,a) in EDGE])*tau)**2))@Q)
    lam_slow=lam[lam>1e-9].min()
    Tmax=min(30.0/lam_slow, 400.0)
    mn=0.0; targ=None
    for t in np.linspace(Tmax/nt,Tmax,nt):
        M=(V@np.diag(np.exp(w*t))@Vi).real[0:8,0:8]
        m=M.min()
        if m<mn: mn=m; targ=t
    return mn,targ

taus=[0.05,0.2,0.5,1.0,2.0,4.0]; oscs=[1.0,0.5,0.2,0.08,0.03]
print("min entry of reduced propagator M(t) over all t (negative = positivity violation)")
print("rows: tau_c ; cols: field scale (omega multiplier); kappa_max = sqrt(2.1)*tau")
hdr="tau\\osc" + "".join(f"{o:>10}" for o in oscs) + "   kappa_max"
print(hdr)
worst=(0,None,None)
for tau in taus:
    row=f"{tau:>7}"
    for osc in oscs:
        mn,tt=min_entry(tau,osc)
        row+=f"{mn:>10.5f}"
        if mn<worst[0]: worst=(mn,tau,osc)
    row+=f"{np.sqrt(2.1)*tau:>10.2f}"
    print(row)
print(f"\nworst violation: min entry = {worst[0]:.5f} at tau={worst[1]}, osc={worst[2]}")
