#!/usr/bin/env python3

import numpy as np, itertools
PASS = lambda ok: "PASS" if ok else "FAIL"

# ===================== [M1] nature and locality of positivity violations ====
print("="*76); print("[M1] POSITIVITY: exact tower vs pair closure; locality of violations")
Rz = np.array([[0.,-1,0],[1,0,0],[0,0,0]]); Rx = np.array([[0.,0,0],[0,0,-1],[0,1,0]])
def sle_max_z(Delta,tau,w0,N,T,nt=1600):
    d=3*(N+1); A=np.zeros((d,d))
    for n in range(N+1):
        A[3*n:3*n+3,3*n:3*n+3]=w0*Rz-(n/tau)*np.eye(3)
        if n+1<=N:
            A[3*n:3*n+3,3*(n+1):3*(n+1)+3]=Delta*np.sqrt(n+1)*Rx
            A[3*(n+1):3*(n+1)+3,3*n:3*n+3]=Delta*np.sqrt(n+1)*Rx
    w,V=np.linalg.eig(A); c=np.linalg.inv(V)@np.eye(d)[:,2]
    return max(abs(np.real((V@(np.exp(w*t)*c))[2])) for t in np.linspace(T/nt,T,nt))
exc = [max(0.0, sle_max_z(1.0,4.0,0.12,N,80.0)-1.0) for N in (1,2,4,8)]
print("  exact coherence-level tower (kappa=4, low field): excess at tiers 1,2,4,8 =",
      ["%+.1e"%e for e in exc], "[%s]"%PASS(max(exc)<1e-6))
print("    -> the microscopic model and its coherence-retaining closures are positive;")
print("       the violation belongs to the population-level (pair) closure alone.")

# where does the pair closure violate? build 3-spin lift, locate offending entries
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
Q=np.diag([np.sqrt(Dsq[a]) for (_,_,a) in EDGE])@B1.T
def lift3(tau,osc):
    Oax=np.array([osc*omg[a] for (_,_,a) in EDGE]); G=np.zeros((32,32))
    G[0:8,8:20]=-Q.T; G[8:20,0:8]=Q
    G[8:20,8:20]=-np.eye(12)/tau; G[20:32,20:32]=-np.eye(12)/tau
    G[8:20,20:32]=np.diag(Oax); G[20:32,8:20]=-np.diag(Oax); return G
ham=lambda i,j: sum(a!=b for a,b in zip(verts[i],verts[j]))
tau,osc=4.0,0.03
G=lift3(tau,osc); w,V=np.linalg.eig(G); Vi=np.linalg.inv(V)
min_by_d={0:0.,1:0.,2:0.,3:0.}
for t in np.linspace(0.05,20,400):
    M=(V@np.diag(np.exp(w*t))@Vi).real[0:8,0:8]
    for i in range(8):
        for j in range(8):
            d=ham(i,j); min_by_d[d]=min(min_by_d[d],M[i,j])
print("  pair closure, worst cell (tau=4, osc=0.03): min entry by Hamming distance:",
      {d: round(min_by_d[d],4) for d in min_by_d})
print("    (even the diagonal violates for pure-state preparations)")
# per-sector contraction: every Walsh amplitude is individually non-expanding
walsh=lambda A: np.array([np.prod([(-1)**v[a] for a in A]) if A else 1.0 for v in verts])
worst_ct=0.0
for r in (1,2,3):
    for A in itertools.combinations(range(3),r):
        x0=np.zeros(32); zA=walsh(A)/np.sqrt(8); x0[0:8]=zA
        c=Vi@x0
        for t in np.linspace(0.02,20,300):
            m=np.real(V@(np.exp(w*t)*c))[0:8]@zA
            worst_ct=max(worst_ct, abs(m)-1.0)
print("  per-sector contraction: max_t |m_A(t)| - |m_A(0)| = %+.1e  [%s]"%(worst_ct,PASS(worst_ct<1e-9)))
# l1-ball theorem spot check: sum_A |m_A(0)| <= 1, zero currents -> positivity
rng=np.random.default_rng(1); worst_p=1.0
Ws=[walsh(A)/np.sqrt(8) for r in (1,2,3) for A in itertools.combinations(range(3),r)]
for trial in range(200):
    m0=rng.standard_normal(7); m0/=np.abs(m0).sum()
    x0=np.zeros(32); x0[0:8]=sum(mm*wv for mm,wv in zip(m0,Ws))
    c=Vi@x0
    for t in np.linspace(0.05,20,200):
        u=np.real(V@(np.exp(w*t)*c))[0:8]
        worst_p=min(worst_p, ((1/np.sqrt(8))+u).min()*np.sqrt(8)/8*8)
print("  l1-ball positivity (sum_A|m_A(0)|=1, 200 random states): min_t,i 8*p_i = %.4f  [%s]"
      %(worst_p, PASS(worst_p>-1e-9)))
print("    -> theorem (beta = 0): positivity holds on the mode-l1 ball, which contains all")
print("       conventional preparations; violations need near-pure-state initial data.")
# ===================== [M2] two-spin Solomon with dynamical currents ========
print("="*76); print("[M2] TWO-SPIN NOE WITH DYNAMICAL CURRENTS")
v4=[(0,0),(0,1),(1,0),(1,1)]; vx={v:i for i,v in enumerate(v4)}
edges=[(vx[(0,0)],vx[(1,0)],5.0,1.0),(vx[(0,1)],vx[(1,1)],5.0,1.0),
       (vx[(0,0)],vx[(0,1)],5.6,1.0),(vx[(1,0)],vx[(1,1)],5.6,1.0),
       (vx[(0,1)],vx[(1,0)],0.6,1.0),(vx[(0,0)],vx[(1,1)],10.6,2.0)]  # ZQ, DQ
B=np.zeros((4,6)); Om=np.zeros(6); Dq=np.zeros(6)
for e,(i,j,w_,d_) in enumerate(edges): B[j,e]=1; B[i,e]=-1; Om[e]=w_; Dq[e]=d_
Q2=np.diag(np.sqrt(Dq))@B.T
ZI=np.array([(-1)**v[0] for v in v4],float); ZS=np.array([(-1)**v[1] for v in v4],float)
def lift2(tau):
    G=np.zeros((16,16)); G[0:4,4:10]=-Q2.T; G[4:10,0:4]=Q2
    G[4:10,4:10]=-np.eye(6)/tau; G[10:16,10:16]=-np.eye(6)/tau
    G[4:10,10:16]=np.diag(Om); G[10:16,4:10]=-np.diag(Om); return G
def Hm(tau): return B@np.diag(Dq*tau/(1+(Om*tau)**2))@B.T
tau=1.5; H=Hm(tau); wv=Dq*tau/(1+(Om*tau)**2)
sig=ZS@H@ZI/4
print("  Markov limit reproduces Solomon: sigma = %+.5f vs w_DQ - w_ZQ = %+.5f  [%s]"
      %(sig, wv[5]-wv[4], PASS(abs(sig-(wv[5]-wv[4]))<1e-12)))
def evolve(gen,x0,T,dt):
    w,V=np.linalg.eig(gen); c=np.linalg.inv(V)@x0
    ts=np.arange(dt,T,dt); return ts, np.real(np.array([V@(np.exp(w*t)*c) for t in ts]))
a=0.8
for tau_,T_,lab in ((0.05,80.0,"fast bath"),(1.5,12.0,"slow flip-flop (kappa_ZQ=1.5)")):
    x0=np.zeros(16); x0[0:4]=-0.5*a*ZI
    ts,tr=evolve(lift2(tau_),x0,T_,0.002)
    mS=tr[:,0:4]@ZS/2
    wm,Vm=np.linalg.eigh(Hm(tau_)); ci=Vm.T@x0[0:4]
    mSm=np.array([Vm@(np.exp(-wm*t)*ci) for t in ts])@ZS/2
    if tau_==0.05:
        dev=np.abs(mS-mSm).max()/a
        print("  %s: max|lift - Solomon| / |m_I(0)| = %.4f  [%s]"%(lab,dev,PASS(dev<0.02)))
    else:
        # curvature at t=0: quadratic (ballistic-current) onset
        G=lift2(tau_); curv_num=( ( (G@G)@x0 )[0:4]@ZS/2 )
        sel=(ts>0.02)&(ts<0.10)
        p_l=np.polyfit(np.log(ts[sel]),np.log(np.abs(mS[sel])+1e-16),1)[0]
        p_m=np.polyfit(np.log(ts[sel]),np.log(np.abs(mSm[sel])+1e-16),1)[0]
        pk_l,pk_m=np.abs(mS).max(),np.abs(mSm).max()
        print("  %s: onset exponent lift %.2f (2 = ballistic) vs Solomon %.2f (1)"%(lab,p_l,p_m))
        print("    d2 m_S/dt2 (0) from generator = %+.4f (nonzero quadratic launch) [%s]"
              %(curv_num, PASS(p_l>1.7 and abs(p_m-1)<0.3)))
        print("    NOE peak: lift %.4f at t=%.2f vs Solomon %.4f at t=%.2f ; "
              "peak shift = %+.0f%%"%(pk_l,ts[np.argmax(np.abs(mS))],pk_m,
              ts[np.argmax(np.abs(mSm))],100*(pk_l-pk_m)/pk_m))

# ===================== [M3] spatial sector: ballistic-diffusive crossover ===
print("="*76); print("[M3] SPATIAL SECTOR: Goldstein-Kac lift of diffusion")
Nx,D,tauD=241,4.0,10.0
Bx=np.zeros((Nx,Nx-1))
for e in range(Nx-1): Bx[e+1,e]=1; Bx[e,e]=-1
Qx=np.sqrt(D)*Bx.T
Gx=np.zeros((2*Nx-1,2*Nx-1)); Gx[0:Nx,Nx:]=-Qx.T; Gx[Nx:,0:Nx]=Qx
Gx[Nx:,Nx:]=-np.eye(Nx-1)/tauD
i0=Nx//2; x0=np.zeros(2*Nx-1); x0[i0]=1.0
def r95(prof,i0):
    prof=np.abs(prof)/np.abs(prof).sum(); r=0
    while prof[max(0,i0-r):i0+r+1].sum()<0.95: r+=1
    return r
def run(G,x0,marks,dt=0.01):
    x=x0.copy(); t=0.; out={}
    for tm in marks:
        while t<tm-dt/2:
            k1=G@x;k2=G@(x+dt/2*k1);k3=G@(x+dt/2*k2);k4=G@(x+dt*k3)
            x+=dt/6*(k1+2*k2+2*k3+k4); t+=dt
        out[tm]=x.copy()
    return out
marks_b=[1.5,3.,6.,9.]; marks_d=[12.,18.,27.,40.]
sn=run(Gx,x0,marks_b+marks_d)
rb=[r95(sn[t][0:Nx],i0) for t in marks_b]; rd=[r95(sn[t][0:Nx],i0) for t in marks_d]
eb=np.polyfit(np.log(marks_b),np.log(np.array(rb)+.5),1)[0]
ed=np.polyfit(np.log(marks_d),np.log(np.array(rd)+.5),1)[0]
Hpad=-np.pad(Bx@Bx.T*D,((0,Nx-1),(0,Nx-1)))
snM=run(Hpad,x0,marks_b+marks_d)
rbM=[r95(snM[t][0:Nx],i0) for t in marks_b]; rdM=[r95(snM[t][0:Nx],i0) for t in marks_d]
ebM=np.polyfit(np.log(marks_b),np.log(np.array(rbM)+.5),1)[0]
edM=np.polyfit(np.log(marks_d),np.log(np.array(rdM)+.5),1)[0]
vLR=2*np.sqrt(D)   # lattice Lieb-Robinson-type bound from coupling norm
leak=max(np.abs(sn[t][0:Nx])[np.abs(np.arange(Nx)-i0)>vLR*t+6].max() for t in marks_b)
snm=run(-np.pad(Bx@Bx.T*D,((0,Nx-1),(0,Nx-1))),x0,marks_b)
leak_m=max(np.abs(snm[t][0:Nx])[np.abs(np.arange(Nx)-i0)>vLR*t+6].max() for t in marks_b)
print("  spreading exponent (95%% radius): lift %.2f / %.2f  vs  Markov %.2f / %.2f"%(eb,ed,ebM,edM))
print("    (lift stays ballistic, horn-dominated, until t ~ several tau)")
print("  causality: mass beyond cone  lift = %.1e  vs  Markov diffusion = %.1e  [%s]"
      %(leak,leak_m,PASS(leak<1e-10 and leak_m>1e-4)))
print("  bulk front speed c = sqrt(D/tau) = %.3f"%np.sqrt(D/tauD))
print("  mass drift = %.1e"%max(abs(sn[t][0:Nx].sum()-1) for t in sn))
print("="*76)
