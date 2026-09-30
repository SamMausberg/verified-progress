"""CPU references for mathematical experiments, NOT production GPU kernels.

Python >=3.10; NumPy. Integer tests deliberately separate algebraic equivalence
from floating-point and serving correctness. No network or GPU access is used.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from typing import Sequence
import math
import numpy as np


def stable_topk(scores: Sequence, k: int, ids: Sequence[int] | None = None):
    """Descending score, then ascending unique ID. Reject NaN explicitly."""
    if ids is None:
        ids = list(range(len(scores)))
    if len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError('scores and unique IDs must have matching lengths')
    if not 0 <= k <= len(scores):
        raise ValueError('invalid k')
    if any(math.isnan(float(x)) for x in scores):
        raise ValueError('NaN has no admitted ranking contract')
    return sorted(zip(scores, ids), key=lambda x: (-x[0], x[1]))[:k]


def tiled_topk(scores: Sequence, k: int, tile: int):
    if tile < 1 or not 0 <= k <= len(scores):
        raise ValueError('invalid tile or k')
    candidates = []
    for start in range(0, len(scores), tile):
        vals = scores[start:start + tile]
        candidates += stable_topk(vals, min(k, len(vals)), list(range(start, start+len(vals))))
    return stable_topk([x[0] for x in candidates], k, [x[1] for x in candidates])


def projection_topk(hidden: np.ndarray, weight: np.ndarray, k: int, tile: int):
    """Complete dot products per vocab tile, never prune partial hidden sums."""
    if hidden.ndim != 1 or weight.ndim != 2 or weight.shape[1] != hidden.size:
        raise ValueError('shape mismatch')
    out = []
    for start in range(0, weight.shape[0], tile):
        z = weight[start:start+tile] @ hidden
        out += stable_topk(z.tolist(), min(k, len(z)), list(range(start, start+len(z))))
    return stable_topk([v for v, _ in out], k, [i for _, i in out])


def score_all(A: np.ndarray, H: np.ndarray, B: np.ndarray, unary: np.ndarray):
    """A,H,B dimensions [L,K,R], [L,R], [L,K,R]; exact for bounded int64."""
    return unary[:,None,:] + np.einsum('lar,lr,lbr->lab', A, H, B)


def _choose(row: np.ndarray, greedy: bool, uniform: float, temperature: float):
    if np.any(~np.isfinite(row)):
        raise ValueError('nonfinite score: dispatch to defined fallback, not silent repair')
    if greedy:
        idx = int(np.argmax(row))
        q = np.zeros(len(row)); q[idx] = 1
        return idx, q
    if not 0 <= uniform < 1 or temperature <= 0:
        raise ValueError('invalid random draw/temperature')
    z = row.astype(np.float64) / temperature
    p = np.exp(z - np.max(z)); p /= np.sum(p)
    idx = min(int(np.searchsorted(np.cumsum(p), uniform, side='right')), len(p)-1)
    return idx, p


def walk_full(scores: np.ndarray, greedy=True, uniforms=None, temperature=1.):
    L, K, _ = scores.shape
    if uniforms is None: uniforms = np.zeros(L)
    previous = 0; path=[]; qs=[]
    for t in range(L):
        previous, q = _choose(scores[t,previous], greedy, float(uniforms[t]), temperature)
        path.append(previous); qs.append(q)
    return tuple(path), np.array(qs)


def walk_lazy(A, H, B, unary, greedy=True, uniforms=None, temperature=1.):
    L, K, _ = A.shape
    if uniforms is None: uniforms = np.zeros(L)
    previous=0; path=[]; qs=[]
    for t in range(L):
        row = unary[t] + B[t] @ (A[t,previous]*H[t])
        previous, q = _choose(row, greedy, float(uniforms[t]), temperature)
        path.append(previous); qs.append(q)
    return tuple(path), np.array(qs)


def lse_summary(xs: Sequence[float]):
    if any(math.isnan(float(x)) or x == math.inf for x in xs):
        raise ValueError('NaN/+inf excluded from finite-score contract')
    if not len(xs) or all(x == -math.inf for x in xs):
        return -math.inf, 0.
    m = max(xs)
    return m, math.fsum(math.exp(x-m) for x in xs)


def merge_lse(a, b):
    ma, za = a; mb, zb = b
    if za == 0: return b
    if zb == 0: return a
    m=max(ma,mb)
    return m, math.exp(ma-m)*za + math.exp(mb-m)*zb


def prefix_reward(r, path, anchor=0):
    v=F(0); alive=F(1); a=anchor
    for t,b in enumerate(path):
        alive *= r[t][a][b]; v += alive; a=b
    return v


def prefix_optimum(r, anchor=0):
    """Exact-rational Bellman DP. Values optimize a supplied Markov surrogate."""
    if not r: return (), F(0)
    K=len(r[0]); L=len(r)
    if any(len(M)!=K or any(len(row)!=K or any(p<0 or p>1 for p in row) or sum(row)>1 for row in M) for M in r):
        raise ValueError('not a substochastic square lattice')
    V=[F(0)]*K; back=[]
    for M in reversed(r):
        vals=[[p*(1+V[b]) for b,p in enumerate(row)] for row in M]
        acts=[max(range(K),key=lambda b: row[b]) for row in vals]
        V=[row[a] for row,a in zip(vals,acts)]; back.append(acts)
    root=V[anchor]; a=anchor; path=[]
    for acts in reversed(back):
        a=acts[a];path.append(a)
    return tuple(path),root


def budget_plan(rewards: Sequence[Sequence[F]], costs: Sequence[F]):
    """Optimize total reward / cost at a snapshot; cost depends ONLY on total depth.

    rewards[i][d] includes request i's mandatory progress token. costs[m] includes
    all mandatory verification and planning cost for total OPTIONAL depth m.
    This is not an optimal online queueing scheduler.
    """
    if not rewards or any(not row for row in rewards): raise ValueError('empty rewards')
    max_m=sum(len(r)-1 for r in rewards)
    if len(costs)!=max_m+1 or any(c<=0 for c in costs): raise ValueError('invalid cost table')
    states={0:(F(0),())}
    for row in rewards:
        nxt={}
        for m,(value,path) in states.items():
            for d,gain in enumerate(row):
                cand=(value+gain,path+(d,))
                if m+d not in nxt or cand[0]>nxt[m+d][0]: nxt[m+d]=cand
        states=nxt
    m=max(states,key=lambda j: states[j][0]/costs[j])
    value,path=states[m]
    return path,value,costs[m]


def rejection_output(p: Sequence[F], actual_q: Sequence[F], reported_q: Sequence[F]|None=None):
    """Analytic one-step rejection sampler law; wrong reported q exposes bias."""
    q=actual_q if reported_q is None else reported_q
    if len(p)!=len(q) or len(p)!=len(actual_q) or any(x<0 for row in (p,q,actual_q) for x in row): raise ValueError('invalid law')
    if any(sum(row)!=1 for row in (p,q,actual_q)): raise ValueError('not normalized')
    if any(a>0 and b==0 for a,b in zip(actual_q,q)): raise ValueError('sample outside reported support')
    accepted=[a*min(F(1),pi/qi) if qi else F(0) for pi,a,qi in zip(p,actual_q,q)]
    rejected=1-sum(accepted)
    residual=[max(F(0),pi-qi) for pi,qi in zip(p,q)]; z=sum(residual)
    if rejected and not z: raise ValueError('invalid correction from inconsistent law')
    return tuple(a+(rejected*r/z if z else 0) for a,r in zip(accepted,residual))


def apply_update(state, record):
    """Toy affine state recurrence; NOT a full GDN implementation."""
    alpha,u,k=record
    return tuple(tuple(alpha*state[i][j]+u[i]*k[j] for j in range(len(k))) for i in range(len(u)))


@dataclass
class ReplayState:
    checkpoint: tuple
    capacity: int
    records: list
    flushes: int=0
    def current(self):
        s=self.checkpoint
        for r in self.records: s=apply_update(s,r)
        return s
    def verify(self, proposals):
        s=self.current();out=[]
        for r in proposals: s=apply_update(s,r);out.append(s)
        return out
    def commit(self, proposals, accepted):
        if not 0<=accepted<=len(proposals): raise ValueError('invalid prefix')
        if self.capacity<1: raise ValueError('invalid capacity')
        # Reference flushes before overwriting live history; real kernels need
        # capacity reservation before verify and distinct tentative/committed cursors.
        for r in proposals[:accepted]:
            if len(self.records)==self.capacity:
                self.checkpoint=self.current();self.records=[];self.flushes+=1
            self.records.append(r)


def certified_l1_screen(hidden, weights, cluster_ids, centers, k):
    """Exact INTEGER screening by center + radius*||h||_1.

    Bounds use ||w-center||_infinity and no floating arithmetic. This is a toy
    exact-search reference, not a usable GPU performance claim or novel bound.
    Returns top-k and numbers of evaluated/skipped vocabulary rows.
    """
    V,D=weights.shape
    if hidden.shape!=(D,) or len(cluster_ids)!=V or not 1<=k<=V: raise ValueError('invalid dimensions')
    groups={c:np.flatnonzero(cluster_ids==c) for c in np.unique(cluster_ids)}
    bounds=[]
    norm=sum(abs(int(x)) for x in hidden)
    for c,idx in groups.items():
        radius=max(abs(int(x)) for x in (weights[idx]-centers[c]).flat)
        upper=int(centers[c]@hidden)+radius*norm
        bounds.append((upper,c,idx))
    found=[]; computed=0; skipped=0
    for upper,c,idx in sorted(bounds,key=lambda x:-x[0]):
        # Strict inequality preserves all score ties, including better token IDs.
        if len(found)>=k and upper<found[-1][0]:
            skipped+=len(idx);continue
        vals=weights[idx]@hidden
        found=stable_topk([x[0] for x in found]+vals.tolist(),min(k,len(found)+len(idx)),[x[1] for x in found]+idx.tolist())
        computed+=len(idx)
    return found,computed,skipped
