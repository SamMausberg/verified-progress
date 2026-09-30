"""Exact CPU reference for The Work a Verifier Needs.

This is not a GPU implementation. Integer logits use mass 2**logit so that every
certificate and categorical law can be tested with Fraction, without trusting a
floating exponential. W and h are bounded synthetic integers in the tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from typing import Callable, Literal, Sequence, TypeVar

Decision = Literal['accept', 'reject', 'unknown']


def mass2(z: int) -> Q:
    if not isinstance(z, int):
        raise TypeError('exact reference requires an integer logit')
    return Q(2**z) if z >= 0 else Q(1, 2**(-z))


def dot(x: Sequence[int], y: Sequence[int]) -> int:
    if len(x) != len(y):
        raise ValueError('dot-product dimensions differ')
    return sum(a*b for a,b in zip(x,y))


def topk_ids(z: Sequence[int], k: int) -> tuple[int,...]:
    if not 0 <= k <= len(z):
        raise ValueError('invalid k')
    return tuple(sorted(range(len(z)), key=lambda i: (-z[i], i))[:k])


@dataclass(frozen=True)
class TopTail:
    """Top-k (logit, unique ID) pairs, and exact mass of every discarded item."""
    k: int
    top: tuple[tuple[int,int],...]
    tail_mass: Q
    count: int

    @staticmethod
    def build(items: Sequence[tuple[int,int]], k: int) -> 'TopTail':
        if k < 0 or len({i for _,i in items}) != len(items):
            raise ValueError('invalid k or duplicate IDs')
        ordered = sorted(items, key=lambda x: (-x[0],x[1]))
        return TopTail(k, tuple(ordered[:k]), sum((mass2(z) for z,_ in ordered[k:]),Q()), len(items))

    def merge(self, other: 'TopTail') -> 'TopTail':
        # Contract: the source domains are disjoint. IDs in the discarded tails
        # are deliberately not retained, so this reference cannot validate that
        # entire precondition at runtime.
        if self.k != other.k:
            raise ValueError('different retained widths')
        joined = self.top + other.top
        if len({i for _,i in joined}) != len(joined):
            raise ValueError('overlapping retained IDs')
        s = TopTail.build(joined,self.k)
        return TopTail(self.k,s.top,self.tail_mass+other.tail_mass+s.tail_mass,self.count+other.count)

    @property
    def total_mass(self) -> Q:
        return self.tail_mass+sum((mass2(z) for z,_ in self.top),Q())


@dataclass(frozen=True)
class TileSummary:
    start: int
    stop: int
    centre: tuple[int,...]
    radius_inf: int
    anchor_max: int
    anchor_mass: Q
    retained: tuple[tuple[int,int],...]
    tail_mass: Q
    tail_max: int | None


@dataclass(frozen=True)
class HeadSummary:
    """Metadata from an already-paid anchor projection, plus weight geometry."""
    anchor: tuple[int,...]
    vocabulary: int
    tiles: tuple[TileSummary,...]
    namespace: str


def make_summary(W: Sequence[Sequence[int]], h0: Sequence[int], tile: int,
                 retained: int=1, namespace: str='test-v1') -> HeadSummary:
    if not W or not h0 or tile < 1 or retained < 0:
        raise ValueError('empty head or invalid tile/retention')
    D = len(h0)
    if any(len(w)!=D for w in W):
        raise ValueError('ragged weights')
    summaries=[]
    for start in range(0,len(W),tile):
        stop=min(start+tile,len(W)); block=W[start:stop]
        # Integer centre is deliberately elementary. Any centre is legal when
        # its radius encloses all rows; centre quality affects pruning only.
        mu=tuple(sum(w[d] for w in block)//len(block) for d in range(D))
        r=max(abs(w[d]-mu[d]) for w in block for d in range(D))
        items=[(dot(w,h0),i) for i,w in enumerate(block,start)]
        tt=TopTail.build(items,retained)
        kept={i for _,i in tt.top}
        tails=[z for z,i in items if i not in kept]
        summaries.append(TileSummary(start,stop,mu,r,max(z for z,_ in items),
                                     tt.total_mass,tt.top,tt.tail_mass,max(tails) if tails else None))
    return HeadSummary(tuple(h0),len(W),tuple(summaries),namespace)


def transported(summary: HeadSummary, h: Sequence[int], namespace: str='test-v1'):
    if namespace != summary.namespace:
        raise ValueError('stale weight/numerical/mask namespace')
    if len(h)!=len(summary.anchor):
        raise ValueError('hidden dimension mismatch')
    delta=tuple(x-y for x,y in zip(h,summary.anchor))
    norm=sum(abs(x) for x in delta)
    out=[]
    for c in summary.tiles:
        shift=dot(delta,c.centre); eps=c.radius_inf*norm
        out.append((c.anchor_max+shift+eps,
                    c.anchor_mass*mass2(shift-eps),
                    c.anchor_mass*mass2(shift+eps),shift,eps))
    return out


class TargetOracle:
    """Only requested rows are projected; counts are arithmetic work, not time."""
    def __init__(self,W: Sequence[Sequence[int]],h: Sequence[int]):
        self.W=W;self.h=tuple(h);self.cache:dict[int,int]={}
        if any(len(w)!=len(h) for w in W):
            raise ValueError('head shape mismatch')

    def row(self,i:int) -> int:
        if not 0 <= i < len(self.W):
            raise IndexError(i)
        if i not in self.cache:
            self.cache[i]=dot(self.W[i],self.h)
        return self.cache[i]

    def tile(self,c:TileSummary) -> tuple[int,Q]:
        z=[self.row(i) for i in range(c.start,c.stop)]
        return max(z),sum((mass2(x) for x in z),Q())

    def all_logits(self) -> tuple[int,...]:
        return tuple(self.row(i) for i in range(len(self.W)))

    @property
    def rows_evaluated(self)->int:
        return len(self.cache)


def certify_greedy(summary: HeadSummary, oracle: TargetOracle,
                   candidate: int, namespace: str='test-v1') -> tuple[int,int,int]:
    """Exact transported max bounds; strict exclusion retains all possible ties.

    Returns winner, distinct target rows projected, whole tiles skipped.
    """
    bounds=transported(summary,oracle.h,namespace)
    winner=candidate;score=oracle.row(candidate);skipped=0
    # A high upper bound is a sensible exact-search order, not a universal best
    # GPU order. Every tile has a certificate or is fully projected.
    order=sorted(range(len(bounds)),key=lambda j:-bounds[j][0])
    for j in order:
        upper=bounds[j][0];c=summary.tiles[j]
        if upper < score:
            skipped+=1;continue
        for i in range(c.start,c.stop):
            z=oracle.row(i)
            if z>score or (z==score and i<winner):
                winner,score=i,z
    return winner,oracle.rows_evaluated,skipped


def interval_decision(weight: Q, zlo: Q, zhi: Q, qx: Q, u: Q) -> Decision:
    """Matches strict u < min(1, weight/(qx*Z)); qx is the ACTUAL proposal."""
    if weight<0 or not 0<zlo<=zhi or not 0<qx<=1 or not 0<=u<1:
        raise ValueError('invalid mass/proposal/uniform interval')
    if u*qx*zhi < weight:
        return 'accept'
    if u*qx*zlo >= weight:
        return 'reject'
    return 'unknown'


def unresolved_probability(weight: Q,zlo: Q,zhi: Q,qx: Q)->Q:
    if weight<0 or not 0<zlo<=zhi or not 0<qx<=1:
        raise ValueError('invalid interval')
    return min(Q(1),weight/(qx*zlo))-min(Q(1),weight/(qx*zhi))


def lazy_accept(summary:HeadSummary,oracle:TargetOracle,x:int,qx:Q,u:Q,
                refine_retained: bool=False,namespace:str='test-v1'):
    """Decide acceptance using progressively refined exact partition bounds.

    The residual branch is NOT handled here. A rejection still needs the true
    residual law; callers must complete evaluation or use a separately proved
    exact residual sampler. No real-model performance claim follows.
    """
    raw=transported(summary,oracle.h,namespace)
    intervals=[]
    for c,(_,lo,hi,shift,eps) in zip(summary.tiles,raw):
        if refine_retained:
            exact=sum((mass2(oracle.row(i)) for _,i in c.retained),Q())
            lo=exact+c.tail_mass*mass2(shift-eps)
            hi=exact+c.tail_mass*mass2(shift+eps)
        intervals.append([lo,hi])
    weight=mass2(oracle.row(x));steps=0
    while True:
        lo=sum((a for a,b in intervals),Q());hi=sum((b for a,b in intervals),Q())
        decision=interval_decision(weight,lo,hi,qx,u)
        if decision != 'unknown':
            return decision,oracle.rows_evaluated,steps
        # Deliberately simple CPU oracle: maximum mass uncertainty. Device
        # scheduling must price common tiles, occupancy and graph barriers.
        j=max(range(len(intervals)),key=lambda j:intervals[j][1]-intervals[j][0])
        _,mass=oracle.tile(summary.tiles[j]);intervals[j]=[mass,mass];steps+=1
        if steps>len(intervals):
            raise AssertionError('exact completion must resolve the decision')


def categorical(weights:Sequence[Q],u:Q)->int:
    if not weights or any(x<0 for x in weights) or sum(weights)<=0 or not 0<=u<1:
        raise ValueError('invalid categorical inputs')
    threshold=u*sum(weights);acc=Q()
    for i,w in enumerate(weights):
        acc+=w
        if threshold<acc:return i
    raise AssertionError('normalised CDF exhausted')


def rejection_law(p:Sequence[Q],q:Sequence[Q])->tuple[Q,...]:
    if len(p)!=len(q) or any(x<0 for row in (p,q) for x in row) or sum(p)!=1 or sum(q)!=1:
        raise ValueError('not probability distributions')
    accepted=[min(a,b) for a,b in zip(p,q)]
    residual=[max(Q(),a-b) for a,b in zip(p,q)]
    assert sum(residual)==1-sum(accepted)
    return tuple(a+r for a,r in zip(accepted,residual))


def couple_one_step(summary:HeadSummary,W,h,q,x,u,residual_u):
    """Lazy accept then FULL exact residual fallback; returns output and work."""
    oracle=TargetOracle(W,h)
    decision,_,_=lazy_accept(summary,oracle,x,q[x],u)
    if decision=='accept':return x,oracle.rows_evaluated
    z=oracle.all_logits();weights=[mass2(v) for v in z];Z=sum(weights)
    residual=[max(Q(),w/Z-qi) for w,qi in zip(weights,q)]
    return categorical(residual,residual_u),oracle.rows_evaluated


def fixed_protocol_frontier(decisions:Sequence[Decision])->tuple[int,int]:
    """(leading certified accepts, earliest certified rejection or n).

    Unknowns earlier than the earliest rejection must be resolved. A later
    rejection can hide work to its right, never the earlier unknown prefix.
    """
    leading=0
    while leading<len(decisions) and decisions[leading]=='accept':leading+=1
    first_reject=next((i for i,x in enumerate(decisions) if x=='reject'),len(decisions))
    return leading,first_reject


def compose_maps(left:Sequence[int],right:Sequence[int])->tuple[int,...]:
    """Chronological composition: execute left, then right."""
    if any(not 0<=x<len(right) for x in left):raise ValueError('invalid map codomain')
    return tuple(right[x] for x in left)


def scan_maps(maps:Sequence[Sequence[int]],anchor:int=0)->tuple[int,...]:
    """Hillis--Steele reference: O(L K log L) work, O(log L) stages.

    A work-efficient tree scan is a separate GPU candidate, O(L K) work. This
    reference intentionally makes no claim to implement that lower-work scan.
    """
    if not maps:return ()
    x=[tuple(m) for m in maps];step=1
    while step<len(x):
        old=x[:]
        for t in range(step,len(x)):
            x[t]=compose_maps(old[t-step],old[t])
        step*=2
    return tuple(m[anchor] for m in x)


def serial_maps(maps:Sequence[Sequence[int]],anchor:int=0)->tuple[int,...]:
    out=[]
    for m in maps:
        anchor=m[anchor];out.append(anchor)
    return tuple(out)


def conditional_depth_bias()->tuple[Q,Q]:
    """p=q=(1/2,1/2). Discard x=0, draw fresh p; keep x=1. Biased."""
    return Q(1,4),Q(3,4)


@dataclass(frozen=True)
class Lease:
    slot:int
    epoch:int
    prefix:int


class EpochStore:
    """A sequential model of a publication guard, NOT a CUDA race-freedom proof."""
    def __init__(self):
        self.epoch:dict[int,int]={};self.state:dict[int,tuple[int,int]]={}
    def allocate(self,slot:int,state:int,prefix:int=0)->Lease:
        e=self.epoch.get(slot,0)+1;self.epoch[slot]=e;self.state[slot]=(state,prefix)
        return Lease(slot,e,prefix)
    def commit(self,lease:Lease,state:int,prefix:int)->bool:
        if self.epoch.get(lease.slot)!=lease.epoch or self.state[lease.slot][1]!=lease.prefix:
            return False
        if prefix<lease.prefix:raise ValueError('prefix regressed')
        self.state[lease.slot]=(state,prefix);return True


def gdn_step(S,alpha,beta,k,v):
    """Exact rational Gated DeltaNet orientation S[value,key]."""
    decayed=[[alpha*x for x in row] for row in S]
    u=[beta*(vi-sum((s*ki for s,ki in zip(row,k)),Q())) for vi,row in zip(v,decayed)]
    nxt=[[x+ui*kj for x,kj in zip(row,k)] for row,ui in zip(decayed,u)]
    return nxt,u


def gdn_fold(S0,records):
    """Correct valid derived records (alpha,u,k) from the same causal run."""
    S=[list(row) for row in S0]
    for a,u,k in records:
        S=[[a*x+ui*kj for x,kj in zip(row,k)] for row,ui in zip(S,u)]
    return S


def snapshot_budget(rewards:Sequence[Sequence[Q]],cost:Callable[[tuple[int,...]],Q]):
    """Exhaustive small-state oracle allowing composition-dependent costs."""
    if not rewards or any(not r for r in rewards):raise ValueError('empty actions')
    best=None
    for d in product(*(range(len(r)) for r in rewards)):
        c=cost(d)
        if c<=0:raise ValueError('nonpositive cost')
        r=sum((rewards[i][a] for i,a in enumerate(d)),Q())
        if best is None or r/c>best[1]/best[2]:best=(d,r,c)
    return best
