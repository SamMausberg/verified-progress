"""Pathwise exact rational priority-race references.

For real categorical sampling, priorities are independent 1/Exp(1), equivalently
exp(Gumbel). Tests use arbitrary positive rational priorities and establish
pathwise equivalence only; they do not discretize an exponential into an exact
continuous sampler. Priors MUST be independent of draft and acceptance draws.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Sequence
from decision_reference import HeadSummary,TargetOracle,make_summary,transported,mass2,dot

@dataclass(frozen=True)
class RaceSummary:
    head: HeadSummary
    priorities: tuple[Q,...]
    top_priorities: tuple[tuple[tuple[Q,int],...],...]
    exclusion_capacity: int

def make_race_summary(W,h0,tile,priorities:Sequence[Q],exclusion_capacity:int=0):
    if len(W)!=len(priorities) or any(x<=0 for x in priorities) or exclusion_capacity<0:
        raise ValueError('invalid priority field or exclusion capacity')
    head=make_summary(W,h0,tile)
    tops=[]
    for c in head.tiles:
        values=[(mass2(dot(W[i],h0))*priorities[i],i) for i in range(c.start,c.stop)]
        tops.append(tuple(sorted(values,key=lambda x:(-x[0],x[1]))[:exclusion_capacity+1]))
    return RaceSummary(head,tuple(priorities),tuple(tops),exclusion_capacity)

def outside_anchor_max(s:RaceSummary,t:int,excluded:set[int]):
    if len(excluded)>s.exclusion_capacity:raise ValueError('exclusion capacity exceeded')
    return next((x for x in s.top_priorities[t] if x[1] not in excluded),None)

def certify_target_race(s:RaceSummary,oracle:TargetOracle):
    bounds=transported(s.head,oracle.h)
    seed=max((top[0] for top in s.top_priorities),key=lambda x:(x[0],-x[1]))[1]
    winner=seed;score=mass2(oracle.row(seed))*s.priorities[seed]
    upper=[top[0][0]*mass2(a+e) for top,(_,lo,hi,a,e) in zip(s.top_priorities,bounds)]
    for t in sorted(range(len(upper)),key=lambda j:-upper[j]):
        if upper[t]<score:continue
        c=s.head.tiles[t]
        for i in range(c.start,c.stop):
            v=mass2(oracle.row(i))*s.priorities[i]
            if v>score or (v==score and i<winner):winner,score=i,v
    return winner,oracle.rows_evaluated

def certify_residual_race(s:RaceSummary,oracle:TargetOracle,q:Sequence[Q]):
    """Exact argmax of (w_i-Z*q_i)_+ * priority_i via valid intervals.

    Uses finite refinement with full completion fallback. Supports arbitrary q
    with support within the summary's exclusion capacity. Retains no claim that
    maximum-width refinement is an efficient GPU policy.
    """
    V=s.head.vocabulary
    if len(q)!=V or any(x<0 for x in q) or sum(q)!=1:raise ValueError('invalid proposal law')
    support={i for i,x in enumerate(q) if x>0}
    if len(support)>s.exclusion_capacity:raise ValueError('proposal support exceeds summary capacity')
    raw=transported(s.head,oracle.h)
    intervals=[[lo,hi] for _,lo,hi,a,e in raw];done=set()
    for i in support:oracle.row(i)
    seeds=[(v,i) for t in range(len(raw)) if (item:=outside_anchor_max(s,t,support)) is not None for v,i in [item]]
    if seeds:oracle.row(max(seeds,key=lambda x:(x[0],-x[1]))[1])
    steps=0
    while True:
        Zlo=sum((a for a,b in intervals),Q());Zhi=sum((b for a,b in intervals),Q())
        lo_score={};hi_score={}
        for i in support:
            w=mass2(oracle.row(i));lo_score[i]=max(Q(),w-Zhi*q[i])*s.priorities[i]
            hi_score[i]=max(Q(),w-Zlo*q[i])*s.priorities[i]
        for i,z in oracle.cache.items():
            if i not in support:lo_score[i]=hi_score[i]=mass2(z)*s.priorities[i]
        winner=max(lo_score,key=lambda i:(lo_score[i],-i));L=lo_score[winner]
        explicit=all(i==winner or L>u or (L==u and winner<i) for i,u in hi_score.items())
        opaque=True
        for t,c in enumerate(s.head.tiles):
            if t in done:continue
            best=outside_anchor_max(s,t,support)
            if best is not None:
                _,_,_,a,e=raw[t];U=best[0]*mass2(a+e)
                if U>=L:opaque=False;break
        if L>0 and explicit and opaque:return winner,oracle.rows_evaluated,steps
        pending=[t for t in range(len(raw)) if t not in done]
        if not pending:
            if max(hi_score.values())<=0:raise ValueError('zero residual; no rejection branch is defined')
            raise AssertionError('complete exact scores must identify their ordered maximizer')
        t=max(pending,key=lambda j:intervals[j][1]-intervals[j][0])
        _,Zc=oracle.tile(s.head.tiles[t]);intervals[t]=[Zc,Zc];done.add(t);steps+=1
