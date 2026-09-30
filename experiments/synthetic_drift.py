"""Synthetic certificate work curves. Outputs are NOT GPU measurements."""
from __future__ import annotations
import sys,csv,json,random
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from decision_reference import *

def run():
    rng=random.Random(90210);rows=[];V,D,tile=128,8,16
    # Construct deliberately favourable, well-separated tiles and an unstructured
    # control. Both are synthetic. Neither estimates the geometry of a real head.
    for family in ('clustered','unstructured'):
        fixtures=[]
        for case in range(36):
            W=[]
            for c in range(V//tile):
                centre=[rng.randint(-4,4) for _ in range(D)]
                W += [[centre[j]+rng.randint(-1,1) if family=='clustered' else rng.randint(-5,5)
                       for j in range(D)] for _ in range(tile)]
            h0=[rng.randint(-2,2) for _ in range(D)];direction=[rng.randint(-1,1) for _ in range(D)]
            # q is explicitly the normalised anchor-head law, not a DFlash2 law.
            zw=[mass2(dot(w,h0)) for w in W];Z0=sum(zw);q=[v/Z0 for v in zw]
            ux=Q(rng.randrange(1,1024),1024);u=Q(rng.randrange(1024),1024)
            x=categorical(q,ux);s=make_summary(W,h0,tile,retained=1)
            fixtures.append((W,h0,direction,q,x,u,s))
        for drift in (0,1,2,4):
            sums={'greedy_rows':0,'accept_rows':0,'retained_accept_rows':0,'initial_resolved':0,'accept':0,'with_residual_rows':0}
            for W,h0,direction,q,x,u,s in fixtures:
                h=[a+drift*d for a,d in zip(h0,direction)]
                seed=topk_ids([dot(w,h0) for w in W],1)[0]
                _,ng,_=certify_greedy(s,TargetOracle(W,h),seed)
                decision,ns,steps=lazy_accept(s,TargetOracle(W,h),x,q[x],u)
                _,nr,_=lazy_accept(s,TargetOracle(W,h),x,q[x],u,True)
                sums['greedy_rows']+=ng;sums['accept_rows']+=ns;sums['retained_accept_rows']+=nr
                sums['initial_resolved']+=int(steps==0);sums['accept']+=int(decision=='accept');sums['with_residual_rows']+=ns if decision=='accept' else V
            n=len(fixtures)
            rows.append(dict(family=family,drift=drift,cases=n,vocabulary=V,dimension=D,tile=tile,
                             greedy_projected_fraction=sums['greedy_rows']/(n*V),
                             accept_projected_fraction=sums['accept_rows']/(n*V),
                             retained_accept_projected_fraction=sums['retained_accept_rows']/(n*V),
                             initially_resolved_fraction=sums['initial_resolved']/n,
                             acceptance_fraction=sums['accept']/n,
                             decision_plus_residual_projected_fraction=sums['with_residual_rows']/(n*V)))
    path=ROOT/'data'/'synthetic_drift.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (ROOT/'evidence'/'synthetic_drift.json').write_text(json.dumps({'seed':90210,'rows':rows,
        'scope':'synthetic integer heads, exact base-2 masses; distinct target rows only, not runtime; includes a separate full-residual row count; bonus, draft-summary and bound costs excluded'},indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':run()
