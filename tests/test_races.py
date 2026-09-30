"""Exact pathwise race equivalence; no claim to sample continuous exponentials."""
import sys,json,random,unittest,time
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from decision_reference import *
from race_reference import *
R=random.Random(314159);COUNTS={}
def count(k):COUNTS[k]=COUNTS.get(k,0)+1
class Races(unittest.TestCase):
    def test_exclusion_summary(self):
        for case in range(300):
            V=R.randint(3,32);K=R.randint(0,V-1);tile=R.randint(1,V)
            W=[[R.randint(-4,4)] for _ in range(V)];prior=[Q(R.randint(1,20),R.randint(1,20)) for _ in range(V)]
            s=make_race_summary(W,[1],tile,prior,K);excluded=set(R.sample(range(V),K))
            for t,c in enumerate(s.head.tiles):
                values=[(mass2(W[i][0])*prior[i],i) for i in range(c.start,c.stop) if i not in excluded]
                ref=max(values,key=lambda x:(x[0],-x[1])) if values else None
                self.assertEqual(outside_anchor_max(s,t,excluded),ref)
            count('exclusion_summaries')
    def test_target_race(self):
        for case in range(250):
            V=R.randint(3,32);D=R.randint(1,6)
            W=[[R.randint(-3,3) for _ in range(D)] for _ in range(V)]
            h0=[R.randint(-2,2) for _ in range(D)];h=[x+R.randint(-1,1) for x in h0]
            prior=[Q(R.randint(1,20),R.randint(1,20)) for _ in range(V)]
            s=make_race_summary(W,h0,R.randint(1,V),prior)
            expected=max(range(V),key=lambda i:(mass2(dot(W[i],h))*prior[i],-i))
            answer,n=certify_target_race(s,TargetOracle(W,h))
            self.assertEqual(answer,expected);self.assertLessEqual(n,V);count('target_race_equivalences')
    def test_residual_race(self):
        for case in range(250):
            V=R.randint(3,32);D=R.randint(1,5);K=R.randint(1,V-1)
            W=[[R.randint(-2,2) for _ in range(D)] for _ in range(V)]
            h0=[R.randint(-1,1) for _ in range(D)];h=[x+R.randint(-1,1) for x in h0]
            support=R.sample(range(V),K);qq=[R.randint(1,9) for _ in support];q=[Q() for _ in range(V)]
            for i,a in zip(support,qq):q[i]=Q(a,sum(qq))
            prior=[Q(R.randint(1,20),R.randint(1,20)) for _ in range(V)]
            s=make_race_summary(W,h0,R.randint(1,V),prior,K)
            w=[mass2(dot(row,h)) for row in W];Z=sum(w)
            expected=max(range(V),key=lambda i:(max(Q(),w[i]-Z*q[i])*prior[i],-i))
            answer,n,steps=certify_residual_race(s,TargetOracle(W,h),q)
            self.assertEqual(answer,expected);self.assertLessEqual(n,V)
            self.assertLessEqual(steps,len(s.head.tiles));count('residual_race_equivalences')
if __name__=='__main__':
    start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Races))
    report={'status':'pass' if result.wasSuccessful() else 'fail','methods':result.testsRun,'seed':314159,
            'counts':COUNTS,'seconds':time.perf_counter()-start,'scope':'exact rational pathwise priorities, not continuous distribution sampling or GPU execution'}
    (ROOT/'evidence'/'race_tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    sys.exit(not result.wasSuccessful())
