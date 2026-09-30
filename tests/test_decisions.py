"""Finite exact-arithmetic tests; run from the bundle root. No GPU required."""
from __future__ import annotations
import json, sys, random, unittest, platform, time, math
from pathlib import Path
from fractions import Fraction as Q
from itertools import product
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from decision_reference import *
R=random.Random(20260930)
COUNTS={}
WITNESSES={}
def count(k,n=1):COUNTS[k]=COUNTS.get(k,0)+n

def head(V=None,D=None):
    V=V or R.randint(3,24);D=D or R.randint(1,6)
    W=[[R.randint(-2,2) for _ in range(D)] for _ in range(V)]
    h0=[R.randint(-2,2) for _ in range(D)]
    h=[a+R.randint(-1,1) for a in h0]
    return W,h0,h

def law(n,positive=False):
    a=[R.randint(1 if positive else 0,8) for _ in range(n)]
    if not sum(a):a[0]=1
    return tuple(Q(v,sum(a)) for v in a)

class Decisions(unittest.TestCase):
    def test_summary_monoid(self):
        for _ in range(250):
            n=R.randint(1,36);k=R.randint(0,n);z=[R.randint(-8,8) for i in range(n)]
            a=R.randint(0,n);b=R.randint(a,n)
            items=list(zip(z,range(n)))
            A=TopTail.build(items[:a],k);B=TopTail.build(items[a:b],k);C=TopTail.build(items[b:],k)
            expected=TopTail.build(items,k)
            self.assertEqual(A.merge(B).merge(C),expected)
            self.assertEqual(A.merge(B.merge(C)),expected)
            self.assertEqual(C.merge(A).merge(B),expected)
            self.assertEqual(expected.total_mass,sum((mass2(x) for x in z),Q()))
            count('top_tail_partitions')

    def test_transport_enclosure(self):
        for _ in range(300):
            W,h0,h=head();s=make_summary(W,h0,R.randint(1,len(W)),R.randint(0,3))
            for c,(U,lo,hi,a,e) in zip(s.tiles,transported(s,h)):
                z=[dot(w,h) for w in W[c.start:c.stop]];Z=sum((mass2(x) for x in z),Q())
                self.assertLessEqual(max(z),U);self.assertLessEqual(lo,Z);self.assertLessEqual(Z,hi)
                for i in range(c.start,c.stop):
                    delta=dot(W[i],h)-dot(W[i],h0)
                    self.assertLessEqual(a-e,delta);self.assertLessEqual(delta,a+e)
                count('transport_tiles');count('transport_weight_rows',len(z))
            count('transport_heads')

    def test_greedy_screen(self):
        for _ in range(250):
            W,h0,h=head();s=make_summary(W,h0,R.randint(1,len(W)))
            oracle=TargetOracle(W,h)
            winner,n,skipped=certify_greedy(s,oracle,R.randrange(len(W)))
            self.assertEqual(winner,topk_ids([dot(w,h) for w in W],1)[0])
            self.assertLessEqual(n,len(W));count('greedy_screens')
        W=[[0],[0],[0],[0]];s=make_summary(W,[0],2)
        self.assertEqual(certify_greedy(s,TargetOracle(W,[0]),3)[0],0)
        W=[[100],[99],[-100],[-99]];s=make_summary(W,[1],2)
        winner,n,skip=certify_greedy(s,TargetOracle(W,[1]),0)
        self.assertLess(n,len(W));self.assertGreater(skip,0)
        WITNESSES['zero_drift_greedy']={'winner':winner,'projected_rows':n,'vocabulary':len(W)}

    def test_retained_mass_tightening(self):
        for _ in range(150):
            W,h0,h=head();s=make_summary(W,h0,R.randint(1,len(W)),R.randint(0,5))
            for c,(_,lo,hi,a,e) in zip(s.tiles,transported(s,h)):
                exact=sum((mass2(dot(W[i],h)) for _,i in c.retained),Q())
                L=exact+c.tail_mass*mass2(a-e);U=exact+c.tail_mass*mass2(a+e)
                Z=sum((mass2(dot(W[i],h)) for i in range(c.start,c.stop)),Q())
                self.assertLessEqual(lo,L);self.assertLessEqual(L,Z)
                self.assertLessEqual(Z,U);self.assertLessEqual(U,hi);count('retained_tightenings')

    def test_acceptance_guards(self):
        for _ in range(400):
            Z=Q(R.randint(1,32));lo=Z/Q(R.randint(1,8));hi=Z*R.randint(1,8)
            w=Q(R.randint(0,int(Z)));q=Q(R.randint(1,16),16)
            for u in (Q(0),Q(1,3),Q(1,2),Q(15,16)):
                d=interval_decision(w,lo,hi,q,u);truth='accept' if u*q*Z<w else 'reject'
                if d!='unknown':self.assertEqual(d,truth)
                self.assertEqual(interval_decision(w,Z,Z,q,u),truth);count('acceptance_guard_instances')
        self.assertEqual(interval_decision(Q(0),Q(1),Q(2),Q(1),Q(0)),'reject')
        with self.assertRaises(ValueError):interval_decision(Q(1),Q(2),Q(1),Q(1),Q(0))

    def test_ambiguity_measure(self):
        for _ in range(160):
            lo=Q(R.randint(1,10));hi=lo+R.randint(0,10);w=Q(R.randint(1,12));q=Q(R.randint(1,10),10)
            # Exact interval length, not an empirical probability estimate.
            cuts=sorted({Q(0),Q(1),min(Q(1),w/(q*hi)),min(Q(1),w/(q*lo))})
            measure=Q()
            for a,b in zip(cuts,cuts[1:]):
                if a<b and interval_decision(w,lo,hi,q,(a+b)/2)=='unknown':measure+=b-a
            self.assertEqual(measure,unresolved_probability(w,lo,hi,q));count('ambiguity_measures')

    def test_lazy_accept_completion(self):
        for _ in range(250):
            W,h0,h=head();s=make_summary(W,h0,R.randint(1,len(W)),R.randint(0,3))
            x=R.randrange(len(W));q=Q(R.randint(1,8),8);u=Q(R.randrange(64),64)
            w=mass2(dot(W[x],h));Z=sum((mass2(dot(wi,h)) for wi in W),Q())
            truth='accept' if u*q*Z<w else 'reject'
            for retain in (False,True):
                d,n,steps=lazy_accept(s,TargetOracle(W,h),x,q,u,retain)
                self.assertEqual(d,truth);self.assertLessEqual(n,len(W));self.assertLessEqual(steps,len(s.tiles))
                count('lazy_accept_decisions')

    def test_coupled_one_step(self):
        for _ in range(200):
            W,h0,h=head(V=R.randint(2,12));s=make_summary(W,h0,R.randint(1,len(W)))
            q=law(len(W),True);x=categorical(q,Q(R.randrange(64),64))
            u=Q(R.randrange(64),64);ur=Q(R.randrange(64),64)
            weights=[mass2(dot(w,h)) for w in W];Z=sum(weights)
            if u*q[x]*Z<weights[x]:ref=x
            else:ref=categorical([max(Q(),w/Z-qi) for w,qi in zip(weights,q)],ur)
            out,n=couple_one_step(s,W,h,q,x,u,ur)
            self.assertEqual(out,ref);self.assertLessEqual(n,len(W));count('coupled_sampling_cases')

    def test_rejection_law(self):
        for _ in range(300):
            n=R.randint(2,8);p=law(n);q=law(n)
            self.assertEqual(rejection_law(p,q),p);count('rejection_law_pairs')

    def test_verification_frontier(self):
        for n in range(1,7):
            for states in product(('accept','reject','unknown'),repeat=n):
                leading,r=fixed_protocol_frontier(states)
                unknown=[i for i,x in enumerate(states) if x=='unknown']
                for bits in product(('accept','reject'),repeat=len(unknown)):
                    full=list(states)
                    for i,b in zip(unknown,bits):full[i]=b
                    true_reject=next((i for i,x in enumerate(full) if x=='reject'),n)
                    self.assertLessEqual(leading,true_reject);self.assertLessEqual(true_reject,r)
                    count('frontier_completions')
                count('partial_frontier_states')

    def test_sample_dependent_depth_bias(self):
        p=q=(Q(1,2),Q(1,2));out=[Q(),Q()]
        for x in range(2):
            if x==0:
                for y in range(2):out[y]+=q[x]*p[y]
            else:out[x]+=q[x] # ordinary p/q test always accepts
        self.assertEqual(tuple(out),conditional_depth_bias());self.assertNotEqual(tuple(out),p)
        WITNESSES['candidate_dependent_depth']={'p':['1/2','1/2'],'q':['1/2','1/2'],'output':list(map(str,out))}
        # The same construction at the second token survives a mandatory first
        # token: conditioning on either first token gives the biased second law.
        joint={(x,y):p[x]*out[y] for x in range(2) for y in range(2)}
        self.assertEqual(sum(v for (x,y),v in joint.items() if y==1),Q(3,4))
        count('depth_bias_witnesses',2)

    def test_map_scan(self):
        for _ in range(400):
            K=R.randint(1,16);L=R.randint(1,20);maps=[tuple(R.randrange(K) for _ in range(K)) for t in range(L)]
            a=R.randrange(K)
            self.assertEqual(scan_maps(maps,a),serial_maps(maps,a));count('map_scan_cases')
            f,g,h=[tuple(R.randrange(K) for _ in range(K)) for t in range(3)]
            self.assertEqual(compose_maps(compose_maps(f,g),h),compose_maps(f,compose_maps(g,h)))
            count('map_associativity_cases')

    def test_gdn_replay(self):
        for _ in range(150):
            D=R.randint(1,4);H=R.randint(1,3)
            S0=[[Q(R.randint(-2,2)) for j in range(D)] for i in range(H)]
            S=S0;records=[]
            for step in range(R.randint(1,10)):
                a=Q(R.randint(0,4),4);b=Q(R.randint(0,4),4)
                k=[Q(R.randint(-2,2),2) for j in range(D)];v=[Q(R.randint(-2,2),2) for i in range(H)]
                S,u=gdn_step(S,a,b,k,v);records.append((a,u,k))
                self.assertEqual(S,gdn_fold(S0,records));count('gdn_prefixes')
            count('gdn_trajectories')
        S,u=gdn_step([[Q(0)]],Q(1),Q(1),[Q(1)],[Q(1)])
        wrong=gdn_fold([[Q(2)]],[(Q(1),u,[Q(1)])])
        true,_=gdn_step([[Q(2)]],Q(1),Q(1),[Q(1)],[Q(1)])
        self.assertNotEqual(wrong,true)
        WITNESSES['derived_record_wrong_state']={'wrong':str(wrong),'true':str(true)}

    def test_epoch_and_prefix_guard(self):
        for _ in range(300):
            store=EpochStore();slot=R.randint(0,5);old=store.allocate(slot,13,2)
            current=store.allocate(slot,23,2)
            self.assertFalse(store.commit(old,99,3));self.assertTrue(store.commit(current,24,3))
            self.assertFalse(store.commit(current,99,4));self.assertEqual(store.state[slot],(24,3))
            count('lease_reuse_schedules')

    def test_cost_counterexamples(self):
        rewards=[[Q(1),Q(19,10),Q(27,10)]];costs=[Q(1),Q(2),Q(2)]
        best=snapshot_budget(rewards,lambda d:costs[d[0]])
        self.assertEqual(best[0],(2,));self.assertLess(rewards[0][1]/costs[1],rewards[0][0]/costs[0])
        rewards=[[Q(1),Q(2)],[Q(1),Q(3,2)]]
        costs={(0,0):Q(1),(1,0):Q(10),(0,1):Q(1),(1,1):Q(11)}
        true=snapshot_budget(rewards,lambda d:costs[d]);self.assertEqual(true[0],(0,1))
        rates=(Q(1,1),Q(1,10));ratio=Q(2,11);self.assertNotEqual(sum(rates)/2,ratio)
        WITNESSES['graph_plateau']={'depth':best[0][0],'reward':str(best[1]),'cost':str(best[2])}
        WITNESSES['same_total_different_cost']={str(k):str(v) for k,v in costs.items()}
        count('cost_counterexamples',3)

    def test_namespace(self):
        W,h0,h=head();s=make_summary(W,h0,4)
        with self.assertRaises(ValueError):transported(s,h,'changed-adapter')
        with self.assertRaises(ValueError):certify_greedy(s,TargetOracle(W,h),0,'changed-mask')
        count('namespace_rejections',2)

    def test_batch_union(self):
        # Exhaustive independent Bernoulli activation, tiny illustrative batch.
        for B in range(1,9):
            f=Q(1,4);unused=Q()
            for flags in product((0,1),repeat=B):
                pr=Q(1)
                for bit in flags:pr*=f if bit else 1-f
                if not any(flags):unused+=pr
            self.assertEqual(1-unused,1-(1-f)**B);count('union_models')
        WITNESSES['batch_union_B32_f1over16']=float(1-(Q(15,16))**32)

    def test_floating_and_support_counterexamples(self):
        import numpy as np
        # Casting a genuine upper real bound downward invalidates certification.
        exact=Q(2**24+1);computed=np.float32(2**24+1)
        self.assertLess(Q(int(computed)),exact)
        # A full-support target and top-1 truncation are distinct distributions.
        p=(Q(3,4),Q(1,4));trunc=(Q(1),Q())
        tv=sum(abs(x-y) for x,y in zip(p,trunc))/2;self.assertEqual(tv,Q(1,4))
        # Partial D comparisons can eliminate the eventual winner.
        partials=((6,-10),(5,5));self.assertEqual(max(range(2),key=lambda i:partials[i][0]),0)
        self.assertEqual(max(range(2),key=lambda i:sum(partials[i])),1)
        WITNESSES['rounded_bound']={'real':str(exact),'float32':float(computed)}
        WITNESSES['truncated_target_TV']=str(tv);count('numeric_support_counterexamples',3)

if __name__=='__main__':
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Decisions)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    import numpy as np
    report={'status':'pass' if result.wasSuccessful() else 'fail','test_methods':result.testsRun,
            'seed':20260930,'python':platform.python_version(),'numpy':np.__version__,
            'counts':COUNTS,'witnesses':WITNESSES,'seconds':time.perf_counter()-start,
            'scope':'bounded synthetic integer logits with exact rational base-2 masses; no GPU, model weights or trained predictor'}
    (ROOT/'evidence'/'decision_tests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));sys.exit(not result.wasSuccessful())
