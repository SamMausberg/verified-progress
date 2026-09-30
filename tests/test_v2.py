"""Finite CPU evidence. Run from bundle root: python tests/test_v2.py."""
from __future__ import annotations
import sys,json,random,math,platform,time
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v2_reference import *
R=random.Random(20260930)
NR=np.random.default_rng(20260930)
COUNTS={}
WITNESSES={}

def count(key,n=1): COUNTS[key]=COUNTS.get(key,0)+n

def law(n):
    x=[R.randrange(10) for _ in range(n)]
    if not sum(x): x[0]=1
    return tuple(F(y,sum(x)) for y in x)

def lattice(L,K):
    return tuple(tuple(tuple(law(K+1)[:K]) for a in range(K)) for t in range(L))

class Evidence(unittest.TestCase):
    def test_lazy_selector(self):
        for case in range(400):
            L=R.randint(1,8);K=R.choice([1,2,3,4,8,16]);rank=R.choice([1,4,16,64])
            A=NR.integers(-3,4,(L,K,rank),dtype=np.int64)
            H=NR.integers(-3,4,(L,rank),dtype=np.int64)
            B=NR.integers(-3,4,(L,K,rank),dtype=np.int64)
            u=NR.integers(-8,9,(L,K),dtype=np.int64)
            if case%20==0: A[:]=0;u[:]=0 # adversarial exact ties
            S=score_all(A,H,B,u)
            uniforms=NR.uniform(0,1,L);uniforms[0]=0 if case%2 else np.nextafter(1.,0.)
            for greedy in (True,False):
                p,q=walk_full(S,greedy,uniforms,.7)
                pp,qq=walk_lazy(A,H,B,u,greedy,uniforms,.7)
                self.assertEqual(p,pp);np.testing.assert_array_equal(q,qq)
                np.testing.assert_allclose(q.sum(1),1,rtol=0,atol=1e-15)
                count('lazy_vs_full_paths')
        with self.assertRaises(ValueError): walk_full(np.full((1,2,2),np.nan))

    def test_tiled_topk(self):
        for _ in range(800):
            V=R.randint(1,160);K=R.randint(1,min(V,20));tile=R.randint(1,50)
            values=[R.randint(-20,20) for _ in range(V)]
            if R.random()<.3: values[R.randrange(V)]=-math.inf
            self.assertEqual(stable_topk(values,K),tiled_topk(values,K,tile))
            count('topk_tile_merges')
        for _ in range(200):
            V=R.randint(2,100);D=R.randint(1,32);K=R.randint(1,min(V,12))
            W=NR.integers(-5,6,(V,D),dtype=np.int64);h=NR.integers(-5,6,D,dtype=np.int64)
            self.assertEqual(stable_topk((W@h).tolist(),K),projection_topk(h,W,K,R.randint(1,31)))
            count('streamed_complete_dot_projection')
        W=np.array([[10,-100],[-100,10],[6,6]],dtype=np.int64)
        local={int(np.argmax(W[:,j])) for j in range(2)}
        self.assertNotIn(int(np.argmax(W@np.ones(2,dtype=np.int64))),local)
        WITNESSES['partial_dot_pruning']={'weights':W.tolist(),'true_winner':2,'partial_winners':sorted(local)}
        # Keeping k-1 per tile is not a legal reduction.
        self.assertNotEqual(stable_topk([10,9,1,0],2),stable_topk([10,1],2,[0,2]))
        with self.assertRaises(ValueError): stable_topk([math.nan,1],1)

    def test_lse_merge(self):
        for _ in range(400):
            x=NR.uniform(-1000,1000,R.randint(1,100)).tolist()
            x[R.randrange(len(x))]=-math.inf
            cut=R.randrange(len(x)+1)
            m,z=merge_lse(lse_summary(x[:cut]),lse_summary(x[cut:]))
            mm,zz=lse_summary(x)
            if z==0: self.assertEqual((m,z),(mm,zz))
            else: self.assertAlmostEqual(m+math.log(z),mm+math.log(zz),places=10)
            count('lse_merges')
        self.assertEqual(merge_lse((-math.inf,0),(-math.inf,0)),(-math.inf,0))

    def test_prefix_dp_and_perturbation(self):
        for _ in range(240):
            L=R.randint(1,5);K=R.randint(1,3);r=lattice(L,K);s=lattice(L,K)
            p,value=prefix_optimum(r)
            paths=list(product(range(K),repeat=L))
            self.assertEqual(value,max(prefix_reward(r,x) for x in paths))
            self.assertEqual(value,prefix_reward(r,p))
            eps=[max(abs(r[t][a][b]-s[t][a][b]) for a in range(K) for b in range(K)) for t in range(L)]
            delta=sum((L-t)*e for t,e in enumerate(eps))
            for x in paths:
                self.assertLessEqual(abs(prefix_reward(r,x)-prefix_reward(s,x)),delta)
            phat,_=prefix_optimum(s)
            self.assertLessEqual(value-prefix_reward(r,phat),2*delta)
            count('bellman_lattices');count('bellman_exhaustive_paths',len(paths))

    def test_scheduler(self):
        for _ in range(300):
            B=R.randint(1,4);L=R.randint(1,4)
            rewards=[]
            for i in range(B):
                surv=F(1);vals=[F(1)]
                for d in range(L):
                    surv*=F(R.randint(0,10),10);vals.append(vals[-1]+surv)
                rewards.append(vals)
            # Positive arbitrary graph-tier-like nondecreasing costs.
            costs=[F(10)];c=10
            for m in range(B*L):
                c+=R.choice([0,0,0,1,4,9]);costs.append(F(c))
            plan,g,t=budget_plan(rewards,costs)
            allplans=list(product(range(L+1),repeat=B))
            optimum=max(sum(rewards[i][d] for i,d in enumerate(p))/costs[sum(p)] for p in allplans)
            self.assertEqual(g/t,optimum)
            self.assertEqual(g,sum(rewards[i][d] for i,d in enumerate(plan)))
            count('scheduler_snapshots');count('scheduler_exhaustive_plans',len(allplans))
        gains=[[F(1),F(19,10),F(27,10)]];costs=[F(1),F(2),F(2)]
        p,g,t=budget_plan(gains,costs)
        self.assertEqual(p,(2,));self.assertLess(gains[0][1]/costs[1],gains[0][0]/costs[0])
        WITNESSES['graph_plateau_greedy_stop']={'rewards':[1,1.9,2.7],'costs':[1,2,2],'optimal_depth':2}

    def test_rejection_law(self):
        for _ in range(600):
            n=R.randint(2,8);p=law(n);q=law(n)
            self.assertEqual(tuple(p),rejection_output(p,q))
            count('exact_rejection_distribution_pairs')
        p=(F(3,4),F(1,4));actual=(F(0),F(1));reported=(F(1,2),F(1,2))
        wrong=rejection_output(p,actual,reported)
        self.assertNotEqual(wrong,p)
        self.assertEqual(rejection_output(p,actual),p)
        WITNESSES['wrong_proposal_law']={'target':['3/4','1/4'],'actual_proposal':[0,1],'wrong_reported_proposal':['1/2','1/2'],'wrong_output':[str(x) for x in wrong]}

    def test_state_replay(self):
        for _ in range(160):
            nv=R.randint(1,3);nk=R.randint(1,3)
            state=tuple(tuple(F(R.randint(-2,2)) for j in range(nk)) for i in range(nv))
            ring=ReplayState(state,R.randint(1,8),[])
            for cycle in range(15):
                length=R.randint(1,8)
                recs=[(F(R.randint(0,4),4),tuple(F(R.randint(-2,2)) for i in range(nv)),tuple(F(R.randint(-2,2)) for j in range(nk))) for _ in range(length)]
                snapshots=[];tmp=state
                for record in recs: tmp=apply_update(tmp,record);snapshots.append(tmp)
                self.assertEqual(ring.verify(recs),snapshots)
                accepted=R.randint(0,length)
                expected=state if accepted==0 else snapshots[accepted-1]
                ring.commit(recs,accepted);state=expected
                self.assertEqual(ring.current(),state)
                self.assertLessEqual(len(ring.records),ring.capacity)
                count('toy_replay_transactions')
            count('toy_replay_flushes',ring.flushes)

    def test_certified_screening(self):
        for _ in range(300):
            C=R.randint(2,12);size=R.randint(2,12);D=R.randint(1,16)
            centers=NR.integers(-30,31,(C,D),dtype=np.int64)
            groups=np.repeat(np.arange(C),size)
            W=centers[groups]+NR.integers(-3,4,(C*size,D),dtype=np.int64)
            h=NR.integers(-5,6,D,dtype=np.int64);k=R.randint(1,min(10,C*size))
            result,computed,skipped=certified_l1_screen(h,W,groups,centers,k)
            self.assertEqual(result,stable_topk((W@h).tolist(),k))
            self.assertEqual(computed+skipped,len(W))
            count('certified_integer_screen_instances');count('screen_toy_rows_evaluated',computed);count('screen_toy_rows_skipped',skipped)
        # Loose bounds can eliminate zero useful work; no unconditional savings.
        W=np.array([[1,0],[-1,0],[0,1],[0,-1]],dtype=np.int64)
        result,computed,skipped=certified_l1_screen(np.array([1,1]),W,np.array([0,0,1,1]),np.zeros((2,2),dtype=np.int64),1)
        self.assertEqual(skipped,0)
        WITNESSES['loose_bound_no_pruning']={'evaluated':computed,'skipped':skipped}

    def test_floating_and_cost_counterexamples(self):
        a,b,c=np.float32(1e20),np.float32(-1e20),np.float32(3.14)
        left=np.float32(np.float32(a+b)+c);right=np.float32(a+np.float32(b+c))
        self.assertNotEqual(left,right)
        WITNESSES['float32_reassociation']={'left':float(left),'right':float(right)}
        vals=np.array([1.0001,1.0002],dtype=np.float32)
        self.assertEqual(int(np.argmax(vals)),1)
        self.assertEqual(int(np.argmax(vals.astype(np.float16))),0)
        WITNESSES['cast_before_rank']={'fp32_winner':1,'fp16_winner':0}
        # E[R/T] is not long-run committed tokens / time.
        reward=[1.,9.];cost=[1.,3.]
        self.assertNotEqual(np.mean(np.array(reward)/cost),sum(reward)/sum(cost))
        WITNESSES['rate_ratio']={'mean_cycle_rates':2.,'ratio_of_totals':2.5}
        # More accepted progress can still mean lower throughput.
        self.assertGreater(F(6),F(5));self.assertLess(F(6,13),F(5,10))
        WITNESSES['acceptance_is_not_speed']={'progress':[5,6],'cost':[10,13]}

if __name__=='__main__':
    start=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Evidence)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    payload={'schema_version':1,'seed':20260930,'python':platform.python_version(),'numpy':np.__version__,'device':'CPU only','unit_test_methods':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'elapsed_seconds':time.perf_counter()-start,'counts':COUNTS,'counterexamples':WITNESSES,'limitations':['Synthetic instances, not target-model workloads.','No CUDA kernel was compiled or run.','No model weights, calibration or serving experiment was run.','Integer/rational equivalence is not a floating-point equivalence proof.','No Lean proof was compiled.']}
    (ROOT/'evidence').mkdir(exist_ok=True)
    (ROOT/'evidence'/'cpu_validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
