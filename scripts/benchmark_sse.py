#!/usr/bin/env python3
"""Closed-loop text-only SSE benchmark client with explicit token accounting.

Uses an already running OpenAI-compatible chat-completions endpoint. It never
launches/reconfigures a server, changes GPU settings or estimates token counts
from chunks. Requires streamed usage.completion_tokens and terminal [DONE].
This is a small auxiliary harness; use a pinned AIPerf as the main comparator.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, os, statistics, sys, time
from pathlib import Path
from typing import Iterable, Any
from urllib.request import Request, urlopen


def events(lines: Iterable[bytes]):
    """Yield SSE data fields, joining multiline data according to SSE framing."""
    pending=[]
    for raw in lines:
        line=raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if pending:yield '\n'.join(pending);pending=[]
        elif line.startswith('data:'):pending.append(line[5:].lstrip(' '))
        # comments, event/id/retry fields do not contain completion payloads
    if pending:yield '\n'.join(pending)


def one_request(url:str,model:str,item:dict[str,Any],index:int,timeout:float,
                temperature:float=0.0,key:str|None=None)->dict[str,Any]:
    prompt=item['prompt'];start=time.perf_counter();first=None;last=None;chunks=[]
    usage=None;done=False;finish=None
    record={'index':index,'prompt_id':str(item.get('id',index)),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'send_s':start,'success':False}
    try:
        payload={'model':model,'messages':[{'role':'user','content':prompt}],
                 'max_tokens':int(item.get('max_tokens',256)), 'temperature':temperature,
                 'stream':True,'stream_options':{'include_usage':True}}
        headers={'Content-Type':'application/json','Accept':'text/event-stream'}
        if key:headers['Authorization']='Bearer '+key
        req=Request(url,json.dumps(payload).encode(),headers,method='POST')
        with urlopen(req,timeout=timeout) as response:
            for data in events(response):
                now=time.perf_counter()
                if data.strip()=='[DONE]':done=True;break
                obj=json.loads(data)
                if 'error' in obj:raise RuntimeError('server error: '+str(obj['error'])[:500])
                if obj.get('usage') is not None:usage=obj['usage']
                for choice in obj.get('choices',[]):
                    delta=choice.get('delta') or {}
                    nonempty=any(isinstance(delta.get(k),str) and bool(delta[k])
                                 for k in ('content','reasoning_content','reasoning'))
                    if nonempty:
                        if first is None:first=now
                        last=now;chunks.append(now-start)
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
        if not done:raise RuntimeError('stream ended without terminal [DONE]')
        n=usage.get('completion_tokens') if isinstance(usage,dict) else None
        if isinstance(n,bool) or not isinstance(n,int) or n<=0:
            raise RuntimeError('missing positive usage.completion_tokens; refusing to count chunks as tokens')
        if first is None or last is None:raise RuntimeError('no nonempty text/reasoning event; text-only contract')
        record.update(success=True,completion_tokens=n,usage=usage,finish_reason=finish,
                      ttft_s=first-start,last_output_s=last,output_e2e_s=last-start,
                      end_to_end_user_tokens_s=n/(last-start),chunk_offsets_s=chunks)
    except Exception as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
        record['partial_chunk_offsets_s']=chunks
    record['done_s']=time.perf_counter();record['request_latency_s']=record['done_s']-start
    return record


def aggregate(records:list[dict],concurrency:int):
    ok=[x for x in records if x['success']]
    elapsed=max(x['done_s'] for x in records)-min(x['send_s'] for x in records)
    total=sum(x['completion_tokens'] for x in ok)
    return {'concurrency':concurrency,'requests':len(records),'completed':len(ok),
            'failed':len(records)-len(ok),'common_wall_s':elapsed,
            'completed_output_tokens':total,'output_tokens_s':total/elapsed,
            'mean_end_to_end_user_tokens_s':statistics.mean(x['end_to_end_user_tokens_s'] for x in ok) if ok else None,
            'median_ttft_s':statistics.median(x['ttft_s'] for x in ok) if ok else None,
            'median_request_latency_s':statistics.median(x['request_latency_s'] for x in ok) if ok else None,
            'valid_comparison':len(ok)==len(records),
            'token_accounting':'server usage.completion_tokens; includes any tokens counted by that field',
            'user_rate_definition':'arithmetic mean completion_tokens/(last nonempty output event - request send); includes TTFT',
            'note':'no decode-only TPOT or per-token timing inferred from SSE chunks; failed requests invalidate matched comparisons'}


def run(url,model,items,concurrency,timeout=120.,temperature=0.,key=None):
    if not items or concurrency<1:raise ValueError('empty workload or nonpositive concurrency')
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(one_request,url,model,it,i,timeout,temperature,key) for i,it in enumerate(items)]
        records=[f.result() for f in futures]
    records.sort(key=lambda x:x['index'])
    return aggregate(records,concurrency),records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:30000/v1/chat/completions')
    p.add_argument('--model',required=True);p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4,8,16,32])
    p.add_argument('--timeout',type=float,default=120.);p.add_argument('--temperature',type=float,default=0.)
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.repeat<1 or a.timeout<=0 or any(c<1 for c in a.concurrency):p.error('positive repeat/timeout/concurrency required')
    items=[json.loads(s) for s in a.workload.read_text().splitlines() if s.strip()]
    if not items:p.error('empty workload')
    for it in items:
        if not isinstance(it.get('prompt'),str) or not it['prompt']:p.error('each row needs a nonempty prompt')
        if not isinstance(it.get('max_tokens',256),int) or it.get('max_tokens',256)<=0:p.error('invalid max_tokens')
    a.out.mkdir(parents=True,exist_ok=True);summaries=[]
    for repeat in range(a.repeat):
        for c in a.concurrency:
            summary,records=run(a.url,a.model,items,c,a.timeout,a.temperature,os.getenv('BENCH_API_KEY'))
            stem=f'r{repeat:02d}_c{c:03d}'
            (a.out/(stem+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in records))
            summary['repeat']=repeat;summaries.append(summary);print(json.dumps(summary))
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return 0 if all(s['valid_comparison'] for s in summaries) else 1
if __name__=='__main__':sys.exit(main())
