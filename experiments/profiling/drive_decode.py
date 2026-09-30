"""Hold C concurrent long greedy generations and profile a steady-state window.

The driver sends C streaming ``/generate`` requests at once (chat-templated
prompts, ``temperature=0``, ``ignore_eos=True``, a large ``max_new_tokens``),
waits until every request has streamed at least ``--warmup-tokens`` tokens so
all prefills are done, lets decode settle, and then opens a profiling window
of ``--window`` seconds. Every request is still running when the window
closes, so the batch size is exactly C for the whole window. The driver then
cancels the requests (SGLang aborts a streaming request when its client
disconnects), sends ``/abort_request`` with ``abort_all`` and waits
``--drain`` seconds for the scheduler to empty.

``meta_info.completion_tokens`` in each streamed chunk gives exact
per-request token counts, so the window's output tokens per second is
measured by the client. SGLang reports ``spec_verify_ct`` only when a request
finishes, so for speculative arms the acceptance length comes from the
server's ``Decode batch`` log lines inside the window (``run_profiles.py``).

Profiler control (``--profiler``):

* ``none``: no profiler; the window only measures throughput.
* ``nsys``: ``nsys start``/``nsys stop`` against a session created by
  ``nsys launch --session-new=<name>`` (see ``run_profiles.py``).
* ``sglang``: SGLang's ``/start_profile`` with the ``CUDA_PROFILER``
  activity (``cudaProfilerStart``/``Stop`` in the scheduler) for
  ``--profile-steps`` forward steps, under
  ``nsys profile --capture-range=cudaProfilerApi``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

MODEL_DIR = Path.home() / (
    '.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/'
    '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
)

TOPICS = (
    'the history of the printing press',
    'how a modern CPU executes out-of-order instructions',
    'the water cycle and its effect on regional climate',
    'designing a relational database schema for a library',
    'the causes and consequences of the French Revolution',
    'how vaccines train the adaptive immune system',
    'implementing a hash map with open addressing in C',
    'the economics of renewable energy auctions',
    'the plot and themes of a classic detective novel',
    'photosynthesis at the level of the electron transport chain',
    'writing a recursive descent parser for arithmetic',
    'urban planning trade-offs between density and green space',
    'how GPS receivers compute their position',
    'the life cycle of a star from nebula to remnant',
    'the difference between TCP congestion control algorithms',
    'baking sourdough bread from starter to crust',
    'the mathematics of public-key cryptography',
    'how glaciers shape valleys and fjords',
    'the rise and fall of the Roman Republic',
    'training a small convolutional network on images',
    'how a bill becomes law in a parliamentary system',
    'the physics of how airplanes generate lift',
    'microservices versus monoliths for a growing startup',
    'the biology of sleep and circadian rhythms',
    "a beginner's guide to linear regression",
    "the architecture of a web browser's rendering engine",
    'the causes of the 2008 financial crisis',
    'how compilers perform register allocation',
    'coral reef ecosystems and ocean acidification',
    'writing unit tests for a date-parsing library',
    'the development of the periodic table',
    'planning a two-week trip through Japan by rail',
)
TASKS = (
    'Write a detailed, well-structured essay about {t}. Use several sections.',
    'Explain {t} step by step to a curious university student, with examples.',
    'Write a long technical tutorial on {t}, including common mistakes.',
    'Give a thorough overview of {t}, then discuss open questions and debates.',
)


def build_prompts(n: int) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    prompts = []
    for i in range(n):
        content = TASKS[(i // len(TOPICS)) % len(TASKS)].format(t=TOPICS[i % len(TOPICS)])
        prompts.append(
            tok.apply_chat_template(
                [{'role': 'user', 'content': content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


@dataclass
class ReqState:
    tokens: int = 0
    done: bool = False
    error: str | None = None
    history: list[tuple[float, int]] = field(default_factory=list)


async def stream_one(
    session: aiohttp.ClientSession, url: str, prompt: str, max_tokens: int, state: ReqState
) -> None:
    payload = {
        'text': prompt,
        'stream': True,
        'sampling_params': {
            'temperature': 0.0,
            'max_new_tokens': max_tokens,
            'ignore_eos': True,
        },
    }
    try:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                body = line[5:].strip()
                if body == '[DONE]':
                    break
                meta = json.loads(body).get('meta_info', {})
                state.tokens = int(meta.get('completion_tokens', state.tokens))
                state.history.append((time.perf_counter(), state.tokens))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        state.error = repr(exc)
    finally:
        state.done = True


def tokens_at(states: list[ReqState], at: float) -> int:
    """Output tokens received by time ``at``, summed over requests."""
    total = 0
    for s in states:
        received = 0
        for when, tokens in s.history:
            if when > at:
                break
            received = tokens
        total += received
    return total


class Profiler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    async def start(self, session: aiohttp.ClientSession) -> None:
        a = self.args
        if a.profiler == 'nsys':
            cmd = ['nsys', 'start', f'--session={a.nsys_session}', '-o', a.output, '-f', 'true']
            await asyncio.to_thread(subprocess.run, cmd, check=True)
        elif a.profiler == 'sglang':
            body = {'activities': ['CUDA_PROFILER'], 'num_steps': a.profile_steps}
            async with session.post(f'{a.base}/start_profile', json=body) as resp:
                print('start_profile:', resp.status, (await resp.text())[:200], flush=True)

    async def stop(self, session: aiohttp.ClientSession) -> None:
        a = self.args
        if a.profiler == 'nsys':
            cmd = ['nsys', 'stop', f'--session={a.nsys_session}']
            await asyncio.to_thread(subprocess.run, cmd, check=True)


async def abort_all(session: aiohttp.ClientSession, base: str) -> int:
    async with session.post(f'{base}/abort_request', json={'abort_all': True}) as resp:
        return resp.status


async def run(args: argparse.Namespace) -> dict:
    prompts = build_prompts(args.concurrency)
    states = [ReqState() for _ in prompts]
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t_submit = time.perf_counter()
        tasks = [
            asyncio.create_task(stream_one(session, f'{args.base}/generate', p, args.max_tokens, s))
            for p, s in zip(prompts, states, strict=True)
        ]
        while not all(s.tokens >= args.warmup_tokens or s.done for s in states):
            await asyncio.sleep(0.05)
        t_all_decoding = time.perf_counter()
        await asyncio.sleep(args.settle)

        load_before = os.getloadavg()
        profiler = Profiler(args)
        t_before_start = time.perf_counter()
        await profiler.start(session)
        t0 = time.perf_counter()
        wall0 = time.time()
        await asyncio.sleep(args.window)
        t1 = time.perf_counter()
        wall1 = time.time()
        load_after = os.getloadavg()
        await profiler.stop(session)
        t_after_stop = time.perf_counter()

        finished_early = [i for i, s in enumerate(states) if s.done]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        abort_status = await abort_all(session, args.base)
        await asyncio.sleep(args.drain)

    tok0 = tokens_at(states, t0)
    tok1 = tokens_at(states, t1)
    dt = t1 - t0
    result = {
        'concurrency': args.concurrency,
        'max_new_tokens': args.max_tokens,
        'profiler': args.profiler,
        'output': args.output,
        'window_s': dt,
        'window_wall_start': wall0,
        'window_wall_end': wall1,
        # Host load matters for host-bound phases (speculative cycles); other jobs'
        # CPU work during a window would inflate host gaps.
        'host_loadavg_1m_before': load_before[0],
        'host_loadavg_1m_after': load_after[0],
        'profiler_start_s': t0 - t_before_start,
        'profiler_stop_s': t_after_stop - t1,
        'prefill_to_all_decoding_s': t_all_decoding - t_submit,
        'window_output_tokens': tok1 - tok0,
        'output_tokens_per_s': (tok1 - tok0) / dt,
        'output_tokens_per_s_per_user': (tok1 - tok0) / dt / args.concurrency,
        'mean_completion_tokens_at_window_start': tok0 / args.concurrency,
        'requests_finished_before_window_end': finished_early,
        'errors': [s.error for s in states if s.error and 'Cancelled' not in s.error],
        'abort_all_status': abort_status,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--port', type=int, default=30020)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--warmup-tokens', type=int, default=16)
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--window', type=float, default=3.0)
    parser.add_argument('--drain', type=float, default=3.0, help='seconds to wait after abort')
    parser.add_argument('--profiler', choices=('none', 'nsys', 'sglang'), default='none')
    parser.add_argument('--nsys-session', default='vp_profile')
    parser.add_argument('--profile-steps', type=int, default=200)
    parser.add_argument('--output', default='', help='nsys report path (nsys mode)')
    parser.add_argument('--summary', type=Path, help='append the JSON summary to this file')
    args = parser.parse_args()
    args.base = f'http://127.0.0.1:{args.port}'
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2), flush=True)
    if args.summary:
        with args.summary.open('a') as fh:
            fh.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
