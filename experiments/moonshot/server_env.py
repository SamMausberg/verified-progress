"""Record each server's environment for the prefixes that change SGLang's numerics or kernels.

`RecordingServer` is bench's Server with one addition: right after launch it reads the
server process's initial environment (/proc/<pid>/environ) and stores every variable whose
name starts with one of PREFIXES in the launch record (`launch.json`, key `env_prefixed`).
A validator can then require that set to equal the one declared for the arm, whatever the
calling shell held.
"""

from __future__ import annotations

from pathlib import Path

from bench.server import Server

PREFIXES = ('SGLANG_', 'FLASHINFER_', 'TRITON_', 'TORCH_', 'PYTORCH_', 'NCCL_')


def prefixed(env: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in sorted(env.items()) if key.startswith(PREFIXES)}


def process_environment(pid: int) -> dict[str, str]:
    raw = Path(f'/proc/{pid}/environ').read_bytes()
    pairs = (item.split(b'=', 1) for item in raw.split(b'\0') if b'=' in item)
    return {key.decode(): value.decode(errors='replace') for key, value in pairs}


class RecordingServer(Server):
    def start(self) -> None:
        super().start()
        assert self.proc is not None
        try:
            env, source = process_environment(self.proc.pid), f'/proc/{self.proc.pid}/environ'
        except OSError:  # the process already exited; record what was passed to it
            env, source = self.environment(), 'Server.environment()'
        self.launch_record['env_prefixed'] = prefixed(env)
        self.launch_record['env_prefixed_source'] = source
        self._write_json('launch.json', self.launch_record)
