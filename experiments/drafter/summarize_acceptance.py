"""Collect acceptance-by-position tables from accept_probe.py runs into CSVs.

Each `--run LABEL:BLOCK:DIR` names a probe output directory (with summary.json)
made with the given drafter label and block size. Writes

    acceptance_by_position.csv  drafter, block_size, position, domain,
                                accept_prob (alpha_k), survival (S(k)), n
    acceptance_summary.csv      drafter, block_size, domain, requests,
                                completion_tokens, cycles, tau_mean_per_request,
                                tau_pooled

where n is the number of verify cycles that reached position k (the
denominator of alpha_k). tau_mean_per_request is the model card's accept length
(completion tokens / verify cycles per request, averaged over requests);
tau_pooled divides the domain's total tokens by its total verify cycles.

    python experiments/drafter/summarize_acceptance.py \
        --run zlab:16:RUN16 --run zlab:8:RUN8 --out evidence/drafter
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def per_request_tau(directory: Path, domain: str) -> float | None:
    """Mean over requests of completion_tokens / spec_verify_ct (the card's metric)."""
    rows = [json.loads(line) for line in (directory / 'requests.jsonl').read_text().splitlines()]
    values = [
        r['completion_tokens'] / r['spec_verify_ct']
        for r in rows
        if r['spec_verify_ct'] and (domain == 'all' or r['domain'] == domain)
    ]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', action='append', required=True, help='LABEL:BLOCK:DIR')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    positions, summary = [], []
    for spec in args.run:
        label, block, directory = spec.split(':', 2)
        data = json.loads((Path(directory) / 'summary.json').read_text())
        for domain, entry in data['domains'].items():
            acc = entry['acceptance']
            cycles = acc.get('cycles', 0)
            summary.append(
                {
                    'drafter': label,
                    'block_size': int(block),
                    'domain': domain,
                    'requests': entry['requests'],
                    'completion_tokens': entry['completion_tokens'],
                    'cycles': cycles,
                    'tau_mean_per_request': entry.get('accept_length_mean_per_request')
                    if 'accept_length_mean_per_request' in entry
                    else per_request_tau(Path(directory), domain),
                    'tau_pooled': entry['accept_length'],
                }
            )
            survival = acc.get('survival') or []
            alpha = acc.get('alpha') or []
            for k in range(1, len(survival)):
                reached = round(survival[k - 1] * cycles)
                positions.append(
                    {
                        'drafter': label,
                        'block_size': int(block),
                        'position': k,
                        'domain': domain,
                        'accept_prob': alpha[k],
                        'survival': survival[k],
                        'n': reached,
                    }
                )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ('acceptance_by_position.csv', positions),
        ('acceptance_summary.csv', summary),
    ):
        with (args.out / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'wrote {len(positions)} position rows and {len(summary)} summary rows to {args.out}')


if __name__ == '__main__':
    main()
