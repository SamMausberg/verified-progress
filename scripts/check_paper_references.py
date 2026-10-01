#!/usr/bin/env python3
"""Check the paper's bibliography and evidence register against the repository.

Four checks, each printed with its failures:

1. every key the paper cites has an entry in ``paper/references.bib``;
2. every entry in ``paper/references.bib`` is cited;
3. each of those entries is identical to its entry in ``sources/references_full.bib``,
   the verified bibliography of the literature review, whose keys must equal those of
   ``sources/source_manifest.json``;
4. every path the evidence register lists (``\\evitem`` in the appendix) exists at a git
   revision, ``origin/main`` by default, because the paper may cite only merged evidence.

Usage: ``python scripts/check_paper_references.py [--rev REV]``. Exit status 1 if any
check fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
REGISTER = PAPER / 'sections' / 'app_register.tex'

CITE = re.compile(r'\\(?:cite[a-z]*|yrcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}')
ENTRY_START = re.compile(r'(?m)^@(\w+)\s*\{\s*([^,\s]+)\s*,')
PATH = re.compile(r'\\nolinkurl\{([^}]*)\}')


def strip_comments(text: str) -> str:
    return re.sub(r'(?m)(?<!\\)%.*$', '', text)


def cited_keys() -> set[str]:
    sources = [PAPER / 'paper.tex', PAPER / 'terminology.tex']
    sources += sorted((PAPER / 'sections').glob('*.tex'))
    sources += sorted((PAPER / 'figures').glob('*.tex'))
    keys: set[str] = set()
    for path in sources:
        for match in CITE.finditer(strip_comments(path.read_text())):
            keys.update(k.strip() for k in match.group(1).split(',') if k.strip())
    return keys


def bib_entries(path: Path) -> dict[str, str]:
    """Map each key to the exact text of its entry, from ``@`` to the closing brace."""
    text = path.read_text()
    entries: dict[str, str] = {}
    for match in ENTRY_START.finditer(text):
        depth = 0
        for end in range(text.index('{', match.start()), len(text)):
            depth += {'{': 1, '}': -1}.get(text[end], 0)
            if depth == 0:
                entries[match.group(2)] = text[match.start() : end + 1]
                break
    return entries


def register_paths() -> list[tuple[str, str]]:
    """(entry key, path) for every file or directory the register lists.

    Within one entry a bare file name is relative to the directory of the previous path
    that has one, as the register's note says.
    """
    text = REGISTER.read_text()
    start = text.index('\\subsection{Evidence register}')
    end = text.index('\\subsection', start + 1)
    found: list[tuple[str, str]] = []
    for line in text[start:end].splitlines():
        if not line.startswith('\\evitem{'):
            continue
        key = line[len('\\evitem{') : line.index('}')]
        base: str | None = None
        for raw in PATH.findall(line):
            if '/' in raw.rstrip('/'):
                full = raw
                base = raw.rstrip('/') if raw.endswith('/') else raw.rsplit('/', 1)[0]
            else:
                full = f'{base}/{raw}' if base else raw
            found.append((key, full))
    return found


def tree_paths(rev: str) -> set[str]:
    names = subprocess.run(
        ['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', rev],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    paths = set(names)
    for name in names:
        parts = name.split('/')
        paths.update('/'.join(parts[:i]) + '/' for i in range(1, len(parts)))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rev', default='origin/main', help='revision the evidence must exist at')
    args = parser.parse_args()

    failures: dict[str, list[str]] = {}
    cited = cited_keys()
    paper_bib = bib_entries(PAPER / 'references.bib')
    full_bib = bib_entries(ROOT / 'sources' / 'references_full.bib')
    manifest = json.loads((ROOT / 'sources' / 'source_manifest.json').read_text())
    manifest_keys = {source['key'] for source in manifest['sources']}

    failures['cited but not in paper/references.bib'] = sorted(cited - set(paper_bib))
    failures['in paper/references.bib but not cited'] = sorted(set(paper_bib) - cited)
    failures['differs from sources/references_full.bib'] = sorted(
        key for key, text in paper_bib.items() if full_bib.get(key) != text
    )
    failures['keys differ between references_full.bib and the manifest'] = sorted(
        set(full_bib) ^ manifest_keys
    )
    existing = tree_paths(args.rev)
    listed = register_paths()
    failures[f'register paths missing at {args.rev}'] = [
        f'{key}: {path}' for key, path in listed if path not in existing
    ]

    print(f'{len(cited)} cited keys, {len(paper_bib)} paper entries, {len(listed)} register paths')
    for check, items in failures.items():
        print(f'{check}: {"none" if not items else ", ".join(items)}')
    return 1 if any(failures.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
