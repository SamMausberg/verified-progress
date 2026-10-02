#!/usr/bin/env python3
"""Check the paper's and the research notes' bibliography and evidence registers.

The paper (``paper/paper.tex``) and the research notes (``paper/notes/research_notes.tex``) share
``paper/references.bib``; each has its own evidence register and pending list. The checks, each
printed with its failures:

1. every key either document cites has an entry in ``paper/references.bib``;
2. every entry in ``paper/references.bib`` is cited by the paper or the notes;
3. each of those entries is identical to its entry in ``sources/references_full.bib``,
   the verified bibliography of the literature review, whose keys must equal those of
   ``sources/source_manifest.json``;
4. every path a register lists (``\\evitem``) exists at a git revision, ``origin/main`` by
   default, because the documents may cite only merged evidence;
5. in each document, every ``\\ev`` key has an ``\\evitem`` and every ``\\pend`` key a
   ``\\devitem`` in that document's register;
6. in each document, every ``\\evitem`` and ``\\devitem`` of its register is cited there, so an
   entry that only the notes need does not stay in the paper's register;
7. an entry in both registers lists the same files and producer in both.
8. the two documents print their register IDs with different prefixes (empty in the paper,
   ``N-`` in the notes), through ``\\regprefix`` in the register macros and in the notes'
   ``\\evref`` and ``\\devref``, so that an ID such as E21 never means two things;
9. every pointer of the paper into the notes, ``\\notessec{label}{number}``, names a label the
   notes define, and, if the notes have been built (``paper/research_notes.aux``), the number
   the notes print for it.

A document's sources are its root file and every file it reaches through ``\\input``.

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
# Document name -> (root file, register file), relative to paper/, where both build.
DOCUMENTS = {
    'paper': ('paper.tex', 'sections/app_register.tex'),
    'notes': ('notes/research_notes.tex', 'notes/register.tex'),
}

CITE = re.compile(r'\\(?:cite[a-z]*|yrcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}')
INPUT = re.compile(r'\\input\{([^}]*)\}')
MARKER = re.compile(r'\\(ev|pend)\{([^}]*)\}')
ENTRY_START = re.compile(r'(?m)^@(\w+)\s*\{\s*([^,\s]+)\s*,')
PATH = re.compile(r'\\nolinkurl\{([^}]*)\}')


def strip_comments(text: str) -> str:
    return re.sub(r'(?m)(?<!\\)%.*$', '', text)


def sources(root: str) -> list[str]:
    """The root file and every file it reaches through \\input, relative to paper/."""
    seen: list[str] = []
    stack = [root]
    while stack:
        name = stack.pop()
        if name in seen or not (PAPER / name).is_file():
            continue
        seen.append(name)
        for match in INPUT.finditer(strip_comments((PAPER / name).read_text())):
            target = match.group(1)
            stack.append(target if target.endswith('.tex') else f'{target}.tex')
    return seen


def cited_keys(files: list[str]) -> set[str]:
    keys: set[str] = set()
    for name in files:
        for match in CITE.finditer(strip_comments((PAPER / name).read_text())):
            # A macro parameter (#1) in a definition such as \renewcommand{\cite} is not a key.
            keys.update(k.strip() for k in match.group(1).split(',') if k.strip() and '#' not in k)
    return keys


def markers(files: list[str], register: str) -> dict[str, set[str]]:
    """Keys of \\ev and \\pend markers outside the register itself."""
    found: dict[str, set[str]] = {'ev': set(), 'pend': set()}
    for name in files:
        if name in (register, 'macros.tex'):
            continue
        for match in MARKER.finditer(strip_comments((PAPER / name).read_text())):
            found[match.group(1)].update(k.strip() for k in match.group(2).split(',') if k.strip())
    return found


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


def brace_args(text: str, start: int, count: int) -> list[str]:
    """The first ``count`` brace-delimited arguments of the macro whose name ends at ``start``."""
    args: list[str] = []
    pos = start
    while len(args) < count:
        pos = text.index('{', pos)
        depth = 0
        for end in range(pos, len(text)):
            depth += {'{': 1, '}': -1}.get(text[end], 0)
            if depth == 0:
                args.append(text[pos + 1 : end])
                pos = end + 1
                break
    return args


def register_entries(register: str) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """(\\evitem key -> (files, producer), set of \\devitem keys) of a register file."""
    text = strip_comments((PAPER / register).read_text())
    evitems: dict[str, tuple[str, str]] = {}
    for match in re.finditer(r'(?m)^\\evitem(?=\{)', text):
        key, _, files, producer = brace_args(text, match.end(), 4)
        evitems[key] = (files, producer)
    devitems = {brace_args(text, m.end(), 1)[0] for m in re.finditer(r'(?m)^\\devitem(?=\{)', text)}
    return evitems, devitems


def register_paths(evitems: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """(entry key, path) for every file or directory a register lists.

    Within one entry a bare file name is relative to the directory of the previous path
    that has one, as the register's note says.
    """
    found: list[tuple[str, str]] = []
    for key, (files, _) in evitems.items():
        base: str | None = None
        for raw in PATH.findall(files):
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


REGISTER_MACROS = 'sections/app_register_macros.tex'
PREFIX_DEF = re.compile(r'\\(?:new|renew|provide)command\{\\regprefix\}\{([^}]*)\}')


def prefix_failures(files: dict[str, list[str]]) -> list[str]:
    """Each document's register-ID prefix, which must differ between the documents."""
    failures: list[str] = []
    macros = strip_comments((PAPER / REGISTER_MACROS).read_text())
    for macro in ('evitem', 'devitem'):
        body = macros[macros.find(f'\\newcommand{{\\{macro}}}') :].split('\n', 1)[0]
        if '\\regprefix' not in body:
            failures.append(f'{REGISTER_MACROS}: \\{macro} does not print \\regprefix')
    prefixes: dict[str, str] = {}
    for doc, names in files.items():
        defined = [
            m.group(1)
            for name in names
            for m in PREFIX_DEF.finditer(strip_comments((PAPER / name).read_text()))
            if name != REGISTER_MACROS
        ]
        prefixes[doc] = defined[-1] if defined else ''
    root = strip_comments((PAPER / DOCUMENTS['notes'][0]).read_text())
    for macro in ('evref', 'devref'):
        found = re.search(r'\\renewcommand\{\\' + macro + r'\}.*', root)
        if not found or '\\regprefix' not in found.group(0):
            failures.append(f'notes: \\{macro} is not redefined to print \\regprefix')
    if prefixes['paper'] or not prefixes['notes'] or prefixes['paper'] == prefixes['notes']:
        failures.append(
            f'register prefixes paper {prefixes["paper"]!r}, notes {prefixes["notes"]!r}'
        )
    return failures


NOTESSEC = re.compile(r'\\notessec\{([^}]*)\}\{([^}]*)\}')
NEWLABEL = re.compile(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}')


def notessec_failures(files: dict[str, list[str]]) -> list[str]:
    """Pointers of the paper into the notes whose label or section number is wrong."""
    defined = {
        m.group(1)
        for name in files['notes']
        for m in re.finditer(r'\\label\{([^}]*)\}', strip_comments((PAPER / name).read_text()))
    }
    aux = PAPER / 'research_notes.aux'
    numbers = dict(NEWLABEL.findall(aux.read_text())) if aux.is_file() else {}
    failures: list[str] = []
    for name in files['paper']:
        for m in NOTESSEC.finditer(strip_comments((PAPER / name).read_text())):
            label, number = m.groups()
            if '#' in label:
                continue  # the macro's own definition
            if label not in defined:
                failures.append(f'{name}: {label} is not a label of the notes')
            elif numbers and numbers.get(label) != number:
                failures.append(
                    f'{name}: {label} is {numbers.get(label)} in the notes, not {number}'
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rev', default='origin/main', help='revision the evidence must exist at')
    args = parser.parse_args()

    failures: dict[str, list[str]] = {}
    files = {doc: sources(root) for doc, (root, _) in DOCUMENTS.items()}
    cited = {doc: cited_keys(names) for doc, names in files.items()}
    all_cited = set().union(*cited.values())
    paper_bib = bib_entries(PAPER / 'references.bib')
    full_bib = bib_entries(ROOT / 'sources' / 'references_full.bib')
    manifest = json.loads((ROOT / 'sources' / 'source_manifest.json').read_text())
    manifest_keys = {source['key'] for source in manifest['sources']}

    failures['cited but not in paper/references.bib'] = [
        f'{doc}: {key}' for doc in DOCUMENTS for key in sorted(cited[doc] - set(paper_bib))
    ]
    failures['in paper/references.bib but cited by neither document'] = sorted(
        set(paper_bib) - all_cited
    )
    failures['differs from sources/references_full.bib'] = sorted(
        key for key, text in paper_bib.items() if full_bib.get(key) != text
    )
    failures['keys differ between references_full.bib and the manifest'] = sorted(
        set(full_bib) ^ manifest_keys
    )

    existing = tree_paths(args.rev)
    registers = {doc: register_entries(reg) for doc, (_, reg) in DOCUMENTS.items()}
    missing_paths: list[str] = []
    unregistered: list[str] = []
    orphans: list[str] = []
    n_paths = 0
    for doc, (_, register) in DOCUMENTS.items():
        evitems, devitems = registers[doc]
        listed = register_paths(evitems)
        n_paths += len(listed)
        missing_paths += [f'{doc} {key}: {path}' for key, path in listed if path not in existing]
        used = markers(files[doc], register)
        unregistered += [f'{doc}: \\ev{{{k}}}' for k in sorted(used['ev'] - set(evitems))]
        unregistered += [f'{doc}: \\pend{{{k}}}' for k in sorted(used['pend'] - devitems)]
        orphans += [f'{doc}: evitem {k}' for k in sorted(set(evitems) - used['ev'])]
        orphans += [f'{doc}: devitem {k}' for k in sorted(devitems - used['pend'])]
    failures[f'register paths missing at {args.rev}'] = missing_paths
    failures['markers without a register entry'] = unregistered
    failures['register entries their document does not cite'] = orphans
    paper_ev, notes_ev = registers['paper'][0], registers['notes'][0]
    failures['entries whose files or producer differ between the registers'] = sorted(
        key for key in set(paper_ev) & set(notes_ev) if paper_ev[key] != notes_ev[key]
    )

    failures['register ID prefixes (paper none, notes distinct)'] = prefix_failures(files)
    aux_note = '' if (PAPER / 'research_notes.aux').is_file() else ' (labels only; notes not built)'
    failures[f'pointers into the notes{aux_note}'] = notessec_failures(files)

    counts = ', '.join(
        f'{doc}: {len(cited[doc])} cited keys, {len(registers[doc][0])} entries, '
        f'{len(registers[doc][1])} pending'
        for doc in DOCUMENTS
    )
    print(f'{counts}; {len(paper_bib)} bib entries, {n_paths} register paths')
    for check, items in failures.items():
        print(f'{check}: {"none" if not items else ", ".join(items)}')
    return 1 if any(failures.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
