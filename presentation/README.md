# Talk slides for the paper

`verifier_paper_talk.pptx` holds the slides for a talk of about 30 minutes on the paper *The Work a
Verifier Needs: Certified Int8 Output Heads That Match a Production Kernel's Rounding*
(`paper/paper.pdf`). `verifier_paper_talk.pdf` is the same deck rendered to PDF, without the
speaker notes.

## What the deck contains

The 16 main slides follow the paper's order: the question (Section 1), what "exact" has to mean
in the engine (Sections 1, 2 and 4.1), the certified head (Section 3), what it gives on recorded
engine states (Sections 4.2 and 4.3), what it bought when served (Section 4.5), transport and why
it fails (Section 5), the engine changes that moved the serving frontier more (Section 6), the
limits (Section 8) and the conclusion (Section 9). Each main slide carries one message in its
title. Its speaker notes give what to say, with the minute marks of its section, and the files the
numbers come from. The scripts add up to about 3,200 words, 21 to 24 minutes at 130 to 150 words a
minute; the minute marks budget 29.5 minutes, which leaves room for interruptions.

The 30 backup slides are grouped by the questions they answer; the first backup slide is the
index:

- exactness and the error model (the references, ties, the served head's exactness and costs, every
  change against stock, BF16 against Hugging Face, GSM8K, seeded sampling);
- why the served gain is small (head-only timing, candidates and fallback growth, every served
  point against its prediction, the method in detail);
- why transport fails (the drift, the bound, the tilings);
- the engine changes and their exactness classes (Table 1, FA4 with the fold, the prefill delay,
  the declared approximations, the first composition, FP8);
- setup, the detailed profile, related work, upstream fixes and the code.

The FP8 and upstream-fixes slides go beyond the paper: FP8 is cited only by the research notes
(`evidence/speed_bytes/`), and the upstream status was read from GitHub on 3 October 2026.

Charts are drawn from the committed files and use the paper's colours (`paper/figures/style.tex`):
plain decoding ink, MTP green, DFlash with 16-token blocks reddish purple, DFlash with 8-token
blocks violet, the certified head blue, fallback vermilion, context grey.

## What it reflects

Every number was computed from the evidence committed at `59cec92` on main (the merge of the
paper's clarity pass) and checked against the paper's text at the same commit; the build stops if
a file or a sentence of the paper no longer says what a slide says. Each slide's footer names its
paper section and evidence files. No experiment was rerun for the deck.

## How it was built

The deck was built with private tooling: python-pptx scripts that start from a slide template,
read the committed CSV and JSON files under `evidence/`, check the paper's sections and the
evidence READMEs, and write the slides and notes; LibreOffice 7.3 rendered the PDF. The tooling is
not in the repository because it depends on a template and earlier drafts that are not public. To
change a number, change the evidence and the paper first; the slides follow them.
