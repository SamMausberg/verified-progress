# Talk slides for the paper

`verifier_paper_talk.pptx` holds the slides for a talk of about 30 minutes on the paper *The Work a
Verifier Needs: Certified Int8 Output Heads That Match a Production Kernel's Rounding*
(`paper/paper.pdf`). `verifier_paper_talk.pdf` is the same deck rendered to PDF, without the
speaker notes.

## What the deck contains

The 17 main slides keep the paper's argument (the question, what "exact" has to mean, the method,
its results, transport, the engine changes) and run in this order:

- the question (slides 1-2; Section 1);
- the setup and the stock baseline: how decisions and speed were measured, and the serving
  frontier with the mean accepted tokens per verify cycle of each drafter arm (3-4; Sections 4.1
  and 4.5);
- where the time goes and what "exact" means: the head's share of a step, BF16 ties, and stock
  SGLang's batch-dependent outputs (5-7; Sections 1, 2 and 4.1, Appendix E.2);
- the certified head: the method, its results on recorded engine states, what it bought when
  served, and why most DFlash verify calls above one request need a fallback (8-11; Sections 3
  and 4);
- transport, what it is and why it fails (12; Section 5);
- the engine changes that gained more than the head (13; Sections 4.5 and 6);
- five other approaches that missed a rule declared before their runs (14; Section 6,
  Appendix E.4, and the research notes for FP8 and the split-KV kernel);
- the stock envelope before and after the two confirmed changes (15; Section 6);
- the quality check: the output comparisons of the head, the fold and the prefill delay against
  stock, beside GSM8K, which ran on four other arms (16; Sections 4.5 and 6);
- the takeaway, with the limits and the next steps that the paper's Limitations and `TASKS.md`
  leave open (17; Sections 8 and 9).

Each main slide states one claim in its title and carries at most three short lines of body text
besides its chart or headline numbers; its footer names the paper's section. Its speaker notes
give what to say, the minute marks of its section, the scope and definitions that did not fit on
the slide, and the files the numbers come from. The scripts add up to about 3,330 words, 22 to 26
minutes at 130 to 150 words a minute; about 220 of those words are marked optional, for when
questions come early. The minute marks budget 29 minutes, which leaves room for interruptions.

The 29 backup slides are grouped by the questions they answer; the first backup slide is the
index:

- exactness and the error model (the references, ties, the served head's exactness and costs, every
  change against stock, BF16 against Hugging Face, GSM8K, seeded sampling);
- why the served gain is small (head-only timing, candidates and fallback growth, every served
  point against its prediction, the method in detail);
- why transport fails (the drift, the bound, the tilings, how much of the vocabulary is left to
  check);
- the engine changes and their exactness classes (every change against the arm it modifies, FA4
  with the fold, the prefill delay, the declared approximations, the first composition, FP8);
- the profile in detail, related work, upstream fixes and the code.

The FP8, split-KV and upstream-fixes material goes beyond the paper: FP8 and the split-KV kernel
are cited only by the research notes (`evidence/speed_bytes/`, `evidence/speed_lowc/`), and the
upstream status was read from GitHub on 3 October 2026.

Charts are drawn from the committed files and use the paper's colours (`paper/figures/style.tex`):
plain decoding ink, MTP green, DFlash with 16-token blocks reddish purple, DFlash with 8-token
blocks violet, the certified head blue, fallback vermilion, context grey.

## What it reflects

Every number was computed from the evidence committed at `02b9c5f`, the paper's second clarity
pass, and checked against the paper's text and the evidence READMEs at that commit. The evidence
files the deck reads are the same there as on main at `59cec92`. Two values follow the evidence
where the paper at that commit had not yet been corrected: 16-token DFlash's certified-head ratio
at one request is 1.013 (+1.3%), rounded once from `evidence/certified_head/served/summary.json`,
and the check-mode counts (592,433 and 196,734) are checked positions, fallback rows included. The
paper's tables are referred to by content, not by number. No experiment was rerun for the deck.

## How it was built

The build scripts are kept outside the repository. They are python-pptx scripts that read the
committed CSV and JSON files under `evidence/`, check each number against the paper's sections and
the evidence READMEs, and write the slides and notes; LibreOffice 7.3 rendered the PDF. To change a
number, change the evidence and the paper first; the slides follow them.
