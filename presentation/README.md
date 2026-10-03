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
- the engine changes that gained more than the head (13; Section 6);
- six approaches that fell short, each with its number and scope (14; Sections 5 and 6,
  Appendix E.4, and the research notes for FP8);
- the stock envelope before and after the two confirmed changes (15; Section 6);
- the quality check: the exactness classes with their scope, beside GSM8K (16; Sections 4.5
  and 6);
- the takeaway, with the limits and the next steps that the paper's Limitations and `TASKS.md`
  leave open (17; Sections 8 and 9).

Each main slide states one claim in its title, and its footer names the paper's section. Its
speaker notes give what to say, the minute marks of its section, and the files the numbers come
from. The scripts add up to about 3,240 words, 22 to 25 minutes at 130 to 150 words a minute;
about 210 of those words are marked optional, for when questions come early. The minute marks
budget 28.5 minutes, which leaves room for interruptions.

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

The FP8 and upstream-fixes slides go beyond the paper: FP8 is cited only by the research notes
(`evidence/speed_bytes/`), and the upstream status was read from GitHub on 3 October 2026.

Charts are drawn from the committed files and use the paper's colours (`paper/figures/style.tex`):
plain decoding ink, MTP green, DFlash with 16-token blocks reddish purple, DFlash with 8-token
blocks violet, the certified head blue, fallback vermilion, context grey.

## Changes from the first version of this deck

- The main deck now runs in the order above: the setup slide moved in from the backups, and two
  main slides are new, the approaches that fell short and the quality check that pairs GSM8K with
  the exactness classes. Transport's two slides became one, and the limits moved onto the closing
  slide with the next steps.
- The frontier slide adds the mean accepted tokens per verify cycle of each drafter arm, from the
  accept-length columns of `evidence/bench/confirm/frontier.csv`.
- The slide on why the served gain is small described the whole-batch fallback; the served runs
  used the per-position fallback, and the slide now says so. The served-head title says the head
  slows the fastest arm at 2 to 8 requests, since plain decoding and MTP keep gaining there. The
  host-gap patches left the gains chart: they were measured against MTP with FlashInfer attention,
  which stays below stock MTP with Triton attention, and the notes say this.
- Main-slide footers name the paper's section only, and the evidence paths moved into the speaker
  notes. GDN, the fold, FA4 and the envelope are defined on the slide where each first appears.
  The paper's tables are referred to by content, not by number.

## What it reflects

Every number was computed from the evidence committed at `59cec92` on main (the merge of the
paper's clarity pass) and checked against the paper's text and the evidence READMEs at the same
commit. No experiment was rerun for the deck.

## How it was built

The build scripts are kept outside the repository. They are python-pptx scripts that read the
committed CSV and JSON files under `evidence/`, check each number against the paper's sections and
the evidence READMEs, and write the slides and notes; LibreOffice 7.3 rendered the PDF. To change a
number, change the evidence and the paper first; the slides follow them.
