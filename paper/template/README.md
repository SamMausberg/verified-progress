# Venue template

The paper is typeset with the official MLSys author kit, the two-column format derived
from the ICML style: a 10-page main body, then references, then appendices. The MLSys
2026 call for papers (<https://mlsys.org/Conferences/2026/CallForPapers>) links the kit at
<https://media.mlsys.org/Conferences/MLSYS2025/mlsys2025style.zip>.

## Files and licences

| File | Source | Licence | In git |
|---|---|---|---|
| `mlsys2025.sty` | MLSys kit, `mlsys2025style/mlsys2025.sty` | none stated | no: fetched by `fetch_mlsys_kit.sh` |
| `mlsys2025.bst` | MLSys kit, `mlsys2025style/mlsys2025.bst`, unmodified | LaTeX Project Public License, version 1 or later (stated in the file's header) | yes |
| `algorithm.sty`, `algorithmic.sty` | CTAN `algorithms` bundle, v0.1 of 2009/08/24, generated from `algorithms.dtx` with `latex algorithms.ins` | GNU LGPL 2.1 or later (stated in each file); text in `COPYING-algorithms-LGPL-2.1` | yes |
| `COPYING-algorithms-LGPL-2.1` | CTAN `algorithms` bundle, `COPYING` | licence text | yes |
| `fancyhdr.sty` | MLSys kit, `mlsys2025style/fancyhdr.sty` (version 3.2), unmodified | LaTeX Project Public License, version 1 or later (stated in the file's header) | yes |

`mlsys2025.sty` has no licence or permission statement, and neither has the ICML style
it derives from, so this repository does not redistribute it. `fetch_mlsys_kit.sh`
downloads the kit, checks both digests below and writes the file here; `paper/latexmkrc`
runs the script before every build and stops with a clear message if the download or a
digest check fails. The kit's own `algorithm.sty` and `algorithmic.sty` are 1996 copies
without licence text; the CTAN versions above are their licensed successors, and
`mlsys2025.sty` loads them unchanged. The kit's `fancyhdr.sty` 3.2 is kept because the
style sets `\footskip` to zero, which TeX Live's fancyhdr 4 reports as a warning on
every page. `natbib`, `eso-pic`, `forloop` and `times` come from TeX Live.

## Digests (SHA-256)

| File | SHA-256 |
|---|---|
| `mlsys2025style.zip` (the kit) | `04e77090038f78c985154c71f7a57b7fbb2553ddd3e4e62f431ed420ba3768ef` |
| `mlsys2025.sty` | `05a9842992b7ef71851fd2380a1058f83b0faafc106602cabc4c169d372ad8e2` |
| `mlsys2025.bst` | `c9c9f1b83e32512b93f6208e28ba2989fc691b6f70763ad0657a77d44bc067a7` |
| `algorithms.zip` from `https://mirrors.ctan.org/macros/latex/contrib/algorithms.zip` | `bc909517bbd254acf13d37ea3a67f7f35c20d355b917044f513916fb1f48c4ce` |
| `algorithm.sty` | `d63fdf24879f0efd535bc6d06bc2a008b79ae77f6d6d2bc96156ff1f35e96bd7` |
| `algorithmic.sty` | `7fe47ed7f8222c56452bd73bf0682e72023ece06d59207653a02e080a341118d` |
| `fancyhdr.sty` | `b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea` |

## How the paper uses the template

`paper.tex` loads `\usepackage[accepted]{mlsys2025}` so that the author block prints. The
paper is a preprint, not an accepted MLSys paper, so its preamble replaces the style's
proceedings notice with a preprint notice. Three further settings live in `paper.tex`,
and the style file itself is not modified:

- `\raggedbottom` replaces the style's `\flushbottom`, which stretches any column that
  holds a large float and reports an underfull page for it (the kit's own example paper
  reports five). It changes no margin, font size or spacing.
- hyperref is loaded with `hypertexnames=false`, because the style's `\@makecaption`
  typesets each caption twice and otherwise produces duplicate PDF destinations.
- `fix-cm` makes Computer Modern scalable, so math at the style's 5.5 pt script size
  needs no font substitution.

There is no affiliation to print, so the name is set in the author block directly rather
than through `\mlsysauthor`, and the first-column footnote is written in `paper.tex`
instead of `\printAffiliationsAndNotice`, which requires a correspondence address.
