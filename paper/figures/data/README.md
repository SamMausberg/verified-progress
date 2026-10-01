# Figure data

Every plot in the paper reads committed data when it is built; no plotted number is typed into a
figure's TeX source. Most figures read CSV files under `evidence/` directly (pgfplots). The few values
that exist only in JSON evidence are copied here, because pgfplots cannot read JSON.

| File | Made by | Contents |
|---|---|---|
| `json_values.csv` | `python paper/figures/data/make_figure_data.py` (from the repository root) | One row per value: key, value at full precision, the evidence file and the path inside its JSON |

`python paper/figures/data/make_figure_data.py --check` exits 1 if the CSV no longer matches the
evidence. Figures read a value with `\vpjson{key}{\macro}` (`paper/figures/style.tex`).

## Where each figure's numbers come from

| Figure | Data |
|---|---|
| `breakdown.tex` | `evidence/profiles/breakdown.csv`, `evidence/profiles/step_share.csv` |
| `mechanism.tex` | none: a schematic with illustrative values |
| `candidates.tex` | `evidence/head_geometry/selfevidence_plain4b_ccdf.csv`; `evidence/certified_head/head_path_time.csv`; `json_values.csv` (`fallback_share_*`, from `evidence/head_geometry/rstock_plain4b.json`) |
| `transport.tex` | `evidence/head_geometry/transport_dflash4b.csv`, `rho_dflash4b_quantiles.csv`, `rho_mtp4b_quantiles.csv`; `json_values.csv` (`dflash4b_cos_hd_ht_median`, from `evidence/head_geometry/stats_dflash4b.json`) |
| `frontier.tex` | `evidence/bench/confirm/frontier.csv`, `evidence/bench/confirm/envelope-all.dat` |
| `divergence.tex` | `evidence/state_safety/first_difference_by_module.csv` |

The commands that produced the evidence files are in the README of each `evidence/` directory.
