# Analysis scripts

Scripts that produce every number and figure of the report from the game logs in
`../Collusion_Liars_Bar/experiments/runs/`. Run them from this folder with Python 3 and numpy, matplotlib and
requests installed.

| Script | What it does | Output (in `results/`) |
|---|---|---|
| `analyze.py` | Per-game measures, paired tests, Holm correction, bootstrap intervals | `games.csv`, `report.txt` |
| `make_figures.py` | Figures of the main measures | `fig1`-`fig3`, `figures_data.csv` |
| `make_figures_extra.py` | Design figure, positive control, target replies, framing pilot | `fig0`, `fig4`-`fig6`, `positive_control.txt`, `figures_extra_data.csv` |
| `llm_label.py` | Labels chats and replies with Claude Haiku 5.5 through OpenRouter (`--validate`, `--all`) | `llm_labels_*.csv`, `llm_labels_summary.txt`, `llm_labels_A_tests.txt` |
| `read_game.py` | Prints one game log as a readable timeline | text |

Order: `python analyze.py`, then `python make_figures.py` and `python make_figures_extra.py`.

`llm_label.py` needs an OpenRouter key in `%OneDrive%\Desktop\secrets\openrouter_key.txt`. Its answers are cached in
`results/llm_labels/`, so the figures can be rebuilt without calling the API. The author's hand labels used by
`--validate` (agreement with the model labels) are not included.

The framing pilot figure (`fig6`) reads development logs that are not in this repository; without them it is drawn
empty. `report.txt` also contains a keyword-rule chat count (section "3.4") that the report does not use; the report
uses the model labels instead.
