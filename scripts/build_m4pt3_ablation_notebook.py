"""Builds and executes notebooks/08_feature_ablation.ipynb.

Reads backtest_results/ablation_summary.csv produced by scripts/ablation.py
and renders the marginal-skill bar chart for the README.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path("notebooks/08_feature_ablation.ipynb")


CELLS = [
    nbf.v4.new_markdown_cell(
        "# M4 part 3 — Feature ablation\n\n"
        "Which feature group actually moves the needle? We trained five LSTM "
        "variants, each adding one feature group on top of the previous, and "
        "scored them on the same 70-date stratified holdout the rest of the "
        "project uses. Same architecture (encoder LSTM(64) → decoder LSTM(64) "
        "→ TimeDistributed Dense), same training recipe (Huber, 60 epochs, "
        "early-stop patience 8), same residual target (`actual − TSO_fc`).\n\n"
        "The ladder:\n\n"
        "| # | Variant | Encoder | Decoder |\n"
        "|---|---------|---------|---------|\n"
        "| A | calendar only | hour/dow sin·cos | hour/dow + holiday |\n"
        "| B | + load history | A + load | A |\n"
        "| C | + residual lag | B + residual_hist | A |\n"
        "| D | + TSO_fc decoder | C | A + tso_load_fc |\n"
        "| E | + weather | D + 4× weather | D + 4× weather |\n\n"
        "All variants train on the residual target so skill is comparable "
        "across the ladder."
    ),

    nbf.v4.new_code_cell(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "_here = Path.cwd()\n"
        "ROOT = _here if (_here / 'pyproject.toml').exists() else _here.parent\n"
        "os.chdir(ROOT)\n"
        "\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
    ),

    nbf.v4.new_markdown_cell("## 1. Results table"),

    nbf.v4.new_code_cell(
        "df = pd.read_csv('backtest_results/ablation_summary.csv')\n"
        "df['marginal_skill'] = df['holdout_skill'].diff().fillna(df['holdout_skill'])\n"
        "df[['variant', 'label', 'enc_features', 'dec_features',\n"
        "    'val_implied_skill', 'holdout_skill', 'marginal_skill']]\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 2. Cumulative holdout skill\n\n"
        "Each bar shows the holdout skill of the model when that feature group "
        "(and all groups to its left) are present."
    ),

    nbf.v4.new_code_cell(
        "fig = go.Figure()\n"
        "fig.add_bar(\n"
        "    x=df['label'], y=df['holdout_skill'],\n"
        "    marker_color=['#888', '#4a90d9', '#2a8c4a', '#2a8c4a', '#cc5500'],\n"
        "    text=[f'{v:+.3f}' for v in df['holdout_skill']],\n"
        "    textposition='outside',\n"
        ")\n"
        "fig.add_hline(y=0, line_color='black', line_width=1)\n"
        "fig.update_layout(\n"
        "    title='Holdout skill vs TSO baseline (cumulative)',\n"
        "    yaxis_title='Skill score (1 - MAE_model / MAE_TSO)',\n"
        "    height=440, width=820, template='plotly_white',\n"
        "    margin=dict(t=60, b=80),\n"
        ")\n"
        "fig.update_yaxes(range=[0, max(df['holdout_skill']) * 1.18])\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 3. Marginal contribution of each feature group\n\n"
        "Each bar = the holdout skill **delta** when adding that feature group "
        "to the previous variant. This is what tells us where the leverage is."
    ),

    nbf.v4.new_code_cell(
        "colors = ['#888'] + [\n"
        "    '#2a8c4a' if v > 0 else '#cc3030'\n"
        "    for v in df['marginal_skill'].iloc[1:]\n"
        "]\n"
        "fig = go.Figure()\n"
        "fig.add_bar(\n"
        "    x=df['label'], y=df['marginal_skill'],\n"
        "    marker_color=colors,\n"
        "    text=[f'{v:+.3f}' for v in df['marginal_skill']],\n"
        "    textposition='outside',\n"
        ")\n"
        "fig.add_hline(y=0, line_color='black', line_width=1)\n"
        "fig.update_layout(\n"
        "    title='Marginal skill contribution per feature group',\n"
        "    yaxis_title='Δ holdout skill vs previous variant',\n"
        "    height=440, width=820, template='plotly_white',\n"
        "    margin=dict(t=60, b=80),\n"
        ")\n"
        "fig.show()\n"
    ),

    nbf.v4.new_markdown_cell(
        "## 4. What this tells us\n\n"
        "**Residual lag is the single biggest lever (+0.139 skill).** This is "
        "residual learning made operational: showing the model the recent "
        "`actual − TSO` error pattern in the encoder lets it learn the *systematic* "
        "bias in the TSO forecast. That's the headline insight of the project — "
        "the TSO already gets the easy 90 % right (calendar, climatology); the "
        "model only has to learn the structured remainder.\n\n"
        "**Calendar features alone clear +0.047 skill.** Even with no load "
        "history at all, the residual target has structure the TSO underspecifies "
        "— mostly holiday-effect bias. Free skill from a 4-feature encoder.\n\n"
        "**Load history is a smaller lever than you might guess (+0.050).** Once "
        "the residual lag is in (variant C), raw load history is largely "
        "redundant with it, but as a step on its own it adds modest level/trend "
        "information.\n\n"
        "**Adding TSO_fc to the decoder is roughly neutral (−0.008).** This is "
        "a worthwhile *negative* result. Since the target is already "
        "`actual − TSO_fc`, exposing TSO_fc again in the decoder is information "
        "the model has already implicitly accounted for. The mild dip is "
        "noise / mild overfit on a tiny ~1000-day training set. The simpler "
        "architecture is fine — and we *checked*, which is the point.\n\n"
        "**Weather adds +0.014 on average.** Matches notebook 06's finding: "
        "the average lift is small, but the *case-study* lift is large — on "
        "1 May 2026 (PV-driven negative-price record) the weather model "
        "improved on the plain LSTM by **+0.51 skill** for that single day. "
        "The average masks the times weather actually matters.\n\n"
        "### Calibration on the design choice\n\n"
        "The residual-learning + lagged-residual feature combo isn't a "
        "decoration — it's the design that buys ~60 % of the project's "
        "skill (0.139 / 0.242 ≈ 57 %). Everything else (calendar, weather, "
        "model architecture) is incremental on top of that one decision."
    ),

    nbf.v4.new_markdown_cell(
        "## 5. What we deliberately did *not* test\n\n"
        "- **Cross-border price lags.** 14 neighbour bidding zones live in the "
        "  parquet but aren't in the windowing pipeline yet. ~30-min plumbing "
        "  job; deferred. Likely small lift on top of weather since prices "
        "  are themselves load-driven.\n"
        "- **Per-Bundesland weather.** Currently 4 nationally-aggregated "
        "  variables. Regional disaggregation could matter for renewable-heavy "
        "  zones (PV in Bayern, wind in Schleswig-Holstein) but quadruples "
        "  feature count on a small training set.\n"
        "- **Different encoder architectures per variant.** All five share the "
        "  same LSTM(64) hidden size. A calendar-only model could probably "
        "  use a smaller encoder, but matching architectures keeps the "
        "  comparison apples-to-apples."
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    NB.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    nbf.write(nb, NB)
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
