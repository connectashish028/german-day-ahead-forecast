"""xAI-inspired dark theme — CSS injected into the Streamlit page.

Palette
-------
- Background:   #1f2228  (warm near-black, blue undertone)
- Text:         #ffffff
- Borders:      rgba(255,255,255,0.10)  default
                rgba(255,255,255,0.20)  emphasised
- Surface:      rgba(255,255,255,0.03)
- Prediction:   #B8A1FF  lilac
- Actual:       #3B82F6  blue (Tailwind blue-500)
- TSO baseline: rgba(255,255,255,0.40)  white-dashed

Typography (via Google Fonts, free fallback for GeistMono / universalSans)
- Display + buttons + numerics: JetBrains Mono (monospace)
- Body + headings:              Inter (geometric sans)
"""
from __future__ import annotations

# Palette — exposed so charts.py can reference the same hex values.
BG = "#1f2228"
TEXT = "#ffffff"
TEXT_70 = "rgba(255,255,255,0.70)"
TEXT_50 = "rgba(255,255,255,0.50)"
TEXT_30 = "rgba(255,255,255,0.30)"
BORDER = "rgba(255,255,255,0.10)"
BORDER_STRONG = "rgba(255,255,255,0.20)"
SURFACE = "rgba(255,255,255,0.03)"

PREDICTION = "#B8A1FF"   # lilac
PREDICTION_FILL = "rgba(184,161,255,0.18)"
ACTUAL = "#3B82F6"       # blue
TSO = "rgba(255,255,255,0.45)"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="st-"], [class*="css-"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

.stApp {{
    background-color: {BG} !important;
}}

/* Hide default Streamlit chrome */
#MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* Top-level container — generous breathing room */
.block-container {{
    padding-top: 3rem !important;
    padding-bottom: 6rem !important;
    max-width: 1200px !important;
}}

/* Headings — Inter, light weights, sharp */
h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    color: {TEXT} !important;
    letter-spacing: -0.01em !important;
}}
h1 {{ font-size: 2.25rem !important; line-height: 1.1 !important; margin-bottom: 0.25rem !important; }}
h2 {{ font-size: 1.5rem !important; line-height: 1.2 !important; margin-top: 3rem !important; margin-bottom: 1rem !important; }}
h3 {{ font-size: 1rem !important; }}

p, li, label, .stMarkdown {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT_70} !important;
    line-height: 1.6 !important;
}}

/* Monospace tags / pills / numerics */
code, .mono {{
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    color: {TEXT} !important;
}}

/* Inputs / selects / date pickers — sharp corners, minimal borders */
input, select, textarea, .stDateInput input, .stSelectbox div[data-baseweb="select"] > div {{
    background-color: transparent !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 0 !important;
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.stDateInput label, .stSelectbox label, .stCheckbox label, .stRadio label {{
    color: {TEXT_70} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

/* Buttons — uppercase monospace, tracked, sharp */
.stButton > button, .stDownloadButton > button {{
    background-color: transparent !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.625rem 1.25rem !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: rgba(255,255,255,0.05) !important;
    border-color: {TEXT} !important;
}}

/* Hero / stat tiles */
.hero-bar {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px solid {BORDER}; padding-bottom: 1rem; margin-bottom: 2.5rem;
}}
.hero-brand {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: {TEXT};
}}
.hero-badge {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    border: 1px solid {BORDER_STRONG}; padding: 0.25rem 0.75rem; color: {TEXT};
}}

.stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: {BORDER}; border: 1px solid {BORDER}; margin: 1.5rem 0 3rem 0; }}
.stat-cell {{ background: {BG}; padding: 1.5rem;
    display: flex; flex-direction: column; justify-content: flex-end; }}
.stat-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: {TEXT_50};
    margin-bottom: 0.5rem; flex: 1 1 auto; }}
.stat-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.75rem;
    font-weight: 300; color: {TEXT}; line-height: 1; }}
.stat-unit {{ font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;
    color: {TEXT_50}; margin-left: 0.25rem; }}

/* Section divider — a hairline, not a bold rule */
hr {{ border: none !important; border-top: 1px solid {BORDER} !important;
    margin: 3rem 0 2rem 0 !important; }}

/* Anchor links */
a {{ color: {TEXT} !important; text-decoration: underline;
    text-decoration-color: {BORDER_STRONG}; text-underline-offset: 4px;
    transition: text-decoration-color 0.15s ease; }}
a:hover {{ text-decoration-color: {TEXT}; color: {TEXT_50} !important; }}

/* Plotly chart toolbar — tone down */
.modebar {{ filter: invert(1) hue-rotate(180deg) opacity(0.4); }}

/* Info tooltip on KPI tiles — pure CSS hover, dark themed */
.info-tip {{
    display: inline-block;
    position: relative;
    margin-left: 0.35rem;
    color: rgba(255,255,255,0.40);
    cursor: help;
    font-size: 0.72rem;
    line-height: 1;
    transition: color 0.15s ease;
}}
.info-tip:hover {{ color: rgba(255,255,255,0.90); }}
.info-tip-content {{
    visibility: hidden;
    opacity: 0;
    position: absolute;
    top: 1.4rem;
    left: -0.5rem;
    z-index: 100;
    width: 240px;
    padding: 0.6rem 0.75rem;
    background: #2a2d35;
    border: 1px solid {BORDER_STRONG};
    color: {TEXT};
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: normal;
    text-transform: none;
    line-height: 1.5;
    transition: opacity 0.15s ease;
    pointer-events: none;
}}
.info-tip:hover .info-tip-content {{
    visibility: visible;
    opacity: 1;
}}
</style>
"""


def inject(st_module) -> None:
    """Call once at the top of app.py: `styles.inject(st)`."""
    st_module.markdown(CSS, unsafe_allow_html=True)
