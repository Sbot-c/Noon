"""Transaction risk desk: a fraud qualifier and a dataset-comparison dashboard.

Run:   streamlit run app.py
Needs: FINAL_dataset.csv next to this file (or upload your dataset in the sidebar).
"""
from __future__ import annotations

import hashlib
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

st.set_page_config(page_title="Transaction risk desk", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "FINAL_dataset.csv"

# ------------------------------------------------------------------ schema
NUMERIC = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d",
           "delivery_distance_km", "prev_refunds"]
BINARY = ["discount_applied", "device_change", "location_mismatch"]
FEATURES = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d", "payment_method",
            "delivery_distance_km", "discount_applied", "prev_refunds", "device_change",
            "location_mismatch"]
TARGET = "is_fraud"
PAYMENTS = ["card", "applepay", "cod", "wallet"]
PRETTY = {"order_amount": "Order amount", "account_age_days": "Account age", "hour_of_day": "Hour of day",
          "num_orders_last_7d": "Orders in last 7 days", "payment_method": "Payment method",
          "delivery_distance_km": "Delivery distance", "discount_applied": "Promo code used",
          "prev_refunds": "Previous refunds", "device_change": "New or changed device",
          "location_mismatch": "Billing / delivery mismatch",
          "pay_card": "Pays by card", "pay_applepay": "Pays by Apple Pay", "pay_cod": "Pays cash on delivery",
          "pay_wallet": "Pays by wallet"}
SHORT = {"order_amount": "amount", "account_age_days": "age", "hour_of_day": "hour",
         "num_orders_last_7d": "orders 7d", "delivery_distance_km": "distance", "discount_applied": "promo",
         "prev_refunds": "refunds", "device_change": "device", "location_mismatch": "mismatch",
         "is_fraud": "fraud"}

# ----------------------------------------------------------------- palette
# Surfaces and ink for the midnight theme. Status colours are reserved for verdicts.
# The two dataset colours were checked for colour-blind separation on the panel surface.
C = dict(bg="#0C1028", panel="#141A3A", panel2="#1B2248", line="#2A3262", ink="#ECEFFA",
         ink2="#A9B2D6", muted="#7A84B3", legit="#3DDC97", review="#FFB84D", fraud="#FF5D73",
         a="#6F87F2", b="#BC8420", low="#46A8D8")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..700&family=Figtree:wght@400;500;600&display=swap');
:root{--bg:#0C1028;--panel:#141A3A;--panel2:#1B2248;--line:#2A3262;--ink:#ECEFFA;--ink2:#A9B2D6;--muted:#7A84B3;
--legit:#3DDC97;--review:#FFB84D;--fraud:#FF5D73;--a:#6F87F2;--b:#BC8420;}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink);
  font-family:'Figtree',system-ui,sans-serif}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.4rem;padding-bottom:4rem;max-width:1280px}
h1,h2,h3,h4{font-family:'Bricolage Grotesque','Figtree',sans-serif!important;color:var(--ink)!important;letter-spacing:-.01em}
p, li, span, label, div{font-family:'Figtree',system-ui,sans-serif}
[data-testid="stSidebar"]{background:#0A0D22;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem}
[data-testid="stWidgetLabel"] p, label p{color:var(--ink2)!important;font-size:.84rem}
.stTabs [data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{padding:10px 2px;background:transparent;color:var(--ink2);font-weight:500;font-size:.98rem}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--ink)}
.stTabs [data-baseweb="tab-highlight"]{background:var(--a);height:2px}
.stTabs [data-baseweb="tab-border"]{display:none}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, textarea{
  background:var(--panel)!important;border-color:var(--line)!important;color:var(--ink)!important}
input, textarea, select{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;background:transparent!important}
[data-baseweb="select"] *{color:var(--ink)!important}
.stTabs [data-baseweb="tab"] p, .stTabs button[data-baseweb="tab"]{color:var(--ink2)!important}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] p{color:var(--ink)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--a)!important}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label{color:var(--ink2)}
[data-testid="stCaptionContainer"], .stCaption{color:var(--muted)!important}
[data-testid="stSliderThumbValue"]{color:var(--a)!important}
[data-baseweb="slider"] div{color:var(--ink2)}
.stButton>button, .stDownloadButton>button{background:var(--panel);border:1px solid var(--line);color:var(--ink);
  border-radius:8px;font-weight:500;transition:border-color .15s}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--a);color:var(--ink)}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{background:var(--a);border-color:var(--a);color:#0C1028}
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1px dashed var(--line);border-radius:10px}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:transparent}
[data-testid="stExpander"] summary p{color:var(--ink2)}
[data-testid="stPlotlyChart"]{border-radius:6px}
hr{border-color:var(--line)}
a{color:var(--a)}

.brand{display:flex;align-items:center;gap:10px;margin-bottom:.4rem}
.brand .mark{width:30px;height:30px}
.brand .t{font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-weight:600;font-size:1.15rem;color:var(--ink)}
.side-note{color:var(--muted);font-size:.78rem;line-height:1.5;margin-top:.4rem}
.page-title{font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-weight:600;font-size:2.5rem;line-height:1.05;margin:0 0 .35rem 0;color:var(--ink)}
.page-sub{color:var(--ink2);font-size:1.02rem;max-width:62ch;margin:0 0 1.4rem 0}

.kpis{display:flex;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:.6rem 0 1.6rem}
.kpi{flex:1 1 150px;padding:14px 20px 14px 18px;border-left:1px solid var(--line)}
.kpi:first-child{border-left:none;padding-left:2px}
.kpi .v{font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-size:1.9rem;font-weight:600;font-variant-numeric:tabular-nums;color:var(--ink);line-height:1.15}
.kpi .l{color:var(--ink2);font-size:.84rem;margin-top:2px}
.kpi .s{color:var(--muted);font-size:.76rem;margin-top:2px}
.sec{margin:1.7rem 0 .5rem}
.sec h3{font-size:1.28rem;margin:0 0 .15rem}
.sec p{color:var(--ink2);font-size:.92rem;margin:0;max-width:78ch}

.badge{display:inline-block;white-space:nowrap;padding:2px 11px;border-radius:99px;font-size:.78rem;font-weight:600;border:1px solid}
.b-legit{color:var(--legit);border-color:rgba(61,220,151,.45);background:rgba(61,220,151,.09)}
.b-review{color:var(--review);border-color:rgba(255,184,77,.45);background:rgba(255,184,77,.09)}
.b-fraud{color:var(--fraud);border-color:rgba(255,93,115,.5);background:rgba(255,93,115,.10)}
.b-neutral{color:var(--ink2);border-color:var(--line);background:rgba(255,255,255,.03)}

.tbl-wrap{margin-bottom:1.1rem;max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:10px}
.tbl{width:100%;border-collapse:collapse;font-size:.86rem}
.tbl th{position:sticky;top:0;background:#10153A;text-align:left;color:var(--ink2);font-weight:500;padding:9px 12px;
  border-bottom:1px solid var(--line);white-space:nowrap}
.tbl td{padding:7px 12px;border-bottom:1px solid rgba(42,50,98,.55);font-variant-numeric:tabular-nums;color:var(--ink)}
.tbl tr:last-child td{border-bottom:none}
.tbl td.k{color:var(--ink2)}
.score-bar{display:inline-block;height:6px;border-radius:3px;vertical-align:middle;margin-right:8px}

.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;font-variant-numeric:tabular-nums;max-width:420px}
.cm .h{color:var(--ink2);font-size:.78rem;align-self:end;padding:0 4px}
.cm .r{color:var(--ink2);font-size:.78rem;align-self:center;padding-right:8px;text-align:right}
.cm .c{border-radius:8px;padding:12px 14px;border:1px solid var(--line);background:var(--panel)}
.cm .c b{display:block;font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-size:1.5rem;font-weight:600}
.cm .c span{color:var(--ink2);font-size:.78rem}
.cm .good{border-color:rgba(61,220,151,.4)}
.cm .bad{border-color:rgba(255,93,115,.45)}

.callout{border-left:2px solid var(--a);padding:6px 16px;color:var(--ink2);margin:.6rem 0 1rem;font-size:.93rem;max-width:90ch}
.callout b{color:var(--ink)}
.empty{border:1px dashed var(--line);border-radius:12px;padding:26px 28px;color:var(--ink2);background:rgba(20,26,58,.5)}
.empty h4{margin:0 0 .4rem 0;font-size:1.1rem}
.chip{display:inline-block;padding:1px 9px;border-radius:6px;font-size:.74rem;font-weight:600;margin-left:8px}
.chip.ok{color:var(--legit);background:rgba(61,220,151,.1)}
.chip.watch{color:var(--review);background:rgba(255,184,77,.1)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.legend-line{color:var(--ink2);font-size:.85rem;margin:.2rem 0 .6rem}

.gauge-wrap{display:flex;justify-content:center}
.verdict{font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-weight:600;font-size:1.5rem}
</style>
"""


# ------------------------------------------------------------ small helpers
def esc(x) -> str:
    return html.escape(str(x))


def section(title: str, sub: str = ""):
    st.markdown(f"<div class='sec'><h3>{esc(title)}</h3>{f'<p>{sub}</p>' if sub else ''}</div>",
                unsafe_allow_html=True)


def kpis(items):
    """items: list of (value, label, sub)."""
    cells = "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{esc(l)}</div>"
                    f"{f'<div class=s>{s}</div>' if s else ''}</div>" for v, l, s in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def callout(text: str):
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def dfkey(df: pd.DataFrame) -> str:
    h = int(pd.util.hash_pandas_object(df, index=False).sum() % (2 ** 63))
    return hashlib.md5(f"{df.shape}-{h}".encode()).hexdigest()


# ---------------------------------------------------------------- plotting
def style(fig: go.Figure, height: int = 340, legend: bool = True, margin=None) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, system-ui, sans-serif", color=C["ink2"], size=12),
        margin=margin or dict(l=8, r=8, t=24, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=C["ink2"], size=12), bgcolor="rgba(0,0,0,0)"),
        showlegend=legend, hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["line"],
                                           font=dict(color=C["ink"], family="Figtree, sans-serif")),
        colorway=[C["a"], C["b"]],
    )
    fig.update_xaxes(automargin=True, gridcolor="rgba(42,50,98,.45)", zeroline=False, linecolor=C["line"],
                     tickfont=dict(color=C["ink2"]), title_font=dict(color=C["ink2"]))
    fig.update_yaxes(automargin=True, gridcolor="rgba(42,50,98,.45)", zeroline=False, linecolor=C["line"],
                     tickfont=dict(color=C["ink2"]), title_font=dict(color=C["ink2"]))
    return fig


def _stretch(fn) -> dict:
    """Full-width keyword that works on both older and newer Streamlit releases."""
    import inspect
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


PLOT_W, BTN_W = _stretch(st.plotly_chart), _stretch(st.button)


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, theme=None, key=key, config=dict(displayModeBar=False), **PLOT_W)


def rounded(fig: go.Figure):
    """Round bar ends where this plotly version supports it."""
    try:
        fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    except Exception:
        pass
    return fig


# --------------------------------------------------------------- data prep
@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse uploaded bytes (re-read safely on every rerun)."""
    return pd.read_csv(io.BytesIO(raw))


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "payment_method" in df.columns:
        df["payment_method"] = (df["payment_method"].astype(str).str.strip().str.lower()
                                .str.replace(r"[\s_\-]+", "", regex=True))
    return df


def prepare(raw: pd.DataFrame, need_label: bool = False):
    """Returns (clean_df, errors, notes, has_label). Errors block use; notes are informational."""
    df = normalise(raw)
    errors, notes = [], []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}."], notes, False
    keep = FEATURES + ([TARGET] if TARGET in df.columns else [])
    df = df[keep].copy()
    for c in NUMERIC + BINARY + ([TARGET] if TARGET in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        notes.append(f"Dropped {before - len(df):,} rows with missing or non-numeric values.")
    if df.empty:
        return None, ["No usable rows after cleaning."], notes, False
    unknown = sorted(set(df["payment_method"]) - set(PAYMENTS))
    if unknown:
        notes.append(f"Unknown payment methods treated as 'other': {', '.join(unknown[:5])}.")
    if not df["hour_of_day"].between(0, 23).all():
        notes.append("Some hour_of_day values fall outside 0-23.")
    for c in BINARY:
        if not set(df[c].unique()) <= {0, 1}:
            errors.append(f"{c} must contain only 0 and 1.")
    has_label = TARGET in df.columns and df[TARGET].nunique() == 2
    if TARGET in df.columns and not has_label:
        notes.append("is_fraud has a single class, so label-based checks are skipped.")
    if need_label and not has_label:
        errors.append("This dataset needs an is_fraud column with both 0 and 1.")
    if has_label:
        df[TARGET] = df[TARGET].astype(int)
    if errors:
        return None, errors, notes, has_label
    return df, errors, notes, has_label


def encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC + BINARY].astype(float).copy()
    for p in PAYMENTS:
        X["pay_" + p] = (df["payment_method"] == p).astype(float)
    return X


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, min_samples_leaf=30,
                                          l2_regularization=1.0, early_stopping=False, random_state=7)


def proba(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(encode(df))[:, 1]


def fast_auc(y, s) -> float:
    """Rank-based AUC; returns 0.5 if only one class."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = pd.Series(np.asarray(s, dtype=float)).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_oof(df: pd.DataFrame, y: np.ndarray, importance: bool = True, seed: int = 7, folds: int = 5):
    """Out-of-fold probabilities, plus permutation importance measured on held-out folds."""
    X = encode(df)
    oof = np.zeros(len(df))
    imp = {f: [] for f in FEATURES}
    rng = np.random.default_rng(seed)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make_model().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        if importance:
            base = fast_auc(y[te], p)
            dte = df.iloc[te]
            for f in FEATURES:
                drops = []
                for _ in range(3):
                    d2 = dte.copy()
                    d2[f] = rng.permutation(d2[f].values)
                    drops.append(base - fast_auc(y[te], proba(m, d2)))
                imp[f].append(float(np.mean(drops)))
    return oof, ({f: float(np.mean(v)) for f, v in imp.items()} if importance else {})


def signature(df: pd.DataFrame) -> pd.Series:
    """Standardised difference (Cohen's d) between fraud and legit rows, per feature."""
    X = encode(df)
    y = df[TARGET].values == 1
    out = {}
    for c in X.columns:
        a, b = X.loc[y, c], X.loc[~y, c]
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(len(a) + len(b) - 2, 1))
        out[c] = 0.0 if sp == 0 else float((a.mean() - b.mean()) / sp)
    return pd.Series(out)


def at_cutoff(y, p, t):
    pred = p >= t
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, flagged=(tp + fp) / len(y))


def tier(p: float, review_from: float, cutoff: float) -> str:
    return "fraud" if p >= cutoff else ("review" if p >= review_from else "legit")


TIER_TEXT = {"legit": "Looks legitimate", "review": "Needs review", "fraud": "Likely fraud"}
TIER_COLOR = {"legit": C["legit"], "review": C["review"], "fraud": C["fraud"]}


# ======================================================================
#  Model bundle (trained once per dataset, cached)
# ======================================================================
@st.cache_resource(show_spinner="Training the qualifier on your dataset...")
def build_bundle(_df: pd.DataFrame, key: str):
    y = _df[TARGET].values
    oof, imp = cv_oof(_df, y)
    model = make_model().fit(encode(_df), y)
    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    cutoff = float(thr[int(np.nanargmax(f1[:-1]))])
    legit = _df[_df[TARGET] == 0]
    return dict(model=model, oof=oof, y=y, imp=imp, cutoff=cutoff,
                bg=legit.sample(min(200, len(legit)), random_state=3).reset_index(drop=True),
                base=float(y.mean()), roc_auc=float(roc_auc_score(y, oof)),
                pr_auc=float(average_precision_score(y, oof)), sig=signature(_df), n=len(_df))


def explain(model, row: dict, bg: pd.DataFrame, n: int = 80, seed: int = 0):
    """Monte-Carlo Shapley values: walk from a typical legit order to this order, one detail at a time,
    in random orders, and average each detail's effect on the fraud probability. Contributions add up
    to (this order's probability minus a typical legit order's). Positive = pushes toward fraud."""
    rng = np.random.default_rng(seed)
    p0 = float(proba(model, pd.DataFrame([row]))[0])
    bgs = bg.sample(min(n, len(bg)), random_state=seed).reset_index(drop=True)
    F = FEATURES
    frames, orders = [], []
    for i in range(len(bgs)):
        order = rng.permutation(len(F))
        cur = bgs.iloc[i][F].to_dict()
        frames.append(dict(cur))
        for k in order:
            cur[F[k]] = row[F[k]]
            frames.append(dict(cur))
        orders.append(order)
    big = pd.DataFrame(frames)
    for c in NUMERIC + BINARY:
        big[c] = big[c].astype(float)
    p = proba(model, big).reshape(len(bgs), len(F) + 1)
    contrib = {f: 0.0 for f in F}
    for i, order in enumerate(orders):
        for step, k in enumerate(order):
            contrib[F[k]] += float(p[i, step + 1] - p[i, step])
    return p0, {f: v / len(bgs) for f, v in contrib.items()}


def fmt_value(f: str, v) -> str:
    if f == "order_amount":
        return f"AED {float(v):,.0f}"
    if f == "account_age_days":
        return f"{int(v):,} days old"
    if f == "hour_of_day":
        return f"{int(v):02d}:00"
    if f == "delivery_distance_km":
        return f"{float(v):.1f} km"
    if f in BINARY:
        return "yes" if int(v) == 1 else "no"
    if f == "payment_method":
        return {"cod": "cash on delivery", "applepay": "Apple Pay"}.get(str(v), str(v))
    return f"{int(v)}"


# ======================================================================
#  Visual components
# ======================================================================
def gauge_svg(p: float, review_from: float, cutoff: float, base: float) -> str:
    W, H, cx, cy, r, sw = 400, 250, 200, 205, 150, 18
    fmax = min(1.0, max(0.5, cutoff * 3))
    if p >= cutoff:
        t = 2 / 3 + (1 / 3) * min((p - cutoff) / max(fmax - cutoff, 1e-9), 1)
    elif p >= review_from:
        t = 1 / 3 + (1 / 3) * (p - review_from) / max(cutoff - review_from, 1e-9)
    else:
        t = (1 / 3) * p / max(review_from, 1e-9)
    t = float(min(max(t, 0.0), 1.0))

    def pt(tt, rad):
        th = np.pi * (1 - tt)
        return cx + rad * np.cos(th), cy - rad * np.sin(th)

    def arc(t0, t1, col):
        gap = 0.014
        x0, y0 = pt(t0 + gap, r)
        x1, y1 = pt(t1 - gap, r)
        return (f'<path d="M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {x1:.1f} {y1:.1f}" stroke="{col}" '
                f'stroke-width="{sw}" fill="none" stroke-linecap="round" opacity=".92"/>')

    tr = tier(p, review_from, cutoff)
    nx0, ny0 = pt(t, r - 44)
    nx1, ny1 = pt(t, r - 14)
    mx, my = pt(t, r)
    labels = "".join(
        f'<text x="{pt(m, r + 26)[0]:.1f}" y="{pt(m, r + 26)[1]:.1f}" fill="{C["ink2"]}" font-size="12" '
        f'text-anchor="middle" font-family="Figtree, sans-serif">{txt}</text>'
        for m, txt in ((1 / 6, "Legit"), (1 / 2, "Review"), (5 / 6, "Fraud")))
    lift = p / base if base else 0
    svg = f"""
<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:430px" role="img"
     aria-label="Fraud probability {pct(p)}, verdict {TIER_TEXT[tr]}">
  {arc(0, 1/3, C['legit'])}{arc(1/3, 2/3, C['review'])}{arc(2/3, 1, C['fraud'])}
  <line x1="{nx0:.1f}" y1="{ny0:.1f}" x2="{nx1:.1f}" y2="{ny1:.1f}" stroke="{C['ink']}" stroke-width="3" stroke-linecap="round"/>
  <circle cx="{mx:.1f}" cy="{my:.1f}" r="10" fill="{C['ink']}" stroke="{C['bg']}" stroke-width="4"/>
  {labels}
  <text x="{cx}" y="{cy - 28}" fill="{C['ink']}" font-size="52" font-weight="600" text-anchor="middle"
        font-family="Bricolage Grotesque, Figtree, sans-serif">{pct(p)}</text>
  <text x="{cx}" y="{cy - 4}" fill="{C['ink2']}" font-size="13" text-anchor="middle"
        font-family="Figtree, sans-serif">chance this order is fraud</text>
  <text x="{cx}" y="{cy + 22}" fill="{C['muted']}" font-size="12.5" text-anchor="middle"
        font-family="Figtree, sans-serif">{lift:.1f}x the {pct(base)} average</text>
</svg>"""
    return f"<div class='gauge-wrap'>{svg}</div>"


def badge(kind: str, text: str | None = None) -> str:
    return f"<span class='badge b-{kind}'>{esc(text or TIER_TEXT.get(kind, kind))}</span>"


def risk_table(out: pd.DataFrame, limit: int = 150) -> str:
    cols = FEATURES
    show_actual = TARGET in out.columns
    head = "<tr><th>Verdict</th><th>Risk</th>" + ("<th>Actual</th>" if show_actual else "") + "".join(
        f"<th>{esc(SHORT.get(c, 'payment'))}</th>" for c in cols) + "</tr>"
    pmax = max(float(out["risk"].max()), 1e-6)
    rows = []
    for _, r in out.head(limit).iterrows():
        tr = r["verdict_key"]
        w = int(70 * min(r["risk"] / pmax, 1))
        cells = (f"<td>{badge(tr)}</td><td><span class='score-bar' style='width:{max(w, 3)}px;background:{TIER_COLOR[tr]}'></span>"
                 f"{pct(r['risk'])}</td>")
        if show_actual:
            cells += f"<td>{badge('fraud', 'fraud') if r[TARGET] == 1 else badge('neutral', 'legit')}</td>"
        for c in cols:
            v = r[c]
            if c == "order_amount":
                s = f"{v:,.0f}"
            elif c == "delivery_distance_km":
                s = f"{v:.1f}"
            elif c == "hour_of_day":
                s = f"{int(v):02d}:00"
            elif c in BINARY:
                s = "yes" if int(v) == 1 else "-"
            elif c == "payment_method":
                s = esc(v)
            else:
                s = f"{int(v):,}"
            cells += f"<td>{s}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='tbl-wrap'><table class='tbl'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"


def confusion_html(m: dict) -> str:
    return f"""
<div class='cm'>
  <div></div><div class='h'>Cleared by the desk</div><div class='h'>Flagged as fraud</div>
  <div class='r'>Actually legit</div>
  <div class='c good'><b>{m['tn']:,}</b><span>correctly cleared</span></div>
  <div class='c bad'><b>{m['fp']:,}</b><span>false alarms</span></div>
  <div class='r'>Actually fraud</div>
  <div class='c bad'><b>{m['fn']:,}</b><span>missed</span></div>
  <div class='c good'><b>{m['tp']:,}</b><span>caught</span></div>
</div>"""


# ======================================================================
#  Tab 1: Overview
# ======================================================================
def tab_overview(B, name, cutoff, review_from, df):
    y, oof = B["y"], B["oof"]
    m = at_cutoff(y, oof, cutoff)
    kpis([(f"{B['n']:,}", "Transactions", f"{int(y.sum()):,} labelled fraud"),
          (pct(B["base"]), "Fraud rate", "share of rows that are fraud"),
          (f"{B['roc_auc']:.2f}", "ROC AUC", "0.50 is a coin flip, 1.00 is perfect"),
          (f"{B['pr_auc']:.2f}", "PR AUC", f"{B['base']:.2f} is what guessing scores"),
          (pct(m["precision"], 0), "Precision at cut-off", f"{m['tp']} of {m['tp'] + m['fp']} flagged are fraud"),
          (pct(m["recall"], 0), "Recall at cut-off", f"{m['tp']} of {m['tp'] + m['fn']} fraud caught")])
    callout(f"<b>How to read this.</b> Every number on this page is out-of-fold: each transaction was scored by a model "
            f"that never saw it. At the current cut-off the desk flags <b>{pct(m['flagged'])}</b> of orders and catches "
            f"<b>{pct(m['recall'], 0)}</b> of the fraud, but only <b>{pct(m['precision'], 0)}</b> of flags are real. "
            f"With fraud this rare, that is normal. Move the cut-off in the sidebar to trade catches for false alarms.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Ranking quality", "ROC curve. The dot marks the current cut-off.")
        fpr, tpr, _ = roc_curve(y, oof)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["muted"], width=1, dash="dot"),
                                 hoverinfo="skip", name="Guessing", showlegend=False))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C["a"], width=2.5), name="Qualifier",
                                 hovertemplate="False alarms %{x:.1%}<br>Fraud caught %{y:.1%}<extra></extra>",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[m["fp"] / max(m["fp"] + m["tn"], 1)], y=[m["recall"]], mode="markers",
                                 marker=dict(size=11, color=C["ink"], line=dict(color=C["bg"], width=3)),
                                 name="Cut-off", hovertemplate="Current cut-off<extra></extra>", showlegend=False))
        fig.update_xaxes(title="Share of legit orders flagged", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Share of fraud caught", tickformat=".0%", range=[0, 1.02])
        show(style(fig, 330, legend=False))
    with c2:
        section("Precision vs recall", "How many flags are real as the desk gets stricter or looser.")
        pr, rc, _ = precision_recall_curve(y, oof)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[B["base"]] * 2, mode="lines", hoverinfo="skip",
                                 line=dict(color=C["muted"], width=1, dash="dot"), showlegend=False))
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=C["a"], width=2.5),
                                 hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers",
                                 marker=dict(size=11, color=C["ink"], line=dict(color=C["bg"], width=3)),
                                 hovertemplate="Current cut-off<extra></extra>", showlegend=False))
        fig.add_annotation(x=0.98, y=B["base"], text="average fraud rate", showarrow=False, yshift=12,
                           xanchor="right", font=dict(color=C["muted"], size=11))
        fig.update_xaxes(title="Recall (fraud caught)", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Precision (flags that are real)", tickformat=".0%", range=[0, min(1.0, max(0.5, pr[:-1].max() * 1.1))])
        show(style(fig, 330, legend=False))

    c3, c4 = st.columns([1, 1.4], gap="large")
    with c3:
        section("What the desk decides", "Out-of-fold results at the current cut-off.")
        st.markdown(confusion_html(m), unsafe_allow_html=True)
    with c4:
        section("Where the scores land", "Share of each group by fraud score. Fraud sits to the right, but the two overlap.")
        hi = float(np.quantile(oof, 0.995))
        edges = np.linspace(0, max(hi, cutoff * 1.5), 36)
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure()
        for lab, col, nm in ((0, C["legit"], "Legit orders"), (1, C["fraud"], "Fraud orders")):
            h, _ = np.histogram(oof[y == lab], bins=edges)
            fig.add_trace(go.Scatter(x=centers, y=h / max((y == lab).sum(), 1), mode="lines", name=nm, line=dict(color=col, width=2.5, shape="spline"),
                                     fill="tozeroy", fillcolor=rgba(col, 0.14),
                                     hovertemplate="Score %{x:.0%}<br>%{y:.1%} of group<extra></extra>"))
        for x, txt in ((review_from, "review"), (cutoff, "fraud")):
            fig.add_vline(x=x, line=dict(color=C["muted"], width=1, dash="dash"))
            fig.add_annotation(x=x, y=1.0, yref="paper", text=txt, showarrow=False, yshift=10,
                               font=dict(color=C["ink2"], size=11))
        fig.update_xaxes(title="Fraud score", tickformat=".0%")
        fig.update_yaxes(title="Share of group", tickformat=".0%")
        show(style(fig, 300))

    c5, c6 = st.columns(2, gap="large")
    with c5:
        section("What the qualifier relies on", "Drop in AUC when a column is shuffled. Bigger means the model leans on it more.")
        imp = pd.Series(B["imp"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[PRETTY[i] for i in imp.index], orientation="h",
                               marker=dict(color=C["a"]), hovertemplate="%{y}<br>AUC drop %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="AUC drop when shuffled")
        show(style(rounded(fig), 380, legend=False))
    with c6:
        section("What fraud looks like here", "Standardised gap between fraud and legit rows. Right: higher in fraud. Left: lower.")
        sig = B["sig"].sort_values()
        names = [PRETTY[i] for i in sig.index]
        fig = go.Figure()
        for nm, col, mask in (("Higher in fraud", C["fraud"], sig.values > 0), ("Lower in fraud", C["low"], sig.values <= 0)):
            fig.add_trace(go.Bar(x=np.where(mask, sig.values, np.nan), y=names, orientation="h", name=nm,
                                 marker=dict(color=col), hovertemplate="%{y}<br>gap %{x:.2f}<extra></extra>"))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="Standardised difference")
        show(style(rounded(fig), 380))

    with st.expander("Preview the dataset"):
        st.markdown(risk_table(df.assign(risk=B["oof"], verdict_key=[tier(p, review_from, cutoff) for p in B["oof"]])
                               .sort_values("risk", ascending=False), limit=60), unsafe_allow_html=True)
        st.caption("Top 60 rows by out-of-fold risk score.")


# ======================================================================
#  Tab 2: Score one transaction
# ======================================================================
PRESETS = {
    "Typical dinner order": dict(order_amount=78, account_age_days=240, hour_of_day=20, num_orders_last_7d=1,
                                 payment_method="card", delivery_distance_km=3.2, discount_applied=0,
                                 prev_refunds=0, device_change=0, location_mismatch=0),
    "Late-night card cash-out": dict(order_amount=420, account_age_days=3, hour_of_day=3, num_orders_last_7d=4,
                                     payment_method="card", delivery_distance_km=4.5, discount_applied=0,
                                     prev_refunds=0, device_change=1, location_mismatch=1),
    "Card-testing probe": dict(order_amount=18, account_age_days=2, hour_of_day=2, num_orders_last_7d=3,
                               payment_method="applepay", delivery_distance_km=2.0, discount_applied=0,
                               prev_refunds=0, device_change=1, location_mismatch=1),
    "Promo-abuse account": dict(order_amount=61, account_age_days=6, hour_of_day=22, num_orders_last_7d=6,
                                payment_method="cod", delivery_distance_km=3.0, discount_applied=1,
                                prev_refunds=0, device_change=1, location_mismatch=0),
    "Serial refund claimant": dict(order_amount=105, account_age_days=420, hour_of_day=14, num_orders_last_7d=2,
                                   payment_method="card", delivery_distance_km=6.0, discount_applied=0,
                                   prev_refunds=4, device_change=0, location_mismatch=0),
    "Far-away cash order": dict(order_amount=160, account_age_days=20, hour_of_day=21, num_orders_last_7d=2,
                                payment_method="cod", delivery_distance_km=11.0, discount_applied=1,
                                prev_refunds=0, device_change=0, location_mismatch=1),
}
FORM_KEYS = dict(order_amount="f_amount", account_age_days="f_age", hour_of_day="f_hour", num_orders_last_7d="f_orders",
                 payment_method="f_pay", delivery_distance_km="f_dist", discount_applied="f_disc",
                 prev_refunds="f_ref", device_change="f_dev", location_mismatch="f_loc")


def apply_preset(name: str):
    for f, v in PRESETS[name].items():
        k = FORM_KEYS[f]
        st.session_state[k] = bool(v) if f in BINARY else (float(v) if f in ("order_amount", "delivery_distance_km") else v)


def tab_score(B, cutoff, review_from):
    if "f_amount" not in st.session_state:
        apply_preset("Typical dinner order")
    section("Score a single order", "Start from an example or enter your own values. The verdict updates as you type.")
    cols = st.columns(3)
    for i, nm in enumerate(PRESETS):
        cols[i % 3].button(nm, key=f"preset_{i}", on_click=apply_preset, args=(nm,), **BTN_W)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        a, b = st.columns(2)
        amount = a.number_input("Order amount (AED)", 1.0, 5000.0, step=5.0, key="f_amount")
        dist = b.number_input("Delivery distance (km)", 0.1, 60.0, step=0.5, key="f_dist")
        age = a.number_input("Account age (days)", 0, 5000, step=1, key="f_age")
        orders = b.number_input("Orders in last 7 days", 0, 60, step=1, key="f_orders")
        refunds = a.number_input("Previous refunds", 0, 30, step=1, key="f_ref")
        pay = b.selectbox("Payment method", PAYMENTS, key="f_pay",
                          format_func=lambda x: {"cod": "Cash on delivery", "applepay": "Apple Pay"}.get(x, x.capitalize()))
        hour = st.slider("Hour of day", 0, 23, key="f_hour", format="%d:00")
        t1, t2, t3 = st.columns(3)
        disc = t1.toggle("Promo code", key="f_disc")
        dev = t2.toggle("New device", key="f_dev")
        loc = t3.toggle("Mismatch", key="f_loc")
    row = dict(order_amount=float(amount), account_age_days=int(age), hour_of_day=int(hour),
               num_orders_last_7d=int(orders), payment_method=pay, delivery_distance_km=float(dist),
               discount_applied=int(disc), prev_refunds=int(refunds), device_change=int(dev), location_mismatch=int(loc))
    p, contrib = explain(B["model"], row, B["bg"])
    tr = tier(p, review_from, cutoff)
    with right:
        st.markdown(gauge_svg(p, review_from, cutoff, B["base"]), unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;margin:-6px 0 4px'><span class='verdict' style='color:{TIER_COLOR[tr]}'>"
                    f"{TIER_TEXT[tr]}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='legend-line' style='text-align:center'>Fraud above {pct(cutoff)}, review above "
                    f"{pct(review_from)}. Change these in the sidebar.</div>", unsafe_allow_html=True)

    section("Why this score", "How much each detail moved the fraud chance compared with an ordinary legit order, in percentage points. "
                              "Details add up to the gap between this order and a typical legit one.")
    s = pd.Series(contrib).reindex(sorted(contrib, key=lambda k: abs(contrib[k]))).astype(float)
    labels = [f"{PRETTY[f]}: {fmt_value(f, row[f])}" for f in s.index]
    fig = go.Figure()
    for nm, col, mask in (("Pushes toward fraud", C["fraud"], s.values > 0), ("Pushes toward legit", C["legit"], s.values <= 0)):
        fig.add_trace(go.Bar(x=np.where(mask, s.values * 100, np.nan), y=labels, orientation="h", name=nm,
                             marker=dict(color=col), hovertemplate="%{y}<br>%{x:+.2f} points<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Change in fraud chance (percentage points)", ticksuffix=" pts")
    show(style(rounded(fig), 390))
    ups = [f for f in sorted(contrib, key=contrib.get, reverse=True) if contrib[f] > 0.002][:3]
    if ups:
        callout("Biggest reasons this looks risky: " + ", ".join(f"<b>{esc(PRETTY[f].lower())}</b> ({esc(fmt_value(f, row[f]))})" for f in ups) + ".")
    else:
        callout("Nothing about this order stands out as risky compared with an ordinary legit order.")


# ======================================================================
#  Tab 3: Batch classify
# ======================================================================
def score_frame(df: pd.DataFrame, p: np.ndarray, cutoff: float, review_from: float) -> pd.DataFrame:
    out = df.copy()
    out["risk"] = p
    out["verdict_key"] = [tier(v, review_from, cutoff) for v in p]
    return out


def tab_batch(B, df_ours, cutoff, review_from):
    section("Classify a file of transactions",
            "Upload a CSV with the ten transaction columns. An is_fraud column is optional; if present, the desk also reports how well it did.")
    up = st.file_uploader("Transactions CSV", type=["csv"], key="batch_up", label_visibility="collapsed")
    c1, c2 = st.columns([1, 3])
    if c1.button("Try 500 rows from our data"):
        idx = np.random.default_rng().choice(len(df_ours), 500, replace=False)
        st.session_state["batch_demo"] = idx
    if up is not None:
        st.session_state.pop("batch_demo", None)

    scored = None
    if up is not None:
        try:
            raw = parse_csv(up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file as a CSV: {e}")
            return
        df, errs, notes, has_label = prepare(raw)
        if errs:
            st.error(" ".join(errs) + " Expected columns: " + ", ".join(FEATURES) + ".")
            return
        for n in notes:
            st.caption(n)
        scored = score_frame(df, proba(B["model"], df), cutoff, review_from)
        source = f"{len(df):,} rows from your file"
    elif "batch_demo" in st.session_state:
        idx = st.session_state["batch_demo"]
        scored = score_frame(df_ours.iloc[idx].reset_index(drop=True), B["oof"][idx], cutoff, review_from)
        has_label = True
        source = "500 random rows from our dataset, scored out-of-fold so the model never saw them"
    if scored is None:
        st.markdown("<div class='empty'><h4>Nothing to classify yet</h4>Drop a CSV above, or use the button to try 500 rows "
                    "from our own dataset. Each row gets a risk score and one of three verdicts.</div>", unsafe_allow_html=True)
        return

    n = len(scored)
    counts = scored["verdict_key"].value_counts()
    st.markdown(f"<div class='legend-line'>{esc(source)}.</div>", unsafe_allow_html=True)
    kpis([(f"{n:,}", "Orders scored", ""),
          (f"{counts.get('fraud', 0):,}", "Likely fraud", pct(counts.get('fraud', 0) / n) + " of orders"),
          (f"{counts.get('review', 0):,}", "Needs review", pct(counts.get('review', 0) / n) + " of orders"),
          (f"{counts.get('legit', 0):,}", "Looks legitimate", pct(counts.get('legit', 0) / n) + " of orders")])
    if has_label and TARGET in scored.columns:
        m = at_cutoff(scored[TARGET].values, scored["risk"].values, cutoff)
        cc1, cc2 = st.columns([1.1, 1], gap="large")
        with cc1:
            section("How the desk did on labelled rows")
            st.markdown(confusion_html(m), unsafe_allow_html=True)
        with cc2:
            section("In plain terms")
            auc = fast_auc(scored[TARGET].values, scored["risk"].values)
            callout(f"Of <b>{m['tp'] + m['fn']}</b> fraud orders the desk caught <b>{m['tp']}</b> ({pct(m['recall'], 0)}). "
                    f"Of <b>{m['tp'] + m['fp']}</b> flagged, <b>{m['tp']}</b> were real ({pct(m['precision'], 0)}). "
                    f"Ranking quality (AUC) on this file: <b>{auc:.2f}</b>.")
    section("Highest-risk orders", "Sorted by risk, top 150 shown. Download the full file below.")
    scored = scored.sort_values("risk", ascending=False)
    st.markdown(risk_table(scored), unsafe_allow_html=True)
    dl = scored.drop(columns=["verdict_key"]).rename(columns={"risk": "fraud_score"})
    dl.insert(0, "verdict", scored["verdict_key"].map(TIER_TEXT).values)
    st.download_button("Download all scored rows (CSV)", dl.to_csv(index=False).encode(), "scored_transactions.csv",
                       "text/csv", type="primary")


# ======================================================================
#  Tab 4: Compare datasets
# ======================================================================
CONTINUOUS = ["order_amount", "account_age_days", "delivery_distance_km"]
CORR_COLS = NUMERIC + BINARY


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def demo_dataset(n: int = 6000, seed: int = 11) -> pd.DataFrame:
    """A deliberately naive generator (independent columns, one logistic fraud rule) for a dry run."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "order_amount": np.round(np.exp(r.normal(4.3, 0.7, n)), 2),
        "account_age_days": r.integers(0, 1500, n),
        "hour_of_day": r.integers(0, 24, n),
        "num_orders_last_7d": r.poisson(2.0, n),
        "payment_method": r.choice(PAYMENTS, n),
        "delivery_distance_km": np.round(r.uniform(0.5, 20, n), 2),
        "discount_applied": (r.random(n) < 0.3).astype(int),
        "prev_refunds": r.poisson(0.3, n),
        "device_change": (r.random(n) < 0.15).astype(int),
        "location_mismatch": (r.random(n) < 0.12).astype(int),
    })
    z = (-5.0 + 2.2 * d.device_change + 1.8 * d.location_mismatch + 0.9 * (d.account_age_days < 30)
         + 0.6 * (d.order_amount > 250) + 0.25 * d.prev_refunds)
    d[TARGET] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return d


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, allv, side="right") / len(a)
                               - np.searchsorted(b, allv, side="right") / len(b))))


def tvd(a: pd.Series, b: pd.Series) -> float:
    ca, cb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    idx = ca.index.union(cb.index)
    return float(0.5 * np.abs(ca.reindex(idx, fill_value=0) - cb.reindex(idx, fill_value=0)).sum())


def gap(a: pd.DataFrame, b: pd.DataFrame, f: str) -> float:
    return ks_stat(a[f].values.astype(float), b[f].values.astype(float)) if f in CONTINUOUS else tvd(a[f], b[f])



def single_aucs(df: pd.DataFrame, y: np.ndarray) -> dict:
    """Best AUC any one column achieves on its own (direction ignored)."""
    X = encode(df)
    cols = {f: ([f] if f != "payment_method" else ["pay_" + p for p in PAYMENTS]) for f in FEATURES}
    return {f: max(max(a, 1 - a) for a in (fast_auc(y, X[c]) for c in cs)) for f, cs in cols.items()}


@st.cache_data(show_spinner="Comparing the two datasets (training both models)...")
def compare_core(_a: pd.DataFrame, _b: pd.DataFrame, ka: str, kb: str, b_labelled: bool):
    out = dict(gaps={f: gap(_a, _b, f) for f in FEATURES})
    ya = _a[TARGET].values
    oof_a, _ = cv_oof(_a, ya, importance=False)
    out["cv_a"] = fast_auc(ya, oof_a)
    out["single_a"] = single_aucs(_a, ya)
    out["sig_a"] = signature(_a)
    model_a = make_model().fit(encode(_a), ya)
    if b_labelled:
        yb = _b[TARGET].values
        oof_b, _ = cv_oof(_b, yb, importance=False)
        out["cv_b"] = fast_auc(yb, oof_b)
        out["single_b"] = single_aucs(_b, yb)
        out["sig_b"] = signature(_b)
        model_b = make_model().fit(encode(_b), yb)
        out["a_on_b"] = fast_auc(yb, proba(model_a, _b))
        out["b_on_a"] = fast_auc(ya, proba(model_b, _a))
    # can a classifier tell the two datasets apart?
    sa = _a.sample(min(4000, len(_a)), random_state=1)
    sb = _b.sample(min(4000, len(_b)), random_state=1)
    D = pd.concat([sa, sb], ignore_index=True)
    yd = np.r_[np.zeros(len(sa)), np.ones(len(sb))].astype(int)
    oof_d, imp_d = cv_oof(D, yd, importance=True)
    out["tell_auc"], out["tell_imp"] = fast_auc(yd, oof_d), imp_d
    cc = CORR_COLS + ([TARGET] if b_labelled else [])
    out["corr_a"] = _a[CORR_COLS + [TARGET]].corr().loc[cc, cc]
    out["corr_b"] = _b[cc].corr().loc[cc, cc] if b_labelled else _b[CORR_COLS].corr()
    return out


def dist_figure(a: pd.DataFrame, b: pd.DataFrame, na: str, nb: str) -> go.Figure:
    titles = [PRETTY[f] for f in FEATURES]
    fig = make_subplots(rows=5, cols=2, subplot_titles=titles, vertical_spacing=0.07, horizontal_spacing=0.08)
    for i, f in enumerate(FEATURES):
        r, c = i // 2 + 1, i % 2 + 1
        first = i == 0
        va, vb = a[f], b[f]
        discrete = f not in CONTINUOUS
        if discrete:
            cap = 10 if f in ("num_orders_last_7d", "prev_refunds") else None
            if cap:
                va, vb = va.clip(upper=cap), vb.clip(upper=cap)
            ca, cb = va.value_counts(normalize=True), vb.value_counts(normalize=True)
            cats = sorted(ca.index.union(cb.index), key=lambda v: (str(type(v)), v))
            xs = [("10+" if cap and v == cap else str(v)) if not isinstance(v, str) else v for v in cats]
            for nm, col, cs in ((na, C["a"], ca), (nb, C["b"], cb)):
                fig.add_trace(go.Bar(x=xs, y=[cs.get(v, 0) for v in cats], name=nm, marker=dict(color=col),
                                     legendgroup=nm, showlegend=first,
                                     hovertemplate=f"{nm}<br>%{{x}}: %{{y:.1%}}<extra></extra>"), row=r, col=c)
        else:
            both = np.concatenate([va.values, vb.values])
            hi = float(np.quantile(both, 0.995))
            edges = np.linspace(float(both.min()), hi, 31)
            ctr = (edges[:-1] + edges[1:]) / 2
            for nm, col, vals in ((na, C["a"], va.values), (nb, C["b"], vb.values)):
                h, _ = np.histogram(vals, bins=edges)
                fig.add_trace(go.Scatter(x=ctr, y=h / max(len(vals), 1), mode="lines", name=nm, legendgroup=nm,
                                         showlegend=first, line=dict(color=col, width=2.2, shape="spline"),
                                         fill="tozeroy", fillcolor=rgba(col, 0.14),
                                         hovertemplate=f"{nm}<br>%{{x:.1f}}: %{{y:.1%}} of rows<extra></extra>"), row=r, col=c)
        fig.update_yaxes(tickformat=".0%", row=r, col=c)
    fig.update_layout(barmode="group", bargap=0.25)
    for ann in fig.layout.annotations:
        ann.font = dict(color=C["ink"], size=13, family="Bricolage Grotesque, Figtree, sans-serif")
        ann.xanchor = "left"
        ann.x = ann.x - 0.205
    return style(rounded(fig), 1180, margin=dict(l=8, r=8, t=40, b=8))


def corr_figure(z: pd.DataFrame, zmax: float, title: str) -> go.Figure:
    labels = [SHORT[c] for c in z.columns]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=labels, y=labels, zmin=-zmax, zmax=zmax, showscale=False, xgap=2, ygap=2,
        colorscale=[[0, C["low"]], [0.5, "#222A57"], [1, C["fraud"]]],
        text=np.where((np.abs(z.values) >= 0.1) & ~np.eye(len(z), dtype=bool), np.round(z.values, 2).astype(str), ""),
        texttemplate="%{text}", textfont=dict(size=9, color=C["ink"]),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_layout(title=dict(text=title, font=dict(color=C["ink"], size=14, family="Bricolage Grotesque, Figtree, sans-serif"), x=0))
    return style(fig, 400, legend=False, margin=dict(l=8, r=8, t=44, b=8))


def chip(ok: bool, good: str = "in range", bad: str = "check") -> str:
    return f"<span class='chip {'ok' if ok else 'watch'}'>{good if ok else bad}</span>"


def grid_html(rows_a: str, rows_b: str, na: str, nb: str, aa, ab, ba, bb) -> str:
    def cell(v, diag):
        return f"<div class='c {'good' if diag else ''}'><b>{v:.2f}</b><span>{'own data (5-fold)' if diag else 'unseen data'}</span></div>"
    return f"""
<div class='cm' style='max-width:560px'>
  <div></div><div class='h'>Tested on {esc(na)}</div><div class='h'>Tested on {esc(nb)}</div>
  <div class='r'>Trained on<br>{esc(na)}</div>{cell(aa, True)}{cell(ab, False)}
  <div class='r'>Trained on<br>{esc(nb)}</div>{cell(ba, False)}{cell(bb, True)}
</div>"""


def tab_compare(df_a: pd.DataFrame, na: str, nb: str):
    section("Compare against another GAN dataset",
            "Upload the other team's CSV. It needs the same ten columns and, for the fraud-pattern checks, an is_fraud column.")
    up = st.file_uploader("Other dataset (CSV)", type=["csv"], key="cmp_up", label_visibility="collapsed")
    c1, c2 = st.columns([1, 3])
    if c1.button("Dry run with a demo dataset"):
        st.session_state["cmp_demo"] = True
    if up is not None:
        st.session_state["cmp_demo"] = False

    df_b, notes, source = None, [], ""
    if up is not None:
        try:
            raw = parse_csv(up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file as a CSV: {e}")
            return
        df_b, errs, notes, _ = prepare(raw)
        if errs:
            st.error(" ".join(errs) + " Expected columns: " + ", ".join(FEATURES + [TARGET]) + ".")
            return
        source = f"{up.name}: {len(df_b):,} rows"
    elif st.session_state.get("cmp_demo"):
        df_b, source = demo_dataset(), "Demo dataset from a naive generator (independent columns, one fraud rule)"
    if df_b is None:
        st.markdown("<div class='empty'><h4>Waiting for the other dataset</h4>Drop their CSV above. You will get a scorecard, "
                    "feature-by-feature distributions, fraud patterns, correlations, and a cross-test that trains a model on one "
                    "dataset and checks it on the other. Use the demo button to see the layout first.</div>", unsafe_allow_html=True)
        return

    for n in notes:
        st.caption(n)
    if len(df_b) > 50000:
        df_b = df_b.sample(50000, random_state=1).reset_index(drop=True)
        st.caption("Their dataset was sampled down to 50,000 rows for speed.")
    labelled = TARGET in df_b.columns and df_b[TARGET].nunique() == 2
    if not labelled:
        st.warning("No usable is_fraud column in this file, so fraud-pattern and cross-test sections are skipped.")
    st.markdown(f"<div class='legend-line'><span class='dot' style='background:{C['a']}'></span>{esc(na)} &nbsp;&nbsp; "
                f"<span class='dot' style='background:{C['b']}'></span>{esc(nb)} &nbsp;&nbsp; {esc(source)}</div>", unsafe_allow_html=True)

    R = compare_core(df_a, df_b, dfkey(df_a), dfkey(df_b), labelled)
    avg_gap = float(np.mean(list(R["gaps"].values())))

    # ---- headline
    items = [(f"{avg_gap:.2f}", "Average distribution gap", "0 = identical, 1 = no overlap"),
             (f"{R['tell_auc']:.2f}", "Can a model tell them apart?", "0.50 = indistinguishable")]
    if labelled:
        items += [(f"{R['a_on_b']:.2f}", f"{na} model on {nb}", f"{R['cv_a']:.2f} on its own data"),
                  (f"{R['b_on_a']:.2f}", f"{nb} model on {na}", f"{R['cv_b']:.2f} on its own data")]
    kpis(items)
    bits = [f"The two datasets differ by <b>{avg_gap:.2f}</b> on average across the ten columns",
            f"a classifier separates their rows with AUC <b>{R['tell_auc']:.2f}</b>"]
    if labelled:
        bits.append(f"a fraud model trained on {esc(na)} reaches AUC <b>{R['a_on_b']:.2f}</b> on {esc(nb)}, versus <b>{R['cv_a']:.2f}</b> on its own data")
    callout("; ".join(bits) + ".")

    # ---- scorecard
    section("Scorecard", "Health checks for each dataset on its own. The bands are rules of thumb for realistic fraud data, not hard limits.")
    sa = dict(rows=len(df_a), fr=float(df_a[TARGET].mean()), dup=float(df_a.duplicated().mean()))
    sb = dict(rows=len(df_b), fr=float(df_b[TARGET].mean()) if labelled else None, dup=float(df_b.duplicated().mean()))
    rows = [("Rows", f"{sa['rows']:,}", f"{sb['rows']:,}", "", ""),
            ("Fraud rate", pct(sa["fr"], 2), pct(sb["fr"], 2) if labelled else "n/a",
             chip(0.01 <= sa["fr"] <= 0.08, "1-8%", "outside 1-8%"),
             chip(labelled and 0.01 <= sb["fr"] <= 0.08, "1-8%", "outside 1-8%") if labelled else ""),
            ("Duplicate rows", pct(sa["dup"], 2), pct(sb["dup"], 2),
             chip(sa["dup"] <= 0.02, "under 2%", "over 2%"), chip(sb["dup"] <= 0.02, "under 2%", "over 2%"))]
    if labelled:
        top_a = max(R["single_a"], key=R["single_a"].get)
        top_b = max(R["single_b"], key=R["single_b"].get)
        rows += [("Fraud learnability (5-fold AUC)", f"{R['cv_a']:.2f}", f"{R['cv_b']:.2f}",
                  chip(0.70 <= R["cv_a"] <= 0.97, "0.70-0.97", "too weak or too easy"),
                  chip(0.70 <= R["cv_b"] <= 0.97, "0.70-0.97", "too weak or too easy")),
                 ("Strongest single column (AUC)", f"{R['single_a'][top_a]:.2f} ({SHORT.get(top_a, 'payment')})",
                  f"{R['single_b'][top_b]:.2f} ({SHORT.get(top_b, 'payment')})",
                  chip(R["single_a"][top_a] <= 0.88, "under 0.88", "gives fraud away"),
                  chip(R["single_b"][top_b] <= 0.88, "under 0.88", "gives fraud away"))]
    body = "".join(f"<tr><td class='k'>{esc(l)}</td><td>{esc(a)}{ca}</td><td>{esc(b)}{cb}</td></tr>" for l, a, b, ca, cb in rows)
    st.markdown(f"<div class='tbl-wrap'><table class='tbl'><thead><tr><th></th><th>{esc(na)}</th><th>{esc(nb)}</th></tr></thead>"
                f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)

    # ---- distributions
    section("Column by column", "Share of rows at each value. Closer curves mean the datasets describe the same kind of orders.")
    mode = "All rows"
    if labelled:
        mode = st.radio("Rows to compare", ["All rows", "Legit only", "Fraud only"], horizontal=True, label_visibility="collapsed")
    fa, fb = df_a, df_b
    if mode != "All rows":
        v = 1 if mode == "Fraud only" else 0
        fa, fb = df_a[df_a[TARGET] == v], df_b[df_b[TARGET] == v]
    if min(len(fa), len(fb)) < 30:
        st.info("Too few rows in one dataset for this slice to be meaningful.")
    else:
        show(dist_figure(fa, fb, na, nb), key="cmp_dist")

    g = pd.Series(R["gaps"]).sort_values()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Which columns differ most", "Distribution gap per column (KS distance for continuous, total variation for the rest).")
        fig = go.Figure(go.Bar(x=g.values, y=[PRETTY[i] for i in g.index], orientation="h", marker=dict(color=C["ink2"]),
                               hovertemplate="%{y}<br>gap %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Gap (0 = identical)", range=[0, max(0.3, float(g.max()) * 1.1)])
        show(style(rounded(fig), 380, legend=False))
    with c2:
        section("What gives the datasets away", "Columns a classifier relies on to tell their rows from ours.")
        imp = pd.Series(R["tell_imp"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[PRETTY[i] for i in imp.index], orientation="h", marker=dict(color=C["ink2"]),
                               hovertemplate="%{y}<br>AUC drop %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="AUC drop when shuffled")
        show(style(rounded(fig), 380, legend=False))

    if not labelled:
        return
    # ---- fraud patterns
    section("Fraud patterns", "How far fraud rows sit from legit rows on each signal, in each dataset. Longer bars mean a stronger tell.")
    sig = pd.DataFrame({na: R["sig_a"], nb: R["sig_b"]})
    sig = sig.reindex(sig.abs().max(axis=1).sort_values().index)
    fig = go.Figure()
    for nm, col in ((na, C["a"]), (nb, C["b"])):
        fig.add_trace(go.Bar(x=sig[nm], y=[PRETTY[i] for i in sig.index], orientation="h", name=nm, marker=dict(color=col),
                             hovertemplate=f"{nm}<br>%{{y}}<br>gap %{{x:.2f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(title="Standardised difference (right: higher in fraud, left: lower)")
    show(style(rounded(fig), 520))
    agree = float(np.corrcoef(R["sig_a"].values, R["sig_b"].reindex(R["sig_a"].index).values)[0, 1])
    callout(f"The two datasets agree on which signals matter with correlation <b>{agree:.2f}</b> "
            f"(1.00 means the same fraud story, 0 means unrelated, negative means opposite).")

    # ---- cross test
    section("Cross-test", "Train a fraud model on one dataset and score the other. High numbers off the diagonal mean the fraud rules transfer.")
    st.markdown(grid_html("", "", na, nb, R["cv_a"], R["a_on_b"], R["b_on_a"], R["cv_b"]), unsafe_allow_html=True)

    # ---- correlations
    section("Correlations", "Pearson correlation between columns. Red: move together. Blue: move opposite.")
    cc = st.columns(3, gap="medium")
    ca, cb = R["corr_a"], R["corr_b"]
    cols_common = [c for c in ca.columns if c in cb.columns]
    with cc[0]:
        show(corr_figure(ca.loc[cols_common, cols_common], 1, na), key="corr_a")
    with cc[1]:
        show(corr_figure(cb.loc[cols_common, cols_common], 1, nb), key="corr_b")
    with cc[2]:
        show(corr_figure(ca.loc[cols_common, cols_common] - cb.loc[cols_common, cols_common], 0.6, f"{na} minus {nb}"), key="corr_d")

    # ---- download
    summary = pd.DataFrame([{"column": f, "gap": R["gaps"][f], f"{na} fraud signal": R["sig_a"].get(f, np.nan),
                             f"{nb} fraud signal": R["sig_b"].get(f, np.nan)} for f in FEATURES])
    st.download_button("Download comparison summary (CSV)", summary.to_csv(index=False).encode(), "dataset_comparison.csv",
                       "text/csv")


# ======================================================================
#  App shell
# ======================================================================
THEME = {"theme.base": "dark", "theme.primaryColor": C["a"], "theme.backgroundColor": C["bg"],
         "theme.secondaryBackgroundColor": C["panel"], "theme.textColor": C["ink"]}


def apply_theme():
    """Make the dark theme work even without a .streamlit/config.toml: set the theme options in
    memory, then rerun once so the browser receives them."""
    try:
        from streamlit import config
        changed = False
        for k, v in THEME.items():
            if config.get_option(k) != v:
                config.set_option(k, v)
                changed = True
        if changed and not st.session_state.get("_themed"):
            st.session_state["_themed"] = True
            st.rerun()
    except Exception:  # noqa: BLE001  (older/newer Streamlit: fall back to the CSS below)
        pass


def load_ours():
    up = st.session_state.get("ours_up")
    if up is not None:
        raw, label = parse_csv(up.getvalue()), up.name
    elif DEFAULT_DATA.exists():
        raw, label = pd.read_csv(DEFAULT_DATA), DEFAULT_DATA.name
    else:
        return None, None, [], "No dataset found"
    df, errs, notes, _ = prepare(raw, need_label=True)
    return df, label, notes, " ".join(errs)


def main():
    apply_theme()
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            "<div class='brand'><svg class='mark' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='#1B2248'/>"
            "<path d='M7 21a9 9 0 0 1 18 0' stroke='#6F87F2' stroke-width='3' fill='none' stroke-linecap='round'/>"
            "<circle cx='16' cy='21' r='2.3' fill='#ECEFFA'/></svg><span class='t'>Risk desk</span></div>",
            unsafe_allow_html=True)
        with st.expander("Use a different dataset as ours"):
            st.file_uploader("CSV with the ten columns and is_fraud", type=["csv"], key="ours_up")
        df, label, notes, err = load_ours()
        if df is None:
            st.error(err or "Could not load the dataset.")
            st.caption(f"Put FINAL_dataset.csv next to app.py, or upload one above.")
            st.stop()
        B = build_bundle(df, dfkey(df))
        name_a = st.text_input("Name of our dataset", "Our GAN dataset")
        name_b = st.text_input("Name of the other dataset", "Their GAN dataset")
        st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)

        if st.session_state.get("_ds") != dfkey(df):
            st.session_state["cut_pct"] = max(1, round(B["cutoff"] * 100))
            st.session_state["rev_pct"] = max(1, round(B["cutoff"] * 50))
            st.session_state["_ds"] = dfkey(df)
        st.session_state["rev_pct"] = min(st.session_state["rev_pct"], max(st.session_state["cut_pct"] - 1, 1))
        cut_pct = st.slider("Fraud cut-off", 1, 95, key="cut_pct", format="%d%%",
                            help="Orders scoring at or above this are called likely fraud. Lower catches more fraud and raises more false alarms.")
        rev_pct = st.slider("Send to review from", 1, max(cut_pct - 1, 1), key="rev_pct", format="%d%%",
                            help="Orders between this and the cut-off are marked for a human to look at.")
        st.caption(f"Suggested cut-off: {pct(B['cutoff'], 0)}, where precision and recall balance best.")
        st.markdown(f"<div class='side-note'>Trained on {B['n']:,} rows from {esc(label)}. "
                    f"The model is gradient boosting on the ten transaction columns.</div>", unsafe_allow_html=True)
        for n in notes:
            st.caption(n)
    cutoff, review_from = cut_pct / 100, rev_pct / 100

    st.markdown("<h1 class='page-title'>Transaction risk desk</h1>"
                f"<p class='page-sub'>A fraud qualifier trained on {esc(name_a)}. Score one order, classify a file, "
                f"or compare against another GAN dataset.</p>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Overview", "Score an order", "Classify a file", "Compare datasets"])
    with t1:
        tab_overview(B, name_a, cutoff, review_from, df)
    with t2:
        tab_score(B, cutoff, review_from)
    with t3:
        tab_batch(B, df, cutoff, review_from)
    with t4:
        tab_compare(df, name_a, name_b)


main()
