"""Noon fraud intelligence desk.

One Streamlit app for the whole challenge: score orders, diagnose the model, put our GAN fraud
next to Noon's, let the real referee data judge both classifiers, price the cut-off in AED, and
export the model for the Phase 4 tool.

Run:   streamlit run app.py
Data:  put these next to app.py (each can also be uploaded from the sidebar):
       FINAL_dataset.csv            our seed + GAN training data          (required)
       B_NOON_synthetic_fraud.csv   Noon's GAN-generated fraud            (Phase 2)
       C_REFEREE_real_test.csv      real referee sample, test only        (Phase 3)
       B_NOON_model.joblib          Noon's classifier                     (Phase 3)
"""
from __future__ import annotations

import hashlib
import html
import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

st.set_page_config(page_title="Noon fraud desk", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ schema
NUMERIC = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d",
           "delivery_distance_km", "prev_refunds"]
BINARY = ["discount_applied", "device_change", "location_mismatch"]
FEATURES = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d", "payment_method",
            "delivery_distance_km", "discount_applied", "prev_refunds", "device_change", "location_mismatch"]
TARGET = "is_fraud"
PAYMENTS = ["card", "applepay", "cod", "wallet"]
EXPORT_COLS = (["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d", "delivery_distance_km",
                "discount_applied", "prev_refunds", "device_change", "location_mismatch"]
               + [f"payment_method_{p}" for p in sorted(PAYMENTS)])
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
         "is_fraud": "fraud", "payment_method": "payment"}

# Noon's published benchmark (B_NOON_benchmark.md), used when their model file is missing
NOON_BENCH = dict(recall=0.82, precision=0.43, f1=0.56, tp=53, fp=71, fn=12, tn=19864)
REAL_ONLY_BENCH = dict(recall=0.14, precision=0.90, f1=0.24, fp=1)

# ----------------------------------------------------------------- palette
# Dataset colours (ours / Noon / real) were validated as a set on the panel surface for
# colour-blind and normal-vision separation. Status colours are reserved for verdicts.
C = dict(bg="#0B0F24", panel="#131938", panel2="#1A2147", line="#283063", ink="#EEF0FA",
         ink2="#AAB3D8", muted="#7A84B3", legit="#3DDC97", review="#FFB84D", fraud="#FF6B5B",
         ours="#3987E5", noon="#C98500", real="#D55181", ghost="#4A5384", low="#46A8D8")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..700&family=Figtree:wght@400;500;600;700&display=swap');
:root{--bg:#0B0F24;--panel:#131938;--panel2:#1A2147;--line:#283063;--ink:#EEF0FA;--ink2:#AAB3D8;--muted:#7A84B3;
--legit:#3DDC97;--review:#FFB84D;--fraud:#FF6B5B;--ours:#3987E5;--noon:#C98500;--real:#D55181;}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink);
  font-family:'Figtree',system-ui,sans-serif}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;padding-bottom:4rem;max-width:1320px}
h1,h2,h3,h4{font-family:'Bricolage Grotesque','Figtree',sans-serif!important;color:var(--ink)!important;letter-spacing:-.01em}
p, li, span, label, div{font-family:'Figtree',system-ui,sans-serif}
[data-testid="stSidebar"]{background:#080B1D;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
[data-testid="stWidgetLabel"] p, label p{color:var(--ink2)!important;font-size:.84rem}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, textarea{
  background:var(--panel)!important;border-color:var(--line)!important;color:var(--ink)!important}
input, textarea, select{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;background:transparent!important}
[data-baseweb="select"] *{color:var(--ink)!important}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label{color:var(--ink2)}
[data-testid="stCaptionContainer"], .stCaption{color:var(--muted)!important}
[data-testid="stSliderThumbValue"]{color:var(--ours)!important}
.stButton>button, .stDownloadButton>button{background:var(--panel);border:1px solid var(--line);color:var(--ink);
  border-radius:9px;font-weight:500;transition:border-color .15s, background .15s}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--ours);color:var(--ink);background:var(--panel2)}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{background:var(--ours);border-color:var(--ours);color:#fff}
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1px dashed var(--line);border-radius:10px}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:transparent}
[data-testid="stExpander"] summary p{color:var(--ink2)}
hr{border-color:var(--line)}
a{color:var(--ours)}

/* sidebar navigation: a radio group restyled as a menu */
[data-testid="stSidebar"] div[role="radiogroup"]{gap:2px;width:100%}
[data-testid="stSidebar"] div[role="radiogroup"] > div{width:100%}
[data-testid="stSidebar"] div[role="radiogroup"] label{padding:8px 12px;border-radius:9px;margin:0;width:100%;
  border:1px solid transparent;transition:background .12s;cursor:pointer}
/* hide the radio circle (older and newer Streamlit markup) */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]){display:none}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(255,255,255,.04)}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:var(--panel2);border-color:var(--line)}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{color:var(--ink)!important;font-weight:600}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span[role="img"]{color:var(--ours)}
[data-testid="stSidebar"] div[role="radiogroup"] p{font-size:.93rem!important;color:var(--ink2)!important}
[data-testid="stSidebar"] div[role="radiogroup"] span[role="img"]{margin-right:6px;font-size:1.15rem}

.brand{display:flex;align-items:center;gap:10px;margin:0 0 .9rem 2px}
.brand .mark{width:32px;height:32px}
.brand .t{font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-weight:600;font-size:1.12rem;color:var(--ink);line-height:1.1}
.brand .t small{display:block;font-family:'Figtree',sans-serif;font-weight:500;font-size:.72rem;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.side-h{color:var(--muted);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;margin:1rem 0 .35rem 2px}
.side-note{color:var(--muted);font-size:.76rem;line-height:1.5;margin-top:.4rem}
.src{display:flex;justify-content:space-between;align-items:center;font-size:.8rem;color:var(--ink2);padding:3px 2px}
.src b{font-weight:500;color:var(--ink2)}

.page-eyebrow{color:var(--muted);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.3rem}
.page-title{font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-weight:600;font-size:2.4rem;line-height:1.05;margin:0 0 .4rem 0;color:var(--ink)}
.page-sub{color:var(--ink2);font-size:1.02rem;max-width:72ch;margin:0 0 1.3rem 0;line-height:1.5}

.hero{position:relative;padding:30px 32px 28px;border:1px solid var(--line);border-radius:20px;margin-bottom:1.3rem;
  background:radial-gradient(900px 280px at 0% 0%, rgba(57,135,229,.18), transparent 62%),
             radial-gradient(700px 260px at 100% 10%, rgba(213,81,129,.13), transparent 60%), var(--panel)}
.hero .page-title{font-size:2.8rem}
.hero .verdict-line{font-size:1.12rem;color:var(--ink);max-width:80ch;line-height:1.55;margin:.2rem 0 0}
.hero .verdict-line b{font-weight:600}

.kpis{display:flex;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:.5rem 0 1.5rem}
.kpi{flex:1 1 150px;padding:14px 20px 14px 18px;border-left:1px solid var(--line)}
.kpi:first-child{border-left:none;padding-left:2px}
.kpi .v{white-space:nowrap;font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-size:clamp(1.3rem,1.7vw,1.85rem);font-weight:600;font-variant-numeric:tabular-nums;color:var(--ink);line-height:1.15}
.kpi .l{color:var(--ink2);font-size:.84rem;margin-top:2px}
.kpi .s{color:var(--muted);font-size:.76rem;margin-top:2px}
.sec{margin:1.8rem 0 .55rem}
.sec h3{font-size:1.26rem;margin:0 0 .15rem}
.sec p{color:var(--ink2);font-size:.92rem;margin:0;max-width:80ch}

.step{border:1px solid var(--line);border-radius:16px;padding:16px 18px 14px;background:var(--panel);height:236px;margin-bottom:8px;
  display:flex;flex-direction:column;gap:4px}
.step .n{font-size:.72rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
.step .h{font-family:'Bricolage Grotesque',sans-serif;font-size:1.12rem;font-weight:600;color:var(--ink)}
.step .big{font-family:'Bricolage Grotesque',sans-serif;font-size:2.1rem;font-weight:600;line-height:1.1;font-variant-numeric:tabular-nums;margin-top:6px}
.step .cap{color:var(--ink2);font-size:.84rem;line-height:1.45}
.step .vs{color:var(--muted);font-size:.8rem}

.badge{display:inline-block;white-space:nowrap;padding:2px 11px;border-radius:99px;font-size:.78rem;font-weight:600;border:1px solid}
.b-legit{color:var(--legit);border-color:rgba(61,220,151,.45);background:rgba(61,220,151,.09)}
.b-review{color:var(--review);border-color:rgba(255,184,77,.45);background:rgba(255,184,77,.09)}
.b-fraud{color:var(--fraud);border-color:rgba(255,107,91,.5);background:rgba(255,107,91,.10)}
.b-neutral{color:var(--ink2);border-color:var(--line);background:rgba(255,255,255,.03)}

.tbl-wrap{margin-bottom:1.1rem;max-height:540px;overflow:auto;border:1px solid var(--line);border-radius:12px}
.tbl{width:100%;border-collapse:collapse;font-size:.86rem}
.tbl th{position:sticky;top:0;background:#10153A;text-align:left;color:var(--ink2);font-weight:500;padding:9px 12px;
  border-bottom:1px solid var(--line);white-space:nowrap;z-index:1}
.tbl td{padding:7px 12px;border-bottom:1px solid rgba(40,48,99,.55);font-variant-numeric:tabular-nums;color:var(--ink)}
.tbl tr:last-child td{border-bottom:none}
.tbl td.k{color:var(--ink2)}
.tbl td.num, .tbl th.num{text-align:right}
.tbl tr.hl td{background:rgba(57,135,229,.07)}
.score-bar{display:inline-block;height:6px;border-radius:3px;vertical-align:middle;margin-right:8px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:-1px}

.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;font-variant-numeric:tabular-nums;max-width:440px}
.cm .h{color:var(--ink2);font-size:.78rem;align-self:end;padding:0 4px}
.cm .r{color:var(--ink2);font-size:.78rem;align-self:center;padding-right:8px;text-align:right}
.cm .c{border-radius:9px;padding:12px 14px;border:1px solid var(--line);background:var(--panel)}
.cm .c b{display:block;font-family:'Bricolage Grotesque','Figtree',system-ui,sans-serif;font-size:1.5rem;font-weight:600}
.cm .c span{color:var(--ink2);font-size:.78rem}
.cm .good{border-color:rgba(61,220,151,.4)}
.cm .bad{border-color:rgba(255,107,91,.45)}
.cm-title{font-weight:600;color:var(--ink);margin:0 0 .5rem;display:flex;align-items:center}

.callout{border-left:2px solid var(--ours);padding:6px 16px;color:var(--ink2);margin:.6rem 0 1rem;font-size:.93rem;max-width:92ch;line-height:1.55}
.callout b{color:var(--ink)}
.story{border:1px solid var(--line);border-radius:16px;background:var(--panel);padding:20px 24px;color:var(--ink2);line-height:1.65;font-size:.96rem}
.story h4{margin:0 0 .5rem;font-size:1.1rem}
.story b{color:var(--ink)}
.story ol{margin:.4rem 0 0 1.1rem;padding:0}
.story li{margin:.35rem 0}
.empty{border:1px dashed var(--line);border-radius:14px;padding:26px 28px;color:var(--ink2);background:rgba(19,25,56,.5)}
.empty h4{margin:0 0 .4rem 0;font-size:1.1rem}
.chip{display:inline-block;padding:1px 9px;border-radius:6px;font-size:.74rem;font-weight:600;margin-left:8px}
.chip.ok{color:var(--legit);background:rgba(61,220,151,.1)}
.chip.watch{color:var(--review);background:rgba(255,184,77,.1)}
.chip.info{color:var(--ink2);background:rgba(255,255,255,.05)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.legend-line{color:var(--ink2);font-size:.85rem;margin:.2rem 0 .6rem}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:.4rem 0 1rem}
.tile{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:var(--panel)}
.tile b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.6rem;font-weight:600;font-variant-numeric:tabular-nums}
.tile span{color:var(--ink2);font-size:.82rem}
@media(max-width:900px){.tiles{grid-template-columns:repeat(2,1fr)}}
.opinion{border:1px solid var(--line);border-radius:12px;padding:12px 16px;background:var(--panel);display:flex;
  justify-content:space-between;align-items:center;gap:12px;margin-top:.6rem}
.opinion .w{color:var(--ink2);font-size:.84rem}
.opinion .p{font-family:'Bricolage Grotesque',sans-serif;font-size:1.3rem;font-weight:600}

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


def page_head(eyebrow: str, title: str, sub: str):
    st.markdown(f"<div class='page-eyebrow'>{esc(eyebrow)}</div><h1 class='page-title'>{esc(title)}</h1>"
                f"<p class='page-sub'>{sub}</p>", unsafe_allow_html=True)


def kpis(items):
    """items: list of (value, label, sub)."""
    cells = "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{esc(l)}</div>"
                    f"{f'<div class=s>{s}</div>' if s else ''}</div>" for v, l, s in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def callout(text: str):
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


def empty(title: str, body: str):
    st.markdown(f"<div class='empty'><h4>{esc(title)}</h4>{body}</div>", unsafe_allow_html=True)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def aed(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"AED {x / 1e6:,.2f}M"
    if abs(x) >= 1e4:
        return f"AED {x / 1e3:,.0f}k"
    return f"AED {x:,.0f}"


def dfkey(df: pd.DataFrame | None) -> str:
    if df is None:
        return "none"
    h = int(pd.util.hash_pandas_object(df, index=False).sum() % (2 ** 63))
    return hashlib.md5(f"{df.shape}-{h}".encode()).hexdigest()


def swatch(color: str) -> str:
    return f"<span class='sw' style='background:{color}'></span>"


def chip(ok: bool, good: str = "in range", bad: str = "check") -> str:
    return f"<span class='chip {'ok' if ok else 'watch'}'>{good if ok else bad}</span>"


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def go_to(page: str):
    st.session_state["nav"] = page


# ---------------------------------------------------------------- plotting
def style(fig: go.Figure, height: int = 340, legend: bool = True, margin=None) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, system-ui, sans-serif", color=C["ink2"], size=12),
        margin=margin or dict(l=8, r=8, t=30, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=C["ink2"], size=12), bgcolor="rgba(0,0,0,0)"),
        showlegend=legend, hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["line"],
                                           font=dict(color=C["ink"], family="Figtree, sans-serif")),
        colorway=[C["ours"], C["noon"], C["real"]],
    )
    fig.update_xaxes(automargin=True, gridcolor="rgba(40,48,99,.45)", zeroline=False, linecolor=C["line"],
                     tickfont=dict(color=C["ink2"]), title_font=dict(color=C["ink2"]))
    fig.update_yaxes(automargin=True, gridcolor="rgba(40,48,99,.45)", zeroline=False, linecolor=C["line"],
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
    except Exception:  # noqa: BLE001
        pass
    return fig


def vline(fig, x, text, row=None, col=None, color=None, log: bool = False):
    """Dashed vertical marker with a label. On a log x-axis the line is drawn as a trace (data units) and the
    label is placed in log10 units, which is how plotly positions annotations on log axes."""
    kw = dict(row=row, col=col) if row else {}
    if log:
        fig.add_trace(go.Scatter(x=[x, x], y=[0, 1], yaxis="y2", mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=color or C["muted"], width=1, dash="dash")))
        fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True, showgrid=False))
        x = float(np.log10(max(x, 1e-12)))
    else:
        fig.add_vline(x=x, line=dict(color=color or C["muted"], width=1, dash="dash"), **kw)
    fig.add_annotation(x=x, y=0.98, yref="paper", text=text, showarrow=False, xanchor="left", xshift=5, yanchor="top",
                       font=dict(color=C["ink2"], size=11), bgcolor=rgba(C["bg"], 0.6))


# --------------------------------------------------------------- data prep
PAY_ALIASES = {"cashondelivery": "cod", "cash": "cod", "creditcard": "card", "debitcard": "card", "credit": "card",
               "debit": "card", "visa": "card", "mastercard": "card", "apple": "applepay", "applepay": "applepay"}
LABEL_ALIASES = ["is_fraud", "isfraud", "fraud", "fraud_flag", "is_fraudulent", "fraudulent", "label", "target",
                 "class", "y", "outcome", "chargeback"]
FEATURE_ALIASES = {
    "order_amount": ["amount", "order_value", "order_total", "total", "basket_value", "value"],
    "account_age_days": ["account_age", "age_days", "customer_age_days", "days_since_signup", "tenure_days"],
    "hour_of_day": ["hour", "order_hour", "time_of_day"],
    "num_orders_last_7d": ["orders_last_7d", "orders_7d", "num_orders_7d", "orders_last_week", "recent_orders"],
    "payment_method": ["payment", "payment_type", "pay_method"],
    "delivery_distance_km": ["distance_km", "distance", "delivery_distance"],
    "discount_applied": ["discount", "promo", "promo_used", "promo_applied", "coupon"],
    "prev_refunds": ["refunds", "previous_refunds", "num_refunds", "refund_count"],
    "device_change": ["new_device", "device_changed"],
    "location_mismatch": ["address_mismatch", "geo_mismatch", "billing_mismatch"],
}


@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "payment_method" in df.columns:
        df["payment_method"] = (df["payment_method"].astype(str).str.strip().str.lower()
                                .str.replace(r"[\s_\-]+", "", regex=True).replace(PAY_ALIASES))
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
    if need_label and not has_label:
        errors.append("This dataset needs a fraud label column with both fraud (1) and legit (0) rows.")
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)
    if errors:
        return None, errors, notes, has_label
    return df, errors, notes, has_label


def _key(x) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def auto_map(columns) -> dict:
    """Best guess of {our column: their column}, matching by name and common aliases."""
    keys = {_key(c): str(c) for c in columns}
    m = {}
    for req in FEATURES:
        for cand in [req] + FEATURE_ALIASES[req]:
            if cand in keys:
                m[req] = keys[cand]
                break
    for cand in LABEL_ALIASES:
        if cand in keys:
            m[TARGET] = keys[cand]
            break
    return m


def to_binary(s: pd.Series) -> pd.Series:
    """Turn a label column (0/1, True/False, yes/no, fraud/legit ...) into 0/1. Unknown values become NaN."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.95:
        u = sorted(num.dropna().unique())
        if len(u) == 1:
            return (num > 0).astype(float).where(num.notna())
        if len(u) == 2:
            return (num == u[1]).astype(float).where(num.notna())
        return pd.Series(np.nan, index=s.index)
    t = s.astype(str).str.strip().str.lower()
    pos = {"1", "true", "yes", "y", "t", "fraud", "fraudulent", "fake", "bad", "positive", "chargeback"}
    neg = {"0", "false", "no", "n", "f", "legit", "legitimate", "genuine", "good", "normal", "negative", "ok", "clean"}
    out = pd.Series(np.nan, index=s.index)
    out[t.isin(pos)] = 1.0
    out[t.isin(neg)] = 0.0
    return out


def canonicalise(raw: pd.DataFrame, m: dict) -> pd.DataFrame:
    out = pd.DataFrame({f: raw[m[f]].values for f in FEATURES})
    if TARGET in m:
        out[TARGET] = to_binary(raw[m[TARGET]]).values
    return out


def label_state(df: pd.DataFrame) -> str:
    """none | both | fraud_only | legit_only"""
    if TARGET not in df.columns:
        return "none"
    u = set(pd.Series(df[TARGET]).dropna().astype(int).unique())
    return "both" if u == {0, 1} else ("fraud_only" if u == {1} else ("legit_only" if u == {0} else "none"))


def ingest(raw: pd.DataFrame, key: str):
    """Let the user match uploaded columns to ours. Returns (canonical_df or None, still_missing_columns)."""
    raw = raw.rename(columns=str)
    auto = auto_map(raw.columns)
    opts = ["(not in file)"] + list(raw.columns)
    fp = hashlib.md5(("|".join(raw.columns) + str(len(raw))).encode()).hexdigest()[:8]
    missing_auto = [f for f in FEATURES if f not in auto]
    renamed = {f: c for f, c in auto.items() if _key(c) != f}
    chosen = {}
    with st.expander("Match columns" + (" (needs your attention)" if missing_auto else ""), expanded=bool(missing_auto)):
        if missing_auto:
            st.caption("Some columns were not found by name. Pick the matching column from the file for each one.")
        else:
            st.caption("Columns were matched by name" + (f"; renamed: {', '.join(f'{c} as {f}' for f, c in renamed.items())}" if renamed else "")
                       + ". Change a match if it is wrong.")
        grid = st.columns(3)
        for i, f in enumerate(FEATURES + [TARGET]):
            d = auto.get(f)
            chosen[f] = grid[i % 3].selectbox(PRETTY.get(f, "Fraud label (is_fraud)"), opts,
                                              index=opts.index(d) if d in opts else 0, key=f"{key}_{fp}_{f}")
    m = {f: c for f, c in chosen.items() if c != opts[0]}
    missing = [f for f in FEATURES if f not in m]
    if missing:
        return None, missing
    return canonicalise(raw, m), []


def load_frame(raw_bytes: bytes, need_label: bool):
    """Auto-mapped load used for the four known files. Returns (df, error)."""
    raw = parse_csv(raw_bytes).rename(columns=str)
    m = auto_map(raw.columns)
    miss = [f for f in FEATURES if f not in m]
    if need_label and TARGET not in m:
        miss.append(TARGET)
    if miss:
        return None, "Could not find columns for: " + ", ".join(miss) + "."
    df, errs, _, _ = prepare(canonicalise(raw, m), need_label=need_label)
    return df, " ".join(errs)


# ---------------------------------------------------------------- modelling
def encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC + BINARY].astype(float).copy()
    for p in PAYMENTS:
        X["pay_" + p] = (df["payment_method"] == p).astype(float)
    return X


def dummies(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Noon-style one-hot encoding (pd.get_dummies), aligned to a column list."""
    X = pd.get_dummies(df[FEATURES], columns=["payment_method"], prefix="payment_method")
    return X.reindex(columns=columns, fill_value=0).astype(float)


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
    y = np.asarray(y)
    pred = np.asarray(p) >= t
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, flagged=(tp + fp) / max(len(y), 1))


def caught_at_budget(y, p, k: int) -> int:
    """Frauds among the k highest-scored orders."""
    k = int(min(max(k, 0), len(y)))
    if k == 0:
        return 0
    idx = np.argsort(-np.asarray(p), kind="stable")[:k]
    return int(np.asarray(y)[idx].sum())


def tier(p: float, review_from: float, cutoff: float) -> str:
    return "fraud" if p >= cutoff else ("review" if p >= review_from else "legit")


TIER_TEXT = {"legit": "Looks legitimate", "review": "Needs review", "fraud": "Likely fraud"}
TIER_COLOR = {"legit": C["legit"], "review": C["review"], "fraud": C["fraud"]}


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, allv, side="right") / len(a)
                               - np.searchsorted(b, allv, side="right") / len(b))))


def tvd(a: pd.Series, b: pd.Series) -> float:
    ca, cb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    idx = ca.index.union(cb.index)
    return float(0.5 * np.abs(ca.reindex(idx, fill_value=0) - cb.reindex(idx, fill_value=0)).sum())


CONTINUOUS = ["order_amount", "account_age_days", "delivery_distance_km"]
CORR_COLS = NUMERIC + BINARY


def gap(a: pd.DataFrame, b: pd.DataFrame, f: str) -> float:
    return ks_stat(a[f].values.astype(float), b[f].values.astype(float)) if f in CONTINUOUS else tvd(a[f], b[f])


def single_aucs(df: pd.DataFrame, y: np.ndarray) -> dict:
    X = encode(df)
    cols = {f: ([f] if f != "payment_method" else ["pay_" + p for p in PAYMENTS]) for f in FEATURES}
    return {f: max(max(a, 1 - a) for a in (fast_auc(y, X[c]) for c in cs)) for f, cs in cols.items()}


# ------------------------------------------------------------- Noon's model
class NoonModel:
    """Wraps Noon's joblib bundle ({'model', 'columns', ...}) or a bare fitted estimator."""

    def __init__(self, obj):
        if isinstance(obj, dict):
            self.model, cols = obj["model"], obj.get("columns")
        else:
            self.model, cols = obj, None
        if cols is None:
            cols = list(getattr(self.model, "feature_names_in_", EXPORT_COLS))
        self.columns = list(cols)
        self.kind = type(self.model).__name__

    def proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(dummies(df, self.columns))[:, 1]


@st.cache_resource(show_spinner="Loading Noon's model...")
def load_noon(key: str, _raw: bytes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NoonModel(joblib.load(io.BytesIO(_raw)))


@dataclass
class Scorer:
    key: str
    name: str
    color: str
    cutoff: float
    review: float
    fn: Callable[[pd.DataFrame], np.ndarray]


# ======================================================================
#  Cached analysis
# ======================================================================
@st.cache_resource(show_spinner="Training the qualifier on our dataset...")
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


@st.cache_data(show_spinner="Scoring the referee set with both models...")
def referee_scores(_ref: pd.DataFrame, _model, _noon, key: str):
    p_ours = proba(_model, _ref)
    p_noon = _noon.proba(_ref) if _noon is not None else None
    return p_ours, p_noon


@st.cache_data(show_spinner="Running the swap experiment...")
def swap_experiment(_ours: pd.DataFrame, _nf: pd.DataFrame, _ref: pd.DataFrame, budget: int, key: str):
    """Keep our legit rows and the model fixed; swap which fraud rows it learns from; score on real data."""
    legit, fraud = _ours[_ours[TARGET] == 0], _ours[_ours[TARGET] == 1]
    nf = _nf.assign(**{TARGET: 1})[FEATURES + [TARGET]]
    same = nf.sample(min(len(fraud), len(nf)), random_state=0)
    variants = [
        ("As built: our legit + our GAN fraud", pd.concat([legit, fraud]), "ours"),
        (f"Our legit + Noon GAN fraud (same {len(same):,} rows)", pd.concat([legit, same]), "swap"),
        (f"Our legit + all {len(nf):,} Noon GAN fraud", pd.concat([legit, nf]), "swap"),
        ("Our legit + our fraud + Noon fraud", pd.concat([legit, fraud, nf]), "mix"),
    ]
    yr = _ref[TARGET].values
    rows = []
    for name, d, kind in variants:
        m = make_model().fit(encode(d), d[TARGET].values)
        p = proba(m, _ref)
        c = caught_at_budget(yr, p, budget)
        rows.append(dict(variant=name, kind=kind, fraud_share=float(d[TARGET].mean()), pr_auc=float(average_precision_score(yr, p)),
                         roc_auc=float(roc_auc_score(yr, p)), caught=c, false_alarms=budget - c))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running the lie-detector...")
def lie_detector(_a: pd.DataFrame, _b: pd.DataFrame, key: str):
    """Can a classifier tell two sets of rows apart? Returns AUC (gradient boosting, 5-fold) and the
    3-fold random-forest accuracy that A_eval_helper.lie_detector reports."""
    sa = _a.sample(min(3000, len(_a)), random_state=1)
    sb = _b.sample(min(3000, len(_b)), random_state=1)
    D = pd.concat([sa[FEATURES], sb[FEATURES]], ignore_index=True)
    y = np.r_[np.zeros(len(sa)), np.ones(len(sb))].astype(int)
    oof, imp = cv_oof(D, y, importance=True)
    acc = float(cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), encode(D), y, cv=3).mean())
    return dict(auc=fast_auc(y, oof), rf_acc=acc, imp=imp)


@st.cache_data(show_spinner="Comparing the two datasets...")
def compare_core(_a: pd.DataFrame, _b: pd.DataFrame, ka: str, kb: str, labelled: bool):
    out = dict(gaps={f: gap(_a, _b, f) for f in FEATURES})
    if labelled:
        ya, yb = _a[TARGET].values, _b[TARGET].values
        oof_a, _ = cv_oof(_a, ya, importance=False)
        oof_b, _ = cv_oof(_b, yb, importance=False)
        out["cv_a"], out["cv_b"] = fast_auc(ya, oof_a), fast_auc(yb, oof_b)
        out["single_a"], out["single_b"] = single_aucs(_a, ya), single_aucs(_b, yb)
        out["sig_a"], out["sig_b"] = signature(_a), signature(_b)
        model_a = make_model().fit(encode(_a), ya)
        model_b = make_model().fit(encode(_b), yb)
        out["a_on_b"] = fast_auc(yb, proba(model_a, _b))
        out["b_on_a"] = fast_auc(ya, proba(model_b, _a))
    cc = CORR_COLS + ([TARGET] if labelled else [])
    out["corr_a"], out["corr_b"] = _a[cc].corr(), _b[cc].corr()
    return out


@st.cache_data(show_spinner=False)
def pca_map(_frames: tuple, key: str, n: int = 1400):
    """Project several frames onto the first two principal components of their pooled, standardised features.
    Skewed counts and amounts are log-scaled first so a few big orders do not dominate."""
    enc = []
    for i, d in enumerate(_frames):
        s = d.sample(min(n, len(d)), random_state=i)
        X = encode(s)
        for c in ["order_amount", "account_age_days", "delivery_distance_km", "num_orders_last_7d", "prev_refunds"]:
            X[c] = np.log1p(X[c].clip(lower=0))
        enc.append(X)
    allx = pd.concat(enc)
    mu, sd = allx.mean(), allx.std().replace(0, 1)
    Z = ((allx - mu) / sd).values
    _, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    out, start = [], 0
    for X in enc:
        z = ((X - mu) / sd).values @ vt[:2].T
        out.append(z)
        start += len(X)
    load = pd.DataFrame(vt[:2].T, index=allx.columns, columns=["pc1", "pc2"])
    return out, var[:2], load


@st.cache_data(show_spinner=False)
def export_bytes(_df: pd.DataFrame, cutoff: float, key: str) -> bytes:
    """A model in the format the Phase 4 shell expects: {'model', 'columns'} with get_dummies columns.
    Wrapped in FixedThresholdClassifier so model.predict() applies our cut-off, not 0.5."""
    X = dummies(_df, EXPORT_COLS)
    y = _df[TARGET].values
    try:
        from sklearn.model_selection import FixedThresholdClassifier
        model = FixedThresholdClassifier(make_model(), threshold=cutoff, response_method="predict_proba").fit(X, y)
    except Exception:  # noqa: BLE001  (older scikit-learn: plain model, 0.5 threshold)
        model = make_model().fit(X, y)
    buf = io.BytesIO()
    joblib.dump({"model": model, "columns": EXPORT_COLS, "cutoff": cutoff}, buf)
    return buf.getvalue()


def explain(model, row: dict, bg: pd.DataFrame, n: int = 80, seed: int = 0):
    """Monte-Carlo Shapley values: walk from a typical legit order to this order, one detail at a time,
    in random orders, and average each detail's effect on the fraud probability."""
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
        gap_ = 0.014
        x0, y0 = pt(t0 + gap_, r)
        x1, y1 = pt(t1 - gap_, r)
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
<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:420px" role="img"
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
        font-family="Figtree, sans-serif">{lift:.1f}x the {pct(base)} training average</text>
</svg>"""
    return f"<div class='gauge-wrap'>{svg}</div>"


def badge(kind: str, text: str | None = None) -> str:
    return f"<span class='badge b-{kind}'>{esc(text or TIER_TEXT.get(kind, kind))}</span>"


def risk_table(out: pd.DataFrame, limit: int = 150, other: str | None = None, other_name: str = "") -> str:
    cols = FEATURES
    show_actual = TARGET in out.columns
    head = ("<tr><th>Verdict</th><th>Risk</th>" + (f"<th>{esc(other_name)}</th>" if other else "")
            + ("<th>Actual</th>" if show_actual else "") + "".join(f"<th>{esc(SHORT.get(c, c))}</th>" for c in cols) + "</tr>")
    pmax = max(float(out["risk"].max()), 1e-6)
    rows = []
    for _, r in out.head(limit).iterrows():
        tr = r["verdict_key"]
        w = int(70 * min(r["risk"] / pmax, 1))
        cells = (f"<td>{badge(tr)}</td><td><span class='score-bar' style='width:{max(w, 3)}px;background:{TIER_COLOR[tr]}'></span>"
                 f"{pct(r['risk'])}</td>")
        if other:
            cells += f"<td>{pct(r[other])}</td>"
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


def confusion_html(m: dict, title: str = "", color: str | None = None) -> str:
    t = f"<div class='cm-title'>{swatch(color) if color else ''}{esc(title)}</div>" if title else ""
    return f"""{t}
<div class='cm'>
  <div></div><div class='h'>Cleared</div><div class='h'>Flagged as fraud</div>
  <div class='r'>Actually legit</div>
  <div class='c good'><b>{m['tn']:,}</b><span>correctly cleared</span></div>
  <div class='c bad'><b>{m['fp']:,}</b><span>false alarms</span></div>
  <div class='r'>Actually fraud</div>
  <div class='c bad'><b>{m['fn']:,}</b><span>missed</span></div>
  <div class='c good'><b>{m['tp']:,}</b><span>caught</span></div>
</div>"""


def html_table(header: list[str], rows: list[list[str]], num_cols: set[int] = frozenset(), hl: set[int] = frozenset()) -> str:
    th = "".join(f"<th class='{'num' if i in num_cols else ''}'>{h}</th>" for i, h in enumerate(header))
    body = "".join(
        f"<tr class='{'hl' if j in hl else ''}'>" + "".join(
            f"<td class='{'num' if i in num_cols else ('k' if i == 0 else '')}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for j, r in enumerate(rows))
    return f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def signal_bars(sig: pd.DataFrame, colors: list[str], xtitle: str, height: int = 500) -> go.Figure:
    """Grouped horizontal bars of standardised fraud signals, one series per column of `sig`."""
    sig = sig.reindex(sig.abs().max(axis=1).sort_values().index)
    fig = go.Figure()
    for nm, col in zip(sig.columns, colors):
        fig.add_trace(go.Bar(x=sig[nm], y=[PRETTY[i] for i in sig.index], orientation="h", name=nm, marker=dict(color=col),
                             hovertemplate=f"{nm}<br>%{{y}}<br>gap %{{x:.2f}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=C["line"], width=1))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(title=xtitle)
    return style(rounded(fig), height)


def dist_figure(frames: list[tuple[str, pd.DataFrame, str]], feats: list[str] | None = None, cols: int = 2) -> go.Figure:
    feats = feats or FEATURES
    rows = int(np.ceil(len(feats) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[PRETTY[f] for f in feats], vertical_spacing=0.42 / rows,
                        horizontal_spacing=0.08)
    for i, f in enumerate(feats):
        r, c = i // cols + 1, i % cols + 1
        first = i == 0
        discrete = f not in CONTINUOUS
        if discrete:
            cap = 10 if f in ("num_orders_last_7d", "prev_refunds") else None
            series = [(nm, d[f].clip(upper=cap) if cap else d[f], col) for nm, d, col in frames]
            counts = [(nm, s.value_counts(normalize=True), col) for nm, s, col in series]
            cats = sorted(set().union(*[set(cn.index) for _, cn, _ in counts]), key=lambda v: (str(type(v)), v))
            xs = [("10+" if cap and v == cap else str(int(v)) if not isinstance(v, str) else v) for v in cats]
            for nm, cn, col in counts:
                fig.add_trace(go.Bar(x=xs, y=[cn.get(v, 0) for v in cats], name=nm, marker=dict(color=col),
                                     legendgroup=nm, showlegend=first,
                                     hovertemplate=f"{nm}<br>%{{x}}: %{{y:.1%}}<extra></extra>"), row=r, col=c)
        else:
            both = np.concatenate([d[f].values for _, d, _ in frames])
            hi = float(np.quantile(both, 0.995))
            edges = np.linspace(float(both.min()), hi, 31)
            ctr = (edges[:-1] + edges[1:]) / 2
            for nm, d, col in frames:
                h, _ = np.histogram(d[f].values, bins=edges)
                fig.add_trace(go.Scatter(x=ctr, y=h / max(len(d), 1), mode="lines", name=nm, legendgroup=nm,
                                         showlegend=first, line=dict(color=col, width=2.2, shape="spline"),
                                         fill="tozeroy", fillcolor=rgba(col, 0.12),
                                         hovertemplate=f"{nm}<br>%{{x:.1f}}: %{{y:.1%}} of rows<extra></extra>"), row=r, col=c)
        fig.update_yaxes(tickformat=".0%", row=r, col=c)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    for ann in fig.layout.annotations:
        ann.font = dict(color=C["ink"], size=13, family="Bricolage Grotesque, Figtree, sans-serif")
    return style(rounded(fig), 236 * rows + 60, margin=dict(l=8, r=8, t=44, b=8))


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


def score_density(groups: list[tuple[str, np.ndarray, str, str]], cuts: list[tuple[float, str]], xmax: float | None = None,
                  height: int = 310, xtitle: str = "Fraud score") -> go.Figure:
    allv = np.concatenate([g[1] for g in groups])
    hi = xmax or float(np.quantile(allv, 0.995))
    hi = max(hi, max(c for c, _ in cuts) * 1.3 if cuts else hi)
    edges = np.linspace(0, min(hi, 1.0), 40)
    ctr = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    for nm, vals, col, dash in groups:
        h, _ = np.histogram(np.clip(vals, 0, edges[-1]), bins=edges)
        fig.add_trace(go.Scatter(x=ctr, y=h / max(len(vals), 1), mode="lines", name=nm,
                                 line=dict(color=col, width=2.4, dash=dash, shape="spline"),
                                 fill="tozeroy" if dash == "solid" else None, fillcolor=rgba(col, 0.12),
                                 hovertemplate=f"{nm}<br>Score %{{x:.0%}}: %{{y:.1%}} of group<extra></extra>"))
    for x, txt in cuts:
        vline(fig, x, txt)
    fig.update_xaxes(title=xtitle, tickformat=".0%")
    fig.update_yaxes(title="Share of group", tickformat=".0%")
    return style(fig, height, margin=dict(l=8, r=8, t=40, b=8))


# ======================================================================
#  Context: everything a page needs
# ======================================================================
@dataclass
class Ctx:
    df: pd.DataFrame
    B: dict
    ours: Scorer
    noon: Scorer | None
    noon_model: NoonModel | None
    nf: pd.DataFrame | None        # Noon synthetic fraud
    ref: pd.DataFrame | None       # real referee set
    label: str
    name_ours: str
    name_noon: str


def ref_eval(ctx: Ctx):
    """Scores, metrics and budget for the referee set. Returns None when the set is missing."""
    if ctx.ref is None:
        return None
    key = dfkey(ctx.ref) + dfkey(ctx.df) + ("noon" if ctx.noon_model is not None else "")
    p_o, p_n = referee_scores(ctx.ref, ctx.B["model"], ctx.noon_model, key)
    y = ctx.ref[TARGET].values
    m_o = at_cutoff(y, p_o, ctx.ours.cutoff)
    m_n = at_cutoff(y, p_n, ctx.noon.cutoff) if p_n is not None else None
    budget = (m_n["tp"] + m_n["fp"]) if m_n else NOON_BENCH["tp"] + NOON_BENCH["fp"]
    return dict(y=y, p_o=p_o, p_n=p_n, m_o=m_o, m_n=m_n, budget=int(budget), n=len(y), nf=int(y.sum()))


# ======================================================================
#  Page: Briefing
# ======================================================================
def page_briefing(ctx: Ctx):
    B, df = ctx.B, ctx.df
    R = ref_eval(ctx)
    m_oof = at_cutoff(B["y"], B["oof"], ctx.ours.cutoff)

    if R:
        mo = R["m_o"]
        mn = R["m_n"] or NOON_BENCH
        same = caught_at_budget(R["y"], R["p_o"], R["budget"])
        line = (f"On <b>{R['n']:,} real orders</b> ({R['nf']} of them fraud), our qualifier caught <b>{mo['tp']} of {R['nf']}</b> "
                f"frauds with <b>{mo['fp']:,}</b> false alarms. Noon's caught <b>{mn['tp']}</b> with <b>{mn['fp']}</b>. "
                f"Given Noon's review budget of {R['budget']} flags, ours would catch <b>{same}</b>.")
    else:
        line = (f"Trained on <b>{B['n']:,}</b> orders with a <b>{pct(B['base'])}</b> fraud rate. "
                f"Add the real referee file to see how it holds up against Noon.")
    st.markdown(f"<div class='hero'><div class='page-eyebrow'>Noon food delivery · fraud challenge</div>"
                f"<h1 class='page-title'>Fraud intelligence desk</h1><p class='verdict-line'>{line}</p></div>",
                unsafe_allow_html=True)

    # ---- the journey, one card per phase
    lie = None
    if ctx.nf is not None:
        fa = df[df[TARGET] == 1]
        lie = lie_detector(fa, ctx.nf, dfkey(fa) + dfkey(ctx.nf))
    cards = [
        ("Phase 1", "Build blind", pct(m_oof["recall"], 0), C["ours"],
         f"of our own fraud caught out-of-fold at the {pct(ctx.ours.cutoff, 0)} cut-off. {B['n']:,} rows, {pct(B['base'])} fraud.",
         "Model lab"),
        ("Phase 2", "GAN face-off", f"{lie['auc']:.2f}" if lie else "–", C["noon"],
         ("AUC for telling our fraud from Noon's. 0.50 would mean identical; ours is "
          + ("easy to tell apart." if lie and lie["auc"] > 0.8 else "fairly close.")) if lie else "Add B_NOON_synthetic_fraud.csv.",
         "GAN face-off"),
        ("Phase 3", "Real-data referee", f"{R['m_o']['tp']} / {R['nf']}" if R else "–", C["real"],
         (f"real frauds caught vs Noon's {(R['m_n'] or NOON_BENCH)['tp']}. "
          f"{R['m_o']['fp']:,} false alarms vs {(R['m_n'] or NOON_BENCH)['fp']}.") if R else "Add C_REFEREE_real_test.csv.",
         "Real-world referee"),
        ("Phase 4", "Ship it", "Live", C["legit"],
         "Score any order with reasons, classify a file, and export fraud_model.joblib for the shell app.",
         "Score an order"),
    ]
    cols = st.columns(4, gap="small")
    for col, (n, h, big, color, cap, target) in zip(cols, cards):
        with col:
            st.markdown(f"<div class='step' style='border-top:3px solid {color}'><div class='n'>{n}</div><div class='h'>{h}</div>"
                        f"<div class='big'>{esc(big)}</div><div class='cap'>{esc(cap)}</div></div>", unsafe_allow_html=True)
            st.button(f"Open {target}  →", key=f"go_{n}", on_click=go_to, args=(target,), **BTN_W)

    if R:
        section("The final scoreboard", "Both classifiers on the real referee set, which neither ever trained on.")
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            show(scoreboard_figure(R, ctx), key="brief_board")
        with c2:
            show(counts_figure(R, ctx), key="brief_counts")
        takeaways(ctx, R, lie)
    else:
        section("Our qualifier at a glance")
        kpis([(f"{B['n']:,}", "Training rows", f"{int(B['y'].sum()):,} fraud"),
              (f"{B['roc_auc']:.2f}", "ROC AUC (out-of-fold)", ""),
              (f"{B['pr_auc']:.2f}", "PR AUC (out-of-fold)", f"guessing scores {B['base']:.2f}"),
              (pct(m_oof["recall"], 0), "Recall at cut-off", ""), (pct(m_oof["precision"], 0), "Precision at cut-off", "")])


def scoreboard_figure(R, ctx: Ctx) -> go.Figure:
    mo = R["m_o"]
    mn = R["m_n"] or NOON_BENCH
    metrics = ["Recall (fraud caught)", "Precision (flags that are real)", "F1"]
    series = [(ctx.name_ours, [mo["recall"], mo["precision"], mo["f1"]], C["ours"]),
              ("Noon: real + GAN" + ("" if R["m_n"] else " (benchmark)"), [mn["recall"], mn["precision"], mn["f1"]], C["noon"]),
              ("Noon: real data only (benchmark)", [REAL_ONLY_BENCH["recall"], REAL_ONLY_BENCH["precision"], REAL_ONLY_BENCH["f1"]], C["ghost"])]
    fig = go.Figure()
    for nm, vals, col in series:
        fig.add_trace(go.Bar(x=metrics, y=vals, name=nm, marker=dict(color=col), text=[f"{v:.2f}" for v in vals],
                             textposition="outside", textfont=dict(color=C["ink2"], size=12), cliponaxis=False,
                             hovertemplate=f"{nm}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", title="Score on real data")
    return style(rounded(fig), 360)


def counts_figure(R, ctx: Ctx) -> go.Figure:
    mo = R["m_o"]
    mn = R["m_n"] or NOON_BENCH
    names = [ctx.name_ours, "Noon"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Real frauds caught (of {R['nf']})", "Innocent orders flagged"],
                        horizontal_spacing=0.18)
    for i, (vals, fmt) in enumerate((([mo["tp"], mn["tp"]], "{:,}"), ([mo["fp"], mn["fp"]], "{:,}"))):
        fig.add_trace(go.Bar(x=names, y=vals, marker=dict(color=[C["ours"], C["noon"]]), text=[fmt.format(v) for v in vals],
                             textposition="outside", textfont=dict(color=C["ink2"]), cliponaxis=False, showlegend=False,
                             hovertemplate="%{x}: %{y:,}<extra></extra>"), row=1, col=i + 1)
        fig.update_yaxes(range=[0, max(vals) * 1.25 + 1], row=1, col=i + 1)
    fig.update_yaxes(range=[0, R["nf"] * 1.18], row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(color=C["ink"], size=13, family="Bricolage Grotesque, Figtree, sans-serif")
    return style(rounded(fig), 360, legend=False, margin=dict(l=8, r=8, t=44, b=8))


def takeaways(ctx: Ctx, R, lie):
    mo, mn = R["m_o"], R["m_n"] or NOON_BENCH
    pr_o = average_precision_score(R["y"], R["p_o"])
    pr_n = average_precision_score(R["y"], R["p_n"]) if R["p_n"] is not None else None
    rank_auc = roc_auc_score(R["y"], R["p_o"])
    bits = [f"<li><b>Ranking is decent, precision is not.</b> Our ROC AUC on real data is {rank_auc:.2f}, but PR AUC is "
            f"{pr_o:.2f}" + (f" against Noon's {pr_n:.2f}" if pr_n is not None else "") +
            ". With fraud at 0.3%, every small overlap between legit and fraud scores becomes hundreds of false alarms.</li>"]
    if lie:
        bits.append(f"<li><b>Our GAN fraud does not look like Noon's.</b> A classifier separates the two with AUC {lie['auc']:.2f} "
                    f"(random-forest accuracy {lie['rf_acc']:.2f}). Our GAN could only learn from our guesses.</li>")
    bits.append(f"<li><b>The gap is the data, not the algorithm.</b> The Real-world referee page swaps Noon's GAN fraud into our "
                f"training set with the model held fixed. That test shows how much of the {mn['tp'] - mo['tp']:+d} difference in "
                f"frauds caught comes from the examples the model learned from.</li>")
    st.markdown(f"<div class='story'><h4>What the numbers say</h4><ol>{''.join(bits)}</ol></div>", unsafe_allow_html=True)


# ======================================================================
#  Page: Model lab (our model on our own data)
# ======================================================================
def page_model_lab(ctx: Ctx):
    B, df = ctx.B, ctx.df
    cutoff, review_from = ctx.ours.cutoff, ctx.ours.review
    y, oof = B["y"], B["oof"]
    m = at_cutoff(y, oof, cutoff)
    page_head("Phase 1 · our qualifier", "Model lab",
              f"Gradient boosting trained on <b>{esc(ctx.label)}</b>. Every number here is out-of-fold: each order was scored by "
              f"a model that never saw it. These numbers describe our own synthetic world; the referee page shows the real one.")
    kpis([(f"{B['n']:,}", "Transactions", f"{int(y.sum()):,} labelled fraud"),
          (pct(B["base"], 2), "Fraud rate", "real referee data: 0.33%"),
          (f"{B['roc_auc']:.2f}", "ROC AUC", "0.50 is a coin flip"),
          (f"{B['pr_auc']:.2f}", "PR AUC", f"{B['base']:.2f} is what guessing scores"),
          (pct(m["precision"], 0), "Precision at cut-off", f"{m['tp']} of {m['tp'] + m['fp']} flags are fraud"),
          (pct(m["recall"], 0), "Recall at cut-off", f"{m['tp']} of {m['tp'] + m['fn']} fraud caught")])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Trade-off by cut-off", "Precision, recall and F1 as the cut-off moves. The dashed line is the current setting.")
        pr, rc, th = precision_recall_curve(y, oof)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        fig = go.Figure()
        for nm, vals, dash, col in (("Recall", rc[:-1], "solid", C["ink"]), ("Precision", pr[:-1], "dash", C["ink2"]),
                                    ("F1", f1[:-1], "dot", C["ours"])):
            fig.add_trace(go.Scatter(x=th, y=vals, mode="lines", name=nm, line=dict(color=col, width=2.2, dash=dash),
                                     hovertemplate=f"Cut-off %{{x:.0%}}<br>{nm} %{{y:.1%}}<extra></extra>"))
        vline(fig, cutoff, "cut-off")
        fig.update_xaxes(title="Cut-off", tickformat=".0%", range=[0, min(1, max(0.6, cutoff * 2.5))])
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        show(style(fig, 330))
    with c2:
        section("Where the scores land", "Share of each group by fraud score. Fraud sits to the right, but the two overlap.")
        show(score_density([("Legit orders", oof[y == 0], C["legit"], "solid"), ("Fraud orders", oof[y == 1], C["fraud"], "solid")],
                           [(review_from, "review"), (cutoff, "fraud")]), key="lab_density")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        section("Ranking quality", "ROC and precision-recall curves. Dots mark the current cut-off.")
        fpr, tpr, _ = roc_curve(y, oof)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC", "Precision vs recall"], horizontal_spacing=0.14)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["muted"], width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C["ours"], width=2.4), showlegend=False,
                                 hovertemplate="False alarms %{x:.1%}<br>Fraud caught %{y:.1%}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[m["fp"] / max(m["fp"] + m["tn"], 1)], y=[m["recall"]], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=C["ink"], line=dict(color=C["bg"], width=3)),
                                 hovertemplate="Current cut-off<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color=C["ours"], width=2.4), showlegend=False,
                                 hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=C["ink"], line=dict(color=C["bg"], width=3)),
                                 hovertemplate="Current cut-off<extra></extra>"), row=1, col=2)
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        for ann in fig.layout.annotations:
            ann.font = dict(color=C["ink"], size=13)
        show(style(fig, 320, legend=False, margin=dict(l=8, r=8, t=40, b=8)))
    with c4:
        section("What the desk decides", "Out-of-fold results at the current cut-off.")
        st.markdown(confusion_html(m), unsafe_allow_html=True)
        callout(f"At this cut-off the desk flags <b>{pct(m['flagged'])}</b> of orders. Move the cut-off in the sidebar to "
                f"trade catches for false alarms; the suggested value balances precision and recall on our data.")

    c5, c6 = st.columns(2, gap="large")
    with c5:
        section("What the qualifier relies on", "Drop in AUC when a column is shuffled. Bigger means the model leans on it more.")
        imp = pd.Series(B["imp"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[PRETTY[i] for i in imp.index], orientation="h",
                               marker=dict(color=C["ours"]), hovertemplate="%{y}<br>AUC drop %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="AUC drop when shuffled")
        show(style(rounded(fig), 380, legend=False))
    with c6:
        section("What fraud looks like in our data", "Standardised gap between fraud and legit rows. Right: higher in fraud.")
        sig = B["sig"].sort_values()
        names = [PRETTY[i] for i in sig.index]
        fig = go.Figure()
        for nm, col, mask in (("Higher in fraud", C["fraud"], sig.values > 0), ("Lower in fraud", C["low"], sig.values <= 0)):
            fig.add_trace(go.Bar(x=np.where(mask, sig.values, np.nan), y=names, orientation="h", name=nm,
                                 marker=dict(color=col), hovertemplate="%{y}<br>gap %{x:.2f}<extra></extra>"))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="Standardised difference")
        show(style(rounded(fig), 380))

    c7, c8 = st.columns(2, gap="large")
    with c7:
        section("Are the scores honest probabilities?", "Average predicted risk vs the observed fraud rate, in ten bins of orders.")
        bins = pd.qcut(pd.Series(oof).rank(method="first"), 10, labels=False)
        cal = pd.DataFrame({"p": oof, "y": y, "b": bins}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        top = float(max(cal.p.max(), cal.y.max()) * 1.1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=C["muted"], dash="dot", width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=cal.p, y=cal.y, mode="lines+markers", line=dict(color=C["ours"], width=2.2),
                                 marker=dict(size=9, color=C["ours"], line=dict(color=C["bg"], width=2)), showlegend=False,
                                 customdata=cal.n, hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>%{customdata} orders<extra></extra>"))
        fig.update_xaxes(title="Average predicted risk", tickformat=".0%", range=[0, top])
        fig.update_yaxes(title="Observed fraud rate", tickformat=".0%", range=[0, top])
        show(style(fig, 330, legend=False))
    with c8:
        section("How one detail moves risk", "Average predicted risk across 400 sampled orders as one detail is set to each value.")
        f = st.selectbox("Detail", [x for x in FEATURES], format_func=lambda x: PRETTY[x], key="pdp_feat", label_visibility="collapsed")
        show(pdp_figure(B["model"], df, f), key="lab_pdp")

    with st.expander("Preview the training data, highest risk first"):
        st.markdown(risk_table(df.assign(risk=oof, verdict_key=[tier(p, review_from, cutoff) for p in oof])
                               .sort_values("risk", ascending=False), limit=60), unsafe_allow_html=True)


SWEEP = {
    "order_amount": np.linspace(10, 500, 40), "account_age_days": np.r_[1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 540, 730, 1000],
    "hour_of_day": np.arange(24), "num_orders_last_7d": np.arange(0, 13), "delivery_distance_km": np.linspace(0.3, 20, 34),
    "prev_refunds": np.arange(0, 8), "discount_applied": np.array([0, 1]), "device_change": np.array([0, 1]),
    "location_mismatch": np.array([0, 1]), "payment_method": np.array(PAYMENTS),
}


def pdp_figure(model, df: pd.DataFrame, f: str) -> go.Figure:
    s = df.sample(min(400, len(df)), random_state=5).reset_index(drop=True)
    vals = SWEEP[f]
    big = pd.concat([s.assign(**{f: v}) for v in vals], ignore_index=True)
    p = proba(model, big).reshape(len(vals), len(s)).mean(axis=1)
    fig = go.Figure()
    if f in BINARY or f == "payment_method":
        xs = [fmt_value(f, v) if f != "payment_method" else fmt_value(f, v) for v in vals]
        fig.add_trace(go.Bar(x=xs, y=p, marker=dict(color=C["ours"]), hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=vals, y=p, mode="lines", line=dict(color=C["ours"], width=2.4, shape="spline"),
                                 fill="tozeroy", fillcolor=rgba(C["ours"], .10), hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        if f == "account_age_days":
            fig.update_xaxes(type="log")
    fig.update_xaxes(title=PRETTY[f])
    fig.update_yaxes(title="Average fraud risk", tickformat=".0%", rangemode="tozero")
    return style(rounded(fig), 300, legend=False)


# ======================================================================
#  Page: Score one order
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
FLIP_GRID = {
    "order_amount": [15, 30, 50, 80, 120, 180, 250, 350, 500], "account_age_days": [1, 3, 7, 14, 30, 90, 180, 365, 730],
    "hour_of_day": list(range(24)), "num_orders_last_7d": [0, 1, 2, 3, 4, 6, 8, 10],
    "delivery_distance_km": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0], "prev_refunds": [0, 1, 2, 3, 5],
    "discount_applied": [0, 1], "device_change": [0, 1], "location_mismatch": [0, 1], "payment_method": PAYMENTS,
}


def apply_preset(name: str):
    for f, v in PRESETS[name].items():
        k = FORM_KEYS[f]
        st.session_state[k] = bool(v) if f in BINARY else (float(v) if f in ("order_amount", "delivery_distance_km") else v)


def load_row_into_form(row: dict):
    for f, k in FORM_KEYS.items():
        v = row[f]
        st.session_state[k] = bool(int(v)) if f in BINARY else (float(v) if f in ("order_amount", "delivery_distance_km")
                                                                 else (str(v) if f == "payment_method" else int(v)))


def flip_finder(scorer: Scorer, row: dict, p0: float) -> list[dict]:
    """Single-detail changes that move this order across the fraud cut-off, best per detail."""
    cands = []
    for f, vals in FLIP_GRID.items():
        for v in vals:
            if (f == "payment_method" and v == row[f]) or (f != "payment_method" and float(v) == float(row[f])):
                continue
            r = dict(row)
            r[f] = v
            cands.append((f, v, r))
    if not cands:
        return []
    frame = pd.DataFrame([c[2] for c in cands])
    for c in NUMERIC + BINARY:
        frame[c] = frame[c].astype(float)
    ps = scorer.fn(frame)
    flagged = p0 >= scorer.cutoff
    best = {}
    for (f, v, _), p in zip(cands, ps):
        crosses = (p < scorer.cutoff) if flagged else (p >= scorer.cutoff)
        if not crosses:
            continue
        if f not in best or (p < best[f]["p"] if flagged else p > best[f]["p"]):
            best[f] = dict(f=f, v=v, p=float(p))
    return sorted(best.values(), key=lambda d: d["p"], reverse=not flagged)[:5]


def page_score(ctx: Ctx):
    B, sc = ctx.B, ctx.ours
    cutoff, review_from = sc.cutoff, sc.review
    if "f_amount" not in st.session_state:
        apply_preset("Typical dinner order")
    page_head("Phase 4 · fraud-check tool", "Score an order",
              "Enter an order's details or start from an example. Our qualifier gives the verdict and the reasons; "
              "Noon's model gives a second opinion.")
    cols = st.columns(len(PRESETS) // 2 + (1 if ctx.ref is not None else 0))
    for i, nm in enumerate(PRESETS):
        cols[i % (len(PRESETS) // 2)].button(nm, key=f"preset_{i}", on_click=apply_preset, args=(nm,), **BTN_W)
    if ctx.ref is not None:
        frauds = ctx.ref[ctx.ref[TARGET] == 1]
        pick = frauds.sample(1, random_state=int(np.random.default_rng().integers(1e9))).iloc[0].to_dict()
        legit = ctx.ref[ctx.ref[TARGET] == 0].sample(1, random_state=int(np.random.default_rng().integers(1e9))).iloc[0].to_dict()
        cols[-1].button("Random real fraud", key="rand_fraud", on_click=load_row_into_form, args=(pick,), **BTN_W)
        cols[-1].button("Random real legit", key="rand_legit", on_click=load_row_into_form, args=(legit,), **BTN_W)

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
        if ctx.noon is not None:
            pn = float(ctx.noon.fn(pd.DataFrame([row]))[0])
            tn = tier(pn, ctx.noon.review, ctx.noon.cutoff)
            agree = (tn == "fraud") == (tr == "fraud")
            st.markdown(f"<div class='opinion'><div><div class='w'>{swatch(C['noon'])}Second opinion: Noon's model</div>"
                        f"<div class='w' style='margin-top:2px'>{'Agrees with ours' if agree else 'Disagrees with ours'}"
                        f" · its cut-off is {pct(ctx.noon.cutoff, 0)}</div></div>"
                        f"<div style='text-align:right'><div class='p'>{pct(pn)}</div>{badge(tn)}</div></div>", unsafe_allow_html=True)

    section("Why this score", "Each detail's push on the fraud chance, starting from a typical legit order. The bars add up to this order's score.")
    base = p - sum(contrib.values())
    order = sorted(contrib, key=lambda k: abs(contrib[k]), reverse=True)
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(order) + ["total"],
        y=["Typical legit order"] + [f"{PRETTY[f]}: {fmt_value(f, row[f])}" for f in order] + ["This order"],
        x=[base * 100] + [contrib[f] * 100 for f in order] + [p * 100],
        connector=dict(line=dict(color=C["line"], width=1)),
        increasing=dict(marker=dict(color=C["fraud"])), decreasing=dict(marker=dict(color=C["legit"])),
        totals=dict(marker=dict(color=C["ours"])),
        hovertemplate="%{y}<br>%{x:+.2f} pts<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Fraud chance (percentage points)", ticksuffix=" pts")
    show(style(fig, 440, legend=False))
    ups = [f for f in sorted(contrib, key=contrib.get, reverse=True) if contrib[f] > 0.002][:3]
    if ups:
        callout("Biggest reasons this looks risky: " + ", ".join(f"<b>{esc(PRETTY[f].lower())}</b> ({esc(fmt_value(f, row[f]))})" for f in ups) + ".")
    else:
        callout("Nothing about this order stands out as risky compared with an ordinary legit order.")

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        section("What if one detail changed?", "Risk from both models as one detail sweeps its range, everything else held.")
        f = st.selectbox("Detail to vary", FEATURES, index=FEATURES.index(ups[0]) if ups else 1, format_func=lambda x: PRETTY[x],
                         key="whatif_f", label_visibility="collapsed")
        vals = SWEEP[f]
        frame = pd.DataFrame([{**row, f: v} for v in vals])
        for c in NUMERIC + BINARY:
            frame[c] = frame[c].astype(float)
        scorers = [ctx.ours] + ([ctx.noon] if ctx.noon is not None else [])
        fig = go.Figure()
        cat = f in BINARY or f == "payment_method"
        xs = [fmt_value(f, v) for v in vals] if cat else vals
        for s_ in scorers:
            ps = s_.fn(frame)
            if cat:
                fig.add_trace(go.Bar(x=xs, y=ps, name=s_.name, marker=dict(color=s_.color),
                                     hovertemplate=f"{s_.name}<br>%{{x}}: %{{y:.1%}}<extra></extra>"))
            else:
                fig.add_trace(go.Scatter(x=xs, y=ps, mode="lines", name=s_.name, line=dict(color=s_.color, width=2.4),
                                         hovertemplate=f"{s_.name}<br>%{{x}}: %{{y:.1%}}<extra></extra>"))
        fig.add_hline(y=cutoff, line=dict(color=C["ours"], width=1, dash="dash"))
        if ctx.noon is not None:
            fig.add_hline(y=ctx.noon.cutoff, line=dict(color=C["noon"], width=1, dash="dash"))
        if not cat:
            is_log = f == "account_age_days"
            vline(fig, max(float(row[f]), 1.0), "this order", color=C["ink2"], log=is_log)
            if is_log:
                fig.update_xaxes(type="log")
        fig.update_layout(barmode="group")
        fig.update_xaxes(title=PRETTY[f])
        fig.update_layout(yaxis=dict(title="Fraud chance", tickformat=".0%", range=[0, 1.02]))
        show(style(rounded(fig), 330), key="whatif")
        st.caption("Dashed lines are each model's fraud cut-off.")
    with c2:
        flagged = p >= cutoff
        section("Smallest change that flips the verdict",
                "Single details that would " + ("clear this order" if flagged else "get this order flagged") + ", by our qualifier.")
        flips = flip_finder(ctx.ours, row, p)
        if flips:
            rows_ = [[f"{PRETTY[d['f']]}", f"{esc(fmt_value(d['f'], row[d['f']]))} → <b>{esc(fmt_value(d['f'], d['v']))}</b>",
                      f"{pct(p)} → <b>{pct(d['p'])}</b>"] for d in flips]
            st.markdown(html_table(["Detail", "Change", "Risk"], rows_, num_cols={2}), unsafe_allow_html=True)
            callout("Read this as a robustness check: if one cheap change (a different hour, a different payment method) clears a "
                    "flagged order, a fraudster could learn that too.")
        else:
            st.markdown("<div class='empty'>No single change crosses the cut-off. The verdict rests on several details at once, "
                        "which makes it harder to game.</div>", unsafe_allow_html=True)


# ======================================================================
#  Page: Classify a file
# ======================================================================
def page_batch(ctx: Ctx):
    page_head("Phase 4 · batch scoring", "Classify a file",
              "Upload a CSV with the ten transaction columns. Column names are matched automatically. An is_fraud column is "
              "optional; when present the desk also reports how each model did.")
    up = st.file_uploader("Transactions CSV", type=["csv"], key="batch_up", label_visibility="collapsed")
    b1, b2, _ = st.columns([1, 1, 2])
    if ctx.ref is not None and b1.button("Try 2,000 real referee rows", **BTN_W):
        st.session_state["batch_demo"] = ("ref", np.random.default_rng().choice(len(ctx.ref), 2000, replace=False))
    if b2.button("Try 500 rows from our data", **BTN_W):
        st.session_state["batch_demo"] = ("ours", np.random.default_rng().choice(len(ctx.df), 500, replace=False))
    if up is not None:
        st.session_state.pop("batch_demo", None)

    df, has_label, source, p_o = None, False, "", None
    if up is not None:
        try:
            raw = parse_csv(up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file as a CSV: {e}")
            return
        canon, miss = ingest(raw, "batch")
        if miss:
            st.error("Pick a column in the file for: " + ", ".join(PRETTY[m] for m in miss) + ".")
            return
        df, errs, notes, has_label = prepare(canon)
        if errs:
            st.error(" ".join(errs))
            return
        for n in notes:
            st.caption(n)
        source = f"{len(df):,} rows from {up.name}"
    elif "batch_demo" in st.session_state:
        kind, idx = st.session_state["batch_demo"]
        if kind == "ref" and ctx.ref is not None:
            df, has_label = ctx.ref.iloc[idx].reset_index(drop=True), True
            source = "2,000 random rows from the real referee set (neither model trained on these)"
        elif kind == "ours":
            df, has_label = ctx.df.iloc[idx].reset_index(drop=True), True
            p_o = ctx.B["oof"][idx]
            source = "500 random rows from our dataset, scored out-of-fold by our qualifier"
    if df is None:
        empty("Nothing to classify yet", "Drop a CSV above, or try a sample. Each row gets a risk score from both models "
                                         "and a verdict from the one you choose.")
        return

    if p_o is None:
        p_o = ctx.ours.fn(df)
    p_n = ctx.noon.fn(df) if ctx.noon is not None else None
    options = ["ours"] + (["noon"] if p_n is not None else [])
    who = st.radio("Decide with", options, horizontal=True, key="batch_who",
                   format_func=lambda k: ctx.name_ours if k == "ours" else "Noon's model")
    dec = ctx.ours if who == "ours" else ctx.noon
    risk = p_o if who == "ours" else p_n
    other = (p_n if who == "ours" else p_o)
    scored = df.copy()
    scored["risk"] = risk
    scored["verdict_key"] = [tier(v, dec.review, dec.cutoff) for v in risk]
    if other is not None:
        scored["other_risk"] = other

    n = len(scored)
    counts = scored["verdict_key"].value_counts()
    st.markdown(f"<div class='legend-line'>{esc(source)}.</div>", unsafe_allow_html=True)
    kpis([(f"{n:,}", "Orders scored", ""),
          (f"{counts.get('fraud', 0):,}", "Likely fraud", pct(counts.get('fraud', 0) / n) + " of orders"),
          (f"{counts.get('review', 0):,}", "Needs review", pct(counts.get('review', 0) / n) + " of orders"),
          (f"{counts.get('legit', 0):,}", "Looks legitimate", pct(counts.get('legit', 0) / n) + " of orders")])

    if has_label and label_state(scored) == "both":
        section("How each model did on the labelled rows")
        cc = st.columns(2 if p_n is not None else 1, gap="large")
        mo = at_cutoff(scored[TARGET].values, p_o, ctx.ours.cutoff)
        with cc[0]:
            st.markdown(confusion_html(mo, ctx.name_ours, C["ours"]), unsafe_allow_html=True)
            st.caption(f"Recall {pct(mo['recall'], 0)} · precision {pct(mo['precision'], 0)} · AUC {fast_auc(scored[TARGET].values, p_o):.2f}")
        if p_n is not None:
            mn = at_cutoff(scored[TARGET].values, p_n, ctx.noon.cutoff)
            with cc[1]:
                st.markdown(confusion_html(mn, "Noon's model", C["noon"]), unsafe_allow_html=True)
                st.caption(f"Recall {pct(mn['recall'], 0)} · precision {pct(mn['precision'], 0)} · AUC {fast_auc(scored[TARGET].values, p_n):.2f}")
    elif TARGET in scored.columns and label_state(scored) in ("fraud_only", "legit_only"):
        only_fraud = label_state(scored) == "fraud_only"
        flagged = float((scored["risk"] >= dec.cutoff).mean())
        callout(f"Every row in this file is labelled <b>{'fraud' if only_fraud else 'legit'}</b>. The deciding model "
                f"{'flagged' if only_fraud else 'raised a false alarm on'} <b>{pct(flagged)}</b> of them.")

    if p_n is not None:
        fo, fn_ = p_o >= ctx.ours.cutoff, p_n >= ctx.noon.cutoff
        section("Where the two models disagree", "Orders flagged by one model but not the other.")
        tiles = [(int((fo & fn_).sum()), "flagged by both"), (int((fo & ~fn_).sum()), f"only {ctx.name_ours}"),
                 (int((~fo & fn_).sum()), "only Noon"), (int((~fo & ~fn_).sum()), "cleared by both")]
        st.markdown("<div class='tiles'>" + "".join(f"<div class='tile'><b>{v:,}</b><span>{esc(t)}</span></div>" for v, t in tiles)
                    + "</div>", unsafe_allow_html=True)
        if has_label and label_state(scored) == "both":
            y = scored[TARGET].values
            ours_only = fo & ~fn_
            noon_only = ~fo & fn_
            callout(f"Of the {int(ours_only.sum()):,} orders only we flagged, <b>{int(y[ours_only].sum())}</b> were fraud. "
                    f"Of the {int(noon_only.sum()):,} only Noon flagged, <b>{int(y[noon_only].sum())}</b> were fraud.")

    section("Highest-risk orders", "Sorted by the deciding model's risk, top 150 shown. Download the full file below.")
    scored = scored.sort_values("risk", ascending=False)
    other_name = ("Noon risk" if who == "ours" else "Our risk") if other is not None else ""
    st.markdown(risk_table(scored, other="other_risk" if other is not None else None, other_name=other_name), unsafe_allow_html=True)
    dl = scored.drop(columns=["verdict_key"]).rename(columns={"risk": "fraud_score", "other_risk": "other_model_score"})
    dl.insert(0, "verdict", scored["verdict_key"].map(TIER_TEXT).values)
    st.download_button("Download all scored rows (CSV)", dl.to_csv(index=False).encode(), "scored_transactions.csv",
                       "text/csv", type="primary")


# ======================================================================
#  Page: GAN face-off (Phase 2)
# ======================================================================
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


def page_faceoff(ctx: Ctx):
    page_head("Phase 2 · the reveal", "GAN face-off",
              "Our GAN's fraud next to Noon's. Noon's GAN learned from real fraud; ours learned from our assumptions. "
              "Every chart here asks the same question: does our fraud look like theirs?")
    opts = (["noon"] if ctx.nf is not None else []) + ["upload", "demo"]
    src = st.radio("Compare against", opts, horizontal=True, key="cmp_src",
                   format_func={"noon": "Noon's GAN fraud", "upload": "Upload another dataset", "demo": "Demo dataset"}.get)
    df_a = ctx.df
    df_b, source, nb = None, "", ctx.name_noon
    if src == "noon":
        df_b, source = ctx.nf, f"B_NOON_synthetic_fraud.csv: {len(ctx.nf):,} rows"
    elif src == "upload":
        up = st.file_uploader("Other dataset (CSV)", type=["csv"], key="cmp_up", label_visibility="collapsed")
        if up is None:
            empty("Waiting for a dataset", "Drop a CSV. A file with only fraud rows (like a GAN's output) works, and so does a "
                                           "full labelled dataset. Columns are matched by name; fix matches if needed.")
            return
        try:
            raw = parse_csv(up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file as a CSV: {e}")
            return
        canon, miss = ingest(raw, "cmp")
        if miss:
            st.error("Pick a column in the file for: " + ", ".join(PRETTY[m] for m in miss) + ".")
            return
        df_b, errs, notes, _ = prepare(canon)
        if errs:
            st.error(" ".join(errs))
            return
        for n in notes:
            st.caption(n)
        source, nb = f"{up.name}: {len(df_b):,} rows", "Their dataset"
    else:
        df_b, source, nb = demo_dataset(), "Demo dataset from a naive generator (independent columns, one fraud rule)", "Demo dataset"
    if len(df_b) > 50000:
        df_b = df_b.sample(50000, random_state=1).reset_index(drop=True)

    state = label_state(df_b)
    labelled, single = state == "both", state in ("fraud_only", "legit_only")
    cls = 1 if state == "fraud_only" else 0
    word = "fraud" if cls == 1 else "legit"
    if state == "none":
        st.info("No fraud label found in that file; label-based sections are skipped. Pick the label under Match columns if it has another name.")
    if single:
        fa = df_a[df_a[TARGET] == cls].reset_index(drop=True)
        fb = df_b.reset_index(drop=True)
        la = f"{ctx.name_ours} ({word} rows)"
    else:
        fa, fb, la = df_a, df_b, ctx.name_ours
    st.markdown(f"<div class='legend-line'><span class='dot' style='background:{C['ours']}'></span>{esc(la)} &nbsp;&nbsp; "
                f"<span class='dot' style='background:{C['noon']}'></span>{esc(nb)} &nbsp;&nbsp; {esc(source)}</div>", unsafe_allow_html=True)

    fa_f, fb_f = fa[FEATURES + ([TARGET] if labelled else [])], fb[FEATURES + ([TARGET] if labelled else [])]
    R = compare_core(fa_f, fb_f, dfkey(fa_f), dfkey(fb_f), labelled)
    L = lie_detector(fa, fb, dfkey(fa) + dfkey(fb))
    avg_gap = float(np.mean(list(R["gaps"].values())))

    items = [(f"{L['auc']:.2f}", "Lie-detector AUC", "0.50 = indistinguishable"),
             (f"{L['rf_acc']:.2f}", "Lie-detector accuracy", "random forest, 3-fold, as in A_eval_helper"),
             (f"{avg_gap:.2f}", "Average distribution gap", "0 = identical, 1 = no overlap")]
    if single:
        p_b = ctx.ours.fn(fb)
        theirs = float((p_b >= ctx.ours.cutoff).mean())
        ours_rate = float((ctx.B["oof"][ctx.B["y"] == cls] >= ctx.ours.cutoff).mean())
        what = "Their fraud flagged by our qualifier" if cls == 1 else "False alarms on their legit rows"
        items.append((pct(theirs, 0), what, f"{pct(ours_rate, 0)} for our own {word} rows"))
    if labelled:
        items += [(f"{R['a_on_b']:.2f}", f"Our model on {nb}", f"{R['cv_a']:.2f} on its own data"),
                  (f"{R['b_on_a']:.2f}", f"{nb} model on ours", f"{R['cv_b']:.2f} on its own data")]
    kpis(items)
    imp = pd.Series(L["imp"]).sort_values(ascending=False)
    top2 = [PRETTY[i].lower() for i in imp.index[:2]]
    verdict = ("very different" if L["auc"] > 0.9 else "clearly different" if L["auc"] > 0.75 else "somewhat different" if L["auc"] > 0.6 else "close")
    callout(f"A classifier tells the two apart with AUC <b>{L['auc']:.2f}</b>, so the fraud is <b>{verdict}</b>. "
            f"It leans most on <b>{esc(top2[0])}</b> and <b>{esc(top2[1])}</b>. "
            + (f"Our own qualifier flags <b>{pct(theirs, 0)}</b> of their fraud: " +
               ("it would miss most of the fraud patterns Noon's GAN learned from real cases." if theirs < 0.5 else
                "it recognises much of their fraud, despite the differences.") if single and cls == 1 else ""))

    # ---- fraud rate realism
    if ctx.ref is not None or src == "noon":
        section("How rare should fraud be?", "Fraud share of each dataset. The real referee sample sets the bar.")
        rows = [(f"{ctx.name_ours} (training data)", float(df_a[TARGET].mean()), C["ours"])]
        if ctx.ref is not None:
            rows.append(("Real referee sample", float(ctx.ref[TARGET].mean()), C["real"]))
        if state in ("both", "fraud_only"):
            rows.append((f"{nb} file", float(df_b[TARGET].mean()), C["noon"]))
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            fig = go.Figure(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
                                   marker=dict(color=[r[2] for r in rows]), text=[pct(r[1], 2) for r in rows],
                                   textposition="outside", textfont=dict(color=C["ink2"]), cliponaxis=False,
                                   hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
            fig.update_xaxes(type="log", title="Fraud share (log scale)", tickformat=".1%", range=[-3.2, 0.45])
            fig.update_yaxes(autorange="reversed")
            show(style(rounded(fig), 220, legend=False), key="rate_bars")
        with c2:
            ours_r = float(df_a[TARGET].mean())
            real_r = float(ctx.ref[TARGET].mean()) if ctx.ref is not None else 0.00325
            callout(f"Our data is <b>{pct(ours_r, 1)}</b> fraud, about <b>{ours_r / real_r:.0f}x</b> the real rate of "
                    f"<b>{pct(real_r, 2)}</b>. More fraud makes the model bolder: it learns that fraud is common and flags "
                    + "freely, which shows up as false alarms on real data. "
                    + ("Noon's file is all fraud because it is the GAN's output, meant to be added to their real data, "
                       "not a picture of real rarity." if state == "fraud_only" else ""))

    # ---- map
    section("The fraud map", "Every order projected onto the two directions that spread these datasets most (PCA). "
                             "Overlapping clouds mean similar fraud; separate clouds mean the GANs learned different things.")
    legit_bg = df_a[df_a[TARGET] == 0]
    frames = (legit_bg, fa if single else fa[fa[TARGET] == 1] if labelled else fa,
              fb if single else fb[fb[TARGET] == 1] if labelled else fb)
    if min(len(f) for f in frames) >= 20:
        Z, var, load = pca_map(tuple(f[FEATURES] for f in frames), "".join(dfkey(f[FEATURES]) for f in frames))
        c1, c2 = st.columns([1.6, 1], gap="large")
        with c1:
            fig = go.Figure()
            for z, nm, col, op, sz in ((Z[0], "Our legit orders", C["ghost"], .35, 4), (Z[1], f"{ctx.name_ours} fraud", C["ours"], .65, 6),
                                       (Z[2], f"{nb} fraud", C["noon"], .55, 6)):
                fig.add_trace(go.Scattergl(x=z[:, 0], y=z[:, 1], mode="markers", name=nm,
                                           marker=dict(color=col, size=sz, opacity=op, line=dict(width=0)),
                                           hovertemplate=f"{nm}<extra></extra>"))
            fig.update_xaxes(title=f"Direction 1 ({pct(var[0], 0)} of spread)", zeroline=False)
            fig.update_yaxes(title=f"Direction 2 ({pct(var[1], 0)} of spread)", zeroline=False)
            show(style(fig, 460), key="pca")
        with c2:
            rows_ = []
            for pc in ("pc1", "pc2"):
                top = load[pc].abs().sort_values(ascending=False).index[:3]
                rows_.append([f"Direction {pc[-1]}", ", ".join(f"{PRETTY[t].lower()} ({'+' if load.loc[t, pc] > 0 else '−'})" for t in top)])
            st.markdown(html_table(["Axis", "Driven mostly by"], rows_), unsafe_allow_html=True)
            cen = [z.mean(0) for z in Z]
            d_ours = float(np.linalg.norm(cen[1] - cen[0]))
            d_noon = float(np.linalg.norm(cen[2] - cen[0]))
            callout(f"Distance of each fraud cloud's centre from our legit orders: <b>{d_ours:.2f}</b> for ours, "
                    f"<b>{d_noon:.2f}</b> for {esc(nb)}. "
                    + ("Noon's fraud sits further from normal orders: their real fraud is more extreme than we assumed."
                       if d_noon > d_ours * 1.2 else "Ours sits further out: we made fraud more extreme than Noon's GAN did."
                       if d_ours > d_noon * 1.2 else "Both sit at similar distances, in different directions."))

    # ---- distributions
    section("Column by column", "Share of rows at each value. Closer curves mean the two GANs describe the same kind of fraud.")
    fa_d, fb_d = fa, fb
    if labelled:
        mode = st.radio("Rows to compare", ["Fraud only", "Legit only", "All rows"], horizontal=True, label_visibility="collapsed")
        if mode != "All rows":
            v = 1 if mode == "Fraud only" else 0
            fa_d, fb_d = df_a[df_a[TARGET] == v], df_b[df_b[TARGET] == v]
    frames_d = [(la, fa_d, C["ours"]), (nb, fb_d, C["noon"])]
    if ctx.ref is not None and st.toggle("Overlay real referee fraud (the answer key)", value=False, key="cmp_real"):
        frames_d.append(("Real fraud (referee)", ctx.ref[ctx.ref[TARGET] == (cls if single else 1)], C["real"]))
    if min(len(f[1]) for f in frames_d) < 30:
        st.info("Too few rows in one dataset for this slice to be meaningful.")
    else:
        show(dist_figure(frames_d), key="cmp_dist")

    g = pd.Series(R["gaps"]).sort_values()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Which columns differ most", "KS distance for continuous columns, total variation for the rest.")
        fig = go.Figure(go.Bar(x=g.values, y=[PRETTY[i] for i in g.index], orientation="h", marker=dict(color=C["ink2"]),
                               hovertemplate="%{y}<br>gap %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Gap (0 = identical)", range=[0, max(0.3, float(g.max()) * 1.1)])
        show(style(rounded(fig), 380, legend=False))
    with c2:
        section("What gives our fraud away", "Columns the lie-detector relies on to tell the two apart.")
        imp_s = pd.Series(L["imp"]).sort_values()
        fig = go.Figure(go.Bar(x=imp_s.values, y=[PRETTY[i] for i in imp_s.index], orientation="h", marker=dict(color=C["ink2"]),
                               hovertemplate="%{y}<br>AUC drop %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="AUC drop when shuffled")
        show(style(rounded(fig), 380, legend=False))

    # ---- means table
    if single and cls == 1:
        section("Side by side", "Average fraud order in each dataset, with our legit orders for reference.")
        refset = [("Our legit", legit_bg, None), (f"{ctx.name_ours} fraud", fa, C["ours"]), (f"{nb} fraud", fb, C["noon"])]
        if ctx.ref is not None:
            refset.append(("Real fraud (referee)", ctx.ref[ctx.ref[TARGET] == 1], C["real"]))
        st.markdown(means_table(refset), unsafe_allow_html=True)

    # ---- fraud patterns
    sig, xtitle = None, ""
    if labelled:
        sig = pd.DataFrame({ctx.name_ours: R["sig_a"], nb: R["sig_b"]})
        xtitle = "Standardised difference (right: higher in fraud)"
    elif single and cls == 1:
        ref_ = encode(legit_bg)
        mu, sd = ref_.mean(), ref_.std().replace(0, 1)
        sig = pd.DataFrame({la: (encode(fa).mean() - mu) / sd, nb: (encode(fb).mean() - mu) / sd})
        xtitle = "Distance from our legit orders, in their standard deviations"
    if sig is not None:
        section("Fraud signals", "How far each dataset's fraud sits from normal orders on each detail. Longer bars mean a stronger tell.")
        show(signal_bars(sig, [C["ours"], C["noon"]], xtitle), key="cmp_sig")
        agree = float(np.corrcoef(sig.iloc[:, 0].values, sig.iloc[:, 1].values)[0, 1])
        callout(f"The two datasets agree on which signals matter with correlation <b>{agree:.2f}</b> "
                f"(1.00 is the same fraud story, 0 is unrelated).")
    if labelled:
        section("Cross-test", "Train on one dataset, score the other. High numbers off the diagonal mean the fraud rules transfer.")
        cells = lambda v, d: f"<div class='c {'good' if d else ''}'><b>{v:.2f}</b><span>{'own data (5-fold)' if d else 'unseen data'}</span></div>"
        st.markdown(f"<div class='cm' style='max-width:560px'><div></div><div class='h'>Tested on {esc(ctx.name_ours)}</div>"
                    f"<div class='h'>Tested on {esc(nb)}</div><div class='r'>Trained on<br>{esc(ctx.name_ours)}</div>{cells(R['cv_a'], True)}"
                    f"{cells(R['a_on_b'], False)}<div class='r'>Trained on<br>{esc(nb)}</div>{cells(R['b_on_a'], False)}{cells(R['cv_b'], True)}</div>",
                    unsafe_allow_html=True)

    section("Correlations", "How details move together. Red: together. Blue: opposite. The third panel is the difference.")
    cc = st.columns(3, gap="medium")
    ca, cb = R["corr_a"], R["corr_b"]
    common = [c for c in ca.columns if c in cb.columns]
    ca, cb = ca.loc[common, common].fillna(0), cb.loc[common, common].fillna(0)
    with cc[0]:
        show(corr_figure(ca, 1, la), key="corr_a")
    with cc[1]:
        show(corr_figure(cb, 1, nb), key="corr_b")
    with cc[2]:
        show(corr_figure(ca - cb, 0.6, "Difference"), key="corr_d")

    summary = pd.DataFrame([{"column": f, "gap": R["gaps"][f], "lie_detector_importance": L["imp"].get(f)} for f in FEATURES])
    st.download_button("Download comparison summary (CSV)", summary.to_csv(index=False).encode(), "gan_faceoff_summary.csv", "text/csv")


def means_table(sets: list[tuple[str, pd.DataFrame, str | None]]) -> str:
    feats = [("order_amount", lambda s: f"AED {s.mean():,.0f}"), ("account_age_days", lambda s: f"{s.median():,.0f} d (median)"),
             ("hour_of_day", lambda s: f"{(s < 6).mean():.0%} at night"), ("num_orders_last_7d", lambda s: f"{s.mean():.1f}"),
             ("delivery_distance_km", lambda s: f"{s.mean():.1f} km"), ("prev_refunds", lambda s: f"{s.mean():.2f}"),
             ("discount_applied", lambda s: f"{s.mean():.0%}"), ("device_change", lambda s: f"{s.mean():.0%}"),
             ("location_mismatch", lambda s: f"{s.mean():.0%}")]
    header = ["Detail"] + [f"{swatch(c) if c else ''}{esc(n)}" for n, _, c in sets]
    rows = [[PRETTY[f]] + [fn(d[f]) for _, d, _ in sets] for f, fn in feats]
    rows.append(["Top payment method"] + [f"{d['payment_method'].value_counts(normalize=True).index[0]} "
                                          f"({d['payment_method'].value_counts(normalize=True).iloc[0]:.0%})" for _, d, _ in sets])
    return html_table(header, rows, num_cols=set(range(1, len(header))))


# ======================================================================
#  Page: Real-world referee (Phase 3)
# ======================================================================
def page_referee(ctx: Ctx):
    page_head("Phase 3 · the referee", "Real-world referee",
              "A sample of real Noon orders, used only as a test set: no model here ever trained on it. "
              "This is the neutral exam for both classifiers.")
    R = ref_eval(ctx)
    if R is None:
        empty("No referee data yet", "Put C_REFEREE_real_test.csv next to app.py or upload it under Data sources in the sidebar.")
        return
    y, p_o, p_n, mo, mn = R["y"], R["p_o"], R["p_n"], R["m_o"], R["m_n"]
    have_noon = mn is not None
    mn_show = mn or NOON_BENCH
    if not have_noon:
        st.info("Noon's model file is missing, so Noon's numbers come from their published benchmark.")
    elif abs(mn["recall"] - NOON_BENCH["recall"]) < 0.02 and mn["fp"] == NOON_BENCH["fp"] and abs(ctx.noon.cutoff - 0.5) < 1e-6:
        st.caption(f"Noon's model reproduces its published benchmark exactly ({mn['tp']} caught, {mn['fp']} false alarms).")

    kpis([(f"{R['n']:,}", "Real orders", f"{R['nf']} fraud ({pct(R['nf'] / R['n'], 2)})"),
          (f"{mo['tp']} vs {mn_show['tp']}", "Frauds caught", f"ours vs Noon, of {R['nf']}"),
          (f"{mo['fp']:,} vs {mn_show['fp']:,}", "False alarms", "innocent orders flagged"),
          (f"{mo['f1']:.2f} vs {mn_show['f1']:.2f}", "F1", "ours vs Noon"),
          (f"{roc_auc_score(y, p_o):.2f}" + (f" vs {roc_auc_score(y, p_n):.2f}" if have_noon else ""), "ROC AUC", "ranking quality"),
          (f"{average_precision_score(y, p_o):.2f}" + (f" vs {average_precision_score(y, p_n):.2f}" if have_noon else ""), "PR AUC",
           f"guessing scores {R['nf'] / R['n']:.3f}")])

    section("The final bar chart", "Fraud-catch results side by side on real data. Grey is Noon's model trained on real data alone, from their benchmark.")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        show(scoreboard_figure(R, ctx), key="ref_board")
    with c2:
        show(counts_figure(R, ctx), key="ref_counts")

    cc = st.columns(2, gap="large")
    with cc[0]:
        st.markdown(confusion_html(mo, f"{ctx.name_ours} · cut-off {pct(ctx.ours.cutoff, 0)}", C["ours"]), unsafe_allow_html=True)
    with cc[1]:
        st.markdown(confusion_html(mn_show, "Noon · " + (f"cut-off {pct(ctx.noon.cutoff, 0)}" if have_noon else "published benchmark"),
                                   C["noon"]), unsafe_allow_html=True)

    # ---- threshold-free comparison
    section("Same review budget, who catches more?",
            "Frauds caught as the team reviews more orders, taking each model's highest-risk orders first. "
            "This removes the cut-off from the comparison.")
    ks = np.unique(np.r_[np.geomspace(5, R["n"] * 0.2, 90).astype(int), R["budget"]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks / R["n"] * 1e4, y=ks * R["nf"] / R["n"], mode="lines", name="Random review",
                             line=dict(color=C["muted"], width=1, dash="dot"), hoverinfo="skip"))
    series = [(ctx.name_ours, p_o, C["ours"])] + ([("Noon", p_n, C["noon"])] if have_noon else [])
    for nm, p, col in series:
        caught = [caught_at_budget(y, p, k) for k in ks]
        fig.add_trace(go.Scatter(x=ks / R["n"] * 1e4, y=caught, mode="lines", name=nm, line=dict(color=col, width=2.6),
                                 customdata=ks, hovertemplate=f"{nm}<br>Review %{{customdata:,}} orders (%{{x:.0f}} per 10k)<br>"
                                                              f"Catch %{{y}} of {R['nf']}<extra></extra>"))
    vline(fig, R["budget"] / R["n"] * 1e4, f"Noon's budget: {R['budget']} flags", log=True)
    fig.update_xaxes(type="log", title="Orders reviewed per 10,000 (log scale)")
    fig.update_layout(yaxis=dict(title=f"Real frauds caught (of {R['nf']})", range=[0, R["nf"] * 1.05]))
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        show(style(fig, 380), key="gains")
    with c2:
        same = caught_at_budget(y, p_o, R["budget"])
        k90 = next((k for k in range(1, R["n"] + 1, max(1, R["n"] // 2000)) if caught_at_budget(y, p_o, k) >= 0.8 * R["nf"]), R["n"])
        rows_ = [["Frauds caught", f"{same}", f"{mn_show['tp']}"],
                 ["False alarms", f"{R['budget'] - same}", f"{mn_show['fp']}"],
                 ["Precision", pct(same / R["budget"], 0), pct(mn_show["tp"] / max(R["budget"], 1), 0)]]
        st.markdown(f"<div class='legend-line'>At Noon's budget of <b>{R['budget']}</b> flags in {R['n']:,} orders:</div>", unsafe_allow_html=True)
        st.markdown(html_table(["", f"{swatch(C['ours'])}{esc(ctx.name_ours)}", f"{swatch(C['noon'])}Noon"], rows_, num_cols={1, 2}),
                    unsafe_allow_html=True)
        callout(f"To catch 80% of real fraud we would need to review about <b>{k90:,}</b> orders "
                f"(<b>{pct(k90 / R['n'], 1)}</b> of traffic).")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Precision vs recall on real data", "Curves for both models. Dots mark each model's cut-off.")
        fig = go.Figure()
        for nm, p, col, cut in [(ctx.name_ours, p_o, C["ours"], ctx.ours.cutoff)] + ([("Noon", p_n, C["noon"], ctx.noon.cutoff)] if have_noon else []):
            pr, rc, _ = precision_recall_curve(y, p)
            m = at_cutoff(y, p, cut)
            fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"{nm} (AP {average_precision_score(y, p):.2f})",
                                     line=dict(color=col, width=2.4), hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", showlegend=False,
                                     marker=dict(size=11, color=col, line=dict(color=C["bg"], width=3)), hovertemplate=f"{nm} cut-off<extra></extra>"))
        fig.update_xaxes(title="Recall", tickformat=".0%", range=[0, 1.02])
        fig.update_yaxes(title="Precision", tickformat=".0%", range=[0, 1.02])
        show(style(fig, 340), key="ref_pr")
    with c2:
        section("Score separation on real data", "Our qualifier's scores for real legit and real fraud orders.")
        show(score_density([("Real legit", p_o[y == 0], C["legit"], "solid"), ("Real fraud", p_o[y == 1], C["fraud"], "solid")],
                           [(ctx.ours.cutoff, "our cut-off")], xmax=1.0, height=340), key="ref_dens")

    # ---- missed fraud
    if have_noon:
        section("Which real frauds did each model catch?", "Every real fraud falls into one of four groups.")
        fy = y == 1
        co, cn = (p_o >= ctx.ours.cutoff) & fy, (p_n >= ctx.noon.cutoff) & fy
        groups = {"Caught by both": co & cn, "Only Noon caught": ~co & cn & fy, "Only we caught": co & ~cn, "Missed by both": ~co & ~cn & fy}
        st.markdown("<div class='tiles'>" + "".join(f"<div class='tile'><b>{int(v.sum())}</b><span>{esc(k)}</span></div>" for k, v in groups.items())
                    + "</div>", unsafe_allow_html=True)
        sets = [("Real legit", ctx.ref[y == 0], None)] + [(k, ctx.ref[v], None) for k, v in groups.items() if v.sum() > 0]
        st.markdown(means_table(sets), unsafe_allow_html=True)
        if groups["Only Noon caught"].sum():
            mf = ctx.ref[groups["Only Noon caught"]]
            callout(f"The frauds only Noon caught average <b>AED {mf.order_amount.mean():,.0f}</b>, "
                    f"<b>{(mf.hour_of_day < 6).mean():.0%}</b> at night and <b>{mf.prev_refunds.mean():.1f}</b> prior refunds. "
                    f"Those are patterns our assumptions under-weighted.")

    explain_gap(ctx, R)


def explain_gap(ctx: Ctx, R):
    st.markdown("<div class='page-eyebrow' style='margin-top:2.2rem'>The most important section</div>"
                "<h2 style='margin:0 0 .3rem'>Explaining the gap</h2>", unsafe_allow_html=True)
    st.markdown("<p class='page-sub'>Three measured reasons, each from the data rather than from opinion.</p>", unsafe_allow_html=True)
    df, ref = ctx.df, ctx.ref
    sig_o, sig_r = signature(df), signature(ref)

    # 1. assumption audit
    section("1 · Our fraud story vs the real one",
            "Standardised fraud-vs-legit gap on each detail, in our data and in real data. Matching bars mean we guessed the signal right.")
    show(signal_bars(pd.DataFrame({ctx.name_ours: sig_o, "Real data (referee)": sig_r}), [C["ours"], C["real"]],
                     "Standardised difference (right: higher in fraud)", 520), key="gap_sig")
    strong = sig_r[sig_r.abs() > 0.15].index
    right_dir = [i for i in strong if np.sign(sig_o[i]) == np.sign(sig_r[i])]
    diff = (sig_r - sig_o)
    under = diff[(sig_r.abs() > 0.15) & (diff.abs() > 0.15) & (np.sign(diff) == np.sign(sig_r))].abs().sort_values(ascending=False)
    over = diff[(diff.abs() > 0.15) & ~diff.index.isin(under.index)].abs().sort_values(ascending=False)
    callout(f"We called the direction of <b>{len(right_dir)} of {len(strong)}</b> real fraud signals correctly. "
            + (f"We under-weighted <b>{', '.join(PRETTY[i].lower() for i in under.index[:3])}</b>. " if len(under) else "")
            + (f"We over-weighted <b>{', '.join(PRETTY[i].lower() for i in over.index[:3])}</b>." if len(over) else ""))

    # 2. normal drift
    section("2 · Our picture of normal was off too",
            "How far our legit orders sit from real legit orders, per detail. A model learns fraud as 'unlike normal', so a wrong normal creates false alarms.")
    lo, lr = df[df[TARGET] == 0], ref[ref[TARGET] == 0]
    drift = pd.Series({f: gap(lo, lr, f) for f in FEATURES}).sort_values()
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        fig = go.Figure(go.Bar(x=drift.values, y=[PRETTY[i] for i in drift.index], orientation="h", marker=dict(color=C["real"]),
                               hovertemplate="%{y}<br>gap %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Gap between our legit and real legit (0 = identical)", range=[0, max(0.3, float(drift.max()) * 1.1)])
        show(style(rounded(fig), 380, legend=False), key="gap_drift")
    with c2:
        top = drift.sort_values(ascending=False).index[:4]
        rows_ = []
        for f in top:
            if f == "payment_method":
                a_ = lo[f].value_counts(normalize=True)
                b_ = lr[f].value_counts(normalize=True)
                rows_.append([PRETTY[f], f"{a_.index[0]} {a_.iloc[0]:.0%}", f"{b_.index[0]} {b_.iloc[0]:.0%}"])
            else:
                rows_.append([PRETTY[f], f"{lo[f].mean():,.2f}", f"{lr[f].mean():,.2f}"])
        st.markdown(html_table(["Detail (legit orders)", f"{swatch(C['ours'])}Ours, average", f"{swatch(C['real'])}Real, average"],
                               rows_, num_cols={1, 2}), unsafe_allow_html=True)
        fp_share = float((R["p_o"][R["y"] == 0] >= ctx.ours.cutoff).mean())
        callout(f"Our qualifier flags <b>{pct(fp_share, 1)}</b> of real legit orders. Real customers simply behave differently "
                f"from the customers we imagined, and some of that difference reads as fraud.")

    # 3. swap experiment
    section("3 · The swap test: same model, different fraud examples",
            "Keep our legit rows and the exact same model. Change only the fraud rows it learns from. Score each version on real data.")
    if ctx.nf is None:
        empty("Needs Noon's GAN fraud", "Add B_NOON_synthetic_fraud.csv to run this test.")
        sw = None
    else:
        sw = swap_experiment(df, ctx.nf, ref, R["budget"], dfkey(df) + dfkey(ctx.nf) + dfkey(ref) + str(R["budget"]))
        rows_ = [[esc(r.variant), pct(r.fraud_share, 1), f"{r.pr_auc:.2f}", f"{r.roc_auc:.2f}", f"{r.caught} / {R['nf']}", f"{r.false_alarms}"]
                 for r in sw.itertuples()]
        if R["m_n"] is not None:
            rows_.append(["Noon's model (real + Noon GAN fraud)", "–", f"{average_precision_score(R['y'], R['p_n']):.2f}",
                          f"{roc_auc_score(R['y'], R['p_n']):.2f}", f"{caught_at_budget(R['y'], R['p_n'], R['budget'])} / {R['nf']}",
                          f"{R['budget'] - caught_at_budget(R['y'], R['p_n'], R['budget'])}"])
        c1, c2 = st.columns([1.9, 1], gap="large")
        with c1:
            st.markdown(html_table(["Training set", "Fraud share", "PR AUC", "ROC AUC", f"Caught at {R['budget']} flags", "False alarms"],
                                   rows_, num_cols={1, 2, 3, 4, 5}, hl={0}), unsafe_allow_html=True)
        with c2:
            fig = go.Figure(go.Bar(x=sw["pr_auc"], y=[f"V{i + 1}" for i in range(len(sw))], orientation="h",
                                   marker=dict(color=[C["ours"] if k == "ours" else C["noon"] if k == "swap" else C["ink2"] for k in sw["kind"]]),
                                   customdata=sw["variant"], text=[f"{v:.2f}" for v in sw["pr_auc"]], textposition="outside",
                                   textfont=dict(color=C["ink2"]), cliponaxis=False,
                                   hovertemplate="%{customdata}<br>PR AUC %{x:.3f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", ticktext=[f"V{i + 1}" for i in range(len(sw))])
            fig.update_xaxes(title="PR AUC on real data", range=[0, max(0.3, float(sw.pr_auc.max()) * 1.25)])
            show(style(rounded(fig), 250, legend=False), key="swap_bar")
            st.caption("V1 to V4 follow the table rows. Blue: as built. Amber: Noon's fraud swapped in. Grey: both.")

    # story
    mo, mn = R["m_o"], R["m_n"] or NOON_BENCH
    story = [f"<li><b>Noon's GAN learned from real fraud; ours learned from guesses.</b> A GAN copies the distribution it is shown, "
             f"faithfully. Ours was shown our assumptions, so it reproduced them, gaps included: we caught {mo['tp']} of {R['nf']} real "
             f"frauds against Noon's {mn['tp']}.</li>",
             f"<li><b>We got the broad strokes right.</b> {len(right_dir)} of {len(strong)} real signals point the way we assumed "
             + (f"(the strongest are {', '.join(PRETTY[i].lower() for i in sig_r.abs().sort_values(ascending=False).index[:2])})" if len(strong) else "")
             + (f", but we under-weighted {', '.join(PRETTY[i].lower() for i in under.index[:2])}." if len(under) else ".") + "</li>",
             f"<li><b>Our normal was wrong, and that costs precision.</b> Real legit customers differ most from ours on "
             f"{', '.join(PRETTY[i].lower() for i in drift.sort_values(ascending=False).index[:2])}. "
             f"With fraud at {pct(R['nf'] / R['n'], 2)}, flagging even {pct(float((R['p_o'][R['y'] == 0] >= ctx.ours.cutoff).mean()), 1)} "
             f"of legit orders swamps the real catches ({mo['fp']:,} false alarms).</li>"]
    if sw is not None:
        v1, v2 = sw.iloc[0], sw.iloc[1]
        if v2.pr_auc > v1.pr_auc:
            story.append(f"<li><b>The swap test settles it.</b> Same model, same legit rows; swapping in {len(ctx.df[ctx.df[TARGET] == 1]):,} of "
                         f"Noon's fraud rows lifts PR AUC from {v1.pr_auc:.2f} to {v2.pr_auc:.2f} and catches {v2.caught} instead of "
                         f"{v1.caught} at Noon's budget. The fraud examples, not the algorithm, explain most of the gap. What remains is "
                         f"our unrealistic legit data and fraud rate.</li>")
        else:
            story.append(f"<li><b>The swap test is mixed.</b> Swapping in Noon's fraud rows moves PR AUC from {v1.pr_auc:.2f} to "
                         f"{v2.pr_auc:.2f}, so our legit data, not only our fraud, is holding the model back.</li>")
    story.append(f"<li><b>Rarity matters.</b> Our data is {pct(float(df[TARGET].mean()), 1)} fraud against {pct(R['nf'] / R['n'], 2)} in "
                 f"reality. A model trained on common fraud is calibrated to flag freely; its scores read about "
                 f"{float(df[TARGET].mean()) / (R['nf'] / R['n']):.0f}x too high on real traffic.</li>")
    st.markdown(f"<div class='story' style='margin-top:1.2rem'><h4>Why Noon beat us</h4><ol>{''.join(story)}</ol></div>", unsafe_allow_html=True)


# ======================================================================
#  Page: Cost & cut-off
# ======================================================================
def cost_curve(y, p, loss: float, review: float, friction: float):
    """Expected cost when the k highest-risk orders are reviewed, for every k."""
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.r_[0, np.cumsum(ys)]
    k = np.arange(len(y) + 1)
    fp = k - tp
    fn = ys.sum() - tp
    cost = fn * loss + k * review + fp * friction
    thr = np.r_[1.0, p[order]]
    return k, cost, thr, tp, fp


def page_economics(ctx: Ctx):
    page_head("Business impact", "Cost & cut-off",
              "Put AED values on a missed fraud and on a false alarm, and find the cut-off that costs Noon least. "
              "Every model faces the same trade-off; this page prices it.")
    use_ref = ctx.ref is not None
    if use_ref:
        R = ref_eval(ctx)
        y = R["y"]
        series = [(ctx.ours, R["p_o"])] + ([(ctx.noon, R["p_n"])] if R["p_n"] is not None else [])
        base_rate = R["nf"] / R["n"]
    else:
        y = ctx.B["y"]
        series = [(ctx.ours, ctx.B["oof"])]
        base_rate = float(y.mean())

    c = st.columns(4)
    loss = c[0].number_input("Loss per missed fraud (AED)", 10, 5000, 250, step=10,
                             help="Order value plus chargeback fees and courier cost. Real fraud orders average about AED 140 before fees.")
    review = c[1].number_input("Review cost per flag (AED)", 0.0, 200.0, 8.0, step=1.0, help="Analyst minutes spent checking each flagged order.")
    friction = c[2].number_input("Extra cost per false alarm (AED)", 0.0, 500.0, 12.0, step=1.0,
                                 help="A delayed or cancelled legit order: support contact, refund credit, churn risk.")
    daily = c[3].number_input("Orders per day", 1000, 2_000_000, 50_000, step=5000)
    month = daily * 30
    n = len(y)

    no_model = y.sum() * loss / n * month
    res = []
    fig = go.Figure()
    for s_, p in series:
        k, cost, thr, tp, fp = cost_curve(y, p, loss, review, friction)
        per_month = cost / n * month
        i_best = int(np.argmin(per_month))
        k_cur = int((p >= s_.cutoff).sum())
        res.append(dict(s=s_, best=per_month[i_best], k_best=int(k[i_best]), thr_best=float(thr[i_best]), cur=per_month[k_cur],
                        k_cur=k_cur, tp_best=int(tp[i_best]), fp_best=int(fp[i_best])))
        sel = np.unique(np.r_[np.geomspace(1, n, 400).astype(int), i_best, k_cur])
        fig.add_trace(go.Scatter(x=k[sel] / n, y=per_month[sel], mode="lines", name=s_.name, line=dict(color=s_.color, width=2.6),
                                 customdata=np.c_[thr[sel], tp[sel], fp[sel]],
                                 hovertemplate=f"{s_.name}<br>Flag top %{{x:.2%}} (cut-off %{{customdata[0]:.1%}})<br>"
                                               f"Caught %{{customdata[1]:.0f}}, false alarms %{{customdata[2]:.0f}}<br>"
                                               f"AED %{{y:,.0f}} per month<extra></extra>"))
        fig.add_trace(go.Scatter(x=[k[i_best] / n], y=[per_month[i_best]], mode="markers", showlegend=False,
                                 marker=dict(size=12, color=s_.color, symbol="star", line=dict(color=C["bg"], width=2)),
                                 hovertemplate=f"{s_.name} cheapest cut-off<extra></extra>"))
        fig.add_trace(go.Scatter(x=[max(k_cur, 1) / n], y=[per_month[k_cur]], mode="markers", showlegend=False,
                                 marker=dict(size=10, color=C["bg"], line=dict(color=s_.color, width=2.5)),
                                 hovertemplate=f"{s_.name} current cut-off<extra></extra>"))
    fig.add_hline(y=no_model, line=dict(color=C["muted"], dash="dot", width=1))
    fig.add_annotation(x=1, xref="paper", y=no_model, text="no model: every fraud lost", showarrow=False, yshift=10,
                       xanchor="right", font=dict(color=C["muted"], size=11))
    ticks = [t for t in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) if t >= 1 / n]
    fig.update_xaxes(type="log", title="Share of orders flagged for review (log scale)", tickvals=ticks,
                     ticktext=[f"{t * 100:g}%" for t in ticks])
    fig.update_yaxes(title="Monthly cost: fraud lost + reviews + friction (AED)", range=[0, min(max([no_model * 1.7] + [r["cur"] * 1.12 for r in res]), no_model * 4)])

    items = [(aed(no_model), "Monthly fraud loss with no model", f"{month:,.0f} orders, {pct(base_rate, 2)} fraud")]
    for r in res:
        items.append((aed(no_model - r["best"]), f"Saved by {r['s'].name} at its best cut-off",
                      f"cut-off {pct(r['thr_best'], 1)}, flags {pct(r['k_best'] / n, 2)} of orders"))
    if len(res) == 2:
        items.append((aed(res[0]["best"] - res[1]["best"]), "Monthly value of Noon's edge", "difference at each model's best cut-off"))
    kpis(items)
    show(style(fig, 430), key="cost_curve")
    st.caption("Stars mark each model's cheapest cut-off; hollow dots mark the cut-off currently set in the sidebar.")

    rows_ = []
    for r in res:
        rows_.append([f"{swatch(r['s'].color)}{esc(r['s'].name)}", pct(r["s"].cutoff, 1), aed(r["cur"]), pct(r["thr_best"], 1), aed(r["best"]),
                      f"{r['tp_best']} / {int(y.sum())}", f"{r['fp_best']:,}"])
    st.markdown(html_table(["Model", "Current cut-off", "Monthly cost now", "Cheapest cut-off", "Monthly cost there", "Caught there", "False alarms there"],
                           rows_, num_cols={1, 2, 3, 4, 5, 6}), unsafe_allow_html=True)
    callout(("Computed on the real referee sample and scaled to your daily volume. Picking a cut-off on the same data you report on "
             "flatters every model; in production you would tune it on a separate validation month. " if use_ref else
             "Computed on our own out-of-fold scores, because the referee file is not loaded. ")
            + "The shape matters more than the totals: when false alarms are cheap, flag more; when they annoy good customers, flag less.")


# ======================================================================
#  Page: Ship it
# ======================================================================
def page_ship(ctx: Ctx):
    page_head("Phase 4 · ship it", "Ship it",
              "Everything you need to hand in or deploy: the model in the format the shell app expects, scored data, and a summary "
              "of the numbers.")
    B, df = ctx.B, ctx.df
    R = ref_eval(ctx)

    section("1 · Model for the fraud-check shell", "A drop-in fraud_model.joblib for the provided app.py shell.")
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        data = export_bytes(df, float(ctx.ours.cutoff), dfkey(df) + f"{ctx.ours.cutoff:.4f}")
        st.download_button("Download fraud_model.joblib", data, "fraud_model.joblib", "application/octet-stream", type="primary", **BTN_W)
        st.caption(f"{len(data) / 1024:,.0f} KB · gradient boosting trained on {B['n']:,} rows · predict() uses the "
                   f"{pct(ctx.ours.cutoff, 1)} cut-off from the sidebar.")
    with c2:
        st.code("import joblib\nM = joblib.load('fraud_model.joblib')\nmodel, COLUMNS = M['model'], M['columns']\n"
                "# COLUMNS match pd.get_dummies(..., columns=['payment_method'])", language="python")

    section("2 · Scored data")
    cc = st.columns(3)
    if R is not None:
        out = ctx.ref.copy()
        out["our_score"] = R["p_o"]
        out["our_flag"] = (R["p_o"] >= ctx.ours.cutoff).astype(int)
        if R["p_n"] is not None:
            out["noon_score"] = R["p_n"]
            out["noon_flag"] = (R["p_n"] >= ctx.noon.cutoff).astype(int)
        cc[0].download_button("Referee set with both models' scores", out.to_csv(index=False).encode(), "referee_scored.csv", "text/csv", **BTN_W)
    our_fraud = df[df[TARGET] == 1]
    cc[1].download_button(f"Our fraud rows ({len(our_fraud):,})", our_fraud.to_csv(index=False).encode(), "our_fraud_rows.csv", "text/csv", **BTN_W)
    cc[2].download_button("Training set with out-of-fold scores", df.assign(oof_score=B["oof"]).to_csv(index=False).encode(),
                          "training_oof_scores.csv", "text/csv", **BTN_W)

    section("3 · Numbers at a glance", "Every headline metric in one table, ready to paste into the report.")
    m_oof = at_cutoff(B["y"], B["oof"], ctx.ours.cutoff)
    rows = [dict(where="Our data (out-of-fold)", model=ctx.name_ours, recall=m_oof["recall"], precision=m_oof["precision"],
                 f1=m_oof["f1"], caught=m_oof["tp"], false_alarms=m_oof["fp"], missed=m_oof["fn"])]
    if R is not None:
        mo = R["m_o"]
        rows.append(dict(where="Real referee", model=ctx.name_ours, recall=mo["recall"], precision=mo["precision"], f1=mo["f1"],
                         caught=mo["tp"], false_alarms=mo["fp"], missed=mo["fn"]))
        mn = R["m_n"] or NOON_BENCH
        rows.append(dict(where="Real referee", model="Noon (real + GAN)" + ("" if R["m_n"] else ", benchmark"), recall=mn["recall"],
                         precision=mn["precision"], f1=mn["f1"], caught=mn["tp"], false_alarms=mn["fp"], missed=mn["fn"]))
        rows.append(dict(where="Real referee", model="Noon (real only), benchmark", recall=REAL_ONLY_BENCH["recall"],
                         precision=REAL_ONLY_BENCH["precision"], f1=REAL_ONLY_BENCH["f1"], caught=None, false_alarms=REAL_ONLY_BENCH["fp"], missed=None))
    tbl = pd.DataFrame(rows)
    st.markdown(html_table(["Evaluated on", "Model", "Recall", "Precision", "F1", "Caught", "False alarms", "Missed"],
                           [[esc(r.where), esc(r.model), f"{r.recall:.2f}", f"{r.precision:.2f}", f"{r.f1:.2f}",
                             "–" if pd.isna(r.caught) else f"{int(r.caught):,}", f"{int(r.false_alarms):,}",
                             "–" if pd.isna(r.missed) else f"{int(r.missed):,}"] for r in tbl.itertuples()],
                           num_cols={2, 3, 4, 5, 6, 7}), unsafe_allow_html=True)
    st.download_button("Download metrics (CSV)", tbl.to_csv(index=False).encode(), "metrics_summary.csv", "text/csv")

    with st.expander("How to run this desk"):
        st.markdown("1. `pip install -r requirements.txt`\n"
                    "2. Put `FINAL_dataset.csv`, `B_NOON_synthetic_fraud.csv`, `C_REFEREE_real_test.csv` and `B_NOON_model.joblib` next to `app.py`.\n"
                    "3. `streamlit run app.py`\n\nAny file can also be uploaded from **Data sources** in the sidebar.")


# ======================================================================
#  App shell
# ======================================================================
THEME = {"theme.base": "dark", "theme.primaryColor": C["ours"], "theme.backgroundColor": C["bg"],
         "theme.secondaryBackgroundColor": C["panel"], "theme.textColor": C["ink"]}

PAGES = {
    "Briefing": (":material/space_dashboard:", page_briefing),
    "Model lab": (":material/science:", page_model_lab),
    "Score an order": (":material/speed:", page_score),
    "Classify a file": (":material/table_view:", page_batch),
    "GAN face-off": (":material/compare:", page_faceoff),
    "Real-world referee": (":material/gavel:", page_referee),
    "Cost & cut-off": (":material/payments:", page_economics),
    "Ship it": (":material/rocket_launch:", page_ship),
}

SOURCES = {
    "ours": ("Our dataset", ["FINAL_dataset.csv", "*FINAL_dataset*.csv"], "csv"),
    "nf": ("Noon GAN fraud", ["B_NOON_synthetic_fraud.csv", "*NOON_synthetic*.csv"], "csv"),
    "ref": ("Real referee set", ["C_REFEREE_real_test.csv", "*REFEREE*.csv"], "csv"),
    "noon_model": ("Noon's model", ["B_NOON_model.joblib", "*NOON_model*.joblib"], "joblib"),
}


def apply_theme():
    """Make the dark theme work even without a .streamlit/config.toml."""
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
    except Exception:  # noqa: BLE001
        pass


@st.cache_data(show_spinner=False)
def read_bytes(path: str, mtime: float) -> bytes:
    return Path(path).read_bytes()


def source_bytes(key: str):
    """Uploaded file wins; otherwise the first matching file next to app.py. Returns (bytes, name, size_key)."""
    up = st.session_state.get(f"src_{key}")
    if up is not None:
        return up.getvalue(), up.name, f"{up.name}-{up.size}"
    for pat in SOURCES[key][1]:
        hits = sorted(HERE.glob(pat))
        if hits:
            p = hits[0]
            stt = p.stat()
            return read_bytes(str(p), stt.st_mtime), p.name, f"{p.name}-{stt.st_size}-{stt.st_mtime}"
    return None, None, None


def main():
    apply_theme()
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            "<div class='brand'><svg class='mark' viewBox='0 0 32 32'><rect width='32' height='32' rx='9' fill='#1A2147'/>"
            "<path d='M7 21a9 9 0 0 1 18 0' stroke='#3987E5' stroke-width='3' fill='none' stroke-linecap='round'/>"
            "<path d='M16 21 L21 14' stroke='#EEF0FA' stroke-width='2.2' stroke-linecap='round'/>"
            "<circle cx='16' cy='21' r='2.4' fill='#EEF0FA'/></svg><span class='t'>Fraud desk<small>Noon food delivery</small></span></div>",
            unsafe_allow_html=True)
        if "nav" not in st.session_state:
            st.session_state["nav"] = "Briefing"
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed",
                        format_func=lambda p: f"{PAGES[p][0]} {p}")

        # ---- data
        loaded = {}
        for key in SOURCES:
            loaded[key] = source_bytes(key)
        st.markdown("<div class='side-h'>Data sources</div>", unsafe_allow_html=True)
        status = "".join(f"<div class='src'><b>{esc(SOURCES[k][0])}</b>{chip(loaded[k][0] is not None, 'loaded', 'missing')}</div>"
                         for k in SOURCES)
        st.markdown(status, unsafe_allow_html=True)
        with st.expander("Upload or replace files"):
            for k, (label, _, kind) in SOURCES.items():
                st.file_uploader(label, type=["csv"] if kind == "csv" else ["joblib", "pkl"], key=f"src_{k}")

        raw, label, _ = loaded["ours"]
        if raw is None:
            st.error("Our dataset is missing. Put FINAL_dataset.csv next to app.py, or upload it above.")
            st.stop()
        df, err = load_frame(raw, need_label=True)
        if df is None:
            st.error(err or "Could not load our dataset.")
            st.stop()
        nf = ref = noon_model = None
        if loaded["nf"][0] is not None:
            nf, e = load_frame(loaded["nf"][0], need_label=False)
            if nf is None:
                st.caption(f"Noon GAN fraud: {e}")
            elif TARGET not in nf.columns:
                nf[TARGET] = 1
        if loaded["ref"][0] is not None:
            ref, e = load_frame(loaded["ref"][0], need_label=True)
            if ref is None:
                st.caption(f"Referee set: {e}")
        if loaded["noon_model"][0] is not None:
            try:
                noon_model = load_noon(loaded["noon_model"][2], loaded["noon_model"][0])
                noon_model.proba(df.head(3))
            except Exception as e:  # noqa: BLE001
                noon_model = None
                st.caption(f"Could not use Noon's model ({type(e).__name__}); its published benchmark is used instead.")

        B = build_bundle(df, dfkey(df))

        # ---- cut-offs
        st.markdown("<div class='side-h'>Decision cut-offs</div>", unsafe_allow_html=True)
        if st.session_state.get("_ds") != dfkey(df):
            st.session_state["cut_pct"] = max(1, round(B["cutoff"] * 100))
            st.session_state["rev_pct"] = max(1, round(B["cutoff"] * 50))
            st.session_state["_ds"] = dfkey(df)
        st.session_state["rev_pct"] = min(st.session_state["rev_pct"], max(st.session_state["cut_pct"] - 1, 1))
        cut_pct = st.slider("Our fraud cut-off", 1, 95, key="cut_pct", format="%d%%",
                            help="Orders scoring at or above this are called likely fraud.")
        rev_pct = st.slider("Send to review from", 1, max(cut_pct - 1, 1), key="rev_pct", format="%d%%",
                            help="Orders between this and the cut-off are marked for a human to look at.")
        st.caption(f"Suggested: {pct(B['cutoff'], 0)}, where precision and recall balance best on our data.")
        noon_cut = 50
        if noon_model is not None:
            noon_cut = st.slider("Noon's cut-off", 1, 95, 50, key="noon_pct", format="%d%%",
                                 help="50% is what Noon's model.predict() uses and what their benchmark reports.")
        with st.expander("Names on charts"):
            name_ours = st.text_input("Our model", "Our model")
            name_noon = st.text_input("Noon's GAN data", "Noon GAN")
        st.markdown(f"<div class='side-note'>Our qualifier: gradient boosting on {B['n']:,} rows from {esc(label)}."
                    + (f" Noon's: {esc(noon_model.kind)}." if noon_model is not None else "") + "</div>", unsafe_allow_html=True)

    ours = Scorer("ours", name_ours, C["ours"], cut_pct / 100, rev_pct / 100, lambda d: proba(B["model"], d))
    # +1e-9 mirrors model.predict(), which only flags probabilities strictly above 0.5
    noon = (Scorer("noon", "Noon's model", C["noon"], noon_cut / 100 + 1e-9, noon_cut / 200, noon_model.proba)
            if noon_model is not None else None)
    ctx = Ctx(df=df, B=B, ours=ours, noon=noon, noon_model=noon_model, nf=nf, ref=ref, label=label,
              name_ours=name_ours, name_noon=name_noon)
    PAGES[page][1](ctx)


main()
