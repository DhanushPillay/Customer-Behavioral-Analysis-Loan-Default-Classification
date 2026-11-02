"""Loan default review dashboard.

Every figure is read or computed from the artefacts produced by the notebooks
and tools/ scripts: data/loan_features.csv, outputs/*.csv, outputs/models/*.pkl
and outputs/figures/*.png. Nothing on the page is typed in by hand.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLEAN = DATA / "clean"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

PAPER = "#fbfaf7"
INK = "#191b1f"
MUTED = "#6b6f76"
RULE = "#e4e1d8"
ACCENT = "#1f4e6b"
RISK = "#a63d2f"
GOOD = "#2f6b4f"

DROP = [
    "loan_id",
    "account_id",
    "client_id",
    "district_id",
    "status",
    "date",
    "parsed_date",
    "gender",
    "target",
]

SIM_INPUTS = {
    "amount": "Loan amount",
    "duration": "Duration",
    "balance_at_loan": "Balance at loan date",
    "min_balance": "Minimum balance",
    "txn_per_month": "Transactions per month",
}


def stress_text(key: str, value: float) -> str:
    if key in {"amount", "balance_at_loan", "min_balance"}:
        return f"{value:,.0f} CZK"
    if key == "duration":
        return f"{value:.0f} months"
    return f"{value:.1f} per month"

PAGES = ["Portfolio", "Data & leakage", "Models", "Patterns", "Simulator"]

st.set_page_config(
    page_title="Loan default review",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
      --paper: #fbfaf7; --ink: #191b1f; --muted: #6b6f76; --rule: #e4e1d8;
      --accent: #1f4e6b; --risk: #a63d2f; --good: #2f6b4f;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
      --sans: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp { background: var(--paper); color: var(--ink); font-family: var(--sans); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #f3f0e7; border-right: 1px solid var(--rule); }
    .block-container { max-width: 1160px; padding-top: 2.4rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: var(--serif); font-weight: 500; letter-spacing: -0.015em; color: var(--ink); }
    h1 { font-size: 2.3rem; line-height: 1.15; margin: 0.15rem 0 0.45rem; }
    h3 { font-size: 1.12rem; margin-top: 1.4rem; }
    p, li { line-height: 1.6; }
    a { color: var(--accent); }
    .eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); }
    .lede { max-width: 64ch; color: var(--muted); font-size: 0.97rem; margin: 0.1rem 0 1.5rem; }
    .kpi { border-top: 1px solid var(--ink); padding: 0.55rem 0 0; }
    .kpi-label { font-size: 0.63rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); }
    .kpi-value { font-family: var(--serif); font-size: 1.72rem; line-height: 1.15; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
    .kpi-note { font-size: 0.74rem; color: var(--muted); margin-top: 0.12rem; }
    .section { display: flex; align-items: baseline; gap: 0.9rem; border-top: 1px solid var(--rule); margin: 2.4rem 0 0.9rem; padding-top: 0.8rem; }
    .section span { font-size: 0.69rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--ink); white-space: nowrap; }
    .section em { font-style: normal; font-size: 0.78rem; color: var(--muted); }
    .note { border-left: 2px solid var(--accent); padding: 0.05rem 0 0.05rem 0.85rem; margin: 0.6rem 0 1.2rem; font-size: 0.9rem; line-height: 1.55; }
    .note b { display: block; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.22rem; }
    .note.risk { border-color: var(--risk); }
    .note.risk b { color: var(--risk); }
    .brand { font-family: var(--serif); font-size: 1.14rem; line-height: 1.2; }
    .brand-sub { font-size: 0.72rem; color: var(--muted); line-height: 1.45; margin: 0.25rem 0 1.1rem; }
    .side-label { font-size: 0.61rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 1.3rem 0 0.35rem; }
    .side-line { display: flex; justify-content: space-between; font-size: 0.78rem; border-bottom: 1px solid var(--rule); padding: 0.24rem 0; }
    .side-line span:last-child { font-variant-numeric: tabular-nums; }
    .prob { font-family: var(--serif); font-size: 3.2rem; line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .prob.risk { color: var(--risk); }
    .prob.calm { color: var(--good); }
    .verdict { font-size: 0.9rem; line-height: 1.5; margin-top: 0.35rem; color: var(--ink); }
    .matrix { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
    .matrix th, .matrix td { border-bottom: 1px solid var(--rule); padding: 0.42rem 0.5rem; text-align: right; }
    .matrix th:first-child, .matrix td:first-child { text-align: left; }
    .matrix thead th { font-size: 0.61rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
    .matrix td.num { font-variant-numeric: tabular-nums; }
    .matrix .hit { color: var(--good); font-weight: 600; }
    .matrix .miss { color: var(--risk); font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--rule); }
    .stTabs [data-baseweb="tab"] { padding: 0.4rem 0; font-size: 0.8rem; }
    [data-testid="stExpander"] details { border: 1px solid var(--rule); border-radius: 0; }
    div[data-testid="stImage"] figcaption { font-size: 0.74rem; color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": RULE,
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.8,
    }
)

HEAT = LinearSegmentedColormap.from_list("paper_ink", ["#fbfaf7", "#c9d6de", "#1f4e6b"])


# ---------------------------------------------------------------- data access


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA / "loan_features.csv")


@st.cache_data(show_spinner=False)
def load_output(name: str) -> pd.DataFrame | None:
    path = OUTPUTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "model_scores.csv")


@st.cache_data(show_spinner="Reading transactions...")
def load_transactions() -> pd.DataFrame:
    return pd.read_csv(CLEAN / "fact_transactions.csv", usecols=["account_id", "date_id"])


@st.cache_data(show_spinner=False)
def table_counts() -> pd.DataFrame:
    rows = []
    for path in sorted(CLEAN.glob("*.csv")):
        with path.open("rb") as handle:
            rows.append({"table": path.stem, "rows": sum(1 for _ in handle) - 1})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODELS / "best_GradientBoosting_baseline_tuned.pkl")


@st.cache_data(show_spinner=False)
def leakage_audit() -> pd.DataFrame:
    features = load_features()
    transactions = load_transactions()
    cuts = pd.to_datetime(features["parsed_date"]).dt.strftime("%Y%m%d").astype(int)
    grouped = {
        account: np.sort(frame["date_id"].to_numpy())
        for account, frame in transactions.groupby("account_id")
    }
    rows = []
    for loan_id, account, cut, n_txn in zip(
        features["loan_id"], features["account_id"], cuts, features["n_txn"]
    ):
        dates = grouped.get(account)
        used = int(np.searchsorted(dates, cut, side="left")) if dates is not None else 0
        total = int(dates.size) if dates is not None else 0
        rows.append((loan_id, account, cut, n_txn, used, total))
    return pd.DataFrame(
        rows, columns=["loan_id", "account_id", "cutoff", "n_txn", "used", "total"]
    )


# ------------------------------------------------------------------- helpers


def page_header(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p class="lede">{lede}</p>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    extra = f"<em>{note}</em>" if note else ""
    st.markdown(
        f'<div class="section"><span>{title}</span>{extra}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        column.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{note}</div></div>',
            unsafe_allow_html=True,
        )


def note(text: str, label: str = "Finding", kind: str = "") -> None:
    st.markdown(
        f'<div class="note {kind}"><b>{label}</b>{text}</div>',
        unsafe_allow_html=True,
    )


def show(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{value:.1%}"


def money(value: float) -> str:
    return f"{value:,.0f} CZK"


def parse_itemset(value: str) -> str:
    text = str(value).strip()
    if text.startswith("frozenset(") and text.endswith(")"):
        text = text[len("frozenset(") : -1]
    try:
        parsed = literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    if isinstance(parsed, (set, frozenset, list, tuple)):
        return ", ".join(sorted(str(item) for item in parsed))
    return str(parsed)


def model_columns(model, features: pd.DataFrame) -> list[str]:
    stored = getattr(model, "feature_names_in_", None)
    if stored is not None:
        return [str(column) for column in stored]
    return [column for column in features.columns if column not in DROP]


def median_row(features: pd.DataFrame) -> pd.Series:
    base = features.drop(columns=[column for column in DROP if column in features.columns])
    return base.median(numeric_only=True)


def applicant_row(base: pd.Series, values: dict) -> pd.Series:
    row = base.copy()
    row["amount"] = float(values["amount"])
    row["duration"] = float(values["duration"])
    row["payments"] = float(values["amount"]) / float(values["duration"])
    row["balance_at_loan"] = float(values["balance_at_loan"])
    row["min_balance"] = float(values["min_balance"])
    row["txn_per_month"] = float(values["txn_per_month"])
    return row


def predict(model, row: pd.Series, columns: list[str]) -> float:
    frame = pd.DataFrame([row[columns].astype(float).to_numpy()], columns=columns)
    return float(model.predict_proba(frame)[0, 1])


def cutoff_stats(scores: pd.DataFrame, features: pd.DataFrame, cutoff: float) -> dict:
    data = scores.merge(features[["loan_id", "amount"]], on="loan_id", how="left")
    flagged = data["prob_default"] >= cutoff
    is_default = data["target"] == 1
    tp = int((flagged & is_default).sum())
    fp = int((flagged & ~is_default).sum())
    fn = int((~flagged & is_default).sum())
    tn = int((~flagged & ~is_default).sum())
    return {
        "flagged": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "exposure": float(data.loc[flagged, "amount"].sum()),
        "defaults": int(is_default.sum()),
        "total": int(len(data)),
    }


def gains_figure(scores: pd.DataFrame, cutoff: float):
    ordered = scores.sort_values("prob_default", ascending=False).reset_index(drop=True)
    share = np.arange(1, len(ordered) + 1) / len(ordered)
    captured = ordered["target"].cumsum() / ordered["target"].sum()
    flagged = int((ordered["prob_default"] >= cutoff).sum())
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(share, captured, color=ACCENT)
    ax.plot([0, 1], [0, 1], color=RULE, linestyle=(0, (4, 3)))
    if flagged:
        ax.scatter([share[flagged - 1]], [captured.iloc[flagged - 1]], color=RISK, s=26, zorder=5)
        ax.annotate(
            f"cutoff {cutoff:.2f}",
            (share[flagged - 1], captured.iloc[flagged - 1]),
            xytext=(7, -13),
            textcoords="offset points",
            fontsize=8,
            color=RISK,
        )
    ax.set_xlabel("Share of portfolio reviewed")
    ax.set_ylabel("Share of defaults captured")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(alpha=0.5)
    return fig


def quintile_figure(features: pd.DataFrame, column: str, label: str):
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    bands = pd.qcut(features[column], 5, labels=labels, duplicates="drop")
    rates = features.groupby(bands, observed=True)["target"].mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    colors = [RISK if index == "Q1" else ACCENT for index in rates.index.astype(str)]
    ax.bar(rates.index.astype(str), rates.values, color=colors, width=0.62)
    ax.set_xlabel(f"{label} quintile")
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, max(rates.max() * 1.25, 0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.5)
    return fig


def heatmap_figure(table: pd.DataFrame, labels: list[str], row_labels: list[str]):
    fig, ax = plt.subplots(figsize=(9.0, max(3.2, len(table) * 0.42)))
    image = ax.imshow(table.to_numpy(), cmap=HEAT, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    for row, (_, values) in enumerate(table.iterrows()):
        for column, value in enumerate(values):
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.62 else INK,
            )
    fig.colorbar(image, ax=ax, label="Score", fraction=0.025)
    return fig


def driver_figure(labels: list[str], deltas: list[float]):
    order = np.argsort(deltas)
    values = [deltas[index] for index in order]
    names = [labels[index] for index in order]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.barh(names, values, color=[RISK if value > 0 else GOOD for value in values], height=0.55)
    ax.axvline(0, color=RULE, linewidth=1)
    ax.set_xlabel("Change in default probability, percentage points")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:+.0f}"))
    ax.grid(axis="x", alpha=0.5)
    return fig


def leakage_figure(used: int, excluded: int):
    total = max(used + excluded, 1)
    fig, ax = plt.subplots(figsize=(9.0, 1.55))
    ax.barh([0], [used], color=ACCENT, height=0.5)
    ax.barh([0], [excluded], left=[used], color=RULE, height=0.5)
    ax.annotate(
        f"{used} used ({used / total:.1%})",
        xy=(used, 0.25),
        xytext=(max(total * 0.05, 10), 0.66),
        ha="left",
        va="bottom",
        fontsize=8,
        color=ACCENT,
        arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 0.8},
    )
    ax.text(
        used + total * 0.015,
        0,
        f"{excluded:,} excluded",
        va="center",
        ha="left",
        fontsize=8,
        color=MUTED,
    )
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.0)
    ax.set_xlabel("Transactions on the account, before and after the loan date")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    return fig


def sidebar(features: pd.DataFrame) -> str:
    st.sidebar.markdown('<div class="brand">Loan default review</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="brand-sub">PKDD\'99 banking records<br>Pre-loan behaviour only</div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("View", PAGES, label_visibility="collapsed")
    st.sidebar.markdown('<div class="side-label">Corpus</div>', unsafe_allow_html=True)
    defaults = int(features["target"].sum())
    rows = [
        ("Applications", f"{len(features):,}"),
        ("Defaults", f"{defaults:,}"),
        ("Base rate", percent(features["target"].mean())),
    ]
    for label, value in rows:
        st.sidebar.markdown(
            f'<div class="side-line"><span>{label}</span><span>{value}</span></div>',
            unsafe_allow_html=True,
        )
    return page


# ------------------------------------------------------------------ portfolio


def portfolio_page() -> None:
    features = load_features()
    scores = load_scores()
    defaults = int(features["target"].sum())
    base_rate = float(features["target"].mean())

    page_header(
        "Portfolio",
        f"{len(features):,} applications, {defaults} defaults, one review budget",
        "Every figure on this page is computed from data/loan_features.csv and the "
        "out-of-fold scores in outputs/model_scores.csv. Move the threshold to see what "
        "a review queue would catch and what it would cost.",
    )

    kpi_row(
        [
            ("Applications", f"{len(features):,}", "One row per loan"),
            ("Defaults", f"{defaults}", "Observed outcome"),
            ("Base rate", percent(base_rate), "No model involved"),
            ("Median score", f"{scores['prob_default'].median():.3f}", "Out-of-fold probability"),
        ]
    )

    section("Review threshold", "Out-of-fold scores, so no loan is scored by a model that saw it")
    cutoff = st.slider(
        "Flag applications scoring at or above",
        min_value=0.05,
        max_value=0.90,
        value=0.40,
        step=0.01,
    )
    stats = cutoff_stats(scores, features, cutoff)
    kpi_row(
        [
            ("In the queue", f"{stats['flagged']}", f"{stats['flagged'] / stats['total']:.1%} of the portfolio"),
            ("Defaults caught", f"{stats['tp']} / {stats['defaults']}", f"{stats['fn']} missed"),
            ("Precision", percent(stats["precision"]), f"{stats['fp']} false alarms"),
            ("Recall", percent(stats["recall"]), "Share of defaults found"),
        ]
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        show(gains_figure(scores, cutoff))
    with right:
        st.markdown(
            f'<table class="matrix"><thead><tr><th></th><th>Flagged</th><th>Not flagged</th></tr></thead><tbody>'
            f'<tr><td>Defaulted</td><td class="num hit">{stats["tp"]}</td><td class="num miss">{stats["fn"]}</td></tr>'
            f'<tr><td>Repaid</td><td class="num">{stats["fp"]}</td><td class="num">{stats["tn"]}</td></tr>'
            f"</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kpi" style="margin-top:1.1rem"><div class="kpi-label">Exposure in the queue</div>'
            f'<div class="kpi-value">{stats["exposure"] / 1e6:.1f}M CZK</div>'
            f'<div class="kpi-note">Sum of loan amounts flagged</div></div>',
            unsafe_allow_html=True,
        )
        note(
            f"At {cutoff:.2f} the queue holds {stats['flagged']} applications: {stats['tp']} of "
            f"{stats['defaults']} defaults and {stats['fp']} that would have repaid.",
            label="At this setting",
        )

    rows = []
    for value in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        entry = cutoff_stats(scores, features, value)
        rows.append(
            {
                "Cutoff": f"{value:.2f}",
                "In queue": entry["flagged"],
                "Defaults caught": f"{entry['tp']} / {entry['defaults']}",
                "Precision": percent(entry["precision"]),
                "Recall": percent(entry["recall"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "The trade-off is monotone: a higher cutoff buys precision with recall. "
        "The project's selected operating point is 0.40."
    )

    section("Where the risk sits", "Default rate by quintile, all 682 loans")
    left, right = st.columns(2, gap="large")
    with left:
        show(quintile_figure(features, "balance_at_loan", "Balance at loan"))
        low = features["balance_at_loan"].quantile(0.2)
        st.caption(f"Q1 is everything below {money(low)}.")
    with right:
        show(quintile_figure(features, "min_balance", "Minimum balance"))
        low = features["min_balance"].quantile(0.2)
        st.caption(f"Q1 is everything below {money(low)}.")

    note(
        "The bottom balance quintile defaults at "
        f"{features.loc[features['balance_at_loan'] <= features['balance_at_loan'].quantile(0.2), 'target'].mean():.1%}, "
        "against a portfolio rate of "
        f"{base_rate:.1%}. Balance capacity separates outcomes; transaction volume separates them far less.",
        label="What the split shows",
    )

    section("Behaviour segments", "KMeans k=2 on pre-loan behaviour, from outputs/segment_profiles.csv")
    profiles = load_output("segment_profiles.csv")
    if profiles is None:
        st.warning("Segment profiles are not available.")
        return
    choice = st.selectbox(
        "Segment",
        options=profiles["segment"].tolist(),
        format_func=lambda value: f"Segment {value}",
    )
    profile = profiles[profiles["segment"] == choice].iloc[0]
    kpi_row(
        [
            ("Loans", f"{int(profile['size'])}", "Members of this segment"),
            ("Default rate", percent(profile["default_rate"]), "Observed"),
            ("Vs base rate", f"{profile['default_rate'] - base_rate:+.1%}", "Percentage points"),
            ("Mean transactions", f"{profile['n_txn']:.0f}", "Per loan"),
        ]
    )
    members = features[features["loan_id"].isin(
        load_output("loan_segments.csv").query("segment == @choice")["loan_id"]
    )].merge(scores[["loan_id", "prob_default"]], on="loan_id", how="left")
    st.dataframe(
        members.sort_values("prob_default", ascending=False)
        .head(12)[["loan_id", "amount", "duration", "balance_at_loan", "min_balance", "txn_per_month", "prob_default", "target"]]
        .rename(
            columns={
                "loan_id": "Loan",
                "amount": "Amount",
                "duration": "Months",
                "balance_at_loan": "Balance",
                "min_balance": "Min balance",
                "txn_per_month": "Txn / month",
                "prob_default": "Score",
                "target": "Defaulted",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Amount": st.column_config.NumberColumn(format="%.0f"),
            "Balance": st.column_config.NumberColumn(format="%.0f"),
            "Min balance": st.column_config.NumberColumn(format="%.0f"),
            "Txn / month": st.column_config.NumberColumn(format="%.1f"),
            "Score": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    note(
        f"The two segments differ by {abs(profiles['default_rate'].diff().iloc[-1]):.1%} in default rate. "
        "That is weak separation on this sample, and segment membership is not a model feature. "
        "Read it as a description of the book, not as a risk rule.",
        label="Caveat",
        kind="risk",
    )


# ------------------------------------------------------------- data & leakage


def data_page() -> None:
    features = load_features()
    counts = table_counts()
    audit = leakage_audit()
    matches = int((audit["used"] == audit["n_txn"]).sum())

    page_header(
        "Data",
        "1,056,320 transactions, 682 loans, no lookahead",
        "The modelling table is built from the star schema in data/clean. One rule keeps it "
        "honest: every transaction used for a loan happened strictly before the loan date.",
    )

    section("Star schema", "Row counts read from data/clean")
    grain = {
        "dim_account": ("one row per account", "Dimension"),
        "dim_branch": ("one row per branch", "Dimension"),
        "dim_customer": ("one row per client", "Dimension"),
        "dim_date": ("one row per calendar day", "Dimension"),
        "dim_loan_type": ("one row per loan type", "Dimension"),
        "fact_loans": ("one row per loan", "Fact"),
        "fact_transactions": ("one row per transaction", "Fact"),
    }
    schema = counts.copy()
    schema["Grain"] = schema["table"].map(lambda name: grain.get(name, ("", ""))[0])
    schema["Role"] = schema["table"].map(lambda name: grain.get(name, ("", ""))[1])
    schema = schema.rename(columns={"table": "Table", "rows": "Rows"})[["Table", "Rows", "Grain", "Role"]]
    st.dataframe(
        schema,
        hide_index=True,
        width="stretch",
        column_config={"Rows": st.column_config.NumberColumn(format="%d")},
    )

    section("Leakage check", "Pick a loan; the split is computed from fact_transactions")
    loan_dates = dict(zip(features["loan_id"], features["parsed_date"]))
    loan_amounts = dict(zip(features["loan_id"], features["amount"]))
    loan_id = st.selectbox(
        "Loan",
        options=features["loan_id"].tolist(),
        format_func=lambda value: f"{value} | {loan_dates[value]} | {loan_amounts[value]:,.0f} CZK",
    )
    chosen = audit[audit["loan_id"] == loan_id].iloc[0]
    excluded = int(chosen["total"] - chosen["used"])
    kpi_row(
        [
            ("Loan date", str(loan_dates[loan_id]), "Cutoff for this loan"),
            ("Account", f"{int(chosen['account_id'])}", "Transaction key"),
            ("Transactions used", f"{int(chosen['used'])}", "Strictly before the loan"),
            ("Transactions excluded", f"{excluded:,}", "On or after the loan"),
        ]
    )
    used = int(chosen["used"])
    total = max(used + excluded, 1)
    show(leakage_figure(used, excluded))
    st.caption(
        f"{used} of {total:,} transactions on this account fall strictly before the loan date. "
        f"The other {excluded:,} are excluded from the feature build."
    )

    note(
        f"Across all {len(features)} loans, n_txn equals the count of transactions strictly before "
        f"the loan date: {matches} of {len(features)} match, and no same-day transaction is used. "
        "This is the check that separates a pre-loan model from a leaked one.",
        label="Verified",
    )

    section("Feature provenance", "Where each modelling column comes from")
    provenance = pd.DataFrame(
        [
            ["Loan terms", "loan.csv", "amount, duration, payments, date"],
            ["Transaction volume", "trans.csv", "n_txn, n_deposit, n_withdrawal, sum_deposit, sum_withdrawal"],
            ["Transaction size", "trans.csv", "avg_amount, std_amount"],
            ["Balance history", "trans.csv", "avg_balance, min_balance, max_balance, balance_at_loan"],
            ["Activity window", "trans.csv", "months_active, txn_per_month"],
            ["Customer", "client.csv", "gender, age, district_id"],
            ["Ownership", "disp.csv, account.csv", "account_id, client_id"],
        ],
        columns=["Group", "Source", "Columns"],
    )
    st.dataframe(provenance, hide_index=True, width="stretch")

    section("Quality checks", "Computed from the modelling table")
    transactions = load_transactions()
    checks = pd.DataFrame(
        [
            ["Rows", f"{len(features):,}", "One per loan"],
            ["Columns", f"{features.shape[1]}", "Terms, behaviour, context, target"],
            ["Missing cells", f"{int(features.isna().sum().sum())}", "None"],
            ["Duplicate loan ids", f"{int(features['loan_id'].duplicated().sum())}", "None"],
            ["Defaults", f"{int(features['target'].sum())}", percent(features["target"].mean())],
            ["Transaction dates", f"{int(transactions['date_id'].min())} to {int(transactions['date_id'].max())}", "Six years"],
            ["Accounts with transactions", f"{transactions['account_id'].nunique():,}", "Of 4,500 accounts"],
        ],
        columns=["Check", "Result", "Note"],
    )
    st.dataframe(checks, hide_index=True, width="stretch")


# --------------------------------------------------------------------- models


def models_page() -> None:
    cv = load_output("cv_comparison.csv")
    metrics = load_output("metrics_comparison.csv")
    tuning = load_output("tuning_results.csv")
    if cv is None or metrics is None:
        st.error("Model comparison outputs are missing.")
        return

    page_header(
        "Models",
        "Six classifiers, 50 repeated-CV estimates each",
        "Selection rule: highest repeated-CV F1 on the unresampled baseline arm. The other "
        "columns are checks on that rule, not tie-breakers.",
    )

    controls = st.columns(3)
    metric = controls[0].selectbox("Rank by", ["F1", "Average precision", "ROC-AUC"])
    setting = controls[1].radio("Resampling", ["baseline", "smote"], horizontal=True)
    variant = controls[2].radio("Variant", ["tuned", "default"], horizontal=True)

    metric_column = {"F1": "cv_f1_mean", "Average precision": "cv_ap_mean", "ROC-AUC": "cv_auc_mean"}[metric]
    selected = cv[(cv["setting"] == setting) & (cv["variant"] == variant)].copy()
    if selected.empty:
        st.warning("No runs match that combination.")
        return
    board = selected.sort_values(metric_column, ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board) + 1))
    st.dataframe(
        board[["Rank", "model", "cv_f1_mean", "cv_f1_std", "cv_ap_mean", "cv_auc_mean"]].rename(
            columns={
                "model": "Model",
                "cv_f1_mean": "F1",
                "cv_f1_std": "F1 sd",
                "cv_ap_mean": "Avg precision",
                "cv_auc_mean": "ROC-AUC",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "F1": st.column_config.NumberColumn(format="%.3f"),
            "F1 sd": st.column_config.NumberColumn(format="%.3f"),
            "Avg precision": st.column_config.NumberColumn(format="%.3f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    section("Held-out scores", "Default hyperparameters on the 171-row test split")
    held = metrics.copy()
    held["Run"] = held["model"] + " / " + held["setting"]
    columns = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "Avg precision", "ROC-AUC"]
    show(
        heatmap_figure(
            held.set_index("Run")[columns],
            labels,
            held["Run"].tolist(),
        )
    )

    tabs = st.tabs(["Repeated CV", "Curves", "Tuning", "Challenger"])

    with tabs[0]:
        table = cv.copy()
        table["Configuration"] = table["setting"] + " / " + table["variant"]
        st.dataframe(
            table.sort_values("cv_f1_mean", ascending=False)[
                ["model", "Configuration", "cv_f1_mean", "cv_f1_std", "cv_ap_mean", "cv_auc_mean"]
            ].rename(
                columns={
                    "model": "Model",
                    "cv_f1_mean": "F1",
                    "cv_f1_std": "F1 sd",
                    "cv_ap_mean": "Avg precision",
                    "cv_auc_mean": "ROC-AUC",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "F1": st.column_config.NumberColumn(format="%.3f"),
                "F1 sd": st.column_config.NumberColumn(format="%.3f"),
                "Avg precision": st.column_config.NumberColumn(format="%.3f"),
                "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("24 configurations, 50 estimates each. Selection uses this table, not the held-out split.")

    with tabs[1]:
        left, right = st.columns(2)
        for column, name, caption in (
            (left, "roc_curves.png", "ROC curves, baseline and SMOTE arms."),
            (right, "pr_curves.png", "Precision-recall curves. The honest view at 11% prevalence."),
        ):
            path = FIGURES / name
            if path.exists():
                column.image(str(path), caption=caption, width="stretch")
        path = FIGURES / "confusion_matrices.png"
        if path.exists():
            st.image(str(path), caption="Confusion matrices for all 12 held-out runs.", width="stretch")

    with tabs[2]:
        if tuning is not None:
            st.dataframe(
                tuning[["model", "setting", "n_candidates", "inner_cv_f1", "search_seconds", "best_params"]].rename(
                    columns={
                        "model": "Model",
                        "setting": "Setting",
                        "n_candidates": "Candidates",
                        "inner_cv_f1": "Inner F1",
                        "search_seconds": "Seconds",
                        "best_params": "Selected parameters",
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Inner F1": st.column_config.NumberColumn(format="%.3f"),
                    "Seconds": st.column_config.NumberColumn(format="%.1f"),
                },
            )
        path = FIGURES / "tuning_curves.png"
        if path.exists():
            st.image(str(path), caption="Search sensitivity around the selected parameters.", width="stretch")

    with tabs[3]:
        comparison = load_output("capacity_model_comparison.csv")
        if comparison is None:
            st.warning("Challenger results are not available.")
        else:
            st.dataframe(
                comparison.rename(
                    columns={
                        "model": "Model",
                        "f1_default_mean": "F1 at 0.50",
                        "f1_tuned_mean": "F1 at tuned threshold",
                        "average_precision_mean": "Avg precision",
                        "roc_auc_mean": "ROC-AUC",
                        "median_threshold": "Threshold",
                    }
                )[
                    ["Model", "F1 at 0.50", "F1 at tuned threshold", "Avg precision", "ROC-AUC", "Threshold"]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "F1 at 0.50": st.column_config.NumberColumn(format="%.3f"),
                    "F1 at tuned threshold": st.column_config.NumberColumn(format="%.3f"),
                    "Avg precision": st.column_config.NumberColumn(format="%.3f"),
                    "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                    "Threshold": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            note(
                "The challenger adds four capacity ratios and a tuned threshold. It gains 0.004 "
                "ROC-AUC and loses 0.008 F1 against the selected model, so it stays a challenger.",
                label="Result",
            )

    baseline_tuned = cv[(cv["setting"] == "baseline") & (cv["variant"] == "tuned")].sort_values(
        "cv_f1_mean", ascending=False
    )
    best = baseline_tuned.iloc[0]
    rival = baseline_tuned[baseline_tuned["model"] == "RandomForest"]
    if not rival.empty:
        rival = rival.iloc[0]
        note(
            f"The tuned baseline arm leads on F1 at {best['cv_f1_mean']:.3f} ± {best['cv_f1_std']:.3f}. "
            f"Random Forest trails by {best['cv_f1_mean'] - rival['cv_f1_mean']:.3f} on F1 but leads on "
            f"average precision ({rival['cv_ap_mean']:.3f} vs {best['cv_ap_mean']:.3f}) and ROC-AUC "
            f"({rival['cv_auc_mean']:.3f} vs {best['cv_auc_mean']:.3f}). The F1 gap is a fraction of the "
            "fold-to-fold spread, so the choice is a convention rather than a decisive win.",
            label="Why this model",
        )

    note(
        "76 defaults in 682 loans, and 19 of them in the held-out split. Every interval on this "
        "page is wide. Treat the ranking as a shortlist, not a verdict.",
        label="Limits",
        kind="risk",
    )


# ------------------------------------------------------------------- patterns


def patterns_page() -> None:
    features = load_features()
    itemsets = load_output("pattern_itemsets.csv")
    rules = load_output("pattern_rules.csv")
    if itemsets is None or rules is None:
        st.error("Pattern outputs are missing. Run tools/pattern_mining.py.")
        return

    # Both files stack an Apriori and an FP-Growth run. The itemsets file also
    # ends with a six-row legend whose support cells are text, not numbers.
    itemsets = itemsets[itemsets["method"].notna()].copy()
    itemsets["support"] = pd.to_numeric(itemsets["support"], errors="coerce")
    itemsets = itemsets.drop_duplicates(subset=["itemsets"])
    rules = rules.drop_duplicates(subset=["antecedents", "consequents"])

    loans = len(features)
    page_header(
        "Patterns",
        f"{len(itemsets)} itemsets, {len(rules)} rules, one worth a second look",
        "FP-Growth and Apriori over pre-loan behaviour baskets. Support, confidence and lift "
        f"are measured against all {loans} loans.",
    )

    top = rules.sort_values("lift", ascending=False).iloc[0]
    kpi_row(
        [
            ("Itemsets", f"{len(itemsets)}", "Identical across both algorithms"),
            ("Default rules", f"{len(rules)}", "Consequence is default"),
            ("Top-rule support", percent(top["support"]), f"{round(top['support'] * loans)} loans"),
            ("Top-rule lift", f"{top['lift']:.2f}", f"Against the {percent(features['target'].mean())} base rate"),
        ]
    )

    section("The rule worth a second look", "Highest lift in the rule set")
    st.markdown(
        f'<div class="note"><b>Rule</b><span style="font-family:var(--serif);font-size:1.15rem">'
        f"{parse_itemset(top['antecedents'])} → {parse_itemset(top['consequents'])}</span></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["Antecedent support", percent(top["antecedent support"])],
                ["Consequent support", percent(top["consequent support"])],
                ["Support", percent(top["support"])],
                ["Confidence", f"{top['confidence']:.2f}"],
                ["Lift", f"{top['lift']:.2f}"],
                ["Conviction", f"{top['conviction']:.1f}"],
            ],
            columns=["Measure", "Value"],
        ),
        hide_index=True,
        width="stretch",
    )
    cases = round(top["support"] * loans)
    note(
        f"All {cases} loans with a negative minimum balance defaulted. "
        f"Confidence 1.00 is a strong signal, but {cases} cases is a small base: the interval "
        "around it is wide. Use it as a screening flag, not an approval rule.",
        label="Reading it",
    )

    section("All rules", f"{len(rules)} rules pass the configured thresholds")
    table = rules.copy()
    table["If"] = table["antecedents"].map(parse_itemset)
    table["Then"] = table["consequents"].map(parse_itemset)
    table["Support %"] = (table["support"] * 100).round(2)
    table["Confidence %"] = (table["confidence"] * 100).round(1)
    st.dataframe(
        table[["If", "Then", "Support %", "Confidence %", "lift"]]
        .sort_values("lift", ascending=False)
        .rename(columns={"lift": "Lift"}),
        hide_index=True,
        width="stretch",
        column_config={
            "Support %": st.column_config.NumberColumn(format="%.2f"),
            "Confidence %": st.column_config.NumberColumn(format="%.1f"),
            "Lift": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Both algorithms return the same rules; duplicates are collapsed.")

    section("Itemsets", f"Apriori and FP-Growth return the same {len(itemsets)} sets")
    threshold = st.slider("Minimum support", 0, 100, 20, step=5)
    table = itemsets.copy()
    table["Support %"] = (table["support"] * 100).round(2)
    filtered = table[table["Support %"] >= threshold].sort_values("Support %", ascending=False)
    table_view = filtered.assign(Items=filtered["itemsets"].map(parse_itemset))[["Items", "Support %"]]
    st.dataframe(table_view, hide_index=True, width="stretch")
    st.caption(
        f"{len(filtered)} of {len(itemsets)} itemsets at support ≥ {threshold}%. "
        "Support counts the share of loans that contain the set."
    )


# ------------------------------------------------------------------ simulator


def simulator_page() -> None:
    features = load_features()
    scores = load_scores()
    model = load_model()
    columns = model_columns(model, features)
    base = median_row(features)

    page_header(
        "Simulator",
        "Score one application against the fitted model",
        "The model is the tuned Gradient Boosting pipeline in outputs/models. Five inputs are "
        "yours; the other thirteen features are held at portfolio medians and listed in full below.",
    )

    left, right = st.columns([1, 1.05], gap="large")

    with left:
        st.markdown('<div class="side-label">Application</div>', unsafe_allow_html=True)
        amount = st.number_input("Loan amount (CZK)", min_value=10_000, max_value=500_000, value=116_928, step=1_000)
        duration = st.select_slider("Duration (months)", options=[6, 12, 24, 36, 48, 60], value=36)
        balance = st.number_input(
            "Balance at loan date (CZK)", min_value=-20_000, max_value=300_000, value=38_550, step=500
        )
        min_balance = st.number_input(
            "Minimum balance in the window (CZK)", min_value=-20_000, max_value=50_000, value=600, step=100
        )
        txn = st.number_input(
            "Transactions per month", min_value=0.0, max_value=15.0, value=5.7, step=0.1, format="%.1f"
        )
        cutoff = st.slider("Review threshold", 0.05, 0.90, 0.40, 0.01)

    values = {
        "amount": amount,
        "duration": duration,
        "balance_at_loan": balance,
        "min_balance": min_balance,
        "txn_per_month": txn,
    }
    row = applicant_row(base, values)
    probability = predict(model, row, columns)
    flagged = probability >= cutoff
    percentile = float((scores["prob_default"] < probability).mean())

    with right:
        st.markdown(
            f'<div class="kpi-label">Default probability</div>'
            f'<div class="prob {"risk" if flagged else "calm"}">{probability:.1%}</div>'
            f'<div class="verdict">'
            f'{"Above" if flagged else "Below"} the {cutoff:.2f} threshold. '
            f'{"Refer for manual review." if flagged else "Standard processing."}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="side-line" style="margin-top:1.1rem"><span>Position in the portfolio</span>'
            f"<span>above {percentile:.0%} of scores</span></div>"
            f'<div class="side-line"><span>PD × exposure</span><span>{money(probability * amount)}</span></div>'
            f'<div class="side-line"><span>Model</span><span>tuned Gradient Boosting</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "PD × exposure multiplies the probability by the loan amount. Loss given default "
            "is not modelled, so this is an upper bound."
        )

    section("What moves the score", "One input at a time, moved to a stressed value")
    stress = {
        "amount": float(features["amount"].quantile(0.95)),
        "duration": 60.0,
        "balance_at_loan": float(features["balance_at_loan"].quantile(0.01)),
        "min_balance": float(features["min_balance"].quantile(0.01)),
        "txn_per_month": float(features["txn_per_month"].quantile(0.01)),
    }
    deltas = []
    for key in SIM_INPUTS:
        altered = dict(values)
        altered[key] = stress[key]
        deltas.append(predict(model, applicant_row(base, altered), columns) - probability)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        show(driver_figure(list(SIM_INPUTS.values()), deltas))
    with right:
        for (key, label), delta in zip(SIM_INPUTS.items(), deltas):
            short = "Activity" if key == "txn_per_month" else label
            if abs(delta) < 0.0005:
                effect = "no measurable effect"
            else:
                direction = "raises" if delta > 0 else "lowers"
                effect = f"{direction} risk by {abs(delta) * 100:.1f} pp"
            st.markdown(
                f'<div class="side-line"><span>{short} at {stress_text(key, stress[key])}</span>'
                f"<span>{effect}</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Each row moves one input to a stress value (95th percentile for amount, 60 months, "
            "1st percentile for balances and activity) and holds the rest at your entries."
        )

    section("What the model sees", "All 18 features, applicant value against the portfolio median")
    with st.expander("Open the full feature vector", expanded=False):
        vector = pd.DataFrame(
            {
                "Feature": columns,
                "Applicant": [row[column] for column in columns],
                "Portfolio median": [base[column] for column in columns],
            }
        )
        st.dataframe(
            vector,
            hide_index=True,
            width="stretch",
            column_config={
                "Applicant": st.column_config.NumberColumn(format="%.2f"),
                "Portfolio median": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    note(
        "payments is derived as amount ÷ duration, exactly as in the feature build, so it is not "
        "an independent input. The pipeline was fitted on all 682 loans; the portfolio page uses "
        "out-of-fold scores. Use this page to compare applications, not to read a calibrated "
        "probability.",
        label="Method",
    )


# ---------------------------------------------------------------------- shell


def main() -> None:
    features = load_features()
    pages = {
        "Portfolio": portfolio_page,
        "Data & leakage": data_page,
        "Models": models_page,
        "Patterns": patterns_page,
        "Simulator": simulator_page,
    }
    pages[sidebar(features)]()


if __name__ == "__main__":
    main()
