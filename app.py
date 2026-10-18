from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

st.set_page_config(
    page_title="Loan Default Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #202124; --blue: #3a86ff; --coral: #ff595e; --citrus: #ffca3a; --paper: #f7f7f2; }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #202124; border-right: 4px solid var(--coral); }
    [data-testid="stSidebar"] * { color: #f7f7f2; }
    [data-testid="stMetricValue"] { color: var(--coral); }
    [data-testid="stMetricLabel"] { color: #5c6268; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    .block-container { max-width: 1320px; padding-top: 2.2rem; }
    .hero { padding: 0.4rem 0 0.8rem; }
    .eyebrow { color: var(--blue); font-size: 0.75rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }
    .muted { color: #5c6268; font-size: 1rem; }
    .signal { background: #eef4ff; border-left: 4px solid var(--blue); padding: 0.85rem 1rem; margin: 0.5rem 0 1.2rem; }
    .sidebar-title { color: var(--citrus); font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.72rem; }
    .sidebar-note { color: #d6dde0; font-size: 0.82rem; line-height: 1.45; }
    [data-testid="stSidebar"] [role="radiogroup"] label p { color: #f7f7f2; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / name)


def load_optional_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUTS / name
    return pd.read_csv(path) if path.exists() else None


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def show_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1><p class='muted'>{description}</p></div>",
        unsafe_allow_html=True,
    )


def show_project_pulse() -> None:
    st.sidebar.markdown('<div class="sidebar-title">Project pulse</div>', unsafe_allow_html=True)
    st.sidebar.metric("Applications", "682")
    pulse_left, pulse_right = st.sidebar.columns(2)
    pulse_left.metric("Defaults", "76")
    pulse_right.metric("Base rate", "11.1%")
    st.sidebar.markdown(
        '<div class="sidebar-note">Historical PKDD\'99 banking data<br>Pre-loan behavioral evidence only</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def comparison_page() -> None:
    metrics = load_csv("metrics_comparison.csv")
    cv = load_csv("cv_comparison.csv")
    segments = load_csv("segment_profiles.csv")

    show_header(
        "01 / Model comparison",
        "Which model earns the review seat?",
        "A single held-out split is shown alongside repeated cross-validation, so model choice is not driven by one lucky test split.",
    )

    metric_options = {
        "F1 score": ("cv_f1_mean", "cv_f1_std"),
        "Average precision": ("cv_ap_mean", "cv_ap_std"),
        "ROC-AUC": ("cv_auc_mean", "cv_auc_std"),
    }
    controls = st.columns([1.2, 1, 1, 1.4])
    metric_label = controls[0].selectbox("Rank models by", list(metric_options))
    setting = controls[1].selectbox("Balance treatment", ["baseline", "smote"], format_func=str.title)
    variant = controls[2].selectbox("Model state", ["tuned", "default"], format_func=str.title)
    metric_column, spread_column = metric_options[metric_label]
    selected = cv[(cv["setting"] == setting) & (cv["variant"] == variant)].copy()
    best = selected.sort_values(metric_column, ascending=False).iloc[0]
    controls[3].caption("Selection protocol")
    controls[3].markdown("Repeated stratified 5-fold x 10 CV")
    cards = st.columns(4)
    cards[0].metric("Best model", best["model"])
    cards[1].metric(metric_label, f"{best[metric_column]:.3f} ± {best[spread_column]:.3f}")
    cards[2].metric("Average precision", format_metric(best["cv_ap_mean"]))
    cards[3].metric("ROC-AUC", format_metric(best["cv_auc_mean"]))
    st.markdown(
        f'<div class="signal"><b>Reading this view:</b> {best["model"]} leads the selected '
        f'{setting} / {variant} slice on {metric_label}. The error range shows how much the score moved across repeated folds.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader(f"Leaderboard by {metric_label}")
        chart_data = selected.sort_values(metric_column)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(chart_data["model"], chart_data[metric_column], color="#3a86ff")
        ax.set_xlabel(metric_label)
        ax.set_xlim(0, min(1, chart_data[metric_column].max() + 0.12))
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.subheader("F1 versus ranking quality")
        tuned_baseline = cv[(cv["setting"] == "baseline") & (cv["variant"] == "tuned")].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(tuned_baseline["cv_f1_mean"], tuned_baseline["cv_auc_mean"], color="#ff595e", s=85)
        for _, row in tuned_baseline.iterrows():
            ax.annotate(row["model"], (row["cv_f1_mean"], row["cv_auc_mean"]), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Tuned F1 mean")
        ax.set_ylabel("Tuned ROC-AUC mean")
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("All model runs")
    display_cv = selected.copy()
    display_cv["configuration"] = display_cv["setting"] + " / " + display_cv["variant"]
    st.dataframe(
        display_cv[["model", "configuration", "cv_f1_mean", "cv_f1_std", "cv_ap_mean", "cv_auc_mean"]]
        .sort_values("cv_f1_mean", ascending=False)
        .style.format({column: "{:.3f}" for column in ["cv_f1_mean", "cv_f1_std", "cv_ap_mean", "cv_auc_mean"]}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("See the held-out default-setting table"):
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Behavior segments")
    segment_cards = st.columns(len(segments))
    for card, (_, segment) in zip(segment_cards, segments.iterrows()):
        card.metric(f"Segment {int(segment['segment'])} default rate", f"{segment['default_rate']:.1%}")
        card.caption(f"{int(segment['size'])} loans | {segment['n_txn']:.0f} average transactions")


def best_model_page() -> None:
    cv = load_csv("cv_comparison.csv")
    metrics = load_csv("metrics_comparison.csv")
    tuned_baseline = cv[(cv["setting"] == "baseline") & (cv["variant"] == "tuned")].copy()
    best = tuned_baseline.sort_values("cv_f1_mean", ascending=False).iloc[0]
    model_metrics = metrics[metrics["model"] == best["model"]]

    show_header(
        "02 / Model rationale",
        f"Why {best['model']} is the headline model",
        "The selection is grounded in repeated validation, then checked against precision, ranking quality, imbalance handling, and business cost.",
    )

    st.info(
        f"The selected configuration is tuned {best['model']} with the unresampled baseline arm. "
        f"It leads on repeated-CV F1 at {best['cv_f1_mean']:.3f} ± {best['cv_f1_std']:.3f}."
    )

    rf = tuned_baseline[tuned_baseline["model"] == "RandomForest"].iloc[0]
    lead_gap = best["cv_f1_mean"] - rf["cv_f1_mean"]
    st.markdown(
        f'<div class="signal"><b>Decision boundary:</b> Gradient Boosting leads Random Forest by only '
        f'{lead_gap:.3f} F1 points, while Random Forest leads on Average Precision and ROC-AUC. '
        "Use Gradient Boosting as the headline classifier and keep Random Forest as the ranking challenger.</div>",
        unsafe_allow_html=True,
    )

    reasons = st.columns(3)
    reasons[0].metric("F1", f"{best['cv_f1_mean']:.3f} ± {best['cv_f1_std']:.3f}")
    reasons[1].metric("Average precision", f"{best['cv_ap_mean']:.3f}")
    reasons[2].metric("ROC-AUC", f"{best['cv_auc_mean']:.3f}")

    st.subheader("The decision in plain language")
    st.markdown(
        """
        - **Repeated validation:** Gradient Boosting is the top configuration on the stated F1 selection rule.
        - **No automatic SMOTE:** the unresampled arm is stronger for the leading ensemble; repeated validation is more reliable than the original single split.
        - **A close alternative:** Random Forest is a defensible ranking model because it leads on Average Precision and ROC-AUC. The margin is smaller than the fold-to-fold spread.
        - **Business signal:** low `min_balance` and `balance_at_loan` are the strongest observed default signals, so the model complements a balance-capacity review rather than replacing it.
        """
    )

    figure = FIGURES / "rf_importance.png"
    if figure.exists():
        st.image(str(figure), caption="Reference feature importance from the tuned Random Forest model.", use_container_width=True)

    st.subheader("Selected model on the held-out split")
    st.dataframe(model_metrics, use_container_width=True, hide_index=True)
    st.caption("The held-out table is a reference check. The model selection itself uses repeated cross-validation.")

    with st.expander("What would change the decision?"):
        st.markdown(
            "Use a recent portfolio, nested cross-validation, and an explicit cost matrix before deployment. "
            "If ranking quality becomes more important than F1 classification, Random Forest is the natural challenger."
        )


def parse_itemset(value: str) -> str:
    try:
        parsed = literal_eval(value)
        return ", ".join(sorted(str(item) for item in parsed))
    except (ValueError, SyntaxError):
        return value


def patterns_page() -> None:
    itemsets = load_optional_csv("pattern_itemsets.csv")
    rules = load_optional_csv("pattern_rules.csv")

    show_header(
        "03 / Pattern recognition",
        "What recurring behaviors travel together?",
        "FP-Growth and Apriori turn pre-loan account activity into interpretable baskets and default-oriented association rules.",
    )

    if itemsets is None or rules is None:
        st.warning("Pattern outputs have not been generated yet. The dashboard is ready, but the mining script still needs to run.")
        st.code("python tools/pattern_mining.py", language="powershell")
        st.markdown("Run the command from the repository root, then reload this page to populate the tables.")
        return

    cards = st.columns(3)
    cards[0].metric("FP-Growth itemsets", int((itemsets["method"] == "fpgrowth").sum()))
    cards[1].metric("Apriori itemsets", int((itemsets["method"] == "apriori").sum()))
    cards[2].metric("Default-oriented rules", len(rules))

    method_counts = itemsets.groupby("method").size().reset_index(name="count")
    st.subheader("Mining coverage")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(method_counts["method"].str.title(), method_counts["count"], color=["#3a86ff", "#ff595e"])
    ax.set_ylabel("Itemsets")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    itemsets = itemsets.copy()
    itemsets["items"] = itemsets["itemsets"].map(parse_itemset)
    st.subheader("Most frequent patterns")
    st.dataframe(
        itemsets[["method", "items", "support"]].sort_values("support", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )

    rules = rules.copy()
    rules["if_behavior"] = rules["antecedents"].map(parse_itemset)
    st.subheader("Rules whose consequence is default")
    if rules.empty:
        st.info("No rule reaches the configured confidence threshold. This is a finding, not a dashboard error.")
        st.caption("Try a lower confidence threshold in tools/pattern_mining.py only as a sensitivity analysis; do not compare thresholds as if they were the same experiment.")
    else:
        st.dataframe(
            rules[["method", "if_behavior", "support", "confidence", "lift"]].sort_values("lift", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


page = st.sidebar.radio(
    "Explore",
    ["Model comparison", "Why the best model", "FP-Growth & patterns"],
)
show_project_pulse()
st.sidebar.divider()
st.sidebar.caption("PKDD'99 loan default classification\nPre-loan behavioral evidence")

if page == "Model comparison":
    comparison_page()
elif page == "Why the best model":
    best_model_page()
else:
    patterns_page()