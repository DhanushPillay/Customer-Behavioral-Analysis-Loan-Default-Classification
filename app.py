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


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "loan_features.csv")


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
        "05 / Model comparison",
        "Six models. One practical shortlist.",
        "Compare every classifier visually, then choose the model that best fits the lending workflow.",
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

    st.subheader("All models at a glance")
    board = selected.sort_values(metric_column, ascending=False).reset_index(drop=True)
    model_columns = st.columns(min(3, len(board)))
    for index, (_, row) in enumerate(board.iterrows()):
        column = model_columns[index % len(model_columns)]
        column.metric(row["model"], f"{row[metric_column]:.3f}", f"F1 {row['cv_f1_mean']:.3f} | AUC {row['cv_auc_mean']:.3f}")

    st.subheader("Metric heatmap")
    heatmap_columns = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
    held_out = metrics[metrics["setting"] == setting].copy()
    available_columns = [column for column in heatmap_columns if column in held_out.columns]
    if available_columns:
        heatmap = held_out.set_index("model")[available_columns].sort_values("f1", ascending=False)
        labels = [column.replace("_", " ").title() for column in available_columns]
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(heatmap) * 0.7)))
        image = ax.imshow(heatmap.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_yticks(range(len(heatmap.index)), heatmap.index)
        for row_index in range(len(heatmap.index)):
            for column_index in range(len(available_columns)):
                ax.text(column_index, row_index, f"{heatmap.iloc[row_index, column_index]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, label="Score")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

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

    st.subheader("Portfolio risk lens")
    st.caption("Adjust the review cutoffs to see how balance capacity and loan size divide the portfolio. This is a screening view, not a lending rule.")
    features = load_features()
    balance_median = round(float(features["balance_at_loan"].median()) / 1000) * 1000.0
    amount_median = round(float(features["amount"].median()) / 1000) * 1000.0
    balance_min = round(float(features["balance_at_loan"].quantile(0.10)) / 1000) * 1000.0
    balance_max = round(float(features["balance_at_loan"].quantile(0.90)) / 1000) * 1000.0
    amount_min = round(float(features["amount"].quantile(0.10)) / 1000) * 1000.0
    amount_max = round(float(features["amount"].quantile(0.90)) / 1000) * 1000.0
    risk_controls = st.columns(2)
    balance_cutoff = risk_controls[0].slider(
        "Balance-at-loan cutoff",
        min_value=balance_min,
        max_value=balance_max,
        value=balance_median,
        step=1000.0,
    )
    amount_cutoff = risk_controls[1].slider(
        "Loan amount cutoff",
        min_value=amount_min,
        max_value=amount_max,
        value=amount_median,
        step=1000.0,
    )
    features = features.copy()
    features["risk_quadrant"] = (
        features["balance_at_loan"].ge(balance_cutoff).map({True: "Higher balance", False: "Lower balance"})
        + " / "
        + features["amount"].ge(amount_cutoff).map({True: "Higher loan", False: "Lower loan"})
    )
    risk_table = (
        features.groupby("risk_quadrant", as_index=False)
        .agg(loans=("target", "size"), defaults=("target", "sum"), default_rate=("target", "mean"))
        .sort_values("default_rate", ascending=False)
    )
    risk_left, risk_right = st.columns([1.1, 1])
    with risk_left:
        st.dataframe(
            risk_table.style.format({"default_rate": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
    with risk_right:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#ff595e" if rate == risk_table["default_rate"].max() else "#3a86ff" for rate in risk_table["default_rate"]]
        ax.barh(risk_table["risk_quadrant"], risk_table["default_rate"], color=colors)
        ax.set_xlabel("Default rate")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    highest_risk = risk_table.iloc[0]
    st.markdown(
        f'<div class="signal"><b>Highest observed quadrant:</b> {highest_risk["risk_quadrant"]} '
        f'with {highest_risk["default_rate"]:.1%} default rate across {int(highest_risk["loans"])} loans. '
        "Use this to prioritize investigation, not to automate approval decisions.</div>",
        unsafe_allow_html=True,
    )


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


def overview_page() -> None:
    features = load_features()
    defaults = int(features["target"].sum())
    good = len(features) - defaults
    show_header(
        "Loan default intelligence",
        "From transaction history to a safer lending conversation",
        "An end-to-end decision-support dashboard for understanding default risk before a loan is approved.",
    )
    cards = st.columns(4)
    cards[0].metric("Loans analysed", f"{len(features):,}")
    cards[1].metric("Default rate", f"{features['target'].mean():.1%}")
    cards[2].metric("Good loans", f"{good:,}")
    cards[3].metric("Default cases", f"{defaults:,}")
    st.markdown(
        '<div class="signal"><b>Business question:</b> Can pre-loan transaction behaviour help a bank identify applications that deserve additional review before approval, without replacing human judgement?</div>',
        unsafe_allow_html=True,
    )
    st.subheader("The project in one view")
    flow = st.columns(5)
    stages = [("01", "Collect", "PKDD'99 banking tables"), ("02", "Prepare", "Clean, join and label loans"), ("03", "Engineer", "Pre-loan behaviour features"), ("04", "Learn", "Compare six classifiers"), ("05", "Act", "Review risk and capacity")]
    for column, (number, title, detail) in zip(flow, stages):
        column.markdown(f"**{number} / {title}**\n\n{detail}")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("What the analysis found")
        st.markdown("- Defaults represent **11.14%** of the modelling population, so accuracy alone is not enough.\n- **Minimum balance** and **balance at loan time** are stronger signals than activity volume alone.\n- Tuned Gradient Boosting leads the repeated-CV F1 selection rule; Random Forest remains a strong ranking challenger.\n- Pattern mining found recurring behaviour baskets, but no default rule met the configured confidence threshold.")
    with right:
        st.subheader("Who can use this dashboard?")
        st.markdown("**Credit analyst:** prioritise manual review.  \n**Risk manager:** test portfolio cut-offs and expected exposure.  \n**Data team:** defend model choice and validation.  \n**Viva panel:** walk from source data to recommendation in one screen.")
    st.subheader("Important boundary")
    st.warning("This is a historical decision-support study, not an automated approval system. Revalidate on current portfolio data, monitor drift, and apply fair-lending and affordability checks before operational use.")


def data_pipeline_page() -> None:
    features = load_features()
    show_header("02 / Data story", "How raw banking records become loan features", "The feature table contains only information available before the loan date, which is the key protection against post-decision leakage.")
    source_table = pd.DataFrame([
        ["loan.csv", "Target and loan terms", "Amount, duration, payments, status"],
        ["trans.csv", "Behavioural source", "Deposits, withdrawals, balances and dates"],
        ["client.csv", "Customer attributes", "Client identity and district"],
        ["account.csv", "Account metadata", "Account creation and district"],
        ["disp.csv", "Relationship bridge", "Links owners to accounts"],
    ], columns=["Source", "Role", "Fields used"])
    st.subheader("Dataset and source")
    st.caption("PKDD'99 Financial Dataset | anonymised Czech banking records covering 1993-1998")
    st.dataframe(source_table, use_container_width=True, hide_index=True)
    st.subheader("Preprocessing pipeline")
    pipeline = st.columns(4)
    for column, title, detail in zip(pipeline, ["Clean", "Join", "Restrict", "Transform"], ["Parse dates, check duplicates and missing values", "Link loan, account and client ownership", "Keep transactions strictly before each loan date", "Encode gender and scale features inside training pipelines"]):
        column.info(f"**{title}**\n\n{detail}")
    quality = pd.DataFrame({"Check": ["Rows in modelling table", "Columns", "Missing values", "Target classes", "Default prevalence"], "Result": [f"{len(features):,}", f"{features.shape[1]}", f"{int(features.isna().sum().sum())}", "Good / Default", f"{features['target'].mean():.2%}"], "Why it matters": ["One row per loan", "Behaviour plus terms and context", "Ready for modelling", "Binary classification target", "Accuracy must be interpreted carefully"]})
    st.subheader("Data-quality checkpoint")
    st.dataframe(quality, use_container_width=True, hide_index=True)


def eda_page() -> None:
    features = load_features()
    show_header("03 / Exploratory analysis", "Where does default risk show up?", "Interactive views of portfolio balance, loan terms, and pre-loan customer behaviour.")
    target_labels = features["target"].map({0: "Good", 1: "Default"})
    left, right = st.columns(2)
    with left:
        st.subheader("Portfolio composition")
        counts = target_labels.value_counts().reindex(["Good", "Default"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
        ax.set_ylabel("Loans")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.subheader("Balance capacity by outcome")
        groups = [features.loc[features["target"] == value, "balance_at_loan"] for value in [0, 1]]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.boxplot(groups, labels=["Good", "Default"], patch_artist=True, boxprops={"facecolor": "#bde0fe"})
        ax.set_ylabel("Balance at loan")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.subheader("Default rate by loan duration")
    duration_summary = features.groupby("duration", as_index=False).agg(loans=("target", "size"), default_rate=("target", "mean"))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(duration_summary["duration"].astype(str), duration_summary["default_rate"], color="#457b9d")
    ax.set_xlabel("Duration (months)")
    ax.set_ylabel("Default rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(duration_summary.style.format({"default_rate": "{:.1%}"}), use_container_width=True, hide_index=True)


def simulator_page() -> None:
    features = load_features()
    show_header("04 / Loan review simulator", "Test a lending policy before it reaches the portfolio", "Change the application and capacity assumptions to see how a review threshold changes exposure. This is a transparent scenario tool, not an approval engine.")
    st.info("Innovation: use this screen for pre-screening, branch coaching, or policy what-if analysis. It converts a model-style risk signal into an operational review queue with a visible rationale.")
    left, right = st.columns(2)
    with left:
        amount = st.number_input("Requested loan amount", min_value=1000.0, value=float(features["amount"].median()), step=5000.0)
        duration = st.select_slider("Loan duration (months)", options=sorted(features["duration"].unique().tolist()), value=int(features["duration"].median()))
        balance = st.number_input("Balance available at application", min_value=0.0, value=float(features["balance_at_loan"].median()), step=1000.0)
    with right:
        min_balance = st.number_input("Minimum observed balance", min_value=0.0, value=float(features["min_balance"].median()), step=500.0)
        txn_rate = st.number_input("Transactions per month", min_value=0.0, value=float(features["txn_per_month"].median()), step=0.5)
        review_cutoff = st.slider("Manual-review cutoff", 0.10, 0.90, 0.45, 0.05)
    amount_pressure = amount / max(float(features["amount"].median()), 1.0)
    balance_strength = balance / max(float(features["balance_at_loan"].median()), 1.0)
    min_balance_strength = min_balance / max(float(features["min_balance"].median()), 1.0)
    duration_pressure = duration / max(float(features["duration"].median()), 1.0)
    activity_support = txn_rate / max(float(features["txn_per_month"].median()), 1.0)
    risk_score = 0.11 + 0.10 * (amount_pressure - 1) + 0.14 * (duration_pressure - 1) - 0.10 * (balance_strength - 1) - 0.08 * (min_balance_strength - 1) - 0.03 * (activity_support - 1)
    risk_score = max(0.01, min(0.95, risk_score))
    decision = "Manual review" if risk_score >= review_cutoff else "Standard review"
    expected_loss = amount * risk_score
    st.subheader("Scenario result")
    cards = st.columns(4)
    cards[0].metric("Estimated risk index", f"{risk_score:.1%}")
    cards[1].metric("Workflow", decision)
    cards[2].metric("Exposure if defaulted", f"{amount:,.0f}")
    cards[3].metric("Expected loss proxy", f"{expected_loss:,.0f}")
    st.progress(risk_score, text=f"Risk index: {risk_score:.1%}")
    rationale = []
    if balance < features["balance_at_loan"].median():
        rationale.append("available balance is below the portfolio median")
    if min_balance < features["min_balance"].median():
        rationale.append("the minimum observed balance is below the portfolio median")
    if amount > features["amount"].median():
        rationale.append("the requested amount is above the portfolio median")
    if not rationale:
        rationale.append("capacity and loan terms are close to or better than portfolio medians")
    st.markdown(f'<div class="signal"><b>Why this result:</b> {"; ".join(rationale)}. Use the result to request documents or affordability evidence, not to auto-reject the applicant.</div>', unsafe_allow_html=True)


def recommendations_page() -> None:
    show_header("05 / Recommendations", "Turn the analysis into a responsible next step", "The dashboard closes the loop between technical findings, credit operations, and future model governance.")
    recommendations = pd.DataFrame([
        ["Credit review", "Prioritise low-balance and high-loan-to-balance applications for manual affordability checks.", "Lower avoidable default exposure"],
        ["Portfolio policy", "Use the simulator to compare review cut-offs before changing approval policy.", "Quantify operational workload and exposure"],
        ["Model governance", "Revalidate with recent data and monitor recall, precision, calibration and drift.", "Prevent historical patterns becoming stale rules"],
        ["Fair lending", "Do not use age or gender as an automatic rejection reason; investigate explainable financial capacity signals.", "Keep decisions reviewable and defensible"],
        ["Further innovation", "Add affordability, bureau, repayment and macroeconomic data when legally and operationally available.", "Move from retrospective classification to live early warning"],
    ], columns=["Audience", "Recommendation", "Future value"])
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.subheader("What can be added next for loan teams?")
    st.markdown("- **Early-warning monitoring:** score existing borrowers monthly and flag deteriorating balances before missed payments.\n- **Offer optimisation:** compare amount, duration and payment scenarios instead of treating risk as a simple yes/no label.\n- **Collections prioritisation:** combine predicted risk with exposure to focus limited outreach capacity.\n- **Decision audit trail:** store the input snapshot, score, threshold, reason codes and human outcome for every reviewed application.")


page = st.sidebar.radio(
    "Explore",
    [
        "Loan review simulator",
        "Model comparison",
        "Project overview",
        "Data & pipeline",
        "EDA & portfolio signals",
        "Recommendations",
    ],
)
show_project_pulse()
st.sidebar.divider()
st.sidebar.caption("PKDD'99 loan default classification\nPre-loan behavioral evidence")

if page == "Loan review simulator":
    simulator_page()
elif page == "Model comparison":
    comparison_page()
elif page == "Project overview":
    overview_page()
elif page == "Data & pipeline":
    data_pipeline_page()
elif page == "EDA & portfolio signals":
    eda_page()
else:
    recommendations_page()