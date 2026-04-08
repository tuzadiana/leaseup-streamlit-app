"""Professional Streamlit app for Vertical Bridge lease-up prediction."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vertical_bridge_leaseup import config as cfg
from vertical_bridge_leaseup.modeling import load_best_models, score_portfolio


st.set_page_config(
    page_title="Vertical Bridge Lease-Up Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: {cfg.BRAND_FONT};
        color: {cfg.BRAND_NEUTRAL};
    }}

    .stApp {{
        background-color: white;
    }}

    h1, h2, h3, h4 {{
        color: {cfg.BRAND_PRIMARY};
        font-family: {cfg.BRAND_FONT};
    }}

    section[data-testid="stSidebar"] {{
        background-color: white;
        border-right: 1px solid {cfg.BRAND_ACCENT};
    }}

    div[data-testid="stMetricValue"] {{
        color: {cfg.BRAND_PRIMARY};
        font-family: {cfg.BRAND_FONT};
        font-weight: 700;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {cfg.BRAND_NEUTRAL};
        font-family: {cfg.BRAND_FONT};
    }}

    .vb-hero {{
        border: 2px solid {cfg.BRAND_ACCENT};
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        background: white;
    }}

    .vb-hero-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {cfg.BRAND_PRIMARY};
        margin-bottom: 0.25rem;
    }}

    .vb-hero-subtitle {{
        font-size: 1rem;
        color: {cfg.BRAND_NEUTRAL};
        line-height: 1.5;
    }}

    .vb-card {{
        border: 1px solid {cfg.BRAND_ACCENT};
        border-radius: 16px;
        padding: 1rem;
        background: white;
        margin-bottom: 0.75rem;
    }}

    .vb-card-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {cfg.BRAND_NEUTRAL};
        margin-bottom: 0.35rem;
    }}

    .vb-card-value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {cfg.BRAND_PRIMARY};
        line-height: 1.1;
    }}

    .vb-chip {{
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        border: 1px solid {cfg.BRAND_ACCENT};
        color: {cfg.BRAND_PRIMARY};
        background: white;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }}

    .vb-section {{
        border-left: 5px solid {cfg.BRAND_PRIMARY};
        padding-left: 0.85rem;
        margin: 0.6rem 0 0.8rem 0;
    }}

    .stButton > button {{
        background-color: {cfg.BRAND_PRIMARY};
        color: white;
        border: 1px solid {cfg.BRAND_PRIMARY};
        border-radius: 10px;
        font-weight: 700;
    }}

    .stDownloadButton > button {{
        background-color: {cfg.BRAND_ACCENT};
        color: white;
        border: 1px solid {cfg.BRAND_ACCENT};
        border-radius: 10px;
        font-weight: 700;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
        font-weight: 700;
        color: {cfg.BRAND_NEUTRAL};
    }}

    .stTabs [aria-selected="true"] {{
        color: {cfg.BRAND_PRIMARY};
        border-bottom: 3px solid {cfg.BRAND_ACCENT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_card(title: str, value: str, subtitle: str = "") -> None:
    """Render a branded KPI card."""
    html = f"""
    <div class="vb-card">
        <div class="vb-card-title">{title}</div>
        <div class="vb-card-value">{value}</div>
        <div style="font-size:0.88rem;color:{cfg.BRAND_NEUTRAL};margin-top:0.35rem;">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def style_fig(fig, title: str):
    """Apply common styling to Plotly charts."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family=cfg.BRAND_FONT, color=cfg.BRAND_NEUTRAL),
        title_font=dict(family=cfg.BRAND_FONT, color=cfg.BRAND_PRIMARY, size=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=60, b=30),
        legend_title_text="",
    )
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def opportunity_band(probability: float, score: float, big3_count: int) -> str:
    """Assign a business-friendly opportunity band."""
    if big3_count >= 3:
        return "Saturated"
    if probability >= 0.70 or score >= 70:
        return "Very High"
    if probability >= 0.50 or score >= 55:
        return "High"
    if probability >= 0.30 or score >= 40:
        return "Medium"
    return "Low"


def recommended_action(probability: float, months: float | None, big3_count: int) -> str:
    """Generate a business-friendly recommendation."""
    if big3_count >= 3:
        return "No immediate Big 3 lease-up opportunity remains on this site. Keep it in portfolio monitoring only."

    if probability >= 0.70 and months is not None and months <= 12:
        return "Prioritize immediately for VI Focus, outbound carrier conversations, and near-term market action."

    if probability >= 0.50 and months is not None and months <= 18:
        return "High-potential site. Keep in the active pipeline and validate carrier need and market conditions."

    if probability >= 0.30:
        return "Moderate opportunity. Keep on a watchlist and review alongside related market and cluster opportunities."

    return "Lower near-term opportunity. Retain for long-term market monitoring rather than immediate sales prioritization."


def build_single_input_df(
    site_no: str,
    site_name: str,
    city: str,
    state: str,
    market: str,
    viable_site_type: str,
    site_status: str,
    site_type: str,
    fiber: str,
    urban_non_urban: str,
    structure_height: float,
    latitude: float,
    longitude: float,
    population_2_mile: float,
    population_per_sq_mile: float,
    aadt_2_mile: float,
    vmt_2_mile: float,
    active_residential_2_mile: float,
    att_nearest: float,
    tmo_nearest: float,
    vzw_nearest: float,
    att_dbm: float | None,
    tmo_dbm: float | None,
    vzw_dbm: float | None,
    big3_count: int,
) -> pd.DataFrame:
    """Create a one-row DataFrame for live prediction."""
    return pd.DataFrame(
        [
            {
                "Site No": site_no,
                "Site Name": site_name,
                "City": city,
                "State": state,
                "Portfolio Market": market,
                "Viable Site Type?": viable_site_type,
                "Site Status": site_status,
                "Site Type": site_type,
                "Fiber": fiber.strip() if fiber.strip() else None,
                "Urban/Non-Urban": urban_non_urban,
                "Structure Height (feet)": structure_height,
                "Latitude": latitude,
                "Longitude": longitude,
                "2 Mile Population": population_2_mile,
                "Population Per Sq. Mile": population_per_sq_mile,
                "AADT 2 Mile": aadt_2_mile,
                "VMT 2 Mile": vmt_2_mile,
                "Active Residential 2 Mile": active_residential_2_mile,
                "AT&T Nearest Site (miles)": att_nearest,
                "TMO Nearest Site (miles)": tmo_nearest,
                "VZW Nearest Site (miles)": vzw_nearest,
                "AT&T dbm Avg": att_dbm,
                "TMO dbm Avg": tmo_dbm,
                "VZW dbm Avg": vzw_dbm,
                cfg.BIG3_COUNT: big3_count,
            }
        ]
    )


def build_batch_template() -> pd.DataFrame:
    """Create a sample batch template."""
    return pd.DataFrame(
        [
            {
                "Site No": "SAMPLE-001",
                "Site Name": "Sample Tower",
                "City": "Dallas",
                "State": "TX",
                "Portfolio Market": "Dallas",
                "Viable Site Type?": "Yes",
                "Site Status": "Existing",
                "Site Type": "Monopole",
                "Fiber": "",
                "Urban/Non-Urban": "Urban",
                "Structure Height (feet)": 150,
                "Latitude": 32.7767,
                "Longitude": -96.7970,
                "2 Mile Population": 5000,
                "Population Per Sq. Mile": 1200,
                "AADT 2 Mile": 15000,
                "VMT 2 Mile": 200000,
                "Active Residential 2 Mile": 100,
                "AT&T Nearest Site (miles)": 1.5,
                "TMO Nearest Site (miles)": 1.3,
                "VZW Nearest Site (miles)": 1.7,
                "AT&T dbm Avg": -103,
                "TMO dbm Avg": -105,
                "VZW dbm Avg": -101,
                cfg.BIG3_COUNT: 0,
            }
        ]
    )


def required_batch_columns() -> list[str]:
    """Return required batch-scoring columns."""
    return cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES


def probability_gauge(probability: float) -> go.Figure:
    """Create a probability gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"family": cfg.BRAND_FONT, "color": cfg.BRAND_PRIMARY}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": cfg.BRAND_PRIMARY},
                "bgcolor": "white",
                "bordercolor": cfg.BRAND_ACCENT,
                "steps": [
                    {"range": [0, 35], "color": "white"},
                    {"range": [35, 65], "color": cfg.BRAND_NEUTRAL},
                    {"range": [65, 100], "color": cfg.BRAND_ACCENT},
                ],
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font=dict(family=cfg.BRAND_FONT, color=cfg.BRAND_NEUTRAL),
    )
    return fig


def score_gauge(score: float) -> go.Figure:
    """Create an overall score gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": cfg.BRAND_ACCENT},
                "bgcolor": "white",
                "bordercolor": cfg.BRAND_PRIMARY,
                "steps": [
                    {"range": [0, 35], "color": "white"},
                    {"range": [35, 65], "color": cfg.BRAND_NEUTRAL},
                    {"range": [65, 100], "color": cfg.BRAND_PRIMARY},
                ],
            },
            number={"font": {"family": cfg.BRAND_FONT, "color": cfg.BRAND_PRIMARY}},
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font=dict(family=cfg.BRAND_FONT, color=cfg.BRAND_NEUTRAL),
    )
    return fig


@st.cache_resource
def get_models():
    """Load trained models once."""
    return load_best_models()


@st.cache_data
def load_csv_if_exists(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists; otherwise return an empty DataFrame."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


try:
    classifier, regressor, metadata = get_models()
except Exception as exc:
    st.error(
        "Trained models were not found. Run the training pipeline first.\n\n"
        f"Details: {exc}"
    )
    st.stop()


scored_portfolio = load_csv_if_exists(cfg.PREDICTIONS_DIR / "scored_portfolio.csv")
vi_focus_df = load_csv_if_exists(cfg.PREDICTIONS_DIR / "vi_focus_sites.csv")
market_rankings_df = load_csv_if_exists(cfg.PREDICTIONS_DIR / "market_rankings.csv")
classification_metrics_df = load_csv_if_exists(cfg.METRICS_DIR / "classification_metrics.csv")
timing_metrics_df = load_csv_if_exists(cfg.METRICS_DIR / "timing_metrics.csv")
rf_importance_df = load_csv_if_exists(cfg.METRICS_DIR / "random_forest_classifier_feature_importance.csv")
xgb_importance_df = load_csv_if_exists(cfg.METRICS_DIR / "xgboost_classifier_feature_importance.csv")


logo_col, title_col = st.columns([1, 3])

with logo_col:
    logo_path = Path("assets/vertical_bridge_logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

with title_col:
    st.markdown(
        """
        <div class="vb-hero">
            <div class="vb-hero-title">📡 Lease-Up Command Center</div>
            <div class="vb-hero-subtitle">
                Welcome to Vertical Bridge’s branded prediction workspace for identifying near-term lease-up opportunities,
                estimating time-to-lease-up, and surfacing the most actionable towers and markets across the portfolio.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.sidebar.header("Workspace")
st.sidebar.markdown(
    "Use the tabs to score a single tower, upload a batch file, or explore portfolio opportunity."
)

model_status = "Ready"
portfolio_status = "Available" if not scored_portfolio.empty else "Not loaded"

st.sidebar.markdown(f'<div class="vb-chip">Model Status: {model_status}</div>', unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="vb-chip">Portfolio Dashboard: {portfolio_status}</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Business framing")
st.sidebar.caption(
    "Classification estimates whether a tower will lease up within 3 years. "
    "Regression estimates time-to-lease-up in months. The app blends both into an overall lease-up score."
)


tab_overview, tab_single, tab_batch, tab_vifocus, tab_intel = st.tabs(
    [
        "Executive Overview",
        "Single Tower Prediction",
        "Batch Scoring",
        "VI Focus & Markets",
        "Model Intelligence",
    ]
)


with tab_overview:
    st.markdown('<div class="vb-section"><h3>Portfolio Overview</h3></div>', unsafe_allow_html=True)

    if scored_portfolio.empty:
        st.info("Run the training pipeline first to populate the portfolio dashboard.")
    else:
        avg_prob = scored_portfolio["3-Year Lease-Up Probability"].mean() * 100
        avg_score = scored_portfolio["Overall Lease-Up Score"].mean()
        high_priority = int((scored_portfolio["Overall Lease-Up Score"] >= 60).sum())
        unsaturated = int((scored_portfolio[cfg.BIG3_COUNT] < 3).sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_card("Sites Scored", f"{len(scored_portfolio):,}", "Current filtered portfolio")
        with c2:
            render_card("Average Probability", f"{avg_prob:.1f}%", "Mean 3-year lease-up likelihood")
        with c3:
            render_card("Average Lease-Up Score", f"{avg_score:.1f}", "Portfolio-wide opportunity score")
        with c4:
            render_card("Unsaturated Sites", f"{unsaturated:,}", "Sites with available Big 3 headroom")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_prob = px.histogram(
                scored_portfolio,
                x="3-Year Lease-Up Probability",
                nbins=30,
                color_discrete_sequence=[cfg.BRAND_PRIMARY],
            )
            st.plotly_chart(style_fig(fig_prob, "Distribution of 3-Year Lease-Up Probability"), use_container_width=True)

        with chart_col2:
            band_df = scored_portfolio.copy()
            band_df["Opportunity Band"] = band_df.apply(
                lambda row: opportunity_band(
                    row["3-Year Lease-Up Probability"],
                    row["Overall Lease-Up Score"],
                    int(row[cfg.BIG3_COUNT]),
                ),
                axis=1,
            )
            band_counts = (
                band_df["Opportunity Band"]
                .value_counts()
                .rename_axis("Opportunity Band")
                .reset_index(name="Site Count")
            )
            fig_bands = px.bar(
                band_counts,
                x="Opportunity Band",
                y="Site Count",
                color="Opportunity Band",
                color_discrete_sequence=[cfg.BRAND_PRIMARY, cfg.BRAND_ACCENT, cfg.BRAND_NEUTRAL],
            )
            st.plotly_chart(style_fig(fig_bands, "Portfolio Opportunity Bands"), use_container_width=True)

        if not market_rankings_df.empty:
            fig_markets = px.bar(
                market_rankings_df.head(15),
                x="Portfolio Market",
                y="avg_score",
                hover_data=["site_count", "avg_probability"],
                color_discrete_sequence=[cfg.BRAND_ACCENT],
            )
            st.plotly_chart(style_fig(fig_markets, "Top Markets by Average Lease-Up Score"), use_container_width=True)

        preview_cols = [
            col
            for col in [
                "VI Focus Rank",
                "Site No",
                "Site Name",
                "Portfolio Market",
                "City",
                "State",
                "Site Type",
                "3-Year Lease-Up Probability",
                "Predicted Lease-Up Months",
                "Overall Lease-Up Score",
            ]
            if col in scored_portfolio.columns
        ]
        st.dataframe(scored_portfolio[preview_cols].head(100), use_container_width=True)


with tab_single:
    st.markdown('<div class="vb-section"><h3>Single Tower Prediction</h3></div>', unsafe_allow_html=True)
    st.caption("Enter a tower’s characteristics to estimate lease-up likelihood, predicted timing, and recommended action.")

    left, mid, right = st.columns(3)

    with left:
        st.markdown("#### Site Context")
        site_no = st.text_input("Site No", value="TEST-001")
        site_name = st.text_input("Site Name", value="Example Tower")
        city = st.text_input("City", value="Dallas")
        state = st.text_input("State", value="TX")
        market = st.text_input("Portfolio Market", value="Dallas")
        viable_site_type = st.selectbox("Viable Site Type?", ["Yes", "No", "Unknown"], index=0)
        site_status = st.selectbox(
            "Site Status",
            ["Existing", "Built", "Not Zoned", "Permitting/Construction", "Pending Anchor", "Unknown"],
            index=0,
        )
        site_type = st.selectbox("Site Type", ["SST", "Guyed Tower", "Monopole", "Stealth"], index=2)
        urban_non_urban = st.selectbox("Urban/Non-Urban", ["Urban", "Non-Urban", "Unknown"], index=0)
        fiber = st.text_input("Fiber", value="")
        big3_count = st.number_input("Current # of Big 3 Tenants", min_value=0, max_value=3, value=0, step=1)

    with mid:
        st.markdown("#### Structure, Location, and Demographics")
        structure_height = st.number_input("Structure Height (feet)", min_value=0.0, value=150.0, step=1.0)
        latitude = st.number_input("Latitude", value=32.776700, format="%.6f")
        longitude = st.number_input("Longitude", value=-96.797000, format="%.6f")
        population_2_mile = st.number_input("2 Mile Population", min_value=0.0, value=5000.0, step=100.0)
        population_per_sq_mile = st.number_input("Population Per Sq. Mile", min_value=0.0, value=1200.0, step=50.0)
        aadt_2_mile = st.number_input("AADT 2 Mile", min_value=0.0, value=15000.0, step=100.0)
        vmt_2_mile = st.number_input("VMT 2 Mile", min_value=0.0, value=200000.0, step=1000.0)
        active_residential_2_mile = st.number_input("Active Residential 2 Mile", min_value=0.0, value=100.0, step=10.0)

    with right:
        st.markdown("#### Carrier Proximity and RF Inputs")
        att_nearest = st.number_input("AT&T Nearest Site (miles)", min_value=0.0, value=1.5, step=0.1)
        tmo_nearest = st.number_input("TMO Nearest Site (miles)", min_value=0.0, value=1.3, step=0.1)
        vzw_nearest = st.number_input("VZW Nearest Site (miles)", min_value=0.0, value=1.7, step=0.1)

        att_dbm_missing = st.checkbox("AT&T dbm Avg missing", value=False)
        att_dbm = None if att_dbm_missing else st.number_input("AT&T dbm Avg", value=-103.0, step=1.0)

        tmo_dbm_missing = st.checkbox("TMO dbm Avg missing", value=False)
        tmo_dbm = None if tmo_dbm_missing else st.number_input("TMO dbm Avg", value=-105.0, step=1.0)

        vzw_dbm_missing = st.checkbox("VZW dbm Avg missing", value=False)
        vzw_dbm = None if vzw_dbm_missing else st.number_input("VZW dbm Avg", value=-101.0, step=1.0)

    if st.button("Generate Prediction"):
        single_df = build_single_input_df(
            site_no=site_no,
            site_name=site_name,
            city=city,
            state=state,
            market=market,
            viable_site_type=viable_site_type,
            site_status=site_status,
            site_type=site_type,
            fiber=fiber,
            urban_non_urban=urban_non_urban,
            structure_height=structure_height,
            latitude=latitude,
            longitude=longitude,
            population_2_mile=population_2_mile,
            population_per_sq_mile=population_per_sq_mile,
            aadt_2_mile=aadt_2_mile,
            vmt_2_mile=vmt_2_mile,
            active_residential_2_mile=active_residential_2_mile,
            att_nearest=att_nearest,
            tmo_nearest=tmo_nearest,
            vzw_nearest=vzw_nearest,
            att_dbm=att_dbm,
            tmo_dbm=tmo_dbm,
            vzw_dbm=vzw_dbm,
            big3_count=int(big3_count),
        )

        prediction_df = score_portfolio(single_df, classifier, regressor)
        row = prediction_df.iloc[0]

        probability = float(row["3-Year Lease-Up Probability"])
        months = None if pd.isna(row["Predicted Lease-Up Months"]) else float(row["Predicted Lease-Up Months"])
        score = float(row["Overall Lease-Up Score"])
        band = opportunity_band(probability, score, int(row[cfg.BIG3_COUNT]))
        action = recommended_action(probability, months, int(row[cfg.BIG3_COUNT]))

        st.success("Prediction complete")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_card("3-Year Lease-Up Probability", f"{probability * 100:.2f}%", "Classification output")
        with k2:
            render_card("Predicted Lease-Up Months", "N/A" if months is None else f"{months:.2f}", "Regression output")
        with k3:
            render_card("Overall Lease-Up Score", f"{score:.2f}", "Blended business score")
        with k4:
            render_card("Opportunity Band", band, "Business priority segment")

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(probability_gauge(probability), use_container_width=True)
        with g2:
            st.plotly_chart(score_gauge(score), use_container_width=True)

        st.markdown("#### Recommendation")
        st.markdown(
            f"""
            <div class="vb-card">
                <div class="vb-card-title">Recommended Action</div>
                <div style="font-size:1rem;color:{cfg.BRAND_NEUTRAL};line-height:1.6;">{action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_cols = [
            col for col in [
                "Site No",
                "Site Name",
                "City",
                "State",
                "Portfolio Market",
                "Site Type",
                "3-Year Lease-Up Probability",
                "Predicted Lease-Up Months",
                "Overall Lease-Up Score",
                "VI Focus Rank",
            ] if col in prediction_df.columns
        ]
        st.dataframe(prediction_df[detail_cols], use_container_width=True)

        st.download_button(
            "Download Prediction CSV",
            data=to_csv_bytes(prediction_df),
            file_name="single_tower_prediction.csv",
            mime="text/csv",
        )


with tab_batch:
    st.markdown('<div class="vb-section"><h3>Batch Scoring</h3></div>', unsafe_allow_html=True)
    st.caption("Upload a CSV or Excel file with model input columns and return a scored output for business use.")

    template_df = build_batch_template()
    st.download_button(
        "Download Batch Template CSV",
        data=to_csv_bytes(template_df),
        file_name="vertical_bridge_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            missing_required = [col for col in required_batch_columns() if col not in batch_df.columns]

            if missing_required:
                st.error("Your uploaded file is missing required columns:")
                st.write(missing_required)
            else:
                if cfg.BIG3_COUNT not in batch_df.columns:
                    batch_df[cfg.BIG3_COUNT] = 0

                scored_batch_df = score_portfolio(batch_df, classifier, regressor)

                st.success("Batch scoring complete")
                st.dataframe(scored_batch_df.head(100), use_container_width=True)

                st.download_button(
                    "Download Scored Batch CSV",
                    data=to_csv_bytes(scored_batch_df),
                    file_name="scored_batch_prediction.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")


with tab_vifocus:
    st.markdown('<div class="vb-section"><h3>VI Focus & Market Opportunity</h3></div>', unsafe_allow_html=True)

    if vi_focus_df.empty and scored_portfolio.empty:
        st.info("Run the training pipeline first to load VI Focus and market views.")
    else:
        base_df = vi_focus_df if not vi_focus_df.empty else scored_portfolio.copy()

        top_n = st.slider("Number of sites to show", min_value=25, max_value=250, value=100, step=25)

        view_cols = [
            col for col in [
                "VI Focus Rank",
                "Site No",
                "Site Name",
                "Portfolio Market",
                "City",
                "State",
                "Site Type",
                "3-Year Lease-Up Probability",
                "Predicted Lease-Up Months",
                "Overall Lease-Up Score",
            ] if col in base_df.columns
        ]
        st.dataframe(base_df[view_cols].head(top_n), use_container_width=True)

        if not market_rankings_df.empty:
            fig_market = px.bar(
                market_rankings_df.head(20),
                x="Portfolio Market",
                y="avg_score",
                hover_data=["site_count", "avg_probability"],
                color_discrete_sequence=[cfg.BRAND_ACCENT],
            )
            st.plotly_chart(style_fig(fig_market, "Top 20 Markets by Average Lease-Up Score"), use_container_width=True)

        if not base_df.empty and {"Latitude", "Longitude"}.issubset(base_df.columns):
            map_df = base_df.dropna(subset=["Latitude", "Longitude"]).head(500)
            if not map_df.empty:
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="Latitude",
                    lon="Longitude",
                    color="Overall Lease-Up Score",
                    size="3-Year Lease-Up Probability",
                    hover_name="Site No" if "Site No" in map_df.columns else None,
                    hover_data={
                        "Site Name": True if "Site Name" in map_df.columns else False,
                        "Portfolio Market": True if "Portfolio Market" in map_df.columns else False,
                        "Overall Lease-Up Score": ":.2f",
                    },
                    mapbox_style="open-street-map",
                    zoom=3,
                    height=650,
                )
                st.plotly_chart(style_fig(fig_map, "Top Opportunity Site Map"), use_container_width=True)


with tab_intel:
    st.markdown('<div class="vb-section"><h3>Model Intelligence</h3></div>', unsafe_allow_html=True)

    left_intel, right_intel = st.columns(2)

    with left_intel:
        st.markdown("#### Classification Metrics")
        if classification_metrics_df.empty:
            st.info("Classification metrics not found.")
        else:
            st.dataframe(classification_metrics_df, use_container_width=True)

    with right_intel:
        st.markdown("#### Timing Metrics")
        if timing_metrics_df.empty:
            st.info("Timing metrics not found.")
        else:
            st.dataframe(timing_metrics_df, use_container_width=True)

    imp_left, imp_right = st.columns(2)

    with imp_left:
        st.markdown("#### Random Forest Feature Importance")
        if rf_importance_df.empty:
            st.info("Random Forest feature importance file not found.")
        else:
            fig_rf = px.bar(
                rf_importance_df.head(20),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[cfg.BRAND_PRIMARY],
            )
            fig_rf.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(style_fig(fig_rf, "Random Forest Feature Importance"), use_container_width=True)

    with imp_right:
        st.markdown("#### XGBoost Feature Importance")
        if xgb_importance_df.empty:
            st.info("XGBoost feature importance file not found.")
        else:
            fig_xgb = px.bar(
                xgb_importance_df.head(20),
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=[cfg.BRAND_ACCENT],
            )
            fig_xgb.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(style_fig(fig_xgb, "XGBoost Feature Importance"), use_container_width=True)

    st.markdown("#### Model Interpretation")
    st.markdown(
        """
        - The **classification model** estimates whether a tower is likely to lease up within 3 years.
        - The **regression model** estimates the expected time-to-lease-up in months.
        - The **overall lease-up score** blends probability and speed into a business-friendly prioritization metric.
        - Sites already carrying all **3 Big 3 tenants** are treated as saturated and scored to zero for new Big 3 lease-up opportunity.
        """
    )
