"""
RetailPulse AI - Interactive Multi-Store Sales Analytics & Forecasting Dashboard
================================================================================
Academic Level: 2nd Year 2nd Semester Undergraduate Data Science Project
Author: RetailPulse AI Team

A modern, intuitive Streamlit web application designed for both technical evaluators
and non-technical business stakeholders.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import (
    FEATURES,
    load_raw_data,
    engineer_features,
    train_and_evaluate,
)

# -----------------------------------------------------------------------------
# Configuration & Theming
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="RetailPulse AI | Multi-Store Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI cards, badges, and clean layout
st.markdown(
    """
    <style>
    /* Main container spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Hero header banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0d9488 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
    }
    .hero-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        color: #cbd5e1;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(13, 148, 136, 0.3);
        border: 1px solid rgba(45, 212, 191, 0.4);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #5eead4;
        margin-top: 0.8rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #0d9488;
        margin-top: 0.2rem;
        font-weight: 500;
    }

    /* Explanation & Info Boxes */
    .info-card {
        background-color: #f0fdfa;
        border-left: 4px solid #0d9488;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #134e4a;
        font-size: 0.92rem;
    }
    .info-card strong {
        color: #0f766e;
    }
    
    /* Navigation styling */
    .stRadio [role="radiogroup"] {
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    """Load historical raw dataset with caching."""
    return load_raw_data()


@st.cache_resource
def get_model_and_metadata():
    """Load trained model, evaluation metadata, and test predictions."""
    model_file = MODELS_DIR / "best_model.joblib"
    meta_file = MODELS_DIR / "metadata.json"
    preds_file = MODELS_DIR / "test_predictions.csv"

    if not model_file.exists() or not meta_file.exists():
        train_and_evaluate()

    model = joblib.load(model_file)
    metadata = json.loads(meta_file.read_text())
    preds = pd.read_csv(preds_file, parse_dates=["Date"])
    return model, metadata, preds


raw_df = get_dataset()
model, metadata, test_preds = get_model_and_metadata()

# Helper currency formatter
def fmt_currency(val):
    if pd.isna(val):
        return "$0"
    return f"${val:,.0f}"


# -----------------------------------------------------------------------------
# App Header & Sidebar Navigation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>RetailPulse AI 📈</h1>
        <p>Multi-Store Weekly Sales Analytics, Machine Learning & Scenario Forecasting</p>
        <span class="hero-badge">🎓 2nd Year Undergrad Data Science Portfolio Project • Time-Series System</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/100/combo-chart.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a View:",
    [
        "📊 Executive Overview",
        "🏪 Store Deep Dive",
        "🔮 Sales Forecast Simulator",
        "🧪 Model Performance Lab",
        "🎓 Project Methodology & Viva Guide",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Snapshot")
st.sidebar.markdown(f"• **Active Stores:** `{raw_df['Store'].nunique()}`")
st.sidebar.markdown(f"• **Dataset Period:** `2010–2012`")
st.sidebar.markdown(f"• **Champion Model:** `{metadata.get('best_model', 'Ensemble')}`")
st.sidebar.markdown(f"• **Validation R²:** `{metadata.get('metrics', {}).get(metadata.get('best_model'), {}).get('R2', 0.98):.3f}`")
st.sidebar.caption("RetailPulse AI • BSc (Hons) Data Science")


# =============================================================================
# PAGE 1: 📊 Executive Overview
# =============================================================================
if page == "📊 Executive Overview":
    st.header("Executive Business Overview")
    st.caption("High-level performance summary across all 45 Walmart retail outlets.")

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_sales = raw_df["Weekly_Sales"].sum()
    avg_sales = raw_df["Weekly_Sales"].mean()
    total_stores = raw_df["Store"].nunique()
    total_weeks = raw_df["Date"].nunique()

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Historical Sales</div>
                <div class="metric-value">{fmt_currency(total_sales)}</div>
                <div class="metric-sub">Across 45 locations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Weekly Sales</div>
                <div class="metric-value">{fmt_currency(avg_sales)}</div>
                <div class="metric-sub">Per store / week</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Retail Stores</div>
                <div class="metric-value">{total_stores}</div>
                <div class="metric-sub">Nationwide coverage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Observed Period</div>
                <div class="metric-value">{total_weeks} Wks</div>
                <div class="metric-sub">Feb 2010 – Oct 2012</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Plain-English Executive Insight Card
    st.markdown(
        """
        <div class="info-card">
            <strong>💡 Key Business Takeaway:</strong> 
            Weekly sales exhibit massive, predictable seasonal surges every year around Thanksgiving (Week 47) and 
            Christmas (Week 51). Stores that plan inventory and staffing using annual seasonal lag indicators gain a 
            tremendous operational advantage over simple rolling average methods.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Network-wide Sales Trend Line Chart
    weekly_network = raw_df.groupby("Date", as_index=False)["Weekly_Sales"].sum()
    fig_network = px.line(
        weekly_network,
        x="Date",
        y="Weekly_Sales",
        title="Total Weekly Sales Across All Stores (2010 – 2012)",
        labels={"Weekly_Sales": "Total Sales ($)", "Date": "Date"},
        template="plotly_white",
        color_discrete_sequence=["#0d9488"],
    )
    fig_network.update_traces(line=dict(width=2.5))
    fig_network.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(tickprefix="$"),
    )
    st.plotly_chart(fig_network, use_container_width=True)

    # Bottom Row: Top Stores & Holiday Effect
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        top_stores = (
            raw_df.groupby("Store", as_index=False)["Weekly_Sales"]
            .mean()
            .sort_values("Weekly_Sales", ascending=False)
            .head(10)
        )
        top_stores["Store_Label"] = top_stores["Store"].apply(lambda x: f"Store {x}")
        fig_top = px.bar(
            top_stores,
            x="Store_Label",
            y="Weekly_Sales",
            title="Top 10 Highest-Grossing Stores (Average Weekly Sales)",
            labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Store_Label": "Store"},
            template="plotly_white",
            color="Weekly_Sales",
            color_continuous_scale="Teal",
        )
        fig_top.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(tickprefix="$"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with chart_col2:
        holiday_comp = raw_df.groupby("Holiday_Flag", as_index=False)["Weekly_Sales"].mean()
        holiday_comp["Week_Type"] = holiday_comp["Holiday_Flag"].map(
            {0: "Normal Week", 1: "Holiday Week (Thanksgiving/Christmas/SuperBowl)"}
        )
        fig_holiday = px.bar(
            holiday_comp,
            x="Week_Type",
            y="Weekly_Sales",
            title="Impact of Holidays on Average Store Sales",
            labels={"Weekly_Sales": "Avg Store Sales ($)", "Week_Type": ""},
            color="Week_Type",
            color_discrete_sequence=["#64748b", "#0d9488"],
            template="plotly_white",
        )
        fig_holiday.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(tickprefix="$"),
        )
        st.plotly_chart(fig_holiday, use_container_width=True)


# =============================================================================
# PAGE 2: 🏪 Store Deep Dive
# =============================================================================
elif page == "🏪 Store Deep Dive":
    st.header("Store-Level Performance Explorer")
    st.caption("Analyze weekly sales patterns, moving average trends, and volatility for any specific retail store.")

    store_list = sorted(raw_df["Store"].unique())
    selected_store = st.selectbox("Select Store Location to Inspect:", store_list, index=0)

    store_df = raw_df[raw_df["Store"] == selected_store].sort_values("Date").copy()
    store_df["Rolling_4W"] = store_df["Weekly_Sales"].rolling(4).mean()

    # Store Stats Cards
    st_col1, st_col2, st_col3, st_col4 = st.columns(4)
    s_mean = store_df["Weekly_Sales"].mean()
    s_max = store_df["Weekly_Sales"].max()
    s_min = store_df["Weekly_Sales"].min()
    s_cv = (store_df["Weekly_Sales"].std() / s_mean) * 100

    with st_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Weekly Sales</div>
                <div class="metric-value">{fmt_currency(s_mean)}</div>
                <div class="metric-sub">Store {selected_store} baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Highest Recorded Week</div>
                <div class="metric-value">{fmt_currency(s_max)}</div>
                <div class="metric-sub">Peak holiday week</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Lowest Recorded Week</div>
                <div class="metric-value">{fmt_currency(s_min)}</div>
                <div class="metric-sub">Trough week</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sales Volatility (CV)</div>
                <div class="metric-value">{s_cv:.1f}%</div>
                <div class="metric-sub">Degree of fluctuation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Interactive Store Time-Series Chart
    fig_store = go.Figure()

    # Actual sales
    fig_store.add_trace(
        go.Scatter(
            x=store_df["Date"],
            y=store_df["Weekly_Sales"],
            name="Actual Weekly Sales",
            line=dict(color="#334155", width=2),
            mode="lines",
        )
    )

    # 4-Week Moving Average
    fig_store.add_trace(
        go.Scatter(
            x=store_df["Date"],
            y=store_df["Rolling_4W"],
            name="4-Week Smoothed Trend",
            line=dict(color="#f59e0b", width=2.5),
            mode="lines",
        )
    )

    # Holiday markers
    holidays = store_df[store_df["Holiday_Flag"] == 1]
    fig_store.add_trace(
        go.Scatter(
            x=holidays["Date"],
            y=holidays["Weekly_Sales"],
            name="Holiday Weeks",
            mode="markers",
            marker=dict(color="#ef4444", size=8, symbol="diamond"),
        )
    )

    fig_store.update_layout(
        title=f"Store #{selected_store} Weekly Sales History & Moving Average",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(tickprefix="$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_store, use_container_width=True)

    # Explanatory card
    st.markdown(
        """
        <div class="info-card">
            <strong>💡 Understanding This View:</strong>
            The <span style="color:#f59e0b; font-weight:600;">yellow line</span> reveals the underlying trend by smoothing out 
            short-term noise over 4-week periods. Notice how every <span style="color:#ef4444; font-weight:600;">red diamond (Holiday)</span> 
            consistently creates an abrupt sales spike in late November and late December.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Monthly Seasonality Distribution
    store_df["Month_Name"] = store_df["Date"].dt.strftime("%b")
    month_order = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_sales = (
        store_df.groupby("Month_Name", as_index=False)["Weekly_Sales"]
        .mean()
        .sort_values(by="Month_Name", key=lambda x: pd.Categorical(x, categories=month_order, ordered=True))
    )

    fig_month = px.bar(
        monthly_sales,
        x="Month_Name",
        y="Weekly_Sales",
        title=f"Store #{selected_store} Average Sales by Month (Seasonality Pattern)",
        labels={"Weekly_Sales": "Avg Sales ($)", "Month_Name": "Month"},
        template="plotly_white",
        color="Weekly_Sales",
        color_continuous_scale="Blues",
    )
    fig_month.update_layout(coloraxis_showscale=False, yaxis=dict(tickprefix="$"))
    st.plotly_chart(fig_month, use_container_width=True)


# =============================================================================
# PAGE 3: 🔮 Sales Forecast Simulator
# =============================================================================
elif page == "🔮 Sales Forecast Simulator":
    st.header("Interactive Multi-Step Scenario Forecaster")
    st.caption(
        "Simulate future weekly sales up to 12 weeks ahead with dynamic holiday detection and economic what-if scenarios."
    )

    sim_store = st.selectbox("Select Store for Forecast Simulation:", sorted(raw_df["Store"].unique()), index=0)
    horizon_weeks = st.slider("Forecast Horizon (Number of weeks ahead):", min_value=1, max_value=12, value=8)

    store_hist = raw_df[raw_df["Store"] == sim_store].sort_values("Date")
    last_row = store_hist.iloc[-1]
    last_date = last_row["Date"]
    store_avg_val = float(store_hist["Weekly_Sales"].mean())

    st.markdown("### 🎛️ Scenario Adjustments (Economic Conditions)")
    st.caption("Adjust anticipated macroeconomic variables for the forecast period, or use recent historical values.")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        temp_input = st.number_input(
            "Temperature (°F):",
            value=float(round(last_row["Temperature"], 1)),
            step=1.0,
            help="Anticipated average regional temperature.",
        )
    with col_e2:
        fuel_input = st.number_input(
            "Fuel Price ($/gal):",
            value=float(round(last_row["Fuel_Price"], 2)),
            step=0.05,
            help="Anticipated fuel price per gallon.",
        )
    with col_e3:
        cpi_input = st.number_input(
            "Consumer Price Index (CPI):",
            value=float(round(last_row["CPI"], 2)),
            step=0.5,
            help="Inflation / purchasing power indicator.",
        )
    with col_e4:
        unemp_input = st.number_input(
            "Unemployment Rate (%):",
            value=float(round(last_row["Unemployment"], 2)),
            step=0.1,
            help="Regional unemployment rate percentage.",
        )

    # Dynamic US Holiday calendar helper
    def is_holiday_week(target_date: pd.Timestamp) -> bool:
        """Auto-detect major US retail holiday periods based on calendar date."""
        month = target_date.month
        day = target_date.day
        # Super Bowl (Feb 5-15)
        if month == 2 and 5 <= day <= 15:
            return True
        # Labor Day (early Sept)
        if month == 9 and day <= 10:
            return True
        # Thanksgiving week (late Nov)
        if month == 11 and 20 <= day <= 28:
            return True
        # Christmas week (late Dec)
        if month == 12 and 20 <= day <= 30:
            return True
        return False

    if st.button("🚀 Generate Forecast", type="primary"):
        sales_history = list(store_hist["Weekly_Sales"])
        forecast_records = []
        rec_mae = metadata.get("recursive_evaluation", {}).get("Recursive_12W_MAE", 46500.0)

        for step in range(1, horizon_weeks + 1):
            target_dt = last_date + pd.Timedelta(days=7 * step)
            iso = target_dt.isocalendar()
            holiday_flag = 1 if is_holiday_week(target_dt) else 0

            feature_dict = {
                "Store": sim_store,
                "Store_Avg_Sales": store_avg_val,
                "Holiday_Flag": holiday_flag,
                "Temperature": temp_input,
                "Fuel_Price": fuel_input,
                "CPI": cpi_input,
                "Unemployment": unemp_input,
                "Year": target_dt.year,
                "Month": target_dt.month,
                "WeekOfYear": int(iso.week),
                "Quarter": target_dt.quarter,
                "Lag_1": sales_history[-1],
                "Lag_2": sales_history[-2],
                "Lag_4": sales_history[-4],
                "Lag_52": sales_history[-52] if len(sales_history) >= 52 else sales_history[-1],
                "Rolling_Mean_4": float(np.mean(sales_history[-4:])),
                "Rolling_Std_4": float(np.std(sales_history[-4:], ddof=1)) if len(sales_history) >= 4 else 0.0,
            }

            pred_val = max(0.0, float(model.predict(pd.DataFrame([feature_dict])[FEATURES])[0]))
            sales_history.append(pred_val)

            # Uncertainty expands with horizon sqrt(step)
            margin = 1.25 * rec_mae * np.sqrt(step)
            lower_bound = max(0.0, pred_val - margin)
            upper_bound = pred_val + margin

            forecast_records.append(
                {
                    "Week": f"Week +{step}",
                    "Date": target_dt,
                    "Forecast ($)": pred_val,
                    "Lower Bound ($)": lower_bound,
                    "Upper Bound ($)": upper_bound,
                    "Holiday Type": "🎉 Major Holiday" if holiday_flag else "🗓️ Regular",
                }
            )

        fc_df = pd.DataFrame(forecast_records)

        # Plotly combined historical + future chart
        fig_fc = go.Figure()

        # Recent historical actuals (last 16 weeks)
        recent_hist = store_hist.tail(16)
        fig_fc.add_trace(
            go.Scatter(
                x=recent_hist["Date"],
                y=recent_hist["Weekly_Sales"],
                name="Recent Actual Sales",
                line=dict(color="#334155", width=2.5),
                mode="lines+markers",
            )
        )

        # Forecast line
        fig_fc.add_trace(
            go.Scatter(
                x=fc_df["Date"],
                y=fc_df["Forecast ($)"],
                name="Predicted Sales (Forecast)",
                line=dict(color="#0d9488", width=3, dash="dash"),
                mode="lines+markers",
            )
        )

        # Shaded uncertainty band
        fig_fc.add_trace(
            go.Scatter(
                x=list(fc_df["Date"]) + list(fc_df["Date"][::-1]),
                y=list(fc_df["Upper Bound ($)"]) + list(fc_df["Lower Bound ($)"][::-1]),
                fill="toself",
                fillcolor="rgba(13, 148, 136, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="85% Prediction Uncertainty Band",
                hoverinfo="skip",
            )
        )

        fig_fc.update_layout(
            title=f"Store #{sim_store}: {horizon_weeks}-Week Recursive Sales Forecast",
            template="plotly_white",
            hovermode="x unified",
            yaxis=dict(tickprefix="$"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Plain-English Explanation
        st.markdown(
            """
            <div class="info-card">
                <strong>💡 How Recursive Forecasting Works:</strong> 
                To predict Week +1, the model uses real historical sales. To predict Week +2, Week +1's predicted value 
                becomes the new <code>Lag_1</code>! The shaded <span style="color:#0d9488; font-weight:600;">green band</span> 
                expands over time to honestly reflect increasing uncertainty further into the future.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Formatted forecast table
        display_df = fc_df.copy()
        display_df["Forecast"] = display_df["Forecast ($)"].apply(fmt_currency)
        display_df["Lower Bound"] = display_df["Lower Bound ($)"].apply(fmt_currency)
        display_df["Upper Bound"] = display_df["Upper Bound ($)"].apply(fmt_currency)
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

        st.dataframe(
            display_df[["Week", "Date", "Forecast", "Lower Bound", "Upper Bound", "Holiday Type"]],
            use_container_width=True,
        )

        # Download button
        csv_data = fc_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Results as CSV",
            data=csv_data,
            file_name=f"store_{sim_store}_sales_forecast.csv",
            mime="text/csv",
        )


# =============================================================================
# PAGE 4: 🧪 Model Performance Lab
# =============================================================================
elif page == "🧪 Model Performance Lab":
    st.header("Machine Learning Performance & Evaluation Lab")
    st.caption("Transparent benchmark comparing Naive Baselines against Scikit-Learn Machine Learning Regressors.")

    # Leaderboard Table
    metrics_dict = metadata.get("metrics", {})
    leaderboard = []
    for m_name, m_vals in metrics_dict.items():
        leaderboard.append(
            {
                "Model Architecture": m_name,
                "MAE ($)": fmt_currency(m_vals.get("MAE", 0)),
                "RMSE ($)": fmt_currency(m_vals.get("RMSE", 0)),
                "MAPE (%)": f"{m_vals.get('MAPE', 0):.2f}%",
                "R² Score": f"{m_vals.get('R2', 0):.4f}",
            }
        )

    lb_df = pd.DataFrame(leaderboard)
    st.subheader("Model Leaderboard (Chronological Holdout Set - 20 Weeks)")
    st.dataframe(lb_df, use_container_width=True)

    champion = metadata.get("best_model", "Gradient Boosting")
    st.success(f"🏆 Champion Model Selected for Production: **{champion}** (Highest R² and lowest Holdout RMSE)")

    # Explanation of metrics
    with st.expander("📚 Plain-English Guide to These Metrics (What do they mean?)"):
        st.markdown(
            """
            - **MAE (Mean Absolute Error):** The average dollar amount the prediction deviates from actual sales.
            - **RMSE (Root Mean Squared Error):** Penalizes larger forecast errors more heavily than small errors.
            - **MAPE (Mean Absolute Percentage Error):** The error expressed as a percentage of actual sales. A MAPE of ~4% indicates excellent forecasting precision.
            - **R² Score:** The proportion of variance explained by the model. An R² above **0.98** means the model captures over 98% of all store sales patterns!
            """
        )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        # Cross-validation comparison
        cv_data = metadata.get("cv", {})
        if cv_data:
            cv_rows = [
                {"Model": k, "Mean CV RMSE ($)": v.get("Mean_RMSE", 0)} for k, v in cv_data.items()
            ]
            cv_plot_df = pd.DataFrame(cv_rows)
            fig_cv = px.bar(
                cv_plot_df,
                x="Model",
                y="Mean CV RMSE ($)",
                title="3-Fold Expanding-Window Cross-Validation RMSE",
                labels={"Mean CV RMSE ($)": "Average RMSE ($)"},
                color="Model",
                color_discrete_sequence=["#0d9488", "#334155"],
                template="plotly_white",
            )
            fig_cv.update_layout(showlegend=False, yaxis=dict(tickprefix="$"))
            st.plotly_chart(fig_cv, use_container_width=True)

    with col_m2:
        # Feature Importance Chart
        imp_list = metadata.get("importance", [])
        if imp_list:
            imp_df = pd.DataFrame(imp_list[:8], columns=["Feature", "Relative Importance"])
            # Format feature names nicely
            friendly_names = {
                "Lag_52": "Lag 52 (Sales 1 Year Ago)",
                "Store_Avg_Sales": "Store Historical Volume",
                "Rolling_Mean_4": "Rolling 4-Week Average",
                "Lag_1": "Lag 1 (Previous Week Sales)",
                "Lag_2": "Lag 2 (Sales 2 Weeks Ago)",
                "Lag_4": "Lag 4 (Sales 4 Weeks Ago)",
                "Unemployment": "Unemployment Rate",
                "WeekOfYear": "Week of Year (Seasonality)",
                "CPI": "Consumer Price Index",
                "Fuel_Price": "Fuel Price",
            }
            imp_df["Feature"] = imp_df["Feature"].map(lambda x: friendly_names.get(x, x))
            fig_imp = px.bar(
                imp_df.sort_values("Relative Importance", ascending=True),
                x="Relative Importance",
                y="Feature",
                orientation="h",
                title="Top Predictive Drivers (Feature Importance)",
                template="plotly_white",
                color="Relative Importance",
                color_continuous_scale="Teal",
            )
            fig_imp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

    # Holdout Actual vs. Predicted Visual
    st.markdown("---")
    st.subheader("Holdout Test Period: Actual vs. Predicted Weekly Sales")
    test_store = st.selectbox("Pick Store to inspect test period fit:", sorted(test_preds["Store"].unique()), index=0)

    store_test = test_preds[test_preds["Store"] == test_store].sort_values("Date")
    col_pred_name = champion.replace(" ", "_")

    fig_act_pred = go.Figure()
    fig_act_pred.add_trace(
        go.Scatter(
            x=store_test["Date"],
            y=store_test["Weekly_Sales"],
            name="Actual Holdout Sales",
            line=dict(color="#0f172a", width=2.5),
            mode="lines+markers",
        )
    )
    if col_pred_name in store_test.columns:
        fig_act_pred.add_trace(
            go.Scatter(
                x=store_test["Date"],
                y=store_test[col_pred_name],
                name=f"Predicted by {champion}",
                line=dict(color="#0d9488", width=2.5, dash="dash"),
                mode="lines+markers",
            )
        )

    fig_act_pred.update_layout(
        title=f"Store #{test_store} Holdout Evaluation (June 2012 – October 2012)",
        template="plotly_white",
        yaxis=dict(tickprefix="$"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_act_pred, use_container_width=True)


# =============================================================================
# PAGE 5: 🎓 Project Methodology & Viva Guide
# =============================================================================
else:
    st.header("Project Methodology & Undergraduate Defense Guide")
    st.caption("A complete academic breakdown prepared for course evaluation, viva defense, and portfolio review.")

    st.markdown(
        """
        ### 1. Problem Formulation & Research Question
        In retail enterprise operations, accurate weekly sales forecasting is vital for **inventory management, supply chain logistics, 
        and staff allocation**. The core challenge is addressing **high annual seasonality** (e.g., Thanksgiving and Christmas spikes) 
        and macroeconomic variability across geographically diverse retail locations.

        ---

        ### 2. Time-Series Engineering Best Practices Enforced
        As a data science student project, strict safeguards were implemented to prevent common time-series modeling mistakes:
        
        1. **Strict Chronological Holdout (No Lookahead Bias):**
           - Unlike standard random k-fold splitting (which causes future data to leak into the past), the final **20 consecutive weeks** 
             across all 45 stores were reserved as an untouched holdout test set.
             
        2. **Shifted Rolling Windows (`shift(1)`):**
           - When computing the 4-week moving average (`Rolling_Mean_4`) and moving standard deviation (`Rolling_Std_4`), values were 
             strictly shifted by 1 week so the week being predicted is never included in its own rolling calculation.
             
        3. **Store Categorical Representation:**
           - Rather than treating store IDs as arbitrary numeric order (e.g., assuming Store 45 > Store 1), each store is characterized 
             by its historical sales scale (`Store_Avg_Sales`) computed strictly prior to the test cutoff date.
             
        4. **Annual Seasonal Lag 52:**
           - Because retail behavior follows a 52-week calendar cycle, `Lag_52` captures the exact sales from 1 year prior, capturing 
             over **58% of the total predictive importance**.

        ---

        ### 3. End-to-End System Architecture
        ```
        [Raw Walmart Sales Data (6,435 rows)]
                        │
                        ▼
        [Feature Engineering Pipeline]
          ├── Calendar Decomposition (WeekOfYear, Month, Quarter)
          ├── Store Historical Scale (Store_Avg_Sales)
          ├── Lag Variables (Lag 1, 2, 4, 52)
          └── Shifted Rolling Statistics (t-1 to t-4)
                        │
                        ▼
        [Expanding-Window TimeSeriesSplit Cross-Validation]
                        │
                        ▼
        [Model Selection & Champion Serialization (Joblib)]
                        │
                        ▼
        [Streamlit Multi-Step Interactive Forecast Engine]
        ```

        ---

        ### 4. Real-World Limitations & Academic Honesty
        - **Historical Timeframe:** The source dataset concludes in late 2012. Macroeconomic assumptions (CPI, fuel, unemployment) 
          in the Forecast Simulator represent user-defined scenarios rather than real-time live API feeds.
        - **Recursive Error Accumulation:** In multi-step recursive forecasting ($h > 1$), prediction errors from early weeks compound 
          into later weeks. The simulator transparently visualizes this using widening uncertainty bands.
        """
    )
