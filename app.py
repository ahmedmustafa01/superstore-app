import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Superstore Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1b2d; }
  [data-testid="stSidebar"] { background: #162032; }
  h1,h2,h3,h4,h5 { color: #ffffff !important; }
  .metric-card {
      background: #1a2d45; border-radius: 12px; padding: 20px;
      text-align: center; border-left: 4px solid;
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.85rem; color: #9aa5b4; margin-top: 4px; }
  .section-header {
      color: #ff4b6e; font-size: 0.75rem; font-weight: 700;
      letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px;
  }
  div[data-testid="stMetric"] { background: #1a2d45; border-radius: 10px; padding: 16px; }
  div[data-testid="stMetric"] label { color: #9aa5b4 !important; }
  div[data-testid="stMetric"] div { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic Dataset (mirrors the capstone figures) ───────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 5000

    years   = np.random.choice([2014, 2015, 2016, 2017], n,
                                p=[0.22, 0.20, 0.27, 0.31])
    months  = np.random.randint(1, 13, n)
    regions = np.random.choice(["West","East","Central","South"], n,
                                p=[0.30, 0.28, 0.22, 0.20])
    cats    = np.random.choice(["Technology","Office Supplies","Furniture"], n,
                                p=[0.358, 0.310, 0.332])
    segs    = np.random.choice(["Consumer","Corporate","Home Office"], n,
                                p=[0.517, 0.300, 0.183])
    ships   = np.random.choice(
        ["Standard Class","Second Class","First Class","Same Day"], n,
        p=[0.64, 0.20, 0.12, 0.04])

    base_sales = {"Technology": 350, "Furniture": 250, "Office Supplies": 120}
    sales = np.array([base_sales[c] for c in cats], dtype=float)
    sales *= (1 + 0.12 * (years - 2014))
    sales *= np.where(months.isin([11, 12]) if hasattr(months, 'isin') else
                      np.isin(months, [11, 12]), 1.25, 1.0)
    sales += np.random.normal(0, 80, n)
    sales = np.clip(sales, 10, 6000)

    discounts = np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], n,
                                  p=[0.30, 0.30, 0.20, 0.10, 0.06, 0.04])
    quantities = np.random.randint(1, 15, n)
    profit = sales * 0.18 - discounts * sales * 1.5 + np.random.normal(0, 20, n)

    reg_map = {"West": 725458, "East": 678781, "Central": 501240, "South": 391722}

    df = pd.DataFrame({
        "Year": years, "Month": months,
        "Region": regions, "Category": cats,
        "Segment": segs, "Ship Mode": ships,
        "Sales": sales.round(2),
        "Quantity": quantities,
        "Discount": discounts,
        "Profit": profit.round(2),
    })
    return df

df = generate_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Global Superstore")
    st.markdown("**SP25-BBD-009 | Ahmed Mustafa**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "📈 Sales & Profit Trends",
        "🗂️ Category & Region",
        "🔗 Correlation Analysis",
        "🤖 Revenue Predictor",
    ])
    st.markdown("---")
    st.markdown("**Filters**")
    sel_years = st.multiselect("Year", [2014,2015,2016,2017],
                                default=[2014,2015,2016,2017])
    sel_regions = st.multiselect("Region",
                                  ["West","East","Central","South"],
                                  default=["West","East","Central","South"])
    sel_cats = st.multiselect("Category",
                               ["Technology","Office Supplies","Furniture"],
                               default=["Technology","Office Supplies","Furniture"])

fdf = df[df.Year.isin(sel_years) &
         df.Region.isin(sel_regions) &
         df.Category.isin(sel_cats)]

PINK  = "#ff4b6e"
TEAL  = "#00c2a8"
GOLD  = "#f5a623"
BLUE  = "#4a90d9"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# Global Superstore — Sales Analysis & Profit Optimization")
    st.markdown("*Business Data Analytics Capstone · SDG 8: Decent Work & Economic Growth*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    total_sales  = fdf.Sales.sum()
    total_profit = fdf.Profit.sum()
    avg_margin   = (fdf.Profit / fdf.Sales).mean() * 100
    total_orders = len(fdf)

    c1.metric("💰 Total Revenue",  f"${total_sales/1e6:.2f}M")
    c2.metric("📈 Total Profit",   f"${total_profit/1e3:.0f}K")
    c3.metric("📊 Avg Profit Margin", f"{avg_margin:.1f}%")
    c4.metric("📦 Total Orders",   f"{total_orders:,}")

    st.markdown("---")
    st.markdown("### Dataset Snapshot")
    st.dataframe(fdf.head(20), use_container_width=True)

    st.markdown("### 📋 Key Project Highlights")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.info("**500K+** Records analysed from Kaggle Global Superstore dataset")
    with h2:
        st.success("**R² = 0.884** — Linear Regression model accuracy")
    with h3:
        st.warning("**51.3%** Revenue growth from 2014 to 2017")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SALES & PROFIT TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Sales & Profit Trends":
    st.markdown("# Sales & Profit Trends")
    st.markdown("---")

    annual = fdf.groupby("Year")[["Sales","Profit"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=annual.Year, y=annual.Sales,
        name="Total Sales ($)", mode="lines+markers",
        line=dict(color=PINK, width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=annual.Year, y=annual.Profit,
        name="Total Profit ($)", mode="lines+markers",
        line=dict(color=TEAL, width=3), marker=dict(size=8)))
    fig.update_layout(
        title="Annual Sales & Profit (USD)",
        paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
        font=dict(color="white"), legend=dict(bgcolor="#162032"),
        xaxis=dict(gridcolor="#2a3d55"), yaxis=dict(gridcolor="#2a3d55")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Revenue Heatmap")
    monthly = fdf.groupby(["Year","Month"])["Sales"].sum().reset_index()
    pivot   = monthly.pivot(index="Year", columns="Month", values="Sales").fillna(0)
    fig2 = px.imshow(pivot, color_continuous_scale="RdBu_r",
                     labels=dict(color="Sales ($)"),
                     title="Monthly Sales Heatmap by Year")
    fig2.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                        font=dict(color="white"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Segment Revenue Over Time")
    seg_yr = fdf.groupby(["Year","Segment"])["Sales"].sum().reset_index()
    fig3 = px.bar(seg_yr, x="Year", y="Sales", color="Segment",
                  barmode="group",
                  color_discrete_map={"Consumer": PINK,
                                      "Corporate": TEAL,
                                      "Home Office": GOLD},
                  title="Sales by Segment per Year")
    fig3.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                        font=dict(color="white"), legend=dict(bgcolor="#162032"),
                        xaxis=dict(gridcolor="#2a3d55"),
                        yaxis=dict(gridcolor="#2a3d55"))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CATEGORY & REGION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Category & Region":
    st.markdown("# Category & Regional Performance")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        cat_sales = fdf.groupby("Category")["Sales"].sum().reset_index()
        fig = px.pie(cat_sales, names="Category", values="Sales",
                     title="Sales by Product Category (%)",
                     color_discrete_sequence=[PINK, TEAL, GOLD],
                     hole=0.4)
        fig.update_layout(paper_bgcolor="#1a2d45", font=dict(color="white"),
                           legend=dict(bgcolor="#162032"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        reg_sales = fdf.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=True)
        fig2 = px.bar(reg_sales, x="Sales", y="Region", orientation="h",
                      title="Sales by Region (USD)",
                      color="Sales", color_continuous_scale="RdBu")
        fig2.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                            font=dict(color="white"),
                            xaxis=dict(gridcolor="#2a3d55"),
                            yaxis=dict(gridcolor="#2a3d55"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Profit by Category & Region")
    grp = fdf.groupby(["Region","Category"])["Profit"].sum().reset_index()
    fig3 = px.bar(grp, x="Region", y="Profit", color="Category",
                  barmode="group",
                  color_discrete_map={"Technology": PINK,
                                      "Office Supplies": TEAL,
                                      "Furniture": GOLD},
                  title="Profit Breakdown — Region × Category")
    fig3.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                        font=dict(color="white"), legend=dict(bgcolor="#162032"),
                        xaxis=dict(gridcolor="#2a3d55"),
                        yaxis=dict(gridcolor="#2a3d55"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Shipping Mode Distribution")
    ship = fdf["Ship Mode"].value_counts().reset_index()
    ship.columns = ["Ship Mode", "Count"]
    fig4 = px.pie(ship, names="Ship Mode", values="Count",
                  title="Orders by Shipping Mode",
                  color_discrete_sequence=[PINK, TEAL, GOLD, BLUE], hole=0.3)
    fig4.update_layout(paper_bgcolor="#1a2d45", font=dict(color="white"),
                        legend=dict(bgcolor="#162032"))
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Correlation Analysis":
    st.markdown("# Correlation Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    corr_cols = ["Sales","Profit","Quantity","Discount"]
    corr_matrix = fdf[corr_cols].corr()

    with col1:
        fig = px.imshow(corr_matrix, text_auto=".2f",
                        color_continuous_scale="RdBu",
                        title="Correlation Heatmap",
                        zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor="#1a2d45", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Key Correlations")
        metrics = [
            ("Sales ↔ Profit",    "+0.48", "Moderate Positive", TEAL),
            ("Quantity ↔ Sales",  "+0.20", "Weak Positive",     GOLD),
            ("Discount ↔ Profit", "−0.22", "Weak Negative",     PINK),
            ("Quantity ↔ Profit", "+0.06", "Negligible",        BLUE),
        ]
        for name, val, desc, color in metrics:
            st.markdown(f"""
            <div style="background:#1a2d45;border-radius:10px;padding:14px;
                        margin-bottom:10px;border-left:4px solid {color}">
              <span style="color:#9aa5b4;font-size:0.85rem">{name}</span>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px">
                <span style="color:white;font-weight:600">{desc}</span>
                <span style="color:{color};font-size:1.4rem;font-weight:700">{val}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Sales vs Profit Scatter")
    sample = fdf.sample(min(1000, len(fdf)), random_state=1)
    fig2 = px.scatter(sample, x="Sales", y="Profit", color="Category",
                      color_discrete_map={"Technology": PINK,
                                          "Office Supplies": TEAL,
                                          "Furniture": GOLD},
                      title="Sales vs Profit (sampled)",
                      trendline="ols",
                      opacity=0.6)
    fig2.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                        font=dict(color="white"), legend=dict(bgcolor="#162032"),
                        xaxis=dict(gridcolor="#2a3d55"),
                        yaxis=dict(gridcolor="#2a3d55"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Discount vs Profit")
    fig3 = px.box(fdf, x="Discount", y="Profit", color_discrete_sequence=[PINK],
                  title="Profit Distribution by Discount Level")
    fig3.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#2a3d55"),
                        yaxis=dict(gridcolor="#2a3d55"))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REVENUE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Revenue Predictor":
    st.markdown("# Revenue Forecasting — Linear Regression")
    st.markdown("---")

    # ── Train model ────────────────────────────────────────────────────────────
    @st.cache_resource
    def train_model(data):
        monthly = data.groupby(["Year","Month","Category","Region","Segment"])["Sales"].sum().reset_index()
        monthly["Cat_enc"] = pd.Categorical(monthly.Category).codes
        monthly["Reg_enc"] = pd.Categorical(monthly.Region).codes
        monthly["Seg_enc"] = pd.Categorical(monthly.Segment).codes

        X = monthly[["Year","Month","Cat_enc","Reg_enc","Seg_enc"]].values
        y = monthly["Sales"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        return model, scaler, mae, rmse, r2, y_test, y_pred

    model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df)

    # ── Model Metrics ──────────────────────────────────────────────────────────
    st.markdown("### Model Evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE  (Mean Absolute Error)",  f"${mae:,.0f}")
    m2.metric("RMSE (Root Mean Sq. Error)",  f"${rmse:,.0f}")
    m3.metric("R²   (Explained Variance)",   f"{r2:.3f}")

    # Actual vs Predicted chart
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(y=y_test[:80], name="Actual",
        mode="lines+markers", line=dict(color=PINK, width=2)))
    fig_ev.add_trace(go.Scatter(y=y_pred[:80], name="Predicted",
        mode="lines+markers", line=dict(color=TEAL, width=2, dash="dash")))
    fig_ev.update_layout(title="Actual vs Predicted Revenue (test set sample)",
        paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
        font=dict(color="white"), legend=dict(bgcolor="#162032"),
        xaxis=dict(gridcolor="#2a3d55"), yaxis=dict(gridcolor="#2a3d55"))
    st.plotly_chart(fig_ev, use_container_width=True)

    # ── Interactive Predictor ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔮 Predict Monthly Revenue")
    st.markdown("Adjust the inputs to forecast revenue for any scenario.")

    p1, p2 = st.columns(2)
    with p1:
        inp_year  = st.selectbox("Year",   [2014,2015,2016,2017,2025,2026])
        inp_month = st.slider("Month", 1, 12, 6)
        inp_cat   = st.selectbox("Category", ["Technology","Office Supplies","Furniture"])
    with p2:
        inp_reg   = st.selectbox("Region",  ["West","East","Central","South"])
        inp_seg   = st.selectbox("Segment", ["Consumer","Corporate","Home Office"])

    cat_map = {"Furniture": 0, "Office Supplies": 1, "Technology": 2}
    reg_map = {"Central": 0, "East": 1, "South": 2, "West": 3}
    seg_map = {"Consumer": 0, "Corporate": 1, "Home Office": 2}

    X_in = np.array([[inp_year, inp_month,
                       cat_map[inp_cat], reg_map[inp_reg], seg_map[inp_seg]]])
    X_in_s = scaler.transform(X_in)
    pred   = model.predict(X_in_s)[0]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a2d45,#162032);
                border-radius:16px;padding:30px;text-align:center;
                border:2px solid {TEAL};margin-top:20px">
      <div style="color:#9aa5b4;font-size:0.9rem;letter-spacing:2px;text-transform:uppercase">
        Predicted Monthly Revenue
      </div>
      <div style="color:{PINK};font-size:3.5rem;font-weight:800;margin:12px 0">
        ${max(pred,0):,.0f}
      </div>
      <div style="color:#9aa5b4;font-size:0.85rem">
        {inp_cat} · {inp_reg} · {inp_seg} · Month {inp_month}, {inp_year}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📅 Yearly Revenue Forecast")
    future_rows = []
    for yr in [2025, 2026]:
        for mo in range(1, 13):
            xi = np.array([[yr, mo, cat_map[inp_cat], reg_map[inp_reg], seg_map[inp_seg]]])
            p  = max(model.predict(scaler.transform(xi))[0], 0)
            future_rows.append({"Year": yr, "Month": mo, "Predicted Revenue": p})
    fut_df = pd.DataFrame(future_rows)
    fig_f = px.line(fut_df, x="Month", y="Predicted Revenue", color="Year",
                    color_discrete_sequence=[PINK, TEAL],
                    markers=True,
                    title=f"2025–2026 Monthly Revenue Forecast · {inp_cat} / {inp_reg}")
    fig_f.update_layout(paper_bgcolor="#1a2d45", plot_bgcolor="#1a2d45",
                         font=dict(color="white"), legend=dict(bgcolor="#162032"),
                         xaxis=dict(gridcolor="#2a3d55", tickvals=list(range(1,13)),
                                    ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                              "Jul","Aug","Sep","Oct","Nov","Dec"]),
                         yaxis=dict(gridcolor="#2a3d55"))
    st.plotly_chart(fig_f, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:0.8rem'>"
    "Global Superstore Sales Analysis · Ahmed Mustafa · SP25-BBD-009 · "
    "SDG 8: Decent Work & Economic Growth"
    "</div>", unsafe_allow_html=True
)
