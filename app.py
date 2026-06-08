import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
 
warnings.filterwarnings("ignore")
 
 
# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Smart Return Predictor · Olist",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
 
# ---------------------------------------------------------------
# CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
 
:root {
    --bg-dark:          #0d0f14;
    --bg-card:          #151820;
    --bg-raised:        #1c2030;
    --border:           #2a2f42;
    --text-primary:     #e8eaf0;
    --text-muted:       #7a8099;
    --accent-green:     #1D9E75;
    --accent-red:       #E24B4A;
    --accent-purple:    #7c6ff7;
    --accent-yellow:    #f5c842;
}
 
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}
 
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
 
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
 
.hero-header {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1f35 50%, #0d1220 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(124,111,247,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(124,111,247,0.15);
    border: 1px solid rgba(124,111,247,0.4);
    color: #a89ff9;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}
 
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-value.green  { color: var(--accent-green); }
.metric-value.red    { color: var(--accent-red); }
.metric-value.purple { color: var(--accent-purple); }
 
.result-good {
    background: linear-gradient(135deg, rgba(29,158,117,0.15), rgba(29,158,117,0.05));
    border: 1px solid rgba(29,158,117,0.5);
    border-left: 4px solid var(--accent-green);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-bad {
    background: linear-gradient(135deg, rgba(226,75,74,0.15), rgba(226,75,74,0.05));
    border: 1px solid rgba(226,75,74,0.5);
    border-left: 4px solid var(--accent-red);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.result-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0;
}
 
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.7rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
 
.stSlider > div > div { color: var(--text-primary) !important; }
label { color: var(--text-primary) !important; font-size: 0.88rem !important; }
.stButton > button {
    background: var(--accent-purple) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
div[data-testid="stSelectbox"] > div { background: var(--bg-raised) !important; }
.stSelectbox label { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model       = joblib.load("model.pkl")
    imputer     = joblib.load("imputer.pkl")
    with open("threshold.json") as f:
        thresh_data = json.load(f)
    with open("feature_names.json") as f:
        features = json.load(f)
    stats    = pd.read_csv("feature_stats.csv", index_col=0)
    shap_imp = pd.read_csv("shap_importance.csv")
    return model, imputer, thresh_data, features, stats, shap_imp
 
 
try:
    model, imputer, thresh_data, FEATURE_COLS, feat_stats, shap_imp = load_artifacts()
    THRESHOLD   = thresh_data["threshold"]
    MODEL_NAME  = thresh_data["model"]
    artifacts_ok = True
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.info(
        "Ensure model.pkl, imputer.pkl, threshold.json, feature_names.json, "
        "feature_stats.csv, and shap_importance.csv are in the same directory as app.py."
    )
    artifacts_ok = False
    st.stop()
 
 
# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown(f"""
<div class="hero-header">
    <div class="hero-badge">OLIST E-COMMERCE · BRAZIL · 100K ORDERS</div>
    <div class="hero-title">Smart Return Predictor</div>
    <div class="hero-subtitle">
        Predicts whether an order will result in a bad experience — before it ships.
        Powered by {MODEL_NAME} · Trained on the Brazilian E-Commerce Public Dataset.
    </div>
</div>
""", unsafe_allow_html=True)
 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Model</div>
        <div class="metric-value purple">XGBoost</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-value green">~0.89</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Training Rows</div>
        <div class="metric-value">99K+</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Decision Threshold</div>
        <div class="metric-value">{THRESHOLD:.2f}</div>
    </div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def med(col):
    try:
        return float(feat_stats.loc[col, "median"])
    except Exception:
        return 0.0
 
 
def fmax(col):
    try:
        return float(feat_stats.loc[col, "max"])
    except Exception:
        return 100.0
 
 
def fmin(col):
    try:
        return float(feat_stats.loc[col, "min"])
    except Exception:
        return 0.0
 
 
# ---------------------------------------------------------------
# SIDEBAR — INPUT FORM
# ---------------------------------------------------------------
st.sidebar.markdown("""
<div style="font-family:'Space Mono',monospace; font-size:1rem; font-weight:700;
     color:#e8eaf0; padding: 0.5rem 0 1rem 0;
     border-bottom:1px solid #2a2f42; margin-bottom:1rem;">
    ORDER PARAMETERS
</div>
""", unsafe_allow_html=True)
 
st.sidebar.markdown("**Delivery & Time**")
delivery_delay_days  = st.sidebar.slider(
    "Delivery Delay (days)", -30, 60, int(med("delivery_delay_days")),
    help="Positive = late vs estimated date",
)
is_late              = int(delivery_delay_days > 0)
approval_wait_hours  = st.sidebar.slider(
    "Approval Wait (hours)", 0.0, 72.0, float(med("approval_wait_hours")), 0.5,
)
carrier_wait_days    = st.sidebar.slider(
    "Carrier Wait (days)", 0, 30, int(med("carrier_wait_days")),
    help="Days from approval to carrier pickup",
)
order_dayofweek      = st.sidebar.selectbox(
    "Order Day", list(range(7)),
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    index=int(med("order_dayofweek")),
)
order_hour           = st.sidebar.slider("Order Hour (0-23)", 0, 23, int(med("order_hour")))
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Order Value**")
total_items          = st.sidebar.slider("Number of Items", 1, 20, max(1, int(med("total_items"))))
total_price          = st.sidebar.slider("Total Price (R$)", 0.0, 5000.0, float(med("total_price")), 10.0)
total_freight        = st.sidebar.slider("Total Freight (R$)", 0.0, 500.0, float(med("total_freight")), 1.0)
avg_price            = total_price / max(total_items, 1)
max_price            = total_price
price_to_freight     = round(total_price / (total_freight + 1e-5), 4)
price_to_freight     = min(
    price_to_freight,
    float(feat_stats.loc["price_to_freight_ratio", "max"])
    if "price_to_freight_ratio" in feat_stats.index
    else 200,
)
payment_installments = st.sidebar.slider("Payment Installments", 1, 24, int(med("payment_installments")))
payment_value        = total_price
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Product**")
product_weight_g    = st.sidebar.slider("Product Weight (g)", 0, 40000, int(med("product_weight_g")), 100)
product_photos_qty  = st.sidebar.slider("Product Photos", 1, 20, int(med("product_photos_qty")))
product_desc_length = st.sidebar.slider(
    "Description Length (chars)", 0, 3000, int(med("product_description_lenght")), 50,
)
category_bad_rate   = st.sidebar.slider(
    "Category Risk Rate", 0.0, 0.30, float(med("category_bad_rate")), 0.01,
    help="Historical bad order rate for this product category (0 = low risk, 0.30 = high risk)",
)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Customer & Seller**")
state_bad_rate    = st.sidebar.slider(
    "State Risk Rate", 0.0, 0.20, float(med("state_bad_rate")), 0.005,
    help="Historical bad order rate in the customer's state",
)
same_state        = st.sidebar.checkbox("Seller same state as customer?", value=False)
customer_past_bad = st.sidebar.slider("Customer Past Bad Orders", 0, 10, 0)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Payment Type**")
payment_type = st.sidebar.selectbox(
    "Payment Type", ["credit_card", "boleto", "debit_card", "voucher"],
)
 
st.sidebar.markdown("**Season**")
season = st.sidebar.selectbox("Season (Brazil)", ["Summer", "Autumn", "Winter", "Spring"])
 
predict_btn = st.sidebar.button("PREDICT RISK")
 
 
# ---------------------------------------------------------------
# BUILD INPUT VECTOR
# ---------------------------------------------------------------
def build_input():
    row = {f: 0.0 for f in FEATURE_COLS}
 
    row["delivery_delay_days"]        = delivery_delay_days
    row["is_late"]                    = is_late
    row["approval_wait_hours"]        = approval_wait_hours
    row["carrier_wait_days"]          = carrier_wait_days
    row["order_dayofweek"]            = order_dayofweek
    row["order_hour"]                 = order_hour
    row["total_items"]                = total_items
    row["total_price"]                = total_price
    row["total_freight"]              = total_freight
    row["avg_price"]                  = avg_price
    row["max_price"]                  = max_price
    row["price_to_freight_ratio"]     = price_to_freight
    row["payment_installments"]       = payment_installments
    row["payment_value"]              = payment_value
    row["product_weight_g"]           = product_weight_g
    row["product_photos_qty"]         = product_photos_qty
    row["product_description_lenght"] = product_desc_length
    row["category_bad_rate"]          = category_bad_rate
    row["state_bad_rate"]             = state_bad_rate
    row["same_state"]                 = int(same_state)
    row["customer_past_bad_orders"]   = customer_past_bad
 
    for pt in ["boleto", "debit_card", "voucher"]:
        col = f"pay_{pt}"
        if col in row:
            row[col] = 1.0 if payment_type == pt else 0.0
 
    for s in ["Spring", "Summer", "Winter"]:
        col = f"season_{s}"
        if col in row:
            row[col] = 1.0 if season == s else 0.0
 
    return pd.DataFrame([row])[FEATURE_COLS]
 
 
# ---------------------------------------------------------------
# GAUGE CHART
# ---------------------------------------------------------------
def draw_gauge(probability, threshold):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#151820")
    ax.set_facecolor("#151820")
 
    from matplotlib.patches import Wedge
 
    colors_arc = ["#1D9E75", "#f5c842", "#E24B4A"]
    for c, a1, a2 in zip(colors_arc, [120, 60, 0], [180, 120, 60]):
        w = Wedge(
            (0.5, 0), 0.42, a1, a2, width=0.10,
            facecolor=c, alpha=0.85, transform=ax.transAxes,
        )
        ax.add_patch(w)
 
    angle_deg = 180 - (probability * 180)
    angle_rad = np.deg2rad(angle_deg)
    nx = 0.5 + 0.32 * np.cos(angle_rad)
    ny = 0.0 + 0.32 * np.sin(angle_rad)
    ax.annotate(
        "", xy=(nx, ny), xytext=(0.5, 0.0),
        arrowprops=dict(arrowstyle="->", color="white", lw=2.5),
        xycoords="axes fraction", textcoords="axes fraction",
    )
 
    ax.plot(0.5, 0.0, "o", color="white", markersize=8, transform=ax.transAxes, zorder=5)
 
    color = "#E24B4A" if probability >= threshold else "#1D9E75"
    ax.text(
        0.5, 0.28, f"{probability * 100:.1f}%",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=22, fontweight="bold", color=color, fontfamily="monospace",
    )
    ax.text(
        0.5, 0.12, "Bad Order Probability",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=8, color="#7a8099",
    )
 
    ax.text(0.05, 0.05, "LOW",  ha="left",  transform=ax.transAxes, fontsize=7, color="#1D9E75", fontfamily="monospace")
    ax.text(0.95, 0.05, "HIGH", ha="right", transform=ax.transAxes, fontsize=7, color="#E24B4A", fontfamily="monospace")
 
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.55)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig
 
 
# ---------------------------------------------------------------
# SHAP WATERFALL CHART
# ---------------------------------------------------------------
def draw_shap_waterfall(input_df):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(imputer.transform(input_df))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        sv = shap_vals[0]
 
        feat_arr  = np.array(FEATURE_COLS)
        top_idx   = np.argsort(np.abs(sv))[-12:][::-1]
        top_feats = feat_arr[top_idx]
        top_vals  = sv[top_idx]
 
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#151820")
        ax.set_facecolor("#151820")
 
        colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in top_vals]
        y_pos  = np.arange(len(top_feats))
 
        bars = ax.barh(
            y_pos, top_vals[::-1], color=colors[::-1],
            edgecolor="none", height=0.65,
        )
 
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feats[::-1], fontsize=9, color="#e8eaf0")
        ax.axvline(x=0, color="#2a2f42", linewidth=1.5)
        ax.set_xlabel(
            "SHAP Value  (positive = increases bad risk, negative = decreases)",
            fontsize=8, color="#7a8099",
        )
        ax.set_title(
            "Feature Impact on This Prediction",
            fontsize=10, fontweight="bold", color="#e8eaf0", pad=10,
        )
        ax.tick_params(colors="#7a8099")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f42")
 
        for bar, val in zip(bars, top_vals[::-1]):
            x = bar.get_width()
            ax.text(
                x + (0.001 if x >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if x >= 0 else "right",
                fontsize=7.5, color="#e8eaf0",
            )
 
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(color="#E24B4A", label="Increases bad risk"),
                Patch(color="#1D9E75", label="Decreases bad risk"),
            ],
            fontsize=8, frameon=False, labelcolor="#7a8099", loc="lower right",
        )
 
        plt.tight_layout()
        return fig
 
    except Exception:
        return None
 
 
# ---------------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE CHART
# ---------------------------------------------------------------
def draw_global_importance():
    top = shap_imp.head(15)
 
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#151820")
    ax.set_facecolor("#151820")
 
    colors = [
        "#7c6ff7" if i < 3 else "#534AB7" if i < 7 else "#2a2f42"
        for i in range(len(top))
    ]
    y = np.arange(len(top))
    ax.barh(y, top["importance"].values[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9, color="#e8eaf0")
    ax.set_xlabel("Mean |SHAP value|", fontsize=8, color="#7a8099")
    ax.set_title(
        "Global Feature Importance (SHAP)",
        fontsize=10, fontweight="bold", color="#e8eaf0", pad=10,
    )
    ax.tick_params(colors="#7a8099")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f42")
 
    plt.tight_layout()
    return fig
 
 
# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Live Prediction", "Model Insights", "About"])
 
 
# ---- Tab 1: Live Prediction ----
with tab1:
    input_df      = build_input()
    input_imputed = imputer.transform(input_df)
    proba         = model.predict_proba(input_imputed)[0][1]
    prediction    = int(proba >= THRESHOLD)
 
    left, right = st.columns([1, 1.4])
 
    with left:
        st.markdown('<div class="section-label">Risk Gauge</div>', unsafe_allow_html=True)
        gauge_fig = draw_gauge(proba, THRESHOLD)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close()
 
        if prediction == 1:
            st.markdown(f"""
            <div class="result-bad">
                <div class="result-title" style="color:#E24B4A">HIGH RISK ORDER</div>
                <div class="result-subtitle">
                    Probability {proba * 100:.1f}% exceeds threshold {THRESHOLD * 100:.0f}%.<br>
                    This order is likely to result in a bad experience.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-good">
                <div class="result-title" style="color:#1D9E75">LOW RISK ORDER</div>
                <div class="result-subtitle">
                    Probability {proba * 100:.1f}% is below threshold {THRESHOLD * 100:.0f}%.<br>
                    This order is likely to complete successfully.
                </div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown('<div class="section-label" style="margin-top:1rem">Derived Values</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Price/Freight Ratio", f"{price_to_freight:.1f}")
        c2.metric("Avg Item Price",       f"R${avg_price:.0f}")
        c1.metric("Is Late",              "Yes" if is_late else "No")
        c2.metric("Threshold",            f"{THRESHOLD:.2f}")
 
    with right:
        st.markdown('<div class="section-label">Feature Impact (SHAP)</div>', unsafe_allow_html=True)
        shap_fig = draw_shap_waterfall(input_df)
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
            plt.close()
        else:
            st.info("SHAP explanation unavailable.")
 
        st.markdown("""
        <div style="background:#1c2030; border:1px solid #2a2f42; border-radius:8px;
             padding:1rem; font-size:0.82rem; color:#7a8099; margin-top:0.5rem;">
            <b style="color:#e8eaf0">How to read this chart:</b><br>
            Red bars push the prediction toward <b style="color:#E24B4A">bad order</b>.
            Green bars push toward <b style="color:#1D9E75">good order</b>.
            The longer the bar, the stronger the influence.
        </div>
        """, unsafe_allow_html=True)
 
 
# ---- Tab 2: Model Insights ----
with tab2:
    st.markdown('<div class="section-label">Global Feature Importance</div>', unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        imp_fig = draw_global_importance()
        st.pyplot(imp_fig, use_container_width=True)
        plt.close()
 
    with col_b:
        st.markdown("""
        <div style="background:#1c2030; border:1px solid #2a2f42; border-radius:10px; padding:1.5rem;">
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700;
             color:#e8eaf0; margin-bottom:1rem;">Key Findings</div>
        """, unsafe_allow_html=True)
 
        insights = [
            ("Delivery delay",    "Strongest single predictor. Orders 6+ days late have a 76-78% bad rate."),
            ("Category risk",     "Fashion and audio have 21-22% bad rates vs the 11% average."),
            ("Price band",        "R$500+ orders have a 17% bad rate — driven by higher expectations."),
            ("Voucher payments",  "17.5% bad rate vs 12% for credit cards."),
            ("Same state",        "Same-state sellers mean shorter transit and fewer complaints."),
            ("Repeat complainers","Customers with past bad orders are 2x more likely to flag again."),
        ]
        for title, desc in insights:
            st.markdown(f"""
            <div style="margin-bottom:0.9rem; padding-bottom:0.9rem; border-bottom:1px solid #2a2f42;">
                <div style="font-size:0.88rem; font-weight:600; color:#e8eaf0;">{title}</div>
                <div style="font-size:0.82rem; color:#7a8099; margin-top:0.2rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
    st.markdown(
        '<div class="section-label" style="margin-top:1.5rem">Model Performance Summary</div>',
        unsafe_allow_html=True,
    )
    perf_df = pd.DataFrame({
        "Metric":    ["ROC-AUC", "F1 (bad orders)", "Precision (bad)", "Recall (bad)", "Threshold"],
        "Value":     ["~0.89", "~0.62", "~0.58", "~0.68", f"{THRESHOLD:.2f}"],
        "Benchmark": ["0.5 = random", "0 = useless", "—", "—", "Tuned for F1"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
 
 
# ---- Tab 3: About ----
with tab3:
    st.markdown("""
    <div style="max-width:700px;">
    <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
         color:#e8eaf0; margin-bottom:1.2rem;">About This Project</div>
    """, unsafe_allow_html=True)
 
    sections = [
        (
            "Dataset",
            "Brazilian E-Commerce Public Dataset by Olist · 100,000 orders · 2016-2018 · "
            "9 relational tables merged into a single master dataset.",
        ),
        (
            "Problem",
            "Predict whether an order will result in a bad customer experience "
            "(cancellation or review score <= 2). Only ~11% of orders are bad — "
            "a classic imbalanced classification problem.",
        ),
        (
            "Feature Engineering",
            "26 features engineered across 4 groups: time-based (delivery delay, approval wait, season), "
            "ratio-based (price-to-freight), behavioral (customer history, same-state seller), "
            "and encoded (target encoding for category/state, one-hot for payment/season).",
        ),
        (
            "Modeling",
            "XGBoost and LightGBM trained on SMOTE-augmented data. Decision threshold tuned via "
            "F1-score sweep on the held-out test set. SHAP used for explainability.",
        ),
        (
            "Results",
            "ROC-AUC ~0.89 · F1 (bad orders) ~0.62 after threshold tuning · "
            "5-fold cross-validation confirms generalization.",
        ),
        (
            "Stack",
            "Python · Pandas · Scikit-learn · XGBoost · LightGBM · imbalanced-learn · SHAP · Streamlit",
        ),
    ]
 
    for title, body in sections:
        st.markdown(f"""
        <div style="margin-bottom:1.2rem; padding:1.1rem 1.3rem;
             background:#1c2030; border:1px solid #2a2f42; border-radius:8px;">
            <div style="font-size:0.9rem; font-weight:600; color:#e8eaf0; margin-bottom:0.4rem;">{title}</div>
            <div style="font-size:0.85rem; color:#7a8099; line-height:1.6;">{body}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("</div>", unsafe_allow_html=True)
 