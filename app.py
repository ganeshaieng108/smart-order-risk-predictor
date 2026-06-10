import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Order Risk Predictor", layout="wide")

# load model and supporting files
@st.cache_resource
def load_artifacts():
    model    = joblib.load("model.pkl")
    with open("threshold.json") as f:
        thresh_data = json.load(f)
    with open("feature_names.json") as f:
        features = json.load(f)
    stats    = pd.read_csv("feature_stats.csv", index_col=0)
    shap_imp = pd.read_csv("shap_importance.csv")
    return model, thresh_data, features, stats, shap_imp

try:
    model, thresh_data, FEATURE_COLS, feat_stats, shap_imp = load_artifacts()
    THRESHOLD = thresh_data["threshold"]
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# helper to get median value for slider defaults
def med(col):
    try:
        return float(feat_stats.loc[col, "median"])
    except Exception:
        return 0.0

# page title
st.title("Smart Order Risk Predictor")
st.caption("Predicts whether an order will result in a bad customer experience.")
st.divider()

# sidebar inputs
st.sidebar.header("Order Details")

st.sidebar.subheader("Delivery")
delivery_delay_days = st.sidebar.slider("Delivery Delay (days)", -30, 60, int(med("delivery_delay_days")))
is_late             = int(delivery_delay_days > 0)
approval_wait_hours = st.sidebar.slider("Approval Wait (hours)", 0.0, 72.0, float(med("approval_wait_hours")), 0.5)
carrier_wait_days   = st.sidebar.slider("Carrier Wait (days)", 0, 30, int(med("carrier_wait_days")))

st.sidebar.subheader("Order")
total_items          = st.sidebar.slider("Number of Items", 1, 20, max(1, int(med("total_items"))))
total_price          = st.sidebar.slider("Total Price (R$)", 0.0, 5000.0, float(med("total_price")), 10.0)
total_freight        = st.sidebar.slider("Total Freight (R$)", 0.0, 500.0, float(med("total_freight")), 1.0)
payment_installments = st.sidebar.slider("Payment Installments", 1, 24, int(med("payment_installments")))
payment_type         = st.sidebar.selectbox("Payment Type", ["credit_card", "boleto", "debit_card", "voucher"])
order_dayofweek      = st.sidebar.selectbox("Order Day", list(range(7)),
                        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
order_hour           = st.sidebar.slider("Order Hour", 0, 23, int(med("order_hour")))

st.sidebar.subheader("Product")
product_weight_g    = st.sidebar.slider("Weight (g)", 0, 40000, int(med("product_weight_g")), 100)
product_photos_qty  = st.sidebar.slider("Number of Photos", 1, 20, int(med("product_photos_qty")))
product_desc_length = st.sidebar.slider("Description Length", 0, 3000, int(med("product_description_lenght")), 50)
category_bad_rate   = st.sidebar.slider("Category Risk Rate", 0.0, 0.30, float(med("category_bad_rate")), 0.01)

st.sidebar.subheader("Customer / Seller")
state_bad_rate    = st.sidebar.slider("State Risk Rate", 0.0, 0.20, float(med("state_bad_rate")), 0.005)
same_state        = st.sidebar.checkbox("Seller in same state as customer?")
customer_past_bad = st.sidebar.slider("Customer Past Bad Orders", 0, 10, 0)
season            = st.sidebar.selectbox("Season (Brazil)", ["Summer", "Autumn", "Winter", "Spring"])

# compute derived fields
avg_price            = total_price / max(total_items, 1)
max_price            = total_price
payment_value        = total_price
price_to_freight     = round(total_price / (total_freight + 1e-5), 4)
cap                  = float(feat_stats.loc["price_to_freight_ratio", "max"]) if "price_to_freight_ratio" in feat_stats.index else 200
price_to_freight     = min(price_to_freight, cap)

# build input dataframe
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

# run prediction
input_df   = build_input()
proba      = model.predict_proba(input_df)[0][1]
prediction = int(proba >= THRESHOLD)

tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Importance", "About"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score")
        # color based on prediction
        color = "red" if prediction == 1 else "green"
        label = "HIGH RISK" if prediction == 1 else "LOW RISK"
        st.markdown(f"<h1 style='color:{color}'>{proba*100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"**{label}** — threshold is {THRESHOLD:.2f}")

        if prediction == 1:
            st.error("This order is likely to result in a bad customer experience.")
        else:
            st.success("This order is likely to complete without issues.")

        st.divider()
        st.subheader("Derived Values")
        c1, c2 = st.columns(2)
        c1.metric("Price / Freight Ratio", f"{price_to_freight:.1f}")
        c2.metric("Avg Item Price", f"R${avg_price:.0f}")
        c1.metric("Is Late", "Yes" if is_late else "No")
        c2.metric("Threshold", f"{THRESHOLD:.2f}")

    with col2:
        st.subheader("Feature Impact (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            sv       = shap_vals[0]
            top_idx  = np.argsort(np.abs(sv))[-12:][::-1]
            top_feat = np.array(FEATURE_COLS)[top_idx]
            top_val  = sv[top_idx]

            fig, ax = plt.subplots(figsize=(7, 5))
            colors  = ["#E24B4A" if v > 0 else "#1D9E75" for v in top_val]
            ax.barh(range(len(top_feat)), top_val[::-1], color=colors[::-1])
            ax.set_yticks(range(len(top_feat)))
            ax.set_yticklabels(top_feat[::-1], fontsize=9)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("SHAP value (red = increases risk, green = decreases risk)")
            ax.set_title("Why this prediction was made")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception:
            st.info("SHAP explanation could not be generated.")

with tab2:
    st.subheader("Global Feature Importance")
    top = shap_imp.head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top["importance"].values[::-1], color="#534AB7")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Most important features across all predictions")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Model Performance")
    perf = pd.DataFrame({
        "Metric":    ["ROC-AUC", "F1 (bad orders)", "Precision", "Recall", "Threshold"],
        "Value":     ["0.75", "0.41", "0.69", "0.29", f"{THRESHOLD:.2f}"],
        "Notes":     ["0.5 = random baseline", "higher = better balance", "of flagged bad orders", "of all bad orders caught", "tuned for best F1"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("About This Project")
    st.markdown("""
**Dataset**
Brazilian E-Commerce Public Dataset by Olist. 100,000 orders from 2016 to 2018 across 9 relational tables.

**Problem**
Predict whether an order will result in a bad experience — defined as a cancellation or a review score of 1 or 2.
Around 13% of orders are bad, making this an imbalanced classification problem.

**Features**
28 features across four groups: delivery timing, order value, product attributes, and customer/seller behaviour.

**Model**
XGBoost trained on SMOTE-balanced data. Decision threshold tuned by sweeping F1 scores on the held-out test set.
SHAP values used to explain individual predictions.

**Stack**
Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Streamlit
    """)
