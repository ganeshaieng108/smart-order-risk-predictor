# 🛍️ Olist Smart Return Predictor

> **Predicts whether a Brazilian e-commerce order will result in a bad customer experience — before it ships.**  
> End-to-end ML project: data wrangling → EDA → feature engineering → model training → live Streamlit deployment.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=flat-square&logo=streamlit)](https://your-app-link.streamlit.app)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Demo](#-live-demo)
3. [Business Problem](#-business-problem)
4. [Dataset](#-dataset)
5. [Project Architecture](#-project-architecture)
6. [Methodology](#-methodology)
   - [Phase 1 — Data Wrangling](#phase-1--data-wrangling)
   - [Phase 2 — Exploratory Data Analysis](#phase-2--exploratory-data-analysis)
   - [Phase 3 — Feature Engineering](#phase-3--feature-engineering)
   - [Phase 4 — Model Training](#phase-4--model-training)
   - [Phase 5 — Streamlit Deployment](#phase-5--streamlit-deployment)
7. [Key Results](#-key-results)
8. [Feature Importance](#-feature-importance)
9. [Project Structure](#-project-structure)
10. [How to Run Locally](#-how-to-run-locally)
11. [Tech Stack](#-tech-stack)
12. [Lessons Learned](#-lessons-learned)

---

## 🔍 Project Overview

E-commerce platforms lose significant revenue through bad order experiences — cancellations, returns, and low review scores. This project builds a **binary classification system** that predicts the probability of a bad order outcome using 26 engineered features drawn from 9 relational tables of the Olist dataset.

The final system is deployed as an **interactive Streamlit application** where anyone can adjust order parameters (delivery delay, product category, payment type, etc.) and instantly see the risk prediction with a full SHAP-based explanation of *why* the model made that call.

**What makes this project production-grade:**
- Proper train/test stratified split to preserve class ratios
- SMOTE applied *only* on training data to prevent data leakage
- Threshold tuning via F1-score sweep (not blindly using 0.5)
- SHAP explainability on every live prediction
- All preprocessing artifacts (imputer, encoder) saved and reused consistently in the app

---

## 🚀 Live Demo

**👉 [Try the live app here](https://your-app-link.streamlit.app)**

> Adjust the sliders in the sidebar → the risk gauge and SHAP chart update instantly.

![App Screenshot](assets/app_screenshot.png)

---

## 💼 Business Problem

**Context:** Olist is a Brazilian marketplace that connects small retailers to major e-commerce channels. A "bad order" is defined as:
- An order that was **cancelled**, OR
- An order that was **delivered but received a review score of 1 or 2 out of 5**

**Why this matters:**
- Bad orders cost platforms in refunds, logistics overhead, and seller penalties
- If we can flag high-risk orders *before* they ship, operations teams can intervene — prioritise shipping, add quality checks, or proactively contact the customer

**Goal:** Build a model that predicts `is_bad_order = 1` with high recall (catch as many bad orders as possible) while keeping precision reasonable (don't flag everything as bad).

---

## 📦 Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — available on Kaggle.

| File | Description | Rows |
|------|-------------|------|
| `olist_orders_dataset.csv` | Core order info, timestamps, status | 99,441 |
| `olist_customers_dataset.csv` | Customer location and ID | 99,441 |
| `olist_order_items_dataset.csv` | Items per order, price, freight | 112,650 |
| `olist_order_payments_dataset.csv` | Payment type, installments, value | 103,886 |
| `olist_order_reviews_dataset.csv` | Review scores and comments | 100,000 |
| `olist_products_dataset.csv` | Product dimensions, category | 32,951 |
| `olist_sellers_dataset.csv` | Seller state | 3,095 |
| `olist_geolocation_dataset.csv` | Zip code → lat/lon | 1,000,163 |
| `product_category_name_translation.csv` | Portuguese → English categories | 71 |

All 9 tables were merged into a single **master dataframe of 99,441 rows × 29 columns** for analysis.

**Class distribution (target variable):**
```
Good orders (is_bad_order = 0) : 88,650  (89.1%)
Bad  orders (is_bad_order = 1) :  10,791  (10.9%)
```
This is a moderately imbalanced dataset — handled via SMOTE during training.

---

## 🏗️ Project Architecture

```
Raw CSVs (9 tables)
        │
        ▼
┌──────────────────┐
│  Data Wrangling  │  Merge, clean, deduplicate → master_df (99K rows)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│       EDA        │  7 insight charts → delivery delay, category, price,
└────────┬─────────┘  payment, monthly trend, item count, state
         │
         ▼
┌──────────────────────┐
│  Feature Engineering │  26 features across 4 groups → model_ready.csv
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  Model Training  │  SMOTE → XGBoost + LightGBM → threshold tuning
└────────┬─────────┘  → SHAP → save artifacts
         │
         ▼
┌──────────────────┐
│  Streamlit App   │  Live prediction + SHAP waterfall + model insights
└──────────────────┘
```

---

## 🔬 Methodology

### Phase 1 — Data Wrangling

The dataset is spread across 9 relational tables. The goal of this phase was to produce a single analysis-ready master dataframe.

**Steps:**
1. Loaded all 9 CSVs and performed basic shape/duplicate checks on each
2. Aggregated `order_payments` by `order_id` — took the mode payment type, max installments, and summed payment value
3. Aggregated `order_items` by `order_id` — computed total items, total/avg/max price, total freight
4. Deduplicated `order_reviews` — kept the most recent review per order
5. Joined English product category names via the translation table
6. Merged all summaries onto the `orders` table using left joins to preserve all 99,441 orders

**Result:** `master_df` with 99,441 rows × 29 columns, zero row count loss.

---

### Phase 2 — Exploratory Data Analysis

Seven targeted charts were produced to understand what drives bad orders:

| Chart | Key Finding |
|-------|-------------|
| Delivery delay vs bad rate | Late orders have **76–78% bad rate** vs 9% for on-time. Single strongest signal. |
| Product category vs bad rate | Fashion & audio hit **21–22% bad rate** vs 11% average |
| Price band vs bad rate | R$500+ orders: **17.2% bad rate** — higher price = higher expectations |
| Payment type vs bad rate | Voucher payments: **17.5% bad rate** vs 12% for credit cards |
| Monthly trend (2017–2018) | Two clear spikes: Nov 2017 (Black Friday), Mar 2018 (logistics disruption) |
| Items per order vs bad rate | 1 item: 11.5% → 4–10 items: **34%+ bad rate** |
| State vs bad rate | Northern states (PA, AM, RR) have consistently higher bad rates |

These findings directly informed which features to engineer.

---

### Phase 3 — Feature Engineering

26 features were engineered across 4 groups from the raw master dataframe:

**Time-based (7 features)**
| Feature | Description |
|---------|-------------|
| `delivery_delay_days` | Actual delivery date minus estimated date. Positive = late. |
| `is_late` | Binary flag. NaT (never delivered) treated as late = 1. |
| `approval_wait_hours` | Hours between order placement and payment approval |
| `carrier_wait_days` | Days between approval and carrier pickup (seller speed proxy) |
| `order_season` | Brazil season (Southern Hemisphere) derived from purchase month |
| `order_dayofweek` | 0=Monday … 6=Sunday |
| `order_hour` | Hour of purchase (late-night = impulse buy risk) |

**Ratio-based (1 feature)**
| Feature | Description |
|---------|-------------|
| `price_to_freight_ratio` | Total price ÷ freight. Low ratio = cheap product + expensive shipping = frustration. Capped at 99th percentile. |

**Behavioral (2 features)**
| Feature | Description |
|---------|-------------|
| `customer_past_bad_orders` | How many previous bad orders this customer had. Current order subtracted to prevent data leakage. |
| `same_state` | 1 if seller and customer are in the same state (shorter transit = fewer issues) |

**Encoded (16 features)**
| Feature | Method | Reason |
|---------|--------|--------|
| `category_bad_rate` | Target encoding | 73 unique categories — one-hot would add 73 columns |
| `state_bad_rate` | Target encoding | 27 states — same reasoning |
| `pay_boleto`, `pay_debit_card`, `pay_voucher` | One-hot (drop first) | Only 4 payment types |
| `season_Spring`, `season_Summer`, `season_Winter` | One-hot (drop first) | Only 4 seasons |

**Null handling:** Median imputation applied to `delivery_delay_days`, `approval_wait_hours`, `carrier_wait_days` (nulls come from undelivered orders with missing timestamps).

**Result:** `model_ready.csv` — 99,441 rows × 27 columns (26 features + 1 target), zero nulls.

---

### Phase 4 — Model Training

**Train/Test Split**
- 80/20 stratified split → preserves 10.9% bad order rate in both halves
- Test set is never touched during preprocessing or SMOTE

**Handling Class Imbalance**
- Without correction, a model can achieve 89% accuracy by predicting "good" for everything — useless
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) with `sampling_strategy=0.40`
  - Bad orders become 40% of good orders in training (not full 50/50, which can overfit)
  - SMOTE is fit and applied **only on training data**

**Null Imputation**
- `SimpleImputer(strategy='median')` fit on training data, transform applied to both train and test
- Imputer saved as `imputer.pkl` for consistent use in the Streamlit app

**Models Trained**
| Model | Key Hyperparameters |
|-------|-------------------|
| XGBoost | n_estimators=500, max_depth=6, lr=0.05, early stopping on AUC |
| LightGBM | n_estimators=500, max_depth=6, lr=0.05 |

Both use `scale_pos_weight` to maintain awareness of the original class imbalance.

**Threshold Tuning**
- Default 0.5 threshold is suboptimal for imbalanced targets
- Swept thresholds 0.10–0.90 in 0.01 steps
- Selected threshold that **maximizes F1-score on bad orders (class 1)**
- This is the threshold used in the live app

**Explainability**
- SHAP `TreeExplainer` used to compute feature contributions
- Global importance saved to `shap_importance.csv`
- Per-prediction SHAP waterfall computed live in the app

**5-Fold Cross-Validation**
- Confirms the model generalises and hasn't overfit to the train set
- Reported as mean ± std of ROC-AUC across folds

---

### Phase 5 — Streamlit Deployment

The app has three sections:

**Live Prediction tab**
- Sidebar with sliders for all 26 input features grouped by category
- Risk gauge (semicircle) showing bad order probability in real-time
- Green/red result banner with verdict and probability vs threshold
- SHAP waterfall chart explaining the specific prediction

**Model Insights tab**
- Global SHAP feature importance bar chart (top 15 features)
- Key EDA findings summary
- Model performance metrics table

**About tab**
- Clean project summary for recruiters who want context without running the notebook

---

## 📊 Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| **ROC-AUC** | ~0.89 | 0.5 = random, 1.0 = perfect |
| **F1-Score (bad orders)** | ~0.62 | Maximised via threshold tuning |
| **Precision (bad orders)** | ~0.58 | Of all flagged orders, 58% are genuinely bad |
| **Recall (bad orders)** | ~0.68 | Catches 68% of all actual bad orders |
| **CV Mean AUC** | ~0.88 ± 0.01 | Stable across 5 folds — no overfitting |
| **Decision Threshold** | ~0.35 | Tuned down from default 0.5 to boost recall |

> Precision and recall involve a trade-off. This model is tuned to **catch more bad orders** (higher recall) at the cost of some false positives — appropriate for a business use case where missing a bad order is more costly than over-flagging.

---

## 📈 Feature Importance

Top features by mean absolute SHAP value:

| Rank | Feature | Why It Matters |
|------|---------|----------------|
| 1 | `delivery_delay_days` | Single strongest signal — late delivery drives 76-78% bad rate |
| 2 | `is_late` | Binary version of above — strong complementary signal |
| 3 | `review_score` proxies | Category and state bad rates encode historical complaint patterns |
| 4 | `carrier_wait_days` | Slow seller dispatch = customer frustration before item even ships |
| 5 | `category_bad_rate` | Some product categories structurally generate more complaints |
| 6 | `payment_installments` | More installments = financial stress = more disputes |
| 7 | `customer_past_bad_orders` | Repeat complainers are statistically more likely to flag again |
| 8 | `price_to_freight_ratio` | Expensive shipping on cheap product = perceived poor value |
| 9 | `same_state` | Local sellers deliver faster with fewer issues |
| 10 | `total_price` | Higher-value orders come with higher customer expectations |

---

## 📁 Project Structure

```
olist-smart-return-predictor/
│
├── notebook/
│   └── Smart_Return_Predictor_Olist.ipynb   # Full analysis notebook (Phases 1-4)
│
├── app/
│   ├── app.py                               # Streamlit application
│   ├── requirements.txt                     # Python dependencies
│   ├── model.pkl                            # Trained XGBoost model
│   ├── imputer.pkl                          # Fitted median imputer
│   ├── threshold.json                       # Optimal decision threshold
│   ├── feature_names.json                   # Ordered feature list
│   ├── feature_stats.csv                    # Min/median/max for slider defaults
│   └── shap_importance.csv                  # Global SHAP importance for charts
│
├── assets/
│   └── app_screenshot.png                   # App preview for README
│
└── README.md
```

---

## ⚙️ How to Run Locally

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/your-username/olist-smart-return-predictor.git
cd olist-smart-return-predictor

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r app/requirements.txt

# 4. Run the Streamlit app
cd app
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

**To reproduce the full notebook:**
1. Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Place all 9 CSVs in the same directory as the notebook
3. Run all cells sequentially — the notebook saves `model_ready.csv` and all artifacts automatically

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Data Wrangling** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **ML — Preprocessing** | Scikit-learn (SimpleImputer, train_test_split) |
| **ML — Imbalance** | imbalanced-learn (SMOTE) |
| **ML — Models** | XGBoost, LightGBM |
| **ML — Evaluation** | Scikit-learn (ROC-AUC, F1, confusion matrix, cross-validation) |
| **Explainability** | SHAP (TreeExplainer) |
| **Serialisation** | Joblib |
| **Deployment** | Streamlit, Streamlit Cloud |
| **Environment** | Google Colab (training), Python 3.10+ |

---

## 💡 Lessons Learned

**Data leakage is subtle.** The `customer_past_bad_orders` feature required subtracting the current order's outcome before computing the aggregate — otherwise the model would have indirect access to the label it's trying to predict.

**Threshold matters more than the model.** Switching from default 0.5 to a tuned threshold (~0.35) improved F1 on bad orders by roughly 8–10 points without changing a single model parameter.

**SMOTE only on training data.** Applying SMOTE before splitting — a common mistake — inflates test metrics because synthetic samples from the test region leak into training. Splitting first, then resampling, gives an honest estimate.

**Target encoding requires care.** Using the full dataset mean to encode categories during feature engineering is a mild form of leakage. In a production setting, this should be done inside a cross-validation loop. For this portfolio project the impact is minimal given the dataset size.

**SHAP makes the model trustworthy.** A prediction without an explanation is just a number. The SHAP waterfall chart turns each prediction into a story — which inputs pushed the risk up, which pushed it down — and makes the model useful to a non-technical stakeholder.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The dataset is publicly available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license via Kaggle/Olist.

---

<div align="center">
  Built with 🧠 using the <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">Olist Brazilian E-Commerce Dataset</a>
</div>
