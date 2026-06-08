# Olist Smart Return Predictor

Machine learning system that predicts whether an e-commerce order is likely to result in a poor customer experience before shipment.

## Project Overview

This project uses the Olist Brazilian E-Commerce dataset to build a binary classification model that predicts bad orders based on customer, payment, delivery, and product-related information.

Bad Order Definition:
- Cancelled order
- Delivered order with review score ≤ 2

## Business Objective

Identify high-risk orders early so operational teams can take preventive actions such as:
- Shipping prioritization
- Additional quality checks
- Customer communication

## Dataset

Source: Olist Brazilian E-Commerce Dataset (Kaggle)

Tables Used:
- Orders
- Customers
- Order Items
- Payments
- Reviews
- Products
- Sellers
- Geolocation
- Category Translation

Final Dataset:
- Rows: 99,441
- Features: 26
- Target: is_bad_order

Class Distribution:
- Good Orders: 89.1%
- Bad Orders: 10.9%

## Methodology

### Data Preparation

- Merged 9 relational tables
- Aggregated payment information
- Aggregated item-level information
- Removed duplicate reviews
- Added translated product categories
- Created master dataset

### Exploratory Data Analysis

Analysis focused on:

- Delivery delays
- Product categories
- Order value
- Payment methods
- Monthly trends
- Item counts
- Geographic patterns

### Feature Engineering

#### Time Features

- delivery_delay_days
- is_late
- approval_wait_hours
- carrier_wait_days
- order_season
- order_dayofweek
- order_hour

#### Behavioral Features

- customer_past_bad_orders
- same_state

#### Ratio Features

- price_to_freight_ratio

#### Encoded Features

- category_bad_rate
- state_bad_rate
- Payment type encoding
- Seasonal encoding

### Model Training

Train/Test Split:
- 80/20 Stratified

Class Imbalance Handling:
- SMOTE (training data only)

Models:
- XGBoost
- LightGBM

Evaluation:
- ROC-AUC
- Precision
- Recall
- F1 Score
- Cross Validation

Threshold Optimization:
- F1-based threshold selection

Explainability:
- SHAP

## Results

| Metric | Value |
|----------|----------|
| ROC-AUC | ~0.89 |
| F1 Score | ~0.62 |
| Precision | ~0.58 |
| Recall | ~0.68 |
| CV AUC | ~0.88 ± 0.01 |
| Threshold | ~0.35 |

## Project Structure

```text
olist-smart-return-predictor/
│
├── notebook/
├── app/
├── assets/
└── README.md