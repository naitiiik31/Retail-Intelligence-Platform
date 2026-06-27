<div align="center">

# 🛒 Retail Intelligence Platform

### End-to-End Data Science & Machine Learning Platform for Retail Analytics

*From raw transaction data to a live, interactive business intelligence dashboard*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-blue?style=for-the-badge)](#-license)

 [📸 Screenshots](#-screenshots) · [⚙️ Setup](#️-setup--installation) · [🧠 Modules](#-core-modules) · [📬 Contact](#-contact)

</div>

---

## 📌 Overview

**Retail Intelligence Platform** is a full-stack data science project built on the real-world **Online Retail II** dataset (~500K+ transactions from a UK-based online retailer). It simulates how a modern e-commerce business turns raw, messy transactional data into **actionable intelligence** — from understanding customers to forecasting future revenue.

The project covers the **entire data science lifecycle**:

> **Raw Data → Cleaning → Feature Engineering → ML Modeling → Evaluation → Interactive Dashboard**

It's designed to demonstrate practical, production-style data science skills: EDA, unsupervised learning (clustering), supervised learning (regression), recommender systems, time-series forecasting, and dashboard deployment — all in one cohesive pipeline.

---

## ✨ Key Highlights

- 🔍 **Exploratory Data Analysis** on 500K+ real transactions — sales trends, top products, geographic spread
- 🧩 **Customer Segmentation** using K-Means Clustering (Premium / Loyal / At Risk / Lost)
- 💰 **Customer Lifetime Value Prediction** using a Random Forest Regressor
- 🎯 **Recommendation Engine** powered by Item-Based Collaborative Filtering
- 📈 **Demand Forecasting** for daily revenue using historical sales patterns
- 🖥️ **Interactive Streamlit Dashboard** tying every model into one live business tool
- 🧪 Handles real-world challenges: cold-start users, sparse data, seasonality, and catalog drift

---

## 📸 Screenshots


<div align="center">

### 🏠 Home / Overview
<img width="1671" height="831" alt="image" src="https://github.com/user-attachments/assets/89a5d95f-5304-4d83-8d4a-c62944a709b2" />


### 🧩 Customer Segmentation
<img width="1603" height="271" alt="image" src="https://github.com/user-attachments/assets/c47531f9-fb1d-4223-a6a8-cff87e6b5fe3" />


### 💰 CLV Prediction
<img width="1607" height="786" alt="image" src="https://github.com/user-attachments/assets/d17633f9-27e0-4d43-8ee7-f2550b412c92" />
<img width="1608" height="252" alt="image" src="https://github.com/user-attachments/assets/d7ac0a3c-5784-4a57-b3f8-8b3f9512117f" />


### 🎯 Recommendation System
<img width="1605" height="747" alt="image" src="https://github.com/user-attachments/assets/f553c8b8-4bc2-4816-bc15-bac9498a03cb" />


### 📈 Demand Forecast
<img width="1588" height="827" alt="image" src="https://github.com/user-attachments/assets/b99659fb-d71d-4c48-ade9-3ba2deb759e1" />
<img width="1610" height="445" alt="image" src="https://github.com/user-attachments/assets/37f4e4c5-54b7-4526-96f8-51d4104d7d9d" />

</div>


---

## 🧠 Core Modules

| # | Module | What It Does | Technique Used |
|---|--------|---------------|-----------------|
| 1 | **Exploratory Data Analysis** | Uncovers monthly sales trends, best-selling products, and geographic distribution of revenue | Pandas, Matplotlib, Seaborn |
| 2 | **Customer Segmentation** | Groups customers into Premium, Loyal, At Risk, and Lost based on buying behavior |  K-Means Clustering |
| 3 | **CLV Prediction** | Predicts the future revenue a customer is likely to generate | Random Forest Regressor |
| 4 | **Recommendation System** | Suggests products tailored to a customer's purchase history | Item-Based Collaborative Filtering |
| 5 | **Demand Forecasting** | Forecasts total daily revenue for the business | Time-Series Regression |

---

## 📊 Model Performance Metrics

Since this project spans Clustering, Regression, and Recommender Systems, each module uses task-appropriate evaluation metrics rather than generic accuracy scores:

| Module | Task Type | Key Metrics | Notes |
|--------|-----------|--------------|-------|
| **CLV Prediction** | Regression | **MAE:** ~₹595.78 &nbsp;·&nbsp; **RMSE:** ~₹5,234.00 &nbsp;·&nbsp; **R²:** ~0.015 | R² is intentionally low — predicting *exact* individual spend is extremely hard; the model still adds value for relative ranking. |
| **Demand Forecasting** | Regression | **MAE:** ~₹17,357.33 &nbsp;·&nbsp; **RMSE:** ~₹19,456.10 | Measures average error in predicting total daily business revenue. |
| **Recommendation System** | Collaborative Filtering | **Precision@10** (see notebooks) | Standard accuracy doesn't apply to ranking thousands of products — Precision@10 measures relevant hits in the top 10 recommendations. |
| **Customer Segmentation** | Clustering | **Silhouette Score** / **Inertia** (Elbow Method) | Unsupervised task — no ground-truth labels, so cluster separation quality is used instead. |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/Retail-Intelligence-Platform.git
cd Retail-Intelligence-Platform
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the full data pipeline

Cleans the raw data, engineers features, and trains all models in one go:

```bash
python run_pipeline.py
```

> Prefer a step-by-step walkthrough? Run the notebooks individually in `notebooks/`, in order from `01` to `06`.

### 4️⃣ Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Quick stats and high-level business overview |
| 🧩 **Customer Segmentation** | Look up any customer's segment — Premium, Loyal, At Risk, or Lost |
| 💰 **CLV Prediction** | View predicted lifetime value for any customer |
| 🎯 **Recommendation System** | Get top 10 personalized product recommendations |
| 📈 **Demand Forecast** | Compare actual vs. predicted daily revenue over time |

---

## 🗂️ Project Structure

```
Retail-Intelligence-Platform/
│
├── data/
│   ├── raw/                  # Original dataset
│   │   └── online_retail_II.xlsx
│   ├── processed/            # Cleaned and feature-engineered data
│   └── external/
│
│
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── clv_model.py
│   ├── recommendation.py
│   └── forecasting.py
│
├── models/                   # Saved .pkl models
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── reports/
│   └── screenshots/          # Dashboard screenshots (for this README)
├── run_pipeline.py           # Single script to run the entire data pipeline
├── requirements.txt
└── README.md
```

---

## ⚠️ Real-World Challenges Addressed

<details>
<summary><b>1. The "Cold Start" Problem (New Customers)</b></summary>
<br>

**Issue:** The recommendation engine and CLV model rely on a customer's purchase history. Brand-new customers have none, so personalized predictions aren't possible.

**Solution:** New customers fall back to **non-personalized recommendations** — Top 10 Global Best-Sellers / trending items — until enough purchase history accumulates to switch them into the personalized model.
</details>

<details>
<summary><b>2. Data Sparsity</b></summary>
<br>

The customer-product matrix is highly sparse — most customers have purchased only a tiny fraction of available products — which makes similarity computation for recommendations harder and requires careful matrix handling.
</details>

<details>
<summary><b>3. Seasonality & Outliers</b></summary>
<br>

Retail sales spike heavily around November–December. Forecasting models need sufficient historical depth to capture yearly seasonality, while bulk-buying wholesale customers can skew CLV predictions and require outlier handling.
</details>

<details>
<summary><b>4. Changing Product Catalogs</b></summary>
<br>

As products are discontinued or added, the recommendation matrix needs periodic retraining to stay aligned with current inventory.
</details>

---

## 📁 Dataset

**Online Retail II** — Transactional data from a UK-based online retailer (2009–2010).

- 📦 **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- 📊 **Records:** ~500,000+ transactions
- 🧾 **Features:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

---

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (K-Means, Random Forest) |
| **Dashboard** | Streamlit |
| **Model Serialization** | Joblib |

---

## ☁️ Deployment on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `dashboard/app.py`
5. Click **Deploy** 🚀

---

## 🔭 Future Improvements

- [ ] Add deep learning–based recommendation (e.g., neural collaborative filtering)
- [ ] Incorporate external features (holidays, promotions, weather) into forecasting
- [ ] A/B testing framework for recommendation strategies
- [ ] Automated model retraining pipeline (CI/CD with scheduled jobs)
- [ ] Containerize with Docker for easier deployment

---


## 📜 License

This project is for **educational purposes**.

---
<div align="center">
  Made with ❤️ by <a href="https://github.com/naitiiik31">naitiiik31</a>
</div>
</div>
