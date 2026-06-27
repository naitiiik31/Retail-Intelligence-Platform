import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Retail Intelligence Platform", page_icon="🛒", layout="wide")

def load_data():
    data = {}
    segments_path = os.path.join(DATA_DIR, "customer_segments.csv")
    if os.path.exists(segments_path):
        data["segments"] = pd.read_csv(segments_path)
    clv_path = os.path.join(DATA_DIR, "clv_predictions.csv")
    if os.path.exists(clv_path):
        data["clv"] = pd.read_csv(clv_path)
    forecast_path = os.path.join(DATA_DIR, "forecast_results.csv")
    if os.path.exists(forecast_path):
        data["forecast"] = pd.read_csv(forecast_path, parse_dates=["Date"])
    return data

def load_models():
    models = {}
    rec_model_path = os.path.join(MODEL_DIR, "recommendation_model.pkl")
    if os.path.exists(rec_model_path):
        models["item_similarity"] = joblib.load(rec_model_path)
    cpm_path = os.path.join(MODEL_DIR, "customer_product_matrix.pkl")
    if os.path.exists(cpm_path):
        models["customer_product_matrix"] = joblib.load(cpm_path)
    return models

data = load_data()
models = load_models()

page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Customer Segmentation", "CLV Prediction", "Recommendation System", "Demand Forecast"]
)

if page == "Home":
    st.title("🛒 Retail Intelligence Platform")
    st.markdown("---")
    st.markdown(
        """
        Welcome to the **Retail Intelligence Platform** — an end-to-end data science project
        built on the Online Retail II dataset.

        ### What This Platform Offers

        | Module | Description |
        |--------|-------------|
        | **Customer Segmentation** | RFM-based clustering into Premium, Loyal, At Risk, and Lost segments |
        | **CLV Prediction** | Predict future customer lifetime value using Random Forest |
        | **Recommendation System** | Item-based collaborative filtering for personalized product suggestions |
        | **Demand Forecast** | Daily revenue forecasting using lag features and Random Forest |

        Use the **sidebar** to navigate between pages.
        """
    )
    if "segments" in data:
        st.markdown("### Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(data['segments']):,}")
        col2.metric("Segments", data["segments"]["Segment"].nunique())
        if "clv" in data:
            col3.metric("Avg CLV", f"£{data['clv']['Monetary'].mean():,.0f}")
        if "forecast" in data:
            col4.metric("Forecast Days", len(data["forecast"]))

elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("---")
    if "segments" not in data:
        st.warning("Run pipeline script first to generate segment data.")
    else:
        segments_df = data["segments"]
        st.markdown("### Segment Distribution")
        segment_counts = segments_df["Segment"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(segment_counts, labels=segment_counts.index, autopct="%1.1f%%",
               colors=["#4CAF50", "#2196F3", "#FF9800", "#F44336"], startangle=140)
        ax.set_title("Customer Segments")
        st.pyplot(fig)
        st.markdown("### Look Up a Customer")
        valid_ids = sorted(segments_df["Customer ID"].unique())
        customer_id = st.selectbox("Select Customer ID", options=valid_ids)
        if customer_id:
            match = segments_df[segments_df["Customer ID"] == customer_id]
            if len(match) > 0:
                row = match.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Segment", row["Segment"])
                col2.metric("Recency", f"{row['Recency']} days")
                col3.metric("Frequency", int(row["Frequency"]))
                col4.metric("Monetary", f"£{row['Monetary']:,.2f}")
            else:
                st.error(f"Customer {customer_id} not found.")

elif page == "CLV Prediction":
    st.title("💰 Customer Lifetime Value Prediction")
    st.markdown("---")
    if "clv" not in data:
        st.warning("Run pipeline script first to generate CLV predictions.")
    else:
        clv_df = data["clv"]
        st.markdown("### CLV Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        clv_capped = clv_df["PredictedCLV"][clv_df["PredictedCLV"] < clv_df["PredictedCLV"].quantile(0.95)]
        ax.hist(clv_capped, bins=50, color="#673AB7", edgecolor="black", alpha=0.8)
        ax.set_title("Predicted CLV Distribution (< 95th Percentile)")
        ax.set_xlabel("Predicted CLV (£)")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)
        st.markdown("### Look Up Customer CLV")
        valid_ids = sorted(clv_df["Customer ID"].unique())
        customer_id = st.selectbox("Select Customer ID", options=valid_ids, key="clv_input")
        if customer_id:
            match = clv_df[clv_df["Customer ID"] == customer_id]
            if len(match) > 0:
                row = match.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted CLV", f"£{row['PredictedCLV']:,.2f}")
                col2.metric("Frequency", int(row["Frequency"]))
                col3.metric("Avg Order Value", f"£{row['AvgOrderValue']:,.2f}")
            else:
                st.error(f"Customer {customer_id} not found.")

elif page == "Recommendation System":
    st.title("🎯 Product Recommendations")
    st.markdown("---")
    if "item_similarity" not in models or "customer_product_matrix" not in models:
        st.warning("Run pipeline script first to generate recommendation models.")
    else:
        item_sim = models["item_similarity"]
        cpm = models["customer_product_matrix"]
        st.markdown("### Get Personalized Recommendations")
        valid_ids = sorted(cpm.index.tolist())
        customer_id = st.selectbox("Select Customer ID", options=valid_ids, key="rec_input")
        if customer_id:
            if customer_id not in cpm.index:
                st.error(f"Customer {customer_id} not found.")
            else:
                customer_purchases = cpm.loc[customer_id]
                purchased_items = customer_purchases[customer_purchases > 0].index.tolist()
                scores = pd.Series(dtype=float)
                for item in purchased_items:
                    if item in item_sim.columns:
                        similar_items = item_sim[item]
                        scores = scores.add(similar_items, fill_value=0)
                scores = scores.drop(labels=purchased_items, errors="ignore")
                recommended = scores.sort_values(ascending=False).head(10)
                st.markdown(f"### Top 10 Recommendations for Customer {customer_id}")
                rec_df = pd.DataFrame({
                    "Rank": range(1, len(recommended) + 1),
                    "Product": recommended.index,
                    "Score": recommended.values
                })
                st.table(rec_df)

elif page == "Demand Forecast":
    st.title("📈 Demand Forecast")
    st.markdown("---")
    if "forecast" not in data:
        st.warning("Run pipeline script first to generate forecast data.")
    else:
        forecast_df = data["forecast"]
        st.markdown("### Future Sales Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(forecast_df["Date"], forecast_df["Actual"], label="Actual", marker="o",
                color="#1976D2", linewidth=2)
        ax.plot(forecast_df["Date"], forecast_df["Predicted"], label="Predicted", marker="s",
                color="#E91E63", linewidth=2, linestyle="--")
        ax.set_title("Demand Forecast — Actual vs Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue (£)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("### Forecast Table")
        display_df = forecast_df.copy()
        display_df["Actual"] = display_df["Actual"].round(2)
        display_df["Predicted"] = display_df["Predicted"].round(2)
        st.dataframe(display_df, use_container_width=True)
