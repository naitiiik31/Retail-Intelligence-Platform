import os
import sys
import joblib

from src.preprocessing import load_and_clean_data
from src.segmentation import perform_rfm
from src.clv_model import calculate_clv
from src.recommendation import recommend_products
from src.forecasting import forecast_sales

def main():
    raw_data = os.path.join("data", "raw", "online_retail_II.xlsx")
    processed_dir = os.path.join("data", "processed")
    models_dir = os.path.join("models")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading and cleaning data...")
    df = load_and_clean_data(raw_data)
    
    print("Performing RFM segmentation...")
    rfm_result = perform_rfm(df)
    rfm_result.to_csv(os.path.join(processed_dir, "customer_segments.csv"), index=False)
    
    print("Calculating CLV...")
    clv_result = calculate_clv(df)
    clv_result.to_csv(os.path.join(processed_dir, "clv_predictions.csv"), index=False)
    
    print("Generating recommendations...")
    item_sim, cpm = recommend_products(df)
    joblib.dump(item_sim, os.path.join(models_dir, "recommendation_model.pkl"))
    joblib.dump(cpm, os.path.join(models_dir, "customer_product_matrix.pkl"))
    
    print("Forecasting sales...")
    forecast_result = forecast_sales(df)
    forecast_result.to_csv(os.path.join(processed_dir, "forecast_results.csv"), index=False)
    
    print("Pipeline run successfully.")

if __name__ == "__main__":
    main()
