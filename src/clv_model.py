import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime as dt

def calculate_clv(df):
    max_date = df['InvoiceDate'].max()
    split_date = max_date - dt.timedelta(days=180)
    
    hist_df = df[df['InvoiceDate'] <= split_date]
    future_df = df[df['InvoiceDate'] > split_date]
    
    snapshot_date = hist_df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    features = hist_df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Total_Price': 'sum'
    }).reset_index()
    features.rename(columns={'InvoiceDate': 'Recency',
                            'Invoice': 'Frequency',
                            'Total_Price': 'Monetary'}, inplace=True)
    
    targets = future_df.groupby('Customer ID').agg({'Total_Price': 'sum'}).reset_index()
    targets.rename(columns={'Total_Price': 'FutureValue'}, inplace=True)
    
    model_df = pd.merge(features, targets, on='Customer ID', how='left').fillna(0)
    
    X = model_df[['Recency', 'Frequency', 'Monetary']]
    y = model_df['FutureValue']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    preds = rf.predict(X)
    
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    
    print(f"CLV - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    model_df['PredictedCLV'] = preds
    model_df['AvgOrderValue'] = model_df['Monetary'] / model_df['Frequency'].replace(0, 1)
    
    return model_df[['Customer ID', 'PredictedCLV', 'Frequency', 'AvgOrderValue', 'Monetary']]
