import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def forecast_sales(df):
    df['Date'] = df['InvoiceDate'].dt.date
    daily_sales = df.groupby('Date')['Total_Price'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date')
    
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['Month'] = daily_sales['Date'].dt.month
    daily_sales['RollingMean7'] = daily_sales['Total_Price'].rolling(window=7).mean().bfill()
    daily_sales['Lag1'] = daily_sales['Total_Price'].shift(1).bfill()
    
    daily_sales['Target'] = daily_sales['Total_Price'].shift(-1)
    daily_sales = daily_sales.dropna()
    
    train_size = int(len(daily_sales) * 0.8)
    train, test = daily_sales.iloc[:train_size], daily_sales.iloc[train_size:]
    
    features = ['DayOfWeek', 'Month', 'RollingMean7', 'Lag1', 'Total_Price']
    X_train, y_train = train[features], train['Target']
    X_test, y_test = test[features], test['Target']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Forecasting - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    forecast_df = pd.DataFrame({
        'Date': test['Date'],
        'Actual': y_test,
        'Predicted': preds
    })
    
    return forecast_df
