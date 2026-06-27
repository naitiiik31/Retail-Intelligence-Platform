import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.contains('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['Total_Price'] = df['Quantity'] * df['Price']
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df
