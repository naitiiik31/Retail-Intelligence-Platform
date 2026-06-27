import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path, report_dir):
    df = pd.read_csv(input_path, parse_dates=['InvoiceDate'])
    
    plt.figure(figsize=(10, 6))
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
    plt.title('Top 10 Products by Quantity Sold')
    plt.xlabel('Quantity')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'top_products.png'))
    plt.close()

    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('MonthYear')['TotalPrice'].sum()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
    plt.title('Monthly Sales Revenue')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'monthly_sales.png'))
    plt.close()

if __name__ == "__main__":
    input_file = os.path.join("data", "processed", "cleaned_retail.csv")
    report_directory = "reports"
    run_eda(input_file, report_directory)
