import pandas as pd
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def perform_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Total_Price': 'sum'
    }).reset_index()
    rfm.rename(columns={'InvoiceDate': 'Recency',
                        'Invoice': 'Frequency',
                        'Total_Price': 'Monetary'}, inplace=True)
    
    features = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_monetary = rfm.groupby('Cluster')['Monetary'].mean().sort_values().index.tolist()
    segment_map = {
        cluster_monetary[0]: 'Lost',
        cluster_monetary[1]: 'At Risk',
        cluster_monetary[2]: 'Loyal',
        cluster_monetary[3]: 'Premium'
    }
    rfm['Segment'] = rfm['Cluster'].map(segment_map)
    
    score = silhouette_score(scaled_features, kmeans.labels_)
    print(f"Segmentation - Silhouette Score: {score:.4f}")
    
    return rfm
