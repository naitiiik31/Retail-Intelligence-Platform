import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_products(df):
    cpm = df.groupby(['Customer ID', 'Description'])['Quantity'].sum().unstack(fill_value=0)
    cpm = (cpm > 0).astype(int)
    
    item_sim = cosine_similarity(cpm.T)
    item_sim_df = pd.DataFrame(item_sim, index=cpm.columns, columns=cpm.columns)
    
    return item_sim_df, cpm
