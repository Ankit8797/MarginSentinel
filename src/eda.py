import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path

# Add project root to sys.path so config can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATASET_PATH, EDA_REPORTS_DIR

def get_population(row):
    if row['is_fraud_ring'] == 1:
        if str(row['ring_id']).startswith('RING_CAMO_'):
            return 'Camouflaged Ring'
        return 'Obvious Ring'
    else:
        if row['customer_order_history_count'] >= 10:
            return 'Power User'
        return 'Legit'

def run_eda():
    os.makedirs(EDA_REPORTS_DIR, exist_ok=True)
    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    df['population'] = df.apply(get_population, axis=1)

    print("Generating class balance plot...")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='population', y='is_rto', estimator=np.mean)
    plt.title('RTO Rate by Population Segment')
    plt.ylabel('RTO Rate')
    plt.savefig(EDA_REPORTS_DIR / 'class_balance.png')
    plt.close()

    print("Generating RTO rate by payment method and category...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='payment_method', y='is_rto', hue='population', estimator=np.mean)
    plt.title('RTO Rate by Payment Method and Population')
    plt.savefig(EDA_REPORTS_DIR / 'rto_by_payment.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='category_volatility', y='is_rto', hue='population', estimator=np.mean)
    plt.title('RTO Rate by Category Volatility and Population')
    plt.savefig(EDA_REPORTS_DIR / 'rto_by_category.png')
    plt.close()

    print("Generating distribution plots...")
    numeric_features = ['address_quality_score', 'checkout_velocity_seconds', 'cart_value_inr']
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x=feature, hue='population', common_norm=False, fill=True)
        plt.title(f'Distribution of {feature} by Population')
        plt.savefig(EDA_REPORTS_DIR / f'dist_{feature}.png')
        plt.close()

    print("Generating correlation heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numeric Features')
    plt.savefig(EDA_REPORTS_DIR / 'correlation_heatmap.png')
    plt.close()

    print("Computing connected-component size distribution (informal)...")
    G = nx.Graph()
    for _, row in df.iterrows():
        customer = row['customer_id']
        G.add_node(customer)
        for identifier in ['client_canvas_hash', 'app_set_id', 'vpa_handle_hash']:
            val = row[identifier]
            if pd.notna(val) and val != 'None':
                G.add_edge(customer, val)
    
    components = list(nx.connected_components(G))
    component_sizes = [len([n for n in comp if str(n).startswith('USR_')]) for comp in components]
    component_sizes = [s for s in component_sizes if s > 0]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(component_sizes, bins=30, kde=False)
    plt.yscale('log')
    plt.title('Distribution of Connected Component Sizes (Customer Nodes)')
    plt.xlabel('Number of Customers in Component')
    plt.ylabel('Count (Log Scale)')
    plt.savefig(EDA_REPORTS_DIR / 'connected_components_dist.png')
    plt.close()

    print("Checking for raw feature AUC leakage...")
    for col in numeric_cols:
        if col not in ['is_rto', 'is_fraud_ring']:
            try:
                # Handle NaNs if any
                clean_df = df.dropna(subset=[col, 'is_rto'])
                auc = roc_auc_score(clean_df['is_rto'], clean_df[col])
                auc = max(auc, 1 - auc)
                if auc > 0.95:
                    print(f"WARNING: Raw feature '{col}' achieves {auc:.4f} AUC against is_rto. Needs more camouflage!")
            except Exception as e:
                pass

    print(f"EDA complete. Plots saved to {EDA_REPORTS_DIR}")

if __name__ == '__main__':
    run_eda()
