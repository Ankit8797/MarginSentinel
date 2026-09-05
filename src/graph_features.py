import networkx as nx
import pandas as pd
import numpy as np

class GraphFeatureExtractor:
    def __init__(self):
        self.G = nx.Graph()
        self.train_txn_labels = {}
        self.global_train_rto_rate = 0.0
        self.global_train_rto_rate = 0.0
        
    def fit(self, df_train: pd.DataFrame):
        """Builds the historical graph and computes global mean."""
        if len(df_train) > 0:
            self.global_train_rto_rate = df_train['is_rto'].mean()
            
        self.G.clear()
        self.train_txn_labels.clear()
        
        self._add_edges(df_train)
        for _, row in df_train.iterrows():
            self.train_txn_labels[row['transaction_id']] = row['is_rto']
            
        return self

    def _add_edges(self, df: pd.DataFrame):
        """Adds edges for the given dataframe."""
        for _, row in df.iterrows():
            txn = row['transaction_id']
            cust = row['customer_id']
            
            # Connect transaction to customer
            self.G.add_edge(txn, cust)
            
            # Connect customer to devices/VPA
            for col in ['client_canvas_hash', 'app_set_id', 'vpa_handle_hash']:
                val = row[col]
                if pd.notna(val) and val != 'NO_VPA':
                    node_id = f"{col}_{val}"
                    self.G.add_edge(cust, node_id)

    def extract_features(self, txn: str) -> dict:
        """Extracts features for a single transaction (assumes it's already in the graph)."""
        if txn not in self.G:
            return {
                'connected_component_size': 1,
                'neighborhood_rto_rate': self.global_train_rto_rate,
                'max_customers_per_device': 1
            }
            
        comp = nx.node_connected_component(self.G, txn)
        
        # 1. Connected component size (number of customers)
        customers_in_comp = [n for n in comp if str(n).startswith('USR_')]
        comp_size = len(customers_in_comp)
        
        # 2. Neighborhood RTO Rate (Leave-one-out, training labels only)
        train_txns = [n for n in comp if str(n).startswith('pay_') and n in self.train_txn_labels]
        rto_sum = 0
        rto_count = 0
        
        for t in train_txns:
            if t != txn: # Leave-one-out
                rto_sum += self.train_txn_labels[t]
                rto_count += 1
                
        if rto_count > 0:
            nbr_rto_rate = rto_sum / rto_count
        else:
            nbr_rto_rate = self.global_train_rto_rate
            
        # 3. Max customers per device (diversity signal)
        # Find all devices connected to the customer of this transaction
        cust_node = None
        for neighbor in self.G.neighbors(txn):
            if str(neighbor).startswith('USR_'):
                cust_node = neighbor
                break
                
        max_cust_on_device = 1
        if cust_node:
            for neighbor in self.G.neighbors(cust_node):
                if str(neighbor).startswith('client_canvas_hash_') or str(neighbor).startswith('app_set_id_'):
                    # How many customers use this device?
                    cust_count = sum(1 for n in self.G.neighbors(neighbor) if str(n).startswith('USR_'))
                    if cust_count > max_cust_on_device:
                        max_cust_on_device = cust_count

        return {
            'connected_component_size': comp_size,
            'neighborhood_rto_rate': nbr_rto_rate,
            'max_customers_per_device': max_cust_on_device
        }

    def fit_transform_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes training data sequentially. 
        For each row:
        1. Add its edges to graph.
        2. Extract features (simulate real-time arrival).
        3. Add its label to train_txn_labels (so future rows can use it).
        """
        self.fit(df)
        self.G.clear()
        self.train_txn_labels.clear()
        
        features_list = []
        for _, row in df.iterrows():
            txn = row['transaction_id']
            # 1. Add to graph so it can see its neighbors
            self._add_edges(pd.DataFrame([row]))
            # 2. Extract features
            features_list.append(self.extract_features(txn))
            # 3. Save label for future transactions to use
            self.train_txn_labels[txn] = row['is_rto']
            
        features_df = pd.DataFrame(features_list, index=df.index)
        return pd.concat([df, features_df], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes incoming data (test/live) sequentially, appending to the existing graph.
        Does NOT record their labels in train_txn_labels, strictly preventing leakage.
        """
        features_list = []
        for _, row in df.iterrows():
            # 1. Add to graph
            self._add_edges(pd.DataFrame([row]))
            # 2. Extract features
            features_list.append(self.extract_features(row['transaction_id']))
            # No label saving!
            
        features_df = pd.DataFrame(features_list, index=df.index)
        return pd.concat([df, features_df], axis=1)
