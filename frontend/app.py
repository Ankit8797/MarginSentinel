import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import sys
import shap
from pathlib import Path
import mlflow
import networkx as nx

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing import load_and_preprocess
from src.graph_features import GraphFeatureExtractor
from config import CATEGORICAL_FEATURES, THRESHOLD_PATH, FP_COST, FN_COST, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.eda import get_population

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Merchant Risk Dashboard", layout="wide")

@st.cache_resource
def load_app_data():
    # Load data, fit graph extractor, load model
    df_train, df_val, df_test = load_and_preprocess(force_resplit=False)
    
    extractor = GraphFeatureExtractor()
    extractor.fit(df_train)
    
    # Load Model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    latest_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{latest_run_id}/catboost_model"
    model = mlflow.catboost.load_model(model_uri)
    
    # Pre-extract test features for performance tab
    test_feat = extractor.transform(df_test)
    test_feat['population'] = test_feat.apply(get_population, axis=1)
    
    return df_test, extractor, model, test_feat

df_test, extractor, model, test_feat = load_app_data()

with open(THRESHOLD_PATH, 'r') as f:
    optimal_threshold = json.load(f)['optimal_threshold']

st.title("🛡️ Merchant Risk Dashboard (Graph-Powered Scorer)")

tab1, tab2, tab3 = st.tabs(["Live Scorer", "Model Performance", "Merchant Impact"])

with tab1:
    st.sidebar.header("Simulate Incoming Order")
    sim_mode = st.sidebar.radio("Simulation Mode", ["Pick from Test Set", "Custom Input"])
    
    if sim_mode == "Pick from Test Set":
        txn_id = st.sidebar.selectbox("Select Transaction ID", df_test['transaction_id'].head(50))
        row = df_test[df_test['transaction_id'] == txn_id].iloc[0]
        
        payload = {
            "transaction_id": str(row['transaction_id']),
            "customer_id": str(row['customer_id']),
            "client_canvas_hash": str(row['client_canvas_hash']),
            "app_set_id": str(row['app_set_id']),
            "vpa_handle_hash": None if pd.isna(row['vpa_handle_hash']) or row['vpa_handle_hash'] == 'NO_VPA' else str(row['vpa_handle_hash']),
            "delivery_pincode": str(row['delivery_pincode']),
            "address_quality_score": float(row['address_quality_score']),
            "payment_method": str(row['payment_method']),
            "cart_value_inr": float(row['cart_value_inr']),
            "category_volatility": str(row['category_volatility']),
            "checkout_velocity_seconds": float(row['checkout_velocity_seconds']),
            "is_discount_applied": int(row['is_discount_applied']),
            "customer_order_history_count": int(row['customer_order_history_count'])
        }
    else:
        payload = {
            "transaction_id": st.sidebar.text_input("Transaction ID", "pay_new_001"),
            "customer_id": st.sidebar.text_input("Customer ID", "USR_new_001"),
            "client_canvas_hash": st.sidebar.text_input("Device Canvas Hash", "hash_abc"),
            "app_set_id": st.sidebar.text_input("App Set ID", "app_abc"),
            "vpa_handle_hash": st.sidebar.text_input("VPA Handle", "user@upi"),
            "delivery_pincode": st.sidebar.text_input("Pincode", "110001"),
            "address_quality_score": st.sidebar.slider("Address Quality", 1.0, 5.0, 4.0),
            "payment_method": st.sidebar.selectbox("Payment Method", ["COD", "UPI", "CARD"]),
            "cart_value_inr": st.sidebar.number_input("Cart Value (INR)", 100, 10000, 2000),
            "category_volatility": st.sidebar.selectbox("Category", ["High (Apparel)", "Medium (Electronics)", "Low (Groceries)"]),
            "checkout_velocity_seconds": st.sidebar.slider("Checkout Velocity (sec)", 1.0, 300.0, 45.0),
            "is_discount_applied": st.sidebar.selectbox("Discount Applied", [1, 0]),
            "customer_order_history_count": st.sidebar.number_input("Order History Count", 0, 100, 5)
        }
        
    st.sidebar.markdown("---")
    graph_features_on = st.sidebar.toggle("Graph Features (Ablation Test)", value=True)
    
    if st.sidebar.button("Analyze Transaction", use_container_width=True, type="primary"):
        # We need to compute features. If graph_features_on is False, we pass a dummy.
        eval_payload = payload.copy()
        if not graph_features_on:
            eval_payload['customer_id'] = "ABLATION_CUST"
            eval_payload['client_canvas_hash'] = "ABLATION_HASH1"
            eval_payload['app_set_id'] = "ABLATION_HASH2"
            eval_payload['vpa_handle_hash'] = "ABLATION_HASH3"
            
        try:
            resp = requests.post(API_URL, json=eval_payload)
            if resp.status_code == 200:
                data = resp.json()
                risk = data['risk_score']
                action = data['defense_action']
                
                # Colors
                if action == "BLOCK_TRANSACTION":
                    color = "#F63366" # matches primary theme
                elif action == "REQUIRE_UPI_PREAUTH":
                    color = "orange"
                else:
                    color = "green"
                    
                col_score, col_graph = st.columns([1, 2])
                with col_score:
                    # Plotly Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Score", 'font': {'size': 24, 'color': "white"}},
                        number = {'font': {'color': "white"}, 'valueformat': '.3f'},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': color},
                            'bgcolor': "#1E1E1E",
                            'steps': [
                                {'range': [0, optimal_threshold*0.7], 'color': "rgba(0,255,0,0.1)"},
                                {'range': [optimal_threshold*0.7, optimal_threshold], 'color': "rgba(255,165,0,0.1)"},
                                {'range': [optimal_threshold, 1], 'color': "rgba(255,0,0,0.1)"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': optimal_threshold
                            }
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="#121212", font={'color': "white"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"<div style='text-align: center; background-color: {color}; padding: 15px; border-radius: 8px; color: white; font-size: 1.2rem; font-weight: bold;'>ACTION: {action}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("Decision Drivers (SHAP)")
                    # Calculate SHAP locally for plotting
                    row_df = pd.DataFrame([eval_payload])
                    row_df['vpa_handle_hash'] = row_df['vpa_handle_hash'].fillna("NO_VPA")
                    feat_df = extractor.transform(row_df)
                    features = ['address_quality_score', 'cart_value_inr', 'checkout_velocity_seconds', 'is_discount_applied', 'customer_order_history_count', 'connected_component_size', 'neighborhood_rto_rate', 'max_customers_per_device'] + CATEGORICAL_FEATURES
                    X_row = feat_df[features]
                    explainer = shap.TreeExplainer(model)
                    shap_val = explainer(X_row)
                    
                    # Bar chart for SHAP
                    shap_df = pd.DataFrame({
                        'Feature': features,
                        'Impact': shap_val.values[0]
                    }).sort_values('Impact', key=abs, ascending=False).head(8)
                    
                    fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                                     color='Impact', color_continuous_scale='RdBu_r', title="Top Influential Features")
                    fig_shap.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#121212", plot_bgcolor="#1E1E1E", font={'color': "white"})
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                with col_graph:
                    st.subheader("Live Entity Resolution Graph")
                    net = Network(height="650px", width="100%", bgcolor="#1E1E1E", font_color="white")
                    
                    # Legend Nodes (floating, unconnected)
                    net.add_node("LEGEND_CUST", label="Customer", color="#F63366", shape="box", x=-300, y=-300, physics=False)
                    net.add_node("LEGEND_DEV", label="Device/AppSet", color="#3366F6", shape="box", x=-150, y=-300, physics=False)
                    net.add_node("LEGEND_VPA", label="VPA Handle", color="#F6A333", shape="box", x=0, y=-300, physics=False)
                    
                    actual_cust = eval_payload['customer_id']
                    net.add_node(actual_cust, label=f"Cust: {actual_cust}", color="#F63366", size=25)
                    
                    net.add_node(eval_payload['client_canvas_hash'], label="Canvas", color="#3366F6")
                    net.add_edge(actual_cust, eval_payload['client_canvas_hash'])
                    
                    net.add_node(eval_payload['app_set_id'], label="AppSet", color="#3366F6")
                    net.add_edge(actual_cust, eval_payload['app_set_id'])
                    
                    if eval_payload['vpa_handle_hash'] and eval_payload['vpa_handle_hash'] != "NO_VPA":
                        net.add_node(eval_payload['vpa_handle_hash'], label="VPA", color="#F6A333")
                        net.add_edge(actual_cust, eval_payload['vpa_handle_hash'])
                        
                    # Also plot the connected component if it's large (to show ring visually)
                    if graph_features_on and feat_df['connected_component_size'].iloc[0] > 1:
                        # Find all neighbors in base graph
                        comp = list(nx.node_connected_component(extractor.G, actual_cust))[:50] # limit rendering
                        for n in comp:
                            if str(n).startswith('USR_') and n != actual_cust:
                                net.add_node(n, label="Other Cust", color="#F63366", size=15)
                            elif str(n).startswith('client_canvas_hash_'):
                                net.add_node(n.replace('client_canvas_hash_', ''), label="Canvas", color="#3366F6")
                            elif str(n).startswith('app_set_id_'):
                                net.add_node(n.replace('app_set_id_', ''), label="AppSet", color="#3366F6")
                            elif str(n).startswith('vpa_handle_hash_'):
                                net.add_node(n.replace('vpa_handle_hash_', ''), label="VPA", color="#F6A333")
                        
                        # Add edges for component
                        subgraph = extractor.G.subgraph(comp)
                        for edge in subgraph.edges():
                            # clean prefixes for visualization
                            u = str(edge[0]).replace('client_canvas_hash_', '').replace('app_set_id_', '').replace('vpa_handle_hash_', '')
                            v = str(edge[1]).replace('client_canvas_hash_', '').replace('app_set_id_', '').replace('vpa_handle_hash_', '')
                            if u in [node['id'] for node in net.nodes] and v in [node['id'] for node in net.nodes]:
                                net.add_edge(u, v)

                    net.save_graph("graph.html")
                    HtmlFile = open("graph.html", 'r', encoding='utf-8')
                    components.html(HtmlFile.read(), height=700)
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")

with tab2:
    st.header("Model Performance & Diagnostics")
    
    # Calculate base metrics
    features = ['address_quality_score', 'cart_value_inr', 'checkout_velocity_seconds', 'is_discount_applied', 'customer_order_history_count', 'connected_component_size', 'neighborhood_rto_rate', 'max_customers_per_device'] + CATEGORICAL_FEATURES
    X_test = test_feat[features]
    y_test = test_feat['is_fraud_ring']
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    col1, col2, col3, col4 = st.columns(4)
    from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score
    y_pred_opt = (y_proba >= optimal_threshold).astype(int)
    
    col1.metric("Precision", f"{precision_score(y_test, y_pred_opt):.3f}")
    col2.metric("Recall", f"{recall_score(y_test, y_pred_opt):.3f}")
    col3.metric("F1 Score", f"{f1_score(y_test, y_pred_opt):.3f}")
    col4.metric("PR-AUC", f"{average_precision_score(y_test, y_proba):.3f}")
    
    st.markdown("---")
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Cost Tradeoff vs. Decision Threshold")
        live_th = st.slider("Simulate Live Threshold Override", 0.0, 1.0, float(optimal_threshold), 0.01)
        
        def get_cost_curve(y_true, y_prob):
            ths = np.linspace(0.01, 0.99, 50)
            fp_costs, fn_costs, total_costs = [], [], []
            for th in ths:
                yp = (y_prob >= th).astype(int)
                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
                fp_costs.append(fp * FP_COST)
                fn_costs.append(fn * FN_COST)
                total_costs.append((fp * FP_COST) + (fn * FN_COST))
            return pd.DataFrame({'Threshold': ths, 'False Positive Cost': fp_costs, 'False Negative Cost': fn_costs, 'Total Financial Cost': total_costs})
            
        cost_df = get_cost_curve(y_test, y_proba)
        
        fig_cost = px.line(cost_df, x='Threshold', y=['False Positive Cost', 'False Negative Cost', 'Total Financial Cost'], 
                           title="Optimization Curve", color_discrete_sequence=["#3366F6", "#F6A333", "#F63366"])
        fig_cost.add_vline(x=live_th, line_dash="dash", line_color="white", annotation_text=f"Selected: {live_th}")
        fig_cost.update_layout(paper_bgcolor="#121212", plot_bgcolor="#1E1E1E", font={'color': "white"}, height=400)
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_table:
        st.subheader("Segment Breakdown")
        st.caption(f"At selected threshold: {live_th}")
        y_pred_live = (y_proba >= live_th).astype(int)
        
        seg_data = []
        for pop in test_feat['population'].unique():
            mask = test_feat['population'] == pop
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_test[mask], y_pred_live[mask], labels=[0,1]).ravel()
            cost = (fp * FP_COST) + (fn * FN_COST)
            seg_data.append({"Segment": pop, "Blocked": tp+fp, "Cost (INR)": cost})
            
        seg_df = pd.DataFrame(seg_data).sort_values("Cost (INR)", ascending=False)
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("Financial Merchant Impact")
    
    # Baseline: Do nothing -> All true rings ship successfully and RTO (causing FN_COST per package)
    total_rto_packages = y_test.sum()
    do_nothing_cost = total_rto_packages * FN_COST
    
    # Model: Total Cost at chosen threshold
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt, labels=[0,1]).ravel()
    model_cost = (fp * FP_COST) + (fn * FN_COST)
    
    savings = do_nothing_cost - model_cost
    
    st.metric("Total ₹ Saved vs Baseline (Test Set Only)", f"₹ {savings:,.2f}", f"{(savings/do_nothing_cost)*100:.1f}% reduction in losses")
    
    impact_df = pd.DataFrame({
        'Scenario': ['Do Nothing Baseline', 'MarginSentinel AI Scorer'],
        'Cost (INR)': [do_nothing_cost, model_cost]
    })
    
    fig_impact = px.bar(impact_df, x='Scenario', y='Cost (INR)', color='Scenario',
                       color_discrete_map={'Do Nothing Baseline': '#333333', 'MarginSentinel AI Scorer': '#F63366'},
                       title="Cost Comparison: Doing Nothing vs. MarginSentinel AI", text_auto='.2s')
    fig_impact.update_layout(paper_bgcolor="#121212", plot_bgcolor="#1E1E1E", font={'color': "white"}, height=500, showlegend=False)
    st.plotly_chart(fig_impact, use_container_width=True)
