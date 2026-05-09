"""
GNN Fraud Detection - Inference & Data Export Pipeline
Loads the trained HyperEliteSAGE model, runs predictions on all players,
and exports structured JSON data for the FastAPI backend / React frontend.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from sklearn.metrics import classification_report
from torch_geometric.utils import to_scipy_sparse_matrix
import json
import os
from collections import Counter

# --- Model Architecture (must match training) ---
class HyperEliteSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HyperEliteSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


def run_inference():
    """Run optimized inference pipeline and export data."""
    
    print("=" * 60)
    print("  GNN FRAUD DETECTION - OPTIMIZED INFERENCE")
    print("=" * 60)
    
    FILE_PATH = 'MMORPG_Medium_Cleaned.csv'
    CHUNK_SIZE = 1_000_000
    
    # --- 1. Global Stats Calculation (Chunked) ---
    print("\n[1/6] Calculating Global Stats (Full Dataset Scan)...")
    total_tx = 0
    total_value = 0
    fraud_tx = 0
    trade_type_counts = Counter()
    fraud_by_type = Counter()
    all_senders = set()
    all_receivers = set()
    
    # Scan file in chunks for aggregate stats to save RAM
    for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):
        total_tx += len(chunk)
        total_value += chunk['In_Game_Currency_Value'].sum()
        fraud_chunk = chunk[chunk['Is_Fraudulent_Trade'] == 1]
        fraud_tx += len(fraud_chunk)
        
        trade_type_counts.update(chunk['Trade_Type'].tolist())
        fraud_by_type.update(fraud_chunk['Trade_Type'].tolist())
        all_senders.update(chunk['Sender_Player_ID'].unique())
        all_receivers.update(chunk['Receiver_Player_ID'].unique())
        print(f"  Processed {total_tx:,} transactions...")

    global_unique_players = len(all_senders.union(all_receivers))
    
    # --- 2. Inference & Feature Engineering (On manageable sample) ---
    print("\n[2/6] Loading GNN Inference Sample (2M rows)...")
    SAMPLE_SIZE = 2_000_000
    df = pd.read_csv(FILE_PATH, nrows=SAMPLE_SIZE)
    
    print("[3/6] Engineering node features & PageRank...")
    stats_out = df.groupby('Sender_Player_ID').agg(
        Total_Sent=('In_Game_Currency_Value', 'sum'),
        Trade_Count_Out=('In_Game_Currency_Value', 'count'),
        Unique_Receivers=('Receiver_Player_ID', 'nunique')
    ).reset_index().rename(columns={'Sender_Player_ID': 'Player_ID'})

    stats_in = df.groupby('Receiver_Player_ID').agg(
        Total_Received=('In_Game_Currency_Value', 'sum'),
        Trade_Count_In=('In_Game_Currency_Value', 'count'),
        Unique_Senders=('Sender_Player_ID', 'nunique')
    ).reset_index().rename(columns={'Receiver_Player_ID': 'Player_ID'})

    unique_players_list = pd.unique(df[['Sender_Player_ID', 'Receiver_Player_ID']].values.ravel('K'))
    unique_players = pd.DataFrame({'Player_ID': unique_players_list})
    player_features = unique_players.merge(stats_out, on='Player_ID', how='left').merge(stats_in, on='Player_ID', how='left').fillna(0)
    
    player_features['Ratio'] = player_features['Total_Sent'] / (player_features['Total_Received'] + 1)
    player_features['Unique_Ratio'] = player_features['Unique_Receivers'] / (player_features['Unique_Senders'] + 1)
    
    # Build Graph & PageRank
    player_to_idx = {player: i for i, player in enumerate(unique_players_list)}
    src = df['Sender_Player_ID'].map(player_to_idx).values
    dst = df['Receiver_Player_ID'].map(player_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=len(unique_players_list))
    pr = np.ones(len(unique_players_list)) / len(unique_players_list)
    adj_T = adj.T.tocsr()
    for _ in range(15):
        pr = 0.85 * adj_T.dot(pr) + 0.15 / len(unique_players_list)
    player_features['PageRank'] = pr

    features_cols = ['Total_Sent', 'Total_Received', 'Trade_Count_Out', 'Trade_Count_In', 'Ratio', 'Unique_Receivers', 'Unique_Senders', 'Unique_Ratio', 'PageRank']
    features_np = np.log1p(player_features[features_cols].values)
    x = torch.tensor(features_np, dtype=torch.float)
    
    # Ground truth
    hacker_ids = set(df[df['Is_Fraudulent_Trade'] == 1]['Sender_Player_ID']).union(
                 set(df[df['Is_Fraudulent_Trade'] == 1]['Receiver_Player_ID']))
    y = torch.tensor([1 if p in hacker_ids else 0 for p in unique_players_list], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # --- 4. Run Model ---
    print("[4/6] Running GNN predictions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperEliteSAGE(in_channels=len(features_cols), hidden_channels=128, out_channels=2).to(device)
    model.load_state_dict(torch.load('hyper_elite_medium_model.pth', map_location=device, weights_only=True))
    model.eval()
    
    opt_threshold = 0.5
    if os.path.exists("threshold.txt"):
        with open("threshold.txt", "r") as f: opt_threshold = float(f.read().strip())
    
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        probs = F.softmax(out, dim=1)
        fraud_probs = probs[:, 1].cpu().numpy()
        preds = (probs[:, 1] >= opt_threshold).cpu().long().numpy()

    # --- 5. Export Optimized JSONs ---
    print("[5/6] Exporting Capped Data (Players & Graph)...")
    os.makedirs('data', exist_ok=True)
    
    # Export only top 5000 players (Risky first)
    players_data = []
    for i, player_id in enumerate(unique_players_list):
        players_data.append({
            'id': str(player_id),
            'risk_score': round(float(fraud_probs[i]), 4),
            'predicted_label': 'Fraudulent' if preds[i] == 1 else 'Safe',
            'ground_truth': 'Fraudulent' if y[i] == 1 else 'Safe',
            'total_sent': round(float(player_features.iloc[i]['Total_Sent']), 2),
            'total_received': round(float(player_features.iloc[i]['Total_Received']), 2),
            'trade_count_out': int(player_features.iloc[i]['Trade_Count_Out']),
            'trade_count_in': int(player_features.iloc[i]['Trade_Count_In']),
            'ratio': round(float(player_features.iloc[i]['Ratio']), 4),
        })
    players_data.sort(key=lambda p: p['risk_score'], reverse=True)
    
    # CAP AT 5000 PLAYERS
    export_players = players_data[:5000]
    with open('data/players.json', 'w', encoding='utf-8') as f:
        json.dump(export_players, f)

    # C) Export Capped Graph (Top 2000 connections)
    fraud_edges_df = df[df['Is_Fraudulent_Trade'] == 1].head(2000)
    graph_nodes = set()
    graph_edges = []
    for _, row in fraud_edges_df.iterrows():
        s, r = str(row['Sender_Player_ID']), str(row['Receiver_Player_ID'])
        graph_nodes.add(s); graph_nodes.add(r)
        graph_edges.append({'source': s, 'target': r, 'value': float(row['In_Game_Currency_Value']), 'type': str(row['Trade_Type'])})
    
    player_lookup = {p['id']: p for p in players_data}
    graph_nodes_data = [{'id': nid, 'risk_score': player_lookup.get(nid, {}).get('risk_score', 0), 'label': player_lookup.get(nid, {}).get('predicted_label', 'Safe')} for nid in graph_nodes]
    
    with open('data/graph.json', 'w') as f:
        json.dump({'nodes': graph_nodes_data, 'edges': graph_edges}, f)

    # --- 6. Dashboard (Global Metrics) ---
    print("[6/6] Finalizing Dashboard (Global Stats)...")
    report = classification_report(y.numpy(), preds, target_names=['Safe', 'Fraudulent'], output_dict=True, zero_division=0)
    
    dashboard = {
        'summary': {
            'total_players': global_unique_players,
            'total_transactions': total_tx,
            'total_flagged': int(np.sum(preds == 1)),
            'total_ground_truth_fraud': fraud_tx,
            'detection_rate': round(report.get('Fraudulent', {}).get('recall', 0) * 100, 2),
            'precision': round(report.get('Fraudulent', {}).get('precision', 0) * 100, 2),
            'accuracy': round(report.get('accuracy', 0) * 100, 2),
        },
        'trade_types': dict(trade_type_counts),
        'fraud_by_trade_type': dict(fraud_by_type),
        'risk_distribution': {
            'Critical (0.8+)': int(np.sum(fraud_probs >= 0.8)),
            'High (0.5-0.8)': int(np.sum((fraud_probs >= 0.5) & (fraud_probs < 0.8))),
            'Safe (< 0.5)': int(np.sum(fraud_probs < 0.5)),
        },
        'top_flagged_players': players_data[:20],
    }
    
    with open('data/dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE - System Optimized")
    print("=" * 60)

if __name__ == '__main__':
    run_inference()

