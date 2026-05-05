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
    """Run full inference pipeline and export data."""
    
    print("=" * 60)
    print("  GNN FRAUD DETECTION - INFERENCE PIPELINE")
    print("=" * 60)
    
    # --- 1. Load & Prepare Data ---
    print("\n[1/6] Loading transaction dataset...")
    
    # Use a sample for speed - load first 2M rows
    SAMPLE_SIZE = 2_000_000
    df = pd.read_csv('MMORPG_Medium_Cleaned.csv', nrows=SAMPLE_SIZE)
    print(f"  Loaded {len(df):,} transactions")
    
    # --- 2. Feature Engineering ---
    print("[2/6] Engineering node features...")
    
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

    unique_players_list = pd.unique(
        df[['Sender_Player_ID', 'Receiver_Player_ID']].values.ravel('K')
    )
    unique_players = pd.DataFrame({'Player_ID': unique_players_list})
    player_features = (
        unique_players
        .merge(stats_out, on='Player_ID', how='left')
        .merge(stats_in, on='Player_ID', how='left')
        .fillna(0)
    )
    player_features['Ratio'] = (
        player_features['Total_Sent'] / (player_features['Total_Received'] + 1)
    )
    player_features['Unique_Ratio'] = player_features['Unique_Receivers'] / (player_features['Unique_Senders'] + 1)
    
    # --- 3. Build Graph ---
    print("[3/6] Building graph topology...")
    player_to_idx = {player: i for i, player in enumerate(unique_players_list)}
    
    src = df['Sender_Player_ID'].map(player_to_idx).values
    dst = df['Receiver_Player_ID'].map(player_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # PageRank
    print("  -> Computing PageRank...")
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=len(unique_players_list))
    pr = np.ones(len(unique_players_list)) / len(unique_players_list)
    adj_T = adj.T.tocsr()
    for _ in range(15):
        pr = 0.85 * adj_T.dot(pr) + 0.15 / len(unique_players_list)
    player_features['PageRank'] = pr

    features_cols = ['Total_Sent', 'Total_Received', 'Trade_Count_Out', 'Trade_Count_In', 'Ratio', 'Unique_Receivers', 'Unique_Senders', 'Unique_Ratio', 'PageRank']
    features_np = np.log1p(player_features[features_cols].values)
    x = torch.tensor(features_np, dtype=torch.float)
    print(f"  Unique players: {len(unique_players_list):,}")
    
    # Ground truth labels
    hacker_ids = set(
        df[df['Is_Fraudulent_Trade'] == 1]['Sender_Player_ID']
    ).union(
        set(df[df['Is_Fraudulent_Trade'] == 1]['Receiver_Player_ID'])
    )
    y = torch.tensor(
        [1 if p in hacker_ids else 0 for p in unique_players_list], dtype=torch.long
    )
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # --- 4. Load Model & Predict ---
    print("[4/6] Loading trained model & running predictions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperEliteSAGE(in_channels=len(features_cols), hidden_channels=128, out_channels=2).to(device)
    
    model_path = 'hyper_elite_medium_model.pth'
    if not os.path.exists(model_path):
        print(f"  ERROR: Model file '{model_path}' not found!")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load optimized threshold dynamically
    opt_threshold = 0.5
    if os.path.exists("threshold.txt"):
        with open("threshold.txt", "r") as f:
            opt_threshold = float(f.read().strip())
    print(f"  Using Optimized Threshold: {opt_threshold:.4f}")
    
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        probs = F.softmax(out, dim=1)
        
        fraud_probs = probs[:, 1].cpu().numpy()
        preds = (probs[:, 1] >= opt_threshold).cpu().long()
    
    predictions = preds.numpy()
    ground_truth = y.numpy()
    
    # --- 5. Generate Classification Report ---
    print("[5/6] Generating classification report...")
    report = classification_report(
        ground_truth, predictions,
        target_names=['Safe', 'Fraudulent'],
        zero_division=0,
        output_dict=True
    )
    report_text = classification_report(
        ground_truth, predictions,
        target_names=['Safe', 'Fraudulent'],
        zero_division=0
    )
    print(report_text)
    
    # --- 6. Export Data for Frontend ---
    print("[6/6] Exporting data for frontend...")
    
    os.makedirs('data', exist_ok=True)
    
    # A) Player list with predictions
    players_data = []
    for i, player_id in enumerate(unique_players_list):
        players_data.append({
            'id': str(player_id),
            'risk_score': round(float(fraud_probs[i]), 4),
            'predicted_label': 'Fraudulent' if predictions[i] == 1 else 'Safe',
            'ground_truth': 'Fraudulent' if ground_truth[i] == 1 else 'Safe',
            'total_sent': round(float(player_features.iloc[i]['Total_Sent']), 2),
            'total_received': round(float(player_features.iloc[i]['Total_Received']), 2),
            'trade_count_out': int(player_features.iloc[i]['Trade_Count_Out']),
            'trade_count_in': int(player_features.iloc[i]['Trade_Count_In']),
            'ratio': round(float(player_features.iloc[i]['Ratio']), 4),
        })
    
    # Sort by risk_score desc
    players_data.sort(key=lambda p: p['risk_score'], reverse=True)
    
    import gzip
    with gzip.open('data/players.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(players_data, f)
    print(f"  -> Exported {len(players_data):,} player profiles to data/players.json.gz")
    
    # B) Dashboard statistics
    total_players = len(unique_players_list)
    total_transactions = len(df)
    total_flagged = int(np.sum(predictions == 1))
    total_safe = int(np.sum(predictions == 0))
    total_ground_truth_fraud = int(np.sum(ground_truth == 1))
    
    # Trade type distribution
    trade_types = df['Trade_Type'].value_counts().to_dict()
    
    # Fraud by trade type
    fraud_df = df[df['Is_Fraudulent_Trade'] == 1]
    fraud_by_type = fraud_df['Trade_Type'].value_counts().to_dict() if len(fraud_df) > 0 else {}
    
    # Transaction value distribution (buckets)
    values = df['In_Game_Currency_Value'].values
    value_buckets = {
        '< 1K': int(np.sum(values < 1000)),
        '1K - 10K': int(np.sum((values >= 1000) & (values < 10000))),
        '10K - 100K': int(np.sum((values >= 10000) & (values < 100000))),
        '100K - 1M': int(np.sum((values >= 100000) & (values < 1000000))),
        '1M - 10M': int(np.sum((values >= 1000000) & (values < 10000000))),
        '> 10M': int(np.sum(values >= 10000000)),
    }
    
    # Risk score distribution
    risk_buckets = {
        'Very Low (0-0.2)': int(np.sum(fraud_probs < 0.2)),
        'Low (0.2-0.4)': int(np.sum((fraud_probs >= 0.2) & (fraud_probs < 0.4))),
        'Medium (0.4-0.6)': int(np.sum((fraud_probs >= 0.4) & (fraud_probs < 0.6))),
        'High (0.6-0.8)': int(np.sum((fraud_probs >= 0.6) & (fraud_probs < 0.8))),
        'Critical (0.8-1.0)': int(np.sum(fraud_probs >= 0.8)),
    }
    
    # Monthly transaction volume (for time-series chart)
    df['Trade_Time'] = pd.to_datetime(df['Trade_Time'], format='mixed', errors='coerce')
    monthly = df.dropna(subset=['Trade_Time']).groupby(
        df['Trade_Time'].dt.to_period('M')
    ).agg(
        tx_count=('In_Game_Currency_Value', 'count'),
        tx_volume=('In_Game_Currency_Value', 'sum'),
        fraud_count=('Is_Fraudulent_Trade', 'sum')
    )
    monthly_data = []
    for period, row in monthly.iterrows():
        monthly_data.append({
            'month': str(period),
            'transactions': int(row['tx_count']),
            'volume': round(float(row['tx_volume']), 2),
            'fraud_count': int(row['fraud_count']),
        })
    
    # Top 20 high-risk players for the dashboard
    top_flagged = players_data[:20]
    
    dashboard = {
        'summary': {
            'total_players': total_players,
            'total_transactions': total_transactions,
            'total_flagged': total_flagged,
            'total_safe': total_safe,
            'total_ground_truth_fraud': total_ground_truth_fraud,
            'detection_rate': round(
                report.get('Fraudulent', {}).get('recall', 0) * 100, 2
            ),
            'precision': round(
                report.get('Fraudulent', {}).get('precision', 0) * 100, 2
            ),
            'f1_score': round(
                report.get('Fraudulent', {}).get('f1-score', 0) * 100, 2
            ),
            'accuracy': round(
                report.get('accuracy', 0) * 100, 2
            ),
        },
        'trade_types': trade_types,
        'fraud_by_trade_type': fraud_by_type,
        'value_distribution': value_buckets,
        'risk_distribution': risk_buckets,
        'monthly_trends': monthly_data,
        'top_flagged_players': top_flagged,
        'classification_report': report,
    }
    
    with open('data/dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2, default=str)
    print(f"  -> Exported dashboard stats to data/dashboard.json")
    
    # C) Sample edges for graph visualization (top 200 fraudulent connections)
    fraud_edges = df[df['Is_Fraudulent_Trade'] == 1].head(500)
    graph_nodes = set()
    graph_edges = []
    
    for _, row in fraud_edges.iterrows():
        s = str(row['Sender_Player_ID'])
        r = str(row['Receiver_Player_ID'])
        graph_nodes.add(s)
        graph_nodes.add(r)
        graph_edges.append({
            'source': s,
            'target': r,
            'value': round(float(row['In_Game_Currency_Value']), 2),
            'type': str(row['Trade_Type']),
        })
    
    # Build node data for visualization
    player_lookup = {p['id']: p for p in players_data}
    graph_nodes_data = []
    for node_id in graph_nodes:
        player = player_lookup.get(node_id, {})
        graph_nodes_data.append({
            'id': node_id,
            'risk_score': player.get('risk_score', 0),
            'label': player.get('predicted_label', 'Unknown'),
        })
    
    graph_data = {
        'nodes': graph_nodes_data,
        'edges': graph_edges,
    }
    
    with open('data/graph.json', 'w') as f:
        json.dump(graph_data, f)
    print(f"  -> Exported graph visualization data ({len(graph_nodes_data)} nodes, {len(graph_edges)} edges)")
    
    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE - Data exported to data/ folder")
    print("=" * 60)


if __name__ == '__main__':
    run_inference()
