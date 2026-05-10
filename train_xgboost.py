import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics import classification_report, precision_recall_curve, recall_score, precision_score
from traingnn import HyperEliteSAGE

# --- 1. SETUP & LOAD DATA (Full 32M Transactions) ---
print("Step 1: Loading Full Dataset (32M Transactions)...")
df = pd.read_csv('MMORPG_Medium_Cleaned.csv')

print("Step 2: Engineering Node Features (Velocity, PageRank, etc.)...")
# Outbound stats
stats_out = df.groupby('Sender_Player_ID').agg(
    Total_Sent=('In_Game_Currency_Value', 'sum'),
    Trade_Count_Out=('In_Game_Currency_Value', 'count'),
    Unique_Receivers=('Receiver_Player_ID', 'nunique')
).reset_index().rename(columns={'Sender_Player_ID': 'Player_ID'})

# Inbound stats
stats_in = df.groupby('Receiver_Player_ID').agg(
    Total_Received=('In_Game_Currency_Value', 'sum'),
    Trade_Count_In=('In_Game_Currency_Value', 'count'),
    Unique_Senders=('Sender_Player_ID', 'nunique')
).reset_index().rename(columns={'Receiver_Player_ID': 'Player_ID'})

# Consolidate player profiles
unique_players_list = pd.unique(df[['Sender_Player_ID', 'Receiver_Player_ID']].values.ravel('K'))
unique_players = pd.DataFrame({'Player_ID': unique_players_list})
player_features = unique_players.merge(stats_out, on='Player_ID', how='left').merge(stats_in, on='Player_ID', how='left').fillna(0)

# Feature Ratios
player_features['Ratio'] = player_features['Total_Sent'] / (player_features['Total_Received'] + 1)
player_features['Unique_Ratio'] = player_features['Unique_Receivers'] / (player_features['Unique_Senders'] + 1)

# Feature: Velocity (Avg Seconds Between Trades)
df['Trade_Time'] = pd.to_datetime(df['Trade_Time'])
df_sorted = df.sort_values(['Sender_Player_ID', 'Trade_Time'])
df_sorted['Time_Diff'] = df_sorted.groupby('Sender_Player_ID')['Trade_Time'].diff().dt.total_seconds()
velocity_stats = df_sorted.groupby('Sender_Player_ID')['Time_Diff'].mean().reset_index().rename(
    columns={'Sender_Player_ID': 'Player_ID', 'Time_Diff': 'Velocity'}
)
player_features = player_features.merge(velocity_stats, on='Player_ID', how='left').fillna(3600)

print("Step 3: Building Graph Topology & PageRank...")
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

features_cols = ['Total_Sent', 'Total_Received', 'Trade_Count_Out', 'Trade_Count_In', 'Ratio', 'Unique_Receivers', 'Unique_Senders', 'Unique_Ratio', 'PageRank', 'Velocity']
x_np = np.log1p(player_features[features_cols].values)

# Labels
hacker_ids = set(df[df['Is_Fraudulent_Trade'] == 1]['Sender_Player_ID']).union(
             set(df[df['Is_Fraudulent_Trade'] == 1]['Receiver_Player_ID']))
y_np = np.array([1 if p in hacker_ids else 0 for p in unique_players_list])

# --- 2. LOAD THE TRAINED GNN ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nLoading trained GraphSAGE model on {device}...")
model = HyperEliteSAGE(in_channels=len(features_cols), hidden_channels=128, out_channels=2).to(device)
model.load_state_dict(torch.load('hyper_elite_medium_model.pth', map_location=device))
model.eval()

# --- 3. EXTRACT DEEP NETWORK EMBEDDINGS (BATCHED) ---
print("Extracting deep network embeddings (Batched for 32M Scale)...")
data = Data(x=torch.tensor(x_np, dtype=torch.float), edge_index=edge_index)
loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=8192, shuffle=False)

embeddings_list = []
with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        # Pass through GNN layers to extract 'h2'
        h1 = model.conv1(batch.x, batch.edge_index)
        h1 = model.bn1(h1)
        h1 = torch.nn.functional.relu(h1)
        
        h2 = model.conv2(h1, batch.edge_index)
        h2 = model.bn2(h2)
        h2 = torch.nn.functional.relu(h2)
        h2 = h2 + h1 # Skip connection
        
        # Combined: Context (128) + Raw (10)
        combined = torch.cat([h2[:batch.batch_size], batch.x[:batch.batch_size]], dim=1)
        embeddings_list.append(combined.cpu().numpy())
        
        if i % 500 == 0:
            print(f"  Processed {i * 8192:,} nodes...")

combined_features = np.concatenate(embeddings_list, axis=0)

# --- 4. TRAIN XGBOOST ENSEMBLE ---
print("\nTraining XGBoost Classifier on Combined Features...")

# Calculate nerfed scale_pos_weight to prevent over-aggression
neg_count = (y_np == 0).sum()
pos_count = (y_np == 1).sum()
scale_weight = (neg_count / pos_count) * 0.25 # Nerfed weight (approx 12x)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,        # More trees to handle the 32M context
    max_depth=9,             # Deeper trees to navigate the 128-dim embeddings
    learning_rate=0.03,      # Slower, more precise learning
    scale_pos_weight=scale_weight,
    tree_method='hist',      # CPU histogram processing for throughput
    n_jobs=-1,               # Utilize all CPU cores
    random_state=42
)

xgb_model.fit(combined_features, y_np)

# --- 5. EVALUATE ENSEMBLE ---
print("\nEvaluating SAGE + XGBoost Ensemble...")
y_prob_xgb = xgb_model.predict_proba(combined_features)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_np, y_prob_xgb)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(f"\n[ENSEMBLE OPTIMIZATION] Best Threshold: {best_threshold:.4f}")

# Print Business Logic Menu for XGBoost
print("\n--- Ensemble Threshold Options ---")
for t in [0.50, 0.70, 0.80, 0.90, 0.95]:
    temp_preds = (y_prob_xgb >= t).astype(int)
    rec = recall_score(y_np, temp_preds, zero_division=0)
    prec = precision_score(y_np, temp_preds, zero_division=0)
    print(f"Threshold > {t:.2f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

y_pred_xgb = (y_prob_xgb >= best_threshold).astype(int)

print("\n--- Final Ensemble Report (SAGE Context + XGBoost Boundaries) ---")
print(classification_report(y_np, y_pred_xgb, target_names=['Safe', 'Hacker'], zero_division=0))

# Save the ensemble components and the optimized threshold
import joblib
joblib.dump(xgb_model, 'xgboost_ensemble_model.pkl')
with open("threshold.txt", "w") as f:
    f.write(str(best_threshold))
print(f"\nProject Complete: Ensemble saved. Threshold {best_threshold:.4f} saved to threshold.txt.")
