import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def plot_training_history(history, reports_dir):
    if not history:
        return

    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, marker='o', label='Loss')
    ax.set_title('Training Loss by Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'training_loss.png'), dpi=160)
    plt.close(fig)

    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    available = [k for k in metric_keys if all(h.get(k) is not None for h in history)]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in available:
        ax.plot(epochs, [h[key] for h in history], marker='o', label=key.replace('_', ' ').title())
    ax.set_title('Training Metrics by Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'training_metrics.png'), dpi=160)
    plt.close(fig)


def evaluate_loader(model, loader, device, threshold=0.5):
    model.eval()
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = F.softmax(out[:batch.batch_size], dim=1)[:, 1]
            y_true_list.append(batch.y[:batch.batch_size].cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    if not y_true_list:
        return {}, None, None

    y_true_all = np.concatenate(y_true_list)
    y_prob_all = np.concatenate(y_prob_list)
    y_pred_all = (y_prob_all >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true_all, y_pred_all) * 100.0,
        'precision': precision_score(y_true_all, y_pred_all, zero_division=0) * 100.0,
        'recall': recall_score(y_true_all, y_pred_all, zero_division=0) * 100.0,
        'f1_score': f1_score(y_true_all, y_pred_all, zero_division=0) * 100.0,
    }
    return metrics, y_true_all, y_prob_all

# --- FOCAL LOSS IMPLEMENTATION ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- 1. MODEL ARCHITECTURE ---
class HyperEliteSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HyperEliteSAGE, self).__init__()
        
        # Graph processing layers (Capture neighborhood topological patterns)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        # THE ROOT INJECTION FIX: The final classifier takes the deep graph features 
        # PLUS the raw original tabular features to prevent oversmoothing
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)
        
        # Layer 2 with Skip Connection
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = h2 + h1 
        
        # ROOT NODE INJECTION: Concatenate the graph context with the raw tabular features
        # This prevents "blurring" of sharp features like Velocity or Ratio
        combined_features = torch.cat([h2, x], dim=1)
        
        # Final decision based on the union of topology and raw stats
        out = self.classifier(combined_features)
        return out

# --- 2. EXECUTION BLOCK (The Windows Multiprocessing Fix) ---
if __name__ == '__main__':
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    ensure_dir(reports_dir)
    num_epochs = int(os.getenv('EPOCHS', '20'))
    eval_sample_size = int(os.getenv('EVAL_SAMPLE_SIZE', '20000'))
    sample_predictions_size = int(os.getenv('SAMPLE_PREDICTIONS', '200'))
    torch.manual_seed(42)

    # A. Load Dataset
    print("Step 1: Loading Full Dataset (32M Transactions)...")
    df = pd.read_csv('MMORPG_Medium_Cleaned.csv')

    # B. Advanced Feature Engineering
    print("Step 2: Engineering Node Features (Total Sent, Received, Ratio)...")
    
    # Calculate outbound stats
    stats_out = df.groupby('Sender_Player_ID').agg(
        Total_Sent=('In_Game_Currency_Value', 'sum'),
        Trade_Count_Out=('In_Game_Currency_Value', 'count'),
        Unique_Receivers=('Receiver_Player_ID', 'nunique')
    ).reset_index().rename(columns={'Sender_Player_ID': 'Player_ID'})

    # Calculate inbound stats
    stats_in = df.groupby('Receiver_Player_ID').agg(
        Total_Received=('In_Game_Currency_Value', 'sum'),
        Trade_Count_In=('In_Game_Currency_Value', 'count'),
        Unique_Senders=('Sender_Player_ID', 'nunique')
    ).reset_index().rename(columns={'Receiver_Player_ID': 'Player_ID'})

    # Consolidate player profiles
    unique_players_list = pd.unique(df[['Sender_Player_ID', 'Receiver_Player_ID']].values.ravel('K'))
    unique_players = pd.DataFrame({'Player_ID': unique_players_list})
    player_features = unique_players.merge(stats_out, on='Player_ID', how='left').merge(stats_in, on='Player_ID', how='left').fillna(0)

    # Feature: Sent-to-Received Ratio (Detects 'Sink' and 'Source' accounts)
    player_features['Ratio'] = player_features['Total_Sent'] / (player_features['Total_Received'] + 1)
    
    # Feature: Hub/Authority Ratio
    player_features['Unique_Ratio'] = player_features['Unique_Receivers'] / (player_features['Unique_Senders'] + 1)

    # Feature: Velocity (Avg Seconds Between Trades) - Phase 3
    # Detects script-based hackers who trade at impossible human speeds
    print("  -> Calculating Trade Velocity...")
    df['Trade_Time'] = pd.to_datetime(df['Trade_Time'])
    df_sorted = df.sort_values(['Sender_Player_ID', 'Trade_Time'])
    df_sorted['Time_Diff'] = df_sorted.groupby('Sender_Player_ID')['Trade_Time'].diff().dt.total_seconds()
    velocity_stats = df_sorted.groupby('Sender_Player_ID')['Time_Diff'].mean().reset_index().rename(
        columns={'Sender_Player_ID': 'Player_ID', 'Time_Diff': 'Velocity'}
    )
    player_features = player_features.merge(velocity_stats, on='Player_ID', how='left').fillna(3600) # Default to 1 hour if only 1 trade

    # C. Build Graph Topology
    print("Step 3: Mapping IDs and Building Adjacency Matrix...")
    player_to_idx = {player: i for i, player in enumerate(unique_players_list)}
    
    # Convert edges to integer indices
    src = df['Sender_Player_ID'].map(player_to_idx).values
    dst = df['Receiver_Player_ID'].map(player_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # PageRank (Fast Power Iteration)
    print("  -> Computing PageRank...")
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=len(unique_players_list))
    pr = np.ones(len(unique_players_list)) / len(unique_players_list)
    adj_T = adj.T.tocsr()
    for _ in range(15):
        pr = 0.85 * adj_T.dot(pr) + 0.15 / len(unique_players_list)
    player_features['PageRank'] = pr

    # Log transformation to normalize wealth distribution
    features_cols = ['Total_Sent', 'Total_Received', 'Trade_Count_Out', 'Trade_Count_In', 'Ratio', 'Unique_Receivers', 'Unique_Senders', 'Unique_Ratio', 'PageRank', 'Velocity']
    features_np = np.log1p(player_features[features_cols].values)
    x = torch.tensor(features_np, dtype=torch.float)

    # D. Prepare Target Labels
    print("Step 4: Extracting Ground-Truth Labels...")
    hacker_ids = set(df[df['Is_Fraudulent_Trade'] == 1]['Sender_Player_ID']).union(
                 set(df[df['Is_Fraudulent_Trade'] == 1]['Receiver_Player_ID']))
    
    y = torch.tensor([1 if p in hacker_ids else 0 for p in unique_players_list], dtype=torch.long)

    # Create PyG Data Object
    data = Data(x=x, edge_index=edge_index, y=y)

    # E. Initialize NeighborLoader (Biased Neighborhood Sampling)
    # We force the model to see a higher concentration of hackers (25%) in every batch
    # to overcome the extreme class imbalance in the 2M node sample.
    print("Step 5: Implementing Biased Sampling (25% Fraud Rate)...")
    hacker_indices = (y == 1).nonzero(as_tuple=False).view(-1)
    safe_indices = (y == 0).nonzero(as_tuple=False).view(-1)

    # Calculate how many safe nodes to include to reach 10% fraud ratio (Phase 1)
    num_hackers = len(hacker_indices)
    num_safe_to_sample = num_hackers * 9 # 1 hacker for every 9 safe players = 10%
    
    # Randomly sample safe nodes
    perm = torch.randperm(len(safe_indices))[:num_safe_to_sample]
    sampled_safe_indices = safe_indices[perm]
    
    # Combine and shuffle
    balanced_input_nodes = torch.cat([hacker_indices, sampled_safe_indices])
    balanced_input_nodes = balanced_input_nodes[torch.randperm(len(balanced_input_nodes))]

    loader = NeighborLoader(
        data,
        num_neighbors=[15, 10, 5], 
        batch_size=4096,
        input_nodes=balanced_input_nodes, # Root nodes for sampling
        shuffle=True,
        num_workers=0,
        persistent_workers=False
    )

    eval_sample_size = max(0, min(eval_sample_size, balanced_input_nodes.numel()))
    eval_sample_nodes = None
    eval_loader = None
    if eval_sample_size > 0:
        sample_perm = torch.randperm(balanced_input_nodes.numel())[:eval_sample_size]
        eval_sample_nodes = balanced_input_nodes[sample_perm]
        eval_loader = NeighborLoader(
            data,
            num_neighbors=[15, 10, 5],
            batch_size=4096,
            input_nodes=eval_sample_nodes,
            shuffle=False,
            num_workers=0
        )

    # F. Model, Optimizer, and Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperEliteSAGE(in_channels=len(features_cols), hidden_channels=128, out_channels=2).to(device)
    
    # Weight decay (L2 Regularization) prevents overfitting on large outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Focal Loss: Dialing back alpha radically because the Biased Sampler is doing the heavy lifting.
    # This prevents the "Probability Shift" where the model thinks everyone is a hacker.
    alpha_weights = torch.tensor([1.0, 1.5]).to(device) 
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

    # Learning Rate Scheduler: Reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # G. Training Loop
    print("\nStep 6: Starting Hyper-Elite Training (Mini-Batch)...")
    model.train()
    training_history = []
    for epoch in range(num_epochs): 
        total_loss = 0
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # Loss is only calculated for target nodes in the current batch
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            
            # Gradient Clipping: Prevents exploding gradients during "surprising" batches
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        metrics = {}
        if eval_loader is not None:
            metrics, _, _ = evaluate_loader(model, eval_loader, device, threshold=0.5)
            model.train()

        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
        })

        if metrics:
            print(
                f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | "
                f"Acc: {metrics['accuracy']:.2f}% | "
                f"Prec: {metrics['precision']:.2f}% | "
                f"Recall: {metrics['recall']:.2f}% | "
                f"F1: {metrics['f1_score']:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']}"
            )
        else:
            print(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")

    if training_history:
        save_json(os.path.join(reports_dir, 'training_history.json'), training_history)
        pd.DataFrame(training_history).to_csv(os.path.join(reports_dir, 'training_history.csv'), index=False)
        plot_training_history(training_history, reports_dir)

    # H. Final Evaluation
    print("\nStep 7: Running Final Evaluation & Generating Report...")
    model.eval()
    y_true_list = []
    y_prob_list = []
    node_id_list = []

    # Use a standard loader for evaluation to get unbiased metrics
    eval_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10, 5],
        batch_size=4096,
        shuffle=False,
        num_workers=0
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = F.softmax(out[:batch.batch_size], dim=1)
            
            y_true_list.append(batch.y[:batch.batch_size].cpu().numpy())
            y_prob_list.append(probs[:, 1].cpu().numpy())
            node_id_list.append(batch.n_id[:batch.batch_size].cpu().numpy())

    y_true_all = np.concatenate(y_true_list)
    y_prob_all = np.concatenate(y_prob_list)
    node_ids_all = np.concatenate(node_id_list) if node_id_list else None

    # 1. Analyze the Probability Distribution
    # This helps us see if Focal Loss is pushing the baseline too high
    print("\n--- Probability Distribution ---")
    print(f"Min Prob:  {y_prob_all.min():.4f}")
    print(f"Max Prob:  {y_prob_all.max():.4f}")
    print(f"Mean Prob: {y_prob_all.mean():.4f}")

    # 2. Automated Threshold Selection (F1 Optimization)
    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_prob_all)
    precisions_t = precisions[1:]
    recalls_t = recalls[1:]
    f1_scores = (2 * (precisions_t * recalls_t) / (precisions_t + recalls_t + 1e-10)) if len(thresholds) else np.array([])
    best_idx = int(np.argmax(f1_scores)) if len(f1_scores) else -1
    best_threshold = thresholds[best_idx] if best_idx >= 0 else 0.5
    
    print(f"\n[OPTIMIZATION] Best Threshold for maximum F1: {best_threshold:.4f}")

    # 3. Manual Threshold Options (Business Logic)
    # We print a menu so the user can see the trade-offs at different confidence levels
    from sklearn.metrics import recall_score, precision_score
    print("\n--- Business Logic Threshold Options ---")
    for t in [0.50, 0.70, 0.80, 0.90, 0.95]:
        temp_preds = (y_prob_all >= t).astype(int)
        rec = recall_score(y_true_all, temp_preds, zero_division=0)
        prec = precision_score(y_true_all, temp_preds, zero_division=0)
        print(f"Threshold > {t:.2f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    
    y_pred_all = (y_prob_all >= best_threshold).astype(int)

    final_metrics = {
        'accuracy': accuracy_score(y_true_all, y_pred_all) * 100.0,
        'precision': precision_score(y_true_all, y_pred_all, zero_division=0) * 100.0,
        'recall': recall_score(y_true_all, y_pred_all, zero_division=0) * 100.0,
        'f1_score': f1_score(y_true_all, y_pred_all, zero_division=0) * 100.0,
        'best_threshold': float(best_threshold),
    }
    save_json(os.path.join(reports_dir, 'final_metrics.json'), final_metrics)

    print("\n--- Final GraphSAGE Report (Medium Dataset) ---")
    report_text = classification_report(y_true_all, y_pred_all, target_names=['Safe', 'Hacker'], zero_division=0)
    print(report_text)
    report_dict = classification_report(
        y_true_all,
        y_pred_all,
        target_names=['Safe', 'Hacker'],
        zero_division=0,
        output_dict=True
    )
    save_json(os.path.join(reports_dir, 'classification_report.json'), report_dict)
    with open(os.path.join(reports_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    pr_curve_rows = []
    for i, t in enumerate(thresholds):
        pr_curve_rows.append({
            'threshold': float(t),
            'precision': float(precisions_t[i]) if i < len(precisions_t) else None,
            'recall': float(recalls_t[i]) if i < len(recalls_t) else None,
            'f1_score': float(f1_scores[i]) if i < len(f1_scores) else None,
        })
    if pr_curve_rows:
        pd.DataFrame(pr_curve_rows).to_csv(os.path.join(reports_dir, 'precision_recall_curve.csv'), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, color='#1f77b4')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'precision_recall_curve.png'), dpi=160)
    plt.close(fig)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Hacker'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'confusion_matrix.png'), dpi=160)
    plt.close(fig)

    if node_ids_all is not None and len(node_ids_all) == len(y_true_all):
        player_ids = np.array(unique_players_list)[node_ids_all]
        sample_size = max(0, min(sample_predictions_size, len(player_ids)))
        if sample_size > 0:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(player_ids), size=sample_size, replace=False)
            sample_df = pd.DataFrame({
                'player_id': player_ids[sample_idx],
                'true_label': y_true_all[sample_idx],
                'predicted_prob': y_prob_all[sample_idx],
                'predicted_label': y_pred_all[sample_idx],
            })
            sample_df.to_csv(os.path.join(reports_dir, 'sample_predictions.csv'), index=False)

    # I. Save Model State and Threshold
    torch.save(model.state_dict(), 'hyper_elite_medium_model.pth')
    # Save the optimized threshold for inference
    with open("threshold.txt", "w") as f:
        f.write(str(best_threshold))
    print(f"\nProject Success: Model saved as 'hyper_elite_medium_model.pth'. Threshold saved.")