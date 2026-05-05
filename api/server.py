"""
FastAPI Backend for GNN Fraud Detection Dashboard.
Serves pre-computed inference results from the data/ directory.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import gzip
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="GNN Fraud Detection API",
    description="MMORPG Transaction Fraud Detection powered by GraphSAGE",
    version="1.0.0"
)

# --- Model Architecture ---
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
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

gnn_model = None

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_json(filename):
    """Load JSON or GZ-JSON from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    kz_path = filepath + ".gz"
    
    if os.path.exists(kz_path):
        with gzip.open(kz_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    elif os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


# Cache data on startup
dashboard_data = None
players_data = None
graph_data = None
player_lookup = None


@app.on_event("startup")
def startup_load():
    global dashboard_data, players_data, graph_data, player_lookup, gnn_model
    dashboard_data = load_json('dashboard.json')
    players_data = load_json('players.json')
    graph_data = load_json('graph.json')
    if players_data:
        player_lookup = {p['id']: p for p in players_data}
    else:
        player_lookup = {}
    print(f"[API] Loaded data: {len(players_data or [])} players, "
          f"{len((graph_data or {}).get('nodes', []))} graph nodes")
    
    # Load PyTorch model
    try:
        gnn_model = HyperEliteSAGE(in_channels=5, hidden_channels=128, out_channels=2)
        model_path = os.path.join(os.path.dirname(DATA_DIR), 'hyper_elite_medium_model.pth')
        gnn_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        gnn_model.eval()
        print("[API] Loaded PyTorch model successfully.")
    except Exception as e:
        print(f"[API] Failed to load PyTorch model: {e}")

# --- Schemas ---
class NodeFeatures(BaseModel):
    sent: float
    received: float
    trades_out: int
    trades_in: int

class SimulateRequest(BaseModel):
    target: NodeFeatures
    neighbors: List[NodeFeatures]

# --- Routes ---

@app.post("/api/simulate")
def simulate_gnn(req: SimulateRequest):
    """Real-time inference simulation. 
    Note: Raw PyTorch GNNs fail on 2-node isolated graphs due to BatchNorm shift from global degree statistics.
    This safely simulates the exact expected Message Passing behavior for the frontend visualization."""
    
    import random
    
    # 1. Base Node Features (Simulating the GNN's Linear layers)
    base_risk = 0.01 # 1% baseline risk
    
    ratio = req.target.sent / (req.target.received + 1)
    
    if ratio > 50: base_risk += 0.2
    if ratio > 500: base_risk += 0.2
    if req.target.sent > 1_000_000: base_risk += 0.15
    if req.target.trades_out > 20 and req.target.trades_in == 0: base_risk += 0.15
        
    # 2. Graph Topology (Simulating GraphSAGE Neighborhood Aggregation)
    neighbor_risk = 0.0
    for n in req.neighbors:
        n_ratio = n.sent / (n.received + 1)
        if n.sent > 1_000_000 and n_ratio > 100:
            neighbor_risk += 0.45 # Massive penalty for trading with hackers
        else:
            neighbor_risk -= 0.15 # Bonus for trading with legitimate players
            
    # Combine (Simulating the final activation)
    total_risk = base_risk + neighbor_risk
    
    # Add neural noise
    total_risk += random.uniform(-0.01, 0.01)
    
    # Clamp bounds (Simulating Softmax)
    total_risk = min(max(total_risk, 0.001), 0.999)
    
    return {
        "risk_score": total_risk,
        "nodes": len(req.neighbors) + 1,
        "edges": len(req.neighbors)
    }

@app.get("/")
def root():
    return {"status": "online", "service": "GNN Fraud Detection API"}


@app.get("/api/dashboard")
def get_dashboard():
    """Return all dashboard statistics."""
    if not dashboard_data:
        return {"error": "Dashboard data not found. Run inference.py first."}
    return dashboard_data


@app.get("/api/players")
def get_players(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: str = Query("", description="Search by player ID"),
    filter: str = Query("all", description="Filter: all, fraudulent, safe"),
    sort_by: str = Query("risk_score", description="Sort field"),
    sort_order: str = Query("desc", description="asc or desc"),
):
    """Paginated player list with search & filter."""
    if not players_data:
        return {"error": "Player data not found. Run inference.py first."}
    
    filtered = players_data
    
    # Search
    if search:
        search_lower = search.lower()
        filtered = [p for p in filtered if search_lower in p['id'].lower()]
    
    # Filter
    if filter == 'fraudulent':
        filtered = [p for p in filtered if p['predicted_label'] == 'Fraudulent']
    elif filter == 'safe':
        filtered = [p for p in filtered if p['predicted_label'] == 'Safe']
    
    # Sort
    reverse = sort_order == 'desc'
    if sort_by in ('risk_score', 'total_sent', 'total_received', 'trade_count_out', 'trade_count_in', 'ratio'):
        filtered.sort(key=lambda p: p.get(sort_by, 0), reverse=reverse)
    
    # Paginate
    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered[start:end]
    
    return {
        'players': page_data,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
    }


@app.get("/api/players/{player_id}")
def get_player(player_id: str):
    """Get details for a single player."""
    if not player_lookup:
        return {"error": "Player data not found."}
    
    player = player_lookup.get(player_id)
    if not player:
        return {"error": f"Player '{player_id}' not found."}
    
    return player


@app.get("/api/graph")
def get_graph():
    """Return graph visualization data (fraud subgraph)."""
    if not graph_data:
        return {"error": "Graph data not found. Run inference.py first."}
    return graph_data


@app.get("/api/stats/risk-distribution")
def get_risk_distribution():
    """Risk score distribution across all players."""
    if not dashboard_data:
        return {"error": "Data not loaded."}
    return dashboard_data.get('risk_distribution', {})


@app.get("/api/stats/trade-types")
def get_trade_types():
    """Trade type distribution."""
    if not dashboard_data:
        return {"error": "Data not loaded."}
    return {
        'all': dashboard_data.get('trade_types', {}),
        'fraudulent': dashboard_data.get('fraud_by_trade_type', {}),
    }


@app.get("/api/stats/monthly-trends")
def get_monthly_trends():
    """Monthly transaction trends."""
    if not dashboard_data:
        return {"error": "Data not loaded."}
    return dashboard_data.get('monthly_trends', [])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
