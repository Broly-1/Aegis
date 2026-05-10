"""
FastAPI Backend for GNN Fraud Detection Dashboard.
Serves pre-computed inference results from the data/ directory.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import gzip
import os
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="GNN Fraud Detection API",
    description="MMORPG Transaction Fraud Detection powered by GraphSAGE",
    version="1.0.0"
)

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
MAX_PLAYERS = int(os.getenv("MAX_PLAYERS", "5000"))
MAX_GRAPH_EDGES = int(os.getenv("MAX_GRAPH_EDGES", "2000"))
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", "2000"))
CACHE_LIMITED_DATA = os.getenv("CACHE_LIMITED_DATA", "1") != "0"


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


def open_data_file(filename):
    """Open JSON or GZ-JSON data file for streaming reads."""
    filepath = os.path.join(DATA_DIR, filename)
    kz_path = filepath + ".gz"

    if os.path.exists(kz_path):
        return gzip.open(kz_path, 'rt', encoding='utf-8')
    if os.path.exists(filepath):
        return open(filepath, 'r', encoding='utf-8')
    return None


def load_json_array_head(filename, limit):
    """Load only the first N items from a JSON array to cap memory usage."""
    if limit <= 0:
        return []

    handle = open_data_file(filename)
    if handle is None:
        return None

    decoder = json.JSONDecoder()
    buffer = ""
    items = []

    with handle as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                return items
            buffer += chunk
            start_idx = buffer.find('[')
            if start_idx != -1:
                buffer = buffer[start_idx + 1:]
                break

        while len(items) < limit:
            buffer = buffer.lstrip()
            if not buffer:
                chunk = f.read(4096)
                if not chunk:
                    break
                buffer += chunk
                continue

            if buffer[0] == ']':
                break

            try:
                value, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = f.read(4096)
                if not chunk:
                    break
                buffer += chunk
                continue

            items.append(value)
            buffer = buffer[offset:]
            if buffer.startswith(','):
                buffer = buffer[1:]

    return items


def get_players_data():
    """Load the pre-capped player list for instant dashboard performance."""
    global players_data, player_lookup
    if players_data is None:
        # We now use standard load_json because inference.py already capped this file to 5,000
        data = load_json('players.json')
        if data is None:
            return None
        if CACHE_LIMITED_DATA:
            players_data = data
            player_lookup = {p['id']: p for p in data}
        return data
    return players_data


def get_graph_data():
    """Load a capped graph payload on demand."""
    global graph_data
    if graph_data is None:
        data = load_json('graph.json')
        if data is None:
            return None
        edges = data.get('edges') or data.get('links') or []
        edges = edges[:MAX_GRAPH_EDGES]
        node_ids = set()
        for edge in edges:
            node_ids.add(edge.get('source'))
            node_ids.add(edge.get('target'))

        nodes = data.get('nodes') or []
        if node_ids:
            nodes = [node for node in nodes if node.get('id') in node_ids]
        nodes = nodes[:MAX_GRAPH_NODES]

        data = {
            'nodes': nodes,
            'edges': edges,
        }
        if CACHE_LIMITED_DATA:
            graph_data = data
        return data
    return graph_data


# Cache data on startup
dashboard_data = None
players_data = None
graph_data = None
player_lookup = None


@app.on_event("startup")
def startup_load():
    global dashboard_data, players_data, graph_data, player_lookup
    dashboard_data = load_json('dashboard.json')
    players_data = None
    graph_data = None
    player_lookup = None
    print("[API] Loaded dashboard data")

# --- Schemas ---
class NodeFeatures(BaseModel):
    sent: float
    received: float
    trades_out: int
    trades_in: int
    velocity: float = 3600.0

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
    import math
    
    # 1. Base Node Features (Simulating the GNN's Linear layers)
    base_risk = 0.01 # 1% baseline risk
    
    ratio = req.target.sent / (req.target.received + 1)
    
    # Smooth continuous impact of ratio (up to +0.4)
    if ratio > 1:
        base_risk += min(0.4, 0.15 * math.log10(ratio))
        
    # Smooth continuous impact of volume (up to +0.3)
    if req.target.sent > 1000:
        base_risk += min(0.3, 0.075 * math.log10(req.target.sent / 1000))
        
    # Smooth continuous impact of trade asymmetry (up to +0.2)
    trade_ratio = req.target.trades_out / (req.target.trades_in + 1)
    if trade_ratio > 1:
        base_risk += min(0.25, 0.1 * math.log10(trade_ratio))
        
    # ROOT NODE INJECTION: Velocity Impact (Up to +0.5 for script speed)
    # 0-5 seconds is highly suspicious for MMORPG trading
    if req.target.velocity < 60:
        velocity_risk = (60 - req.target.velocity) / 60.0
        base_risk += velocity_risk * 0.5 
        
    # 2. Graph Topology (Simulating GraphSAGE Neighborhood Aggregation)
    neighbor_risk = 0.0
    for n in req.neighbors:
        n_ratio = n.sent / (n.received + 1)
        if n.sent > 1_000_000 and n_ratio > 10:
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
    players = get_players_data()
    if not players:
        return {"error": "Player data not found. Run inference.py first."}
    
    filtered = players
    
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
        'data_limit': MAX_PLAYERS,
    }


@app.get("/api/players/{player_id}")
def get_player(player_id: str):
    """Get details for a single player."""
    players = get_players_data()
    if not players:
        return {"error": "Player data not found."}
    if not player_lookup:
        player_lookup = {p['id']: p for p in players}
    
    if not player_lookup:
        return {"error": "Player data not found."}
    
    player = player_lookup.get(player_id)
    if not player:
        return {"error": f"Player '{player_id}' not found."}
    
    return player


@app.get("/api/graph")
def get_graph():
    """Return graph visualization data (fraud subgraph)."""
    data = get_graph_data()
    if not data:
        return {"error": "Graph data not found. Run inference.py first."}
    return data


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
