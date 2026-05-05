# Project Overview: GNN Fraud Detection

This document explains the architecture and data flow of the MMORPG Fraud Detection system, detailing how the Python AI backend and the React frontend work together.

## System Architecture

The project is structured into three main phases: **Data Preparation & AI Training**, **Inference & API Serving**, and the **React Dashboard**.

### 1. Data Preparation & AI Training
- **`test.py` (Data Cleaner)**
  This script processes the raw massive datasets from the `archive (1)` folder (like `HI-Medium_Trans.csv`). It maps columns to standard names (`Sender_Player_ID`, `In_Game_Currency_Value`, etc.), drops missing values, and exports a unified `MMORPG_Medium_Cleaned.csv`.
- **`traingnn.py` (Model Trainer)**
  This is the core AI training script. 
  1. It loads the cleaned dataset and groups transactions by player to engineer node features (Total Sent, Total Received, Trade Counts, and Sent/Received Ratio).
  2. It builds an adjacency matrix (`edge_index`) connecting players who traded with each other.
  3. It defines the `HyperEliteSAGE` model—a 3-layer GraphSAGE (Graph Sample and Aggregate) neural network.
  4. It uses a heavily weighted `CrossEntropyLoss` (1:5 ratio) to handle the extreme class imbalance (millions of safe players vs. hundreds of hackers).
  5. It outputs the trained weights to `hyper_elite_medium_model.pth`.

### 2. Inference & API Serving
- **`inference.py` (Data Pipeline)**
  Instead of running the heavy PyTorch model dynamically on every web request, this script pre-computes everything. It loads the trained `hyper_elite_medium_model.pth`, runs the GraphSAGE model on the entire network of 1.3 million players, and saves the predictions and aggregations into fast, lightweight JSON files in the `data/` directory (`players.json`, `dashboard.json`, `graph.json`).
- **`api/server.py` (FastAPI Backend)**
  A lightweight REST API using FastAPI. When the server starts, it loads the JSON files into memory. It exposes several endpoints (`/api/dashboard`, `/api/players`, `/api/graph`) which the React frontend queries to fetch paginated data, search results, and analytical statistics instantly without needing to wake up PyTorch.

### 3. React Frontend Dashboard
- **`frontend/src/App.jsx` & `main.jsx`**
  The entry points for the React + Vite application. They set up the React Router for navigating between pages.
- **`frontend/src/components/`**
  Contains reusable UI elements:
  - `Sidebar.jsx`: The main navigation sidebar.
  - `StatCard.jsx`: Reusable animated cards for displaying top-level metrics.
- **`frontend/src/pages/`**
  - `Dashboard.jsx`: The main landing page showing model accuracy, risk pie charts, and top flagged players.
  - `Players.jsx`: A searchable, paginated table of all 1.3M players with their specific risk scores.
  - `GraphView.jsx`: An interactive force-directed canvas (using `react-force-graph-2d`) that visualizes the network topology of high-risk transactions.
  - `Analytics.jsx`: Deep dive charts (using `recharts`) showing fraud rates across different payment methods (ACH, Wire, Cheque) and transaction volume over time.

## How It All Works Together
1. **Offline**: You run `traingnn.py` to train the neural network on the transaction graph.
2. **Offline**: You run `inference.py` to generate the predictions and stats.
3. **Runtime**: You start the `api/server.py` backend.
4. **Runtime**: You start the React frontend. The frontend makes HTTP requests to the FastAPI backend, fetching the pre-computed graph data and displaying it in a highly responsive, modern interface.
