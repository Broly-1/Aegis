# Aegis GNN Fraud Detection (MMORPG)

End-to-end fraud detection for MMORPG transaction networks using GraphSAGE + optional XGBoost ensemble, with a FastAPI backend and a React dashboard.

## What This Project Does
- Builds player-level graph features from raw transaction logs.
- Trains a GraphSAGE model to detect fraudulent players.
- Optionally trains an XGBoost stage-2 ensemble on learned embeddings.
- Runs offline inference to generate dashboard-ready JSON artifacts.
- Serves analytics and graph visualizations via a FastAPI API and React UI.

## Repository Layout
- `traingnn.py` - GNN training (GraphSAGE) + evaluation artifacts.
- `train_xgboost.py` - Optional stage-2 ensemble training (uses GNN embeddings).
- `inference.py` - Full-scale inference + data export for UI.
- `api/server.py` - FastAPI backend serving precomputed data.
- `frontend/` - React + Vite dashboard.
- `data/` - Precomputed JSON outputs for the UI.
- `reports/` - Training charts and evaluation outputs (created by `traingnn.py`).
- `TRAINING_RESULTS.md` - Detailed explanation of training outputs.
- `project_overview.md` - Architecture and pipeline overview.
- `DEPLOYMENT_GUIDE.md` - GCP + Vercel deployment steps.

## Setup
### Python
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend
```powershell
cd frontend
npm install
```

## Data Preparation
If you already have the cleaned dataset, place it at the project root:
- `MMORPG_Medium_Cleaned.csv`

If you need to clean raw data first:
```powershell
python test.py
```
This will generate `MMORPG_Medium_Cleaned.csv` based on your raw dataset paths.

## Train the GNN Model
```powershell
python traingnn.py
```
Outputs:
- `hyper_elite_medium_model.pth`
- `threshold.txt`
- `reports/` (training charts, confusion matrix, PR curve, metrics, sample predictions)

Optional environment variables:
- `EPOCHS` (default 20)
- `EVAL_SAMPLE_SIZE` (default 20000)
- `SAMPLE_PREDICTIONS` (default 200)

## Optional: Train the XGBoost Ensemble
```powershell
python train_xgboost.py
```
Outputs:
- `xgboost_ensemble_model.pkl`
- `threshold.txt` (overwrites the existing threshold)

Note: `inference.py` will automatically use the XGBoost model if `xgboost_ensemble_model.pkl` is present.

## Run Inference (Generate Dashboard Data)
```powershell
python inference.py
```
Outputs in `data/`:
- `players.json`
- `dashboard.json`
- `graph.json`

## Start the API
```powershell
uvicorn api.server:app --reload
```
API endpoints:
- `GET /api/dashboard`
- `GET /api/players`
- `GET /api/players/{player_id}`
- `GET /api/graph`
- `POST /api/simulate`

## Start the Frontend
```powershell
cd frontend
npm run dev
```
Update [frontend/src/config.js](frontend/src/config.js) if your API URL changes.

## Training Reports and Charts
Training outputs are stored in `reports/` and explained in [TRAINING_RESULTS.md](TRAINING_RESULTS.md). These include:
- Training loss and metrics curves
- Precision-recall curve
- Confusion matrix
- Final classification report
- Sample prediction outputs

## Deployment
See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for GCP + Vercel instructions.

## Notes
- The dataset is large. Expect training and inference to take significant time and memory.
- The API serves precomputed JSON for speed. Re-run `inference.py` after retraining.
