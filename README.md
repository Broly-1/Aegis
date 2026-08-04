# Aegis — Graph Neural Network Fraud Detection

Detecting circular currency laundering in a 32M-transaction MMORPG economy using a two-stage GraphSAGE + XGBoost ensemble, served through a FastAPI backend and a React analytics dashboard.

![Python](https://img.shields.io/badge/Python-PyTorch_Geometric-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EC4E20?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite_Dashboard-61DAFB?style=flat-square&logo=react&logoColor=black)

Artificial Intelligence semester project, FAST NUCES (Spring 2026).

---

## The problem

Gold-farming operations launder in-game currency through circular trade rings — A pays B pays C pays A — to disguise the origin of illicitly generated funds. On a per-transaction basis these trades look ordinary. The signal only exists in the topology of the trade graph, which is exactly what a graph neural network is for.

Two things made this hard beyond the usual classification setup:

**Extreme class imbalance.** Roughly 1 fraudulent account per 1,500 legitimate ones. A model that predicts "clean" for everything scores 99.93% accuracy and is worthless.

**Oversmoothing.** This was the real problem. Message passing over many hops makes neighbouring node embeddings converge, and legitimate high-volume traders — "safe whales" — have graph neighbourhoods that look structurally identical to laundering rings. Pure topology cannot separate them.

## Approach

A two-stage pipeline that deliberately does not rely on the graph alone:

**Stage 1 — GraphSAGE.** Learns 128-dimensional node embeddings capturing each account's position in the trade network. Trained with 10% biased neighbourhood sampling and focal loss to keep the minority class from being drowned out.

**Stage 2 — XGBoost.** Consumes the 128-dim graph embeddings *concatenated with* engineered tabular features — trade velocity, sent/received ratio, counterparty diversity, transaction timing. This is what resolves the oversmoothing problem: safe whales and laundering bots have similar topology but very different behavioural signatures, and the gradient-boosted trees draw a decision boundary the GNN alone could not.

**Threshold calibration.** The operating point is tuned explicitly against a precision-recall curve rather than left at 0.5, since the cost of a false positive (banning a paying customer) and a false negative (missing a bot) are not symmetric.

## Results

| Metric | Value |
|---|---|
| Precision | 36.2% |
| Recall | 41.0% |
| F1 | 38.4% |

Read those against the base rate: with roughly 1 fraudulent account in 1,500, random flagging yields ~0.07% precision. The ensemble is about **540× better than chance** while still recovering 41% of laundering accounts. On a real moderation queue that is the difference between an unusable alert stream and a reviewable one.

The dashboard exposes the full PR curve and confusion matrix with an interactive threshold slider, so the precision/recall trade-off can be set by whoever owns the moderation policy rather than baked into the model.

## Pipeline

```
raw transaction logs
  → test.py              clean + aggregate to player level
  → traingnn.py          GraphSAGE training, embeddings, eval artifacts
  → train_xgboost.py     stage-2 ensemble on embeddings + tabular features
  → inference.py         full-scale scoring, exports dashboard JSON
  → api/server.py        FastAPI serving precomputed artifacts
  → frontend/            React + Vite dashboard
```

Inference output is precomputed to JSON and served statically — scoring 32M transactions on request is not a viable API contract, and separating training from serving is how this would be deployed in practice.

## Repository layout

| Path | Purpose |
|---|---|
| `traingnn.py` | GraphSAGE training and evaluation artifacts |
| `train_xgboost.py` | Stage-2 ensemble training on learned embeddings |
| `inference.py` | Full-scale inference and dashboard data export |
| `api/server.py` | FastAPI backend |
| `frontend/` | React + Vite dashboard |
| `data/` | Precomputed JSON consumed by the UI |
| `reports/` | Training charts, PR curve, confusion matrix |
| `TRAINING_RESULTS.md` | Detailed breakdown of training outputs |
| `project_overview.md` | Architecture and pipeline notes |
| `DEPLOYMENT_GUIDE.md` | GCP + Vercel deployment |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

Place the cleaned dataset at the project root as `MMORPG_Medium_Cleaned.csv`, or run `python test.py` to generate it from raw data.

## Running

```bash
python traingnn.py          # → hyper_elite_medium_model.pth, threshold.txt, reports/
python train_xgboost.py     # → xgboost_ensemble_model.pkl  (optional but recommended)
python inference.py         # → data/players.json, dashboard.json, graph.json

uvicorn api.server:app --reload
cd frontend && npm run dev
```

`inference.py` picks up the XGBoost model automatically if `xgboost_ensemble_model.pkl` is present. Training accepts `EPOCHS` (default 20), `EVAL_SAMPLE_SIZE` (default 20000), and `SAMPLE_PREDICTIONS` (default 200) as environment variables. Point the dashboard at a different API host via `frontend/src/config.js`.

**API endpoints:** `GET /api/dashboard` · `GET /api/players` · `GET /api/players/{id}` · `GET /api/graph` · `POST /api/simulate`

> The dataset is large. Training and inference take significant time and memory.

## What I took away from it

I set out to avoid building a classroom demo, and the thing that made it feel real was discovering that the graph model alone was structurally incapable of the task. Oversmoothing wasn't a tuning problem — no amount of epochs was going to fix it — and the fix came from feature engineering and architecture rather than from the model. That, plus calibrating a threshold against actual cost asymmetry instead of reporting accuracy, was the whole education.

---

**Hassan Kamran** · [LinkedIn](https://www.linkedin.com/in/hassankamran3) · hassangaming111@gmail.com
