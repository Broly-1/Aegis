# Deployment Guide: Aegis GNN Fraud Detection (GCP + Vercel)

This guide explains how to deploy your high-scale GNN ensemble system:
- **Backend (FastAPI):** Google Cloud Run (Containerized).
- **Frontend (React/Vite):** Vercel.

---

## 1. Backend: Google Cloud Run
Cloud Run is perfect for this because it handles containerized scaling automatically.

### Step A: Preparation
Ensure your `requirements.txt` includes `xgboost`, `scikit-learn`, and `joblib`. (Already updated).

### Step B: Deploy to GCloud
Run these commands from your root directory:

1. **Build and Submit to Artifact Registry:**
   ```powershell
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/aegis-api
   ```
   *(Replace `[PROJECT-ID]` with your actual GCP project ID)*

2. **Deploy to Cloud Run:**
   ```powershell
   gcloud run deploy aegis-api --image gcr.io/[PROJECT-ID]/aegis-api --platform managed --allow-unauthenticated --memory 4Gi
   ```
   *Note: We request 4Gi memory to handle the model loading.*

3. **Get your URL**: Once deployed, GCP will give you a Service URL like `https://aegis-api-xyz.a.run.app`.

---

## 2. Frontend: Vercel

### Step A: Update Config
Update `frontend/src/config.js` to point to your Cloud Run URL:
```javascript
export const API_URL = "https://aegis-api-xyz.a.run.app/api";
```

### Step B: Deploy
1. Install Vercel CLI: `npm i -g vercel`
2. Run deployment from the root:
   ```powershell
   vercel
   ```
3. When prompted:
   - **Link to existing project?** No
   - **Project name?** `aegis-frontend`
   - **Root directory?** `frontend/`
   - **Framework?** `Vite`

---

## 3. Production Weights & Data
For the project to be "Live," ensure these files are included in your container:
- `hyper_elite_medium_model.pth` (GNN Weights)
- `xgboost_ensemble_model.pkl` (Ensemble Sniper)
- `threshold.txt` (Optimized Decision Boundary)
- `data/` folder (Pre-computed JSON results)

---
**Your enterprise-grade fraud detection system is now live!** 🚀
