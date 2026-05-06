# Deployment Guide: Aegis GNN Fraud Detection

This guide explains how to deploy your GNN-powered fraud detection system using a **Serverless/PaaS** architecture:
- **Backend (FastAPI):** Deployed on **Render** (Web Service).
- **Frontend (React/Vite):** Deployed on **Vercel**.

---

## 1. Backend Deployment (Render)

Render will host your FastAPI server and serve the pre-computed GNN results.

### Step A: Create `requirements.txt`
Ensure you have a `requirements.txt` in your root folder with these dependencies:
```text
fastapi
uvicorn
torch
torch-geometric
pydantic
pandas
numpy
```

### Step B: Setup on Render
1.  Go to [Render.com](https://render.com/) and log in.
2.  Click **New +** > **Web Service**.
3.  Connect your GitHub repository.
4.  **Configure the service:**
    *   **Name:** `aegis-api`
    *   **Environment:** `Python 3`
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
5.  **Environment Variables:**
    *   Click **Advanced** > **Add Environment Variable**.
    *   Key: `PYTHON_VERSION` | Value: `3.10.x` (or your local version).

---

## 2. Frontend Deployment (Vercel)

Vercel is optimized for React/Vite apps.

### Step A: Update API URL
In your React code (likely `frontend/src/App.jsx` or a config file), update the base URL to point to your new Render URL:
```javascript
// Example:
const API_BASE_URL = "https://aegis-api.onrender.com"; 
```

### Step B: Setup on Vercel
1.  Go to [Vercel.com](https://vercel.com/) and log in.
2.  Click **Add New** > **Project**.
3.  Import your GitHub repository.
4.  **Edit Project Settings:**
    *   **Root Directory:** Select `frontend/`.
    *   **Framework Preset:** `Vite`.
    *   **Build Command:** `npm run build`.
    *   **Output Directory:** `dist`.
5.  Click **Deploy**.

---

## 3. Post-Deployment Checklist

1.  **CORS Settings:** Ensure `api/server.py` allows your Vercel URL:
    ```python
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://your-app-name.vercel.app"], 
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ```
2.  **Model Weights:** Verify `hyper_elite_medium_model.pth` is in your GitHub repo, as Render needs it to initialize the API.
3.  **Data Path:** The API looks for `data/players.json.gz`. Ensure the `data/` folder was pushed to GitHub.

---
**Your app is now multi-cloud serverless!** 🚀
