# Customer Churn Intelligence Portal

A full-stack predictive evaluation and analysis portal built for the Telco Customer Churn test cohort, powered directly by a trained Decision Tree classification pipeline.

---

## Key Capabilities

- **On-Demand Inference**:
  - **Predict All Cohort (1,409)**: Batch inference over all 1,409 evaluation customers with a single click.
  - **Individual Predict**: Instant real-time prediction for any customer directly from the table row.
  - **Reset**: Reset all predictions back to unpredicted state for demonstrations.
- **Test Cohort Explorer**: Browse, filter (by risk tier, contract, internet service), and sort the 1,409 stratified test customers.
- **One-Click Customer Dossier**: Inspect any customer's churn risk probability, model verdict, and comparison against ground truth.
- **Explainability & Signal Breakdown**: Feature contribution breakdown (`tenure`, `Contract_Two year`, `InternetService_Fiber optic`, `PaymentMethod_Electronic check`).
- **Decision Tree Path Trace**: Step-by-step audit of tree traversal from root split down to the exact leaf node and sample class distribution.
- **Interactive "What-If" Counterfactual Simulator**: Real-time evaluation of contract, payment method, or tenure adjustments.
- **Actionable Retention Playbook**: Personalized, rule-driven retention interventions for at-risk accounts.

---

## Project Structure

```
churn-portal/
├── backend/
│   ├── data/                 # Evaluation dataset (self-contained)
│   ├── libs/                 # Custom transformers (clean_cls, ctf, etc.)
│   ├── models/               # Serialized Decision Tree pipeline (pipeline_dt.pkl)
│   ├── model_service.py      # Core inference & analysis service
│   └── server.py             # Flask API & static file server
├── frontend/
│   ├── index.html            # Main UI layout
│   ├── style.css             # Editorial financial console design system
│   └── app.js                # Interactive client logic & API bindings
├── requirements.txt          # Python dependencies
├── Procfile                  # Production WSGI process declaration
├── .gitignore
└── README.md
```

---

## Running Locally

```bash
# 1. Navigate to the repository root
cd churn-portal

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
python backend/server.py
```

Then open **[http://localhost:5050](http://localhost:5050)** in your browser.

---

## 1-Click Cloud Deployment (Render / Railway)

### Deploying on Render.com
1. Push this repository to GitHub.
2. Go to [Render.com](https://render.com) and click **New > Web Service**.
3. Connect your GitHub repository.
4. Set:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT backend.server:app`
5. Click **Deploy Web Service**.
