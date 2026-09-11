# ML Price Predictor — Web App

A project cost estimator: a static HTML/JS frontend backed by a scikit-learn
RandomForestRegressor model, served as a Python serverless function on Vercel.

## Structure

```
.
├── index.html            # Frontend (static, served by Vercel)
├── api/
│   ├── predict_cost.py   # Flask app -> Vercel Python serverless function
│   └── model_cost.pkl    # Trained model (bundled with the function)
├── train_model.py        # Retrain the model on new/real data
├── projects.csv          # Training data
├── training_report.json  # Metrics from the last training run
├── requirements.txt      # Python deps for the serverless function
└── vercel.json           # Vercel config
```

## How it works

- `index.html` is plain HTML/CSS/JS — no build step. It POSTs form data to `/api/predict_cost`.
- `api/predict_cost.py` is a Flask app. Vercel's Python runtime automatically
  detects the `app` WSGI object and wraps it as a serverless function —
  no extra glue code needed.
- The model is loaded once per cold start from `model_cost.pkl`, which sits
  next to the function so it's bundled into the deployment automatically.

## Deploy to Vercel

1. Push this folder to a GitHub repo.
2. Go to [vercel.com/new](https://vercel.com/new) and import the repo.
3. Framework preset: **Other** (no build step needed).
4. Deploy. Vercel will detect `api/predict_cost.py` and `requirements.txt`
   automatically and provision the Python function.

Or via CLI:

```bash
npm i -g vercel
vercel        # deploy a preview
vercel --prod # deploy to production
```

## Local development

```bash
pip install -r requirements.txt
cd api && python predict.py   # runs Flask dev server on :5000
```

Then open `index.html` directly, or point the frontend's fetch URL to
`http://localhost:5000/api/predict_cost` for local testing.

## Retraining the model

```bash
pip install -r requirements.txt
python train_model.py
```

This regenerates `model_cost.pkl` and `training_report.json`. Copy the new
`model_cost.pkl` into `api/` before redeploying.

## API

**POST** `/api/predict_cost`

```json
{
  "task_complexity": 3,
  "team_size": 5,
  "effective_hours": 12.5,
  "experience": 2
}
```

Response:

```json
{ "success": true, "predicted_cost": 1827.32 }
```
