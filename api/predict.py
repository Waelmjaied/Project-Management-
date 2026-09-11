import os
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load both models relative to this file (serverless functions can have any CWD)
COST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cost.pkl")
DURATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_duration.pkl")

try:
    cost_model = joblib.load(COST_MODEL_PATH)
except Exception as e:
    cost_model = None
    print(f"[ERROR] Could not load cost model from '{COST_MODEL_PATH}': {e}")

try:
    duration_model = joblib.load(DURATION_MODEL_PATH)
except Exception as e:
    duration_model = None
    print(f"[ERROR] Could not load duration model from '{DURATION_MODEL_PATH}': {e}")

COST_FEATURES = ["task_complexity", "team_size", "effective_hours", "experience", "hourly_rate"]
DURATION_FEATURES = ["task_complexity", "team_size", "effective_hours", "experience"]


@app.route("/api/predict", methods=["GET"])
def get_predict():
    return jsonify({
        "message": "Send a POST request with task_complexity, team_size, effective_hours, experience, hourly_rate",
        "example_payload": {
            "task_complexity": 3,
            "team_size": 5,
            "effective_hours": 12.5,
            "experience": 2,
            "hourly_rate": 22,
        },
    })


@app.route("/api/predict", methods=["POST"])
def post_predict():
    if cost_model is None or duration_model is None:
        return jsonify({
            "success": False,
            "error": "One or more ML models failed to load. Check deployment bundle.",
        }), 500

    try:
        data = request.get_json(force=True)

        task_complexity = int(data.get("task_complexity", 0))
        team_size = int(data.get("team_size", 0))
        effective_hours = float(data.get("effective_hours", 0))
        experience = int(data.get("experience", 0))
        hourly_rate = float(data.get("hourly_rate", 0))

        X_cost = pd.DataFrame(
            [[task_complexity, team_size, effective_hours, experience, hourly_rate]],
            columns=COST_FEATURES,
        )
        X_duration = pd.DataFrame(
            [[task_complexity, team_size, effective_hours, experience]],
            columns=DURATION_FEATURES,
        )

        predicted_cost = cost_model.predict(X_cost)[0]
        predicted_duration = duration_model.predict(X_duration)[0]

        return jsonify({
            "success": True,
            "predicted_cost": round(float(predicted_cost), 2),
            "predicted_duration_days": round(float(predicted_duration), 1),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400


# Local dev only — Vercel imports `app` directly as a WSGI callable
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)