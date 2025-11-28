from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import classify_expense, load_model

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
absolute_data_path = os.path.join(script_dir, "models", "expense_classifier.pkl")
# Load model at startup
MODEL_PATH = absolute_data_path
model = load_model(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the category of an expense based on its description.

    Request format:
    {
        "activity": "makan nasi goreng"
    }

    Response format:
    {
        "category": "Makanan",
        "confidence": 0.92
    }
    """
    # Get request data
    data = request.get_json()

    if not data or "activity" not in data:
        return jsonify({"error": "Missing required parameter: activity"}), 400

    activity = data["activity"]

    # Make prediction
    category, confidence = classify_expense(activity, model=model)
    recogzined = True if confidence > 0.8 else False
    # Return prediction
    return jsonify(
        {
            "activity": activity,
            "category": category,
            "confidence": float(confidence),
            "recognized": recogzined,
        }
    )


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """
    Predict categories for multiple expenses at once.

    Request format:
    {
        "activities": ["makan nasi goreng", "beli sepatu olahraga", ...]
    }

    Response format:
    {
        "predictions": [
            {"activity": "makan nasi goreng", "category": "Makanan", "confidence": 0.92},
            {"activity": "beli sepatu olahraga", "category": "Belanja", "confidence": 0.87},
            ...
        ]
    }
    """
    # Get request data
    data = request.get_json()

    if not data or "activities" not in data or not isinstance(data["activities"], list):
        return (
            jsonify(
                {"error": "Missing or invalid parameter: activities (should be a list)"}
            ),
            400,
        )

    activities = data["activities"]
    print(f"Activities: {activities}")

    # Make predictions
    predictions = []
    for activity in activities:
        category, confidence = classify_expense(activity, model=model)
        predictions.append(
            {
                "activity": activity,
                "category": category,
                "confidence": float(confidence),
            }
        )

    # Return predictions
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    # Start the Flask application
    print("Starting AutoExpense Categorizer API...")
    app.run(debug=True, host="0.0.0.0", port=8080)
