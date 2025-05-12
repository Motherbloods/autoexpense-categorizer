import pickle
import os


def load_model(file_path="../models/expense_classifier.pkl"):
    """
    Load a trained expense classifier model from file.

    Args:
        file_path (str): Path to the saved model file

    Returns:
        object: Loaded model
    """
    try:
        with open(file_path, "rb") as file:
            loaded_model = pickle.load(file)
        print(f"Model loaded successfully from {file_path}")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def classify_expense(
    activity, model=None, model_path="../models/expense_classifier.pkl"
):
    """
    Classify an expense activity into a category.

    Args:
        activity (str): Description of the expense activity
        model: Pre-loaded model (optional)
        model_path (str): Path to the model file if model not provided

    Returns:
        tuple: (prediction, confidence) - Category prediction and confidence score
    """
    # Load model if not provided
    if model is None:
        model = load_model(model_path)
        if model is None:
            return "Unknown", 0.0

    # Make prediction
    prediction = model.predict([activity])[0]
    probabilities = model.predict_proba([activity])[0]
    confidence = max(probabilities)

    return prediction, confidence


def batch_classify(activities, model_path="../models/expense_classifier.pkl"):
    """
    Classify a batch of expense activities.

    Args:
        activities (list): List of expense activity descriptions
        model_path (str): Path to the model file

    Returns:
        list: List of (activity, category, confidence) tuples
    """
    model = load_model(model_path)
    results = []

    if model is None:
        return [(activity, "Unknown", 0.0) for activity in activities]

    for activity in activities:
        category, confidence = classify_expense(activity, model)
        results.append((activity, category, confidence))

    return results


if __name__ == "__main__":
    # Test the classification function with some examples
    test_activities = [
        "makan nasi goreng",
        "beli sepatu olahraga",
        "nonton konser musik",
        "bayar tagihan listrik",
        "service motor",
    ]

    # Individual classification
    print("Testing individual classification:")
    model = load_model()
    for activity in test_activities[:2]:
        category, confidence = classify_expense(activity, model)
        print(f"Activity: '{activity}'")
        print(f"Predicted category: '{category}' with {confidence:.2f} confidence\n")

    # Batch classification
    print("Testing batch classification:")
    results = batch_classify(test_activities)
    for activity, category, confidence in results:
        print(f"Activity: '{activity}'")
        print(f"Predicted category: '{category}' with {confidence:.2f} confidence\n")
