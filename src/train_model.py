import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk

from prepare_data import prepare_dataset


def train_model(X, y, output_path="../models/expense_classifier.pkl"):
    """
    Train an expense classifier model using TF-IDF and RandomForest.

    Args:
        X: Feature data (expense descriptions)
        y: Target data (expense categories)
        output_path: Path to save the trained model

    Returns:
        best_model: The trained model with best parameters
    """
    # Download necessary NLTK resources if not already downloaded
    try:
        stop_words = stopwords.words("indonesian")
    except LookupError:
        nltk.download("stopwords")
        stop_words = stopwords.words("indonesian")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create the pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"\b\w+\b",
                    stop_words=stop_words,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    # Define parameters for grid search
    parameters = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
    }

    # Perform grid search
    print("Starting grid search for optimal hyperparameters...")
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the model
    print(f"Saving the model to {output_path}...")
    with open(output_path, "wb") as file:
        pickle.dump(best_model, file)

    return best_model


def test_model_with_examples(model):
    """
    Test the trained model with example activities.

    Args:
        model: Trained model for expense classification
    """
    print("\nTesting with example activities:")
    test_activities = [
        "makan nasi goreng",
        "beli sepatu olahraga",
        "nonton konser musik",
        "bayar tagihan listrik",
        "service motor",
        "beli buku novel",
        "makan pizza",
        "nonton drama Korea",
        "bayar asuransi mobil",
    ]

    for activity in test_activities:
        prediction = model.predict([activity])[0]
        probabilities = model.predict_proba([activity])[0]
        confidence = max(probabilities)
        print(f"Activity: '{activity}'")
        print(f"Predicted category: '{prediction}' with {confidence:.2f} confidence\n")


if __name__ == "__main__":
    # Prepare data
    df, X, y = prepare_dataset()

    # Train model
    trained_model = train_model(X, y)

    # Test model with examples
    test_model_with_examples(trained_model)

    print("Model training and evaluation complete.")
