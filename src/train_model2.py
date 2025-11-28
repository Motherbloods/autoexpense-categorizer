import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import nltk
import numpy as np
import pandas as pd

from prepare_data import prepare_dataset


# ============================================================================
# ENHANCED CATEGORY KEYWORDS
# ============================================================================
CATEGORY_KEYWORDS = {
    "transportasi": {
        "strong": [
            "bensin",
            "solar",
            "pertamax",
            "parkir",
            "tol",
            "grab",
            "gojek",
            "ojek",
            "taxi",
            "kereta",
            "pesawat",
            "busway",
            "spbu",
            "motor",
            "mobil",
            "berkendara",
            "terbang",
            "naik",
            "pergi",
            "angkot",
            "bis",
            "bus",
        ],
        "medium": ["kendaraan", "transportasi", "jalan"],
        "weak": [],
    },
    "tagihan": {
        "strong": [
            "listrik",
            "token",
            "pln",
            "cicilan",
            "angsuran",
            "kpr",
            "pdam",
            "internet",
            "wifi",
            "indihome",
            "air",
            "gas",
            "iuran",
            "premi",
            "tagihan",
            "bayar",
            "membayar",
            "biaya",
            "asuransi",
        ],
        "medium": ["bulanan", "pinjaman", "hutang", "kredit", "cicil"],
        "weak": [],
    },
    "langganan": {
        "strong": [
            "pulsa",
            "paket",
            "data",
            "kuota",
            "netflix",
            "spotify",
            "youtube",
            "premium",
            "membership",
            "berlangganan",
            "mendaftar",
            "daftar",
            "subscribe",
            "subscription",
        ],
        "medium": ["voucher", "tri", "telkomsel", "xl", "indosat", "operator"],
        "weak": [],
    },
    "makanan": {
        "strong": [
            "nasi",
            "ayam",
            "bakso",
            "soto",
            "mie",
            "restoran",
            "warteg",
            "cafe",
            "gofood",
            "grabfood",
            "kopi",
            "makan",
            "sarapan",
            "jajan",
            "snack",
            "minuman",
            "makanan",
        ],
        "medium": ["pizza", "burger", "sate", "rendang", "goreng", "roti", "seafood"],
        "weak": [],
    },
    "belanja": {
        "strong": [
            "baju",
            "celana",
            "sepatu",
            "tas",
            "kaos",
            "jaket",
            "shopee",
            "tokopedia",
            "lazada",
            "pakaian",
            "fashion",
            "beli",
            "belanja",
        ],
        "medium": ["market", "supermarket", "indomaret", "alfamart", "mall", "toko"],
        "weak": [],
    },
    "hiburan": {
        "strong": [
            "bioskop",
            "film",
            "cinema",
            "cgv",
            "xxi",
            "konser",
            "karaoke",
            "game",
            "ps5",
            "ps4",
            "bowling",
            "nonton",
            "main",
            "hiburan",
            "tengok",
            "tonton",
        ],
        "medium": ["movie", "musik", "lagu", "timezone", "arcade"],
        "weak": [],
    },
    "rekreasi": {
        "strong": [
            "liburan",
            "wisata",
            "tour",
            "pantai",
            "gunung",
            "hotel",
            "villa",
            "penginapan",
            "resort",
            "traveling",
            "vacation",
            "rekreasi",
            "piknik",
            "jalan",
            "berjalan",
            "berkemah",
            "mendaki",
            "hutan",
        ],
        "medium": ["perjalanan", "alam", "destinasi", "kawasan", "desa", "danau"],
        "weak": [],
    },
    "perawatan": {
        "strong": [
            "salon",
            "rambut",
            "spa",
            "massage",
            "facial",
            "dokter",
            "obat",
            "apotek",
            "skincare",
            "potong",
            "creambath",
            "perawatan",
            "treatment",
            "kosmetik",
            "vitamin",
            "gigi",
            "kuku",
        ],
        "medium": ["medical", "kesehatan", "sehat", "tubuh", "kulit"],
        "weak": [],
    },
    "elektronik": {
        "strong": [
            "laptop",
            "hp",
            "handphone",
            "smartphone",
            "komputer",
            "iphone",
            "macbook",
            "asus",
            "samsung",
            "sony",
            "monitor",
            "keyboard",
            "charger",
            "headset",
            "speaker",
            "mouse",
            "gadget",
            "drone",
            "kamera",
            "printer",
        ],
        "medium": ["elektronik", "digital", "teknologi", "perangkat", "device"],
        "weak": [],
    },
    "jasa_profesional": {
        "strong": [
            "konsultan",
            "lawyer",
            "pengacara",
            "notaris",
            "akuntan",
            "desainer",
            "pajak",
            "jasa",
            "service",
            "kursus",
            "les",
            "mentor",
            "profesional",
            "ahli",
            "konsultasi",
        ],
        "medium": ["layanan", "pembuatan", "perencanaan", "manajemen", "konten"],
        "weak": [],
    },
}


# ============================================================================
# CONTEXT-AWARE RULE ENGINE (IMPROVED)
# ============================================================================
class ContextAwareKeywordExtractor:
    """Rule-based context analyzer for disambiguating predictions"""

    @staticmethod
    def apply_rules(activity):
        """
        Apply context rules BEFORE model prediction.
        Returns category if rule matches, None otherwise.
        """
        text = activity.lower().strip()

        # RULE 0: Check for negations first - handle "tidak jadi" patterns
        negation_words = ["tidak", "bukan", "jangan"]
        if any(word in text for word in negation_words):
            # If negation present, let model decide (rules don't apply)
            return None

        # RULE 1: "beli pulsa" or "top up" = langganan (NOT belanja/elektronik)
        if any(
            word in text
            for word in ["pulsa", "top up", "top-up", "kuota", "paket data"]
        ):
            if any(action in text for action in ["beli", "isi", "top"]):
                return "langganan"

        # RULE 1B: "voucher game" = langganan (NOT belanja/elektronik)
        if "voucher" in text and "game" in text:
            return "langganan"

        # RULE 2: "bayar cicilan X" = tagihan (where X is vehicle/property)
        if any(word in text for word in ["cicilan", "angsuran", "kredit", "cicil"]):
            if any(action in text for action in ["bayar", "membayar"]):
                return "tagihan"

        # RULE 3: "bayar token/listrik/air/gas/wifi" = tagihan (NOT elektronik)
        if any(action in text for action in ["bayar", "membayar", "isi", "top"]):
            if any(
                utility in text
                for utility in [
                    "token",
                    "listrik",
                    "air",
                    "pdam",
                    "gas",
                    "internet",
                    "wifi",
                    "indihome",
                ]
            ):
                return "tagihan"

        # RULE 3B: Asuransi = tagihan
        if "asuransi" in text:
            if any(action in text for action in ["bayar", "membayar", "bayarin"]):
                return "tagihan"

        # RULE 4: "konsultan pajak" = jasa_profesional (NOT tagihan)
        if any(
            word in text
            for word in [
                "konsultan",
                "pengacara",
                "lawyer",
                "akuntan",
                "notaris",
                "desainer",
            ]
        ):
            if any(action in text for action in ["bayar", "ke", "konsultasi", "jasa"]):
                return "jasa_profesional"

        # RULE 5: "belanja" alone (without payment keywords) = belanja
        if "belanja" in text:
            payment_words = ["bayar", "cicilan", "tagihan", "angsuran", "kredit"]
            if not any(word in text for word in payment_words):
                return "belanja"

        # RULE 5B: "beli" + shopping keywords = belanja
        if "beli" in text:
            shopping_keywords = [
                "baju",
                "celana",
                "sepatu",
                "tas",
                "pakaian",
                "toko",
                "supermarket",
            ]
            if any(word in text for word in shopping_keywords):
                payment_words = ["bayar", "cicilan", "tagihan", "angsuran"]
                if not any(word in text for word in payment_words):
                    return "belanja"

        # RULE 6: Payment + vehicle (but not cicilan) = transportasi
        if any(payment in text for payment in ["bayar", "membayar", "isi"]):
            if any(
                word in text
                for word in [
                    "parkir",
                    "tol",
                    "bensin",
                    "solar",
                    "motor",
                    "mobil",
                    "kereta",
                    "bus",
                    "angkot",
                ]
            ):
                if "cicilan" not in text and "angsuran" not in text:
                    return "transportasi"

        # RULE 7: Rekreasi context (piknik, liburan dengan makanan) = rekreasi
        if any(word in text for word in ["piknik", "liburan", "rekreasi", "wisata"]):
            if not any(word in text for word in ["bayar", "cicilan", "tagihan"]):
                return "rekreasi"

        return None


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract keyword-based features with weighted scoring
    """

    def __init__(self):
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for text in X:
            text_lower = text.lower().strip()
            feature_vector = []

            for category, keyword_dict in CATEGORY_KEYWORDS.items():
                score = 0

                # Strong keywords: weight 9
                for keyword in keyword_dict["strong"]:
                    if keyword in text_lower:
                        if keyword == text_lower or f" {keyword} " in f" {text_lower} ":
                            score += 11
                        else:
                            score += 9

                # Medium keywords: weight 4
                for keyword in keyword_dict["medium"]:
                    if keyword in text_lower:
                        score += 4

                # Weak keywords: weight 1
                for keyword in keyword_dict["weak"]:
                    if keyword in text_lower:
                        score += 1

                feature_vector.append(score)

            features.append(feature_vector)

        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return [f"keyword_{cat}" for cat in CATEGORY_KEYWORDS.keys()]


# ============================================================================
# RULE-AWARE CLASSIFIER WRAPPER
# ============================================================================
class RuleAwareClassifier:
    """Wrapper that applies rule engine before model prediction"""

    def __init__(self, model):
        self.model = model
        self.rule_engine = ContextAwareKeywordExtractor()

    def predict(self, X):
        """Apply rules first, then model"""
        predictions = []
        for activity in X:
            # Try rule first
            rule_prediction = self.rule_engine.apply_rules(activity)
            if rule_prediction is not None:
                predictions.append(rule_prediction)
            else:
                # Fall back to model
                pred = self.model.predict([activity])[0]
                predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X):
        """Return model probabilities"""
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes_


# ============================================================================
# TRAIN MODEL FUNCTION
# ============================================================================
def train_model(X, y, output_path="../models/expense_classifier.pkl"):
    """
    Train an improved expense classifier with weighted keyword features
    and rule-based disambiguation.
    """
    # Download NLTK resources
    try:
        stop_words = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("indonesian"))

    # Add custom stopwords (NOT including action verbs like beli, bayar, isi)
    custom_stopwords = {
        "ke",
        "di",
        "dari",
        "dengan",
        "dan",
        "yang",
        "ini",
        "itu",
        "ada",
        "rumah",
        "atau",
        "pada",
        "oleh",
        "sama",
        "hal",
    }
    stop_words.update(custom_stopwords)
    stop_words = list(stop_words)

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())

    # Create pipeline with feature union
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                token_pattern=r"\b\w+\b",
                                stop_words=stop_words,
                                ngram_range=(1, 3),
                                min_df=1,
                                max_df=0.9,
                                max_features=1000,
                            ),
                        ),
                        ("keywords", KeywordFeatureExtractor()),
                    ],
                    transformer_weights={
                        "tfidf": 1.0,
                        "keywords": 15.0,  # Increased from 12.0
                    },
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    n_estimators=300,  # Increased from 250
                    max_depth=35,  # Increased from 30
                    min_samples_split=2,  # Decreased from 3
                    min_samples_leaf=1,
                ),
            ),
        ]
    )

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # Wrap with rule engine
    print("Applying rule-based wrapper...\n")
    final_model = RuleAwareClassifier(pipeline)

    # Evaluate
    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{'='*60}")
    print("TEST SET PERFORMANCE (WITH RULES)")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving model to {output_path}...")
    with open(output_path, "wb") as file:
        pickle.dump(final_model, file)

    return final_model


# ============================================================================
# TEST MODEL WITH EXAMPLES
# ============================================================================
def test_model_with_examples(model):
    """Test with comprehensive examples"""
    print(f"\n{'='*60}")
    print("TESTING WITH EXAMPLE ACTIVITIES")
    print(f"{'='*60}\n")

    test_cases = [
        ("beli nasi goreng", "makanan"),
        ("bayar bensin", "transportasi"),
        ("beli baju", "belanja"),
        ("bayar cicilan motor", "tagihan"),
        ("nonton bioskop", "hiburan"),
        ("beli pulsa", "langganan"),
        ("bayar parkir", "transportasi"),
        ("makan di restoran", "makanan"),
        ("bayar token listrik", "tagihan"),
        ("belanja bulanan", "belanja"),
    ]

    correct = 0
    total = len(test_cases)

    for activity, expected in test_cases:
        prediction = model.predict([activity])[0]
        is_correct = prediction == expected
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} '{activity}' → {prediction} (expected: {expected})")

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*60}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Loading dataset...")
    df, X, y = prepare_dataset()

    print(f"\nDataset Overview:")
    print(f"Total samples: {len(X)}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().sort_index())

    # Train model
    trained_model = train_model(X, y)

    # Test model
    test_model_with_examples(trained_model)

    print("\n✓ Training complete!")
