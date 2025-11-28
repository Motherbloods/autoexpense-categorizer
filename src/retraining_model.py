import json
import os
from datetime import datetime
from threading import Thread
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from src.train_model import train_model


class MetadataTracker:
    """Track metadata untuk setiap retraining cycle"""

    def __init__(self, metadata_file="data/retraining_metadata.jsonl"):
        self.metadata_file = metadata_file
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        if not os.path.exists(metadata_file):
            Path(metadata_file).touch()

    def save_retraining_metadata(self, metadata):
        """Save metadata ke file"""
        with open(self.metadata_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

    def get_all_metadata(self):
        """Get semua metadata"""
        metadata_list = []
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        metadata_list.append(json.loads(line))
        except Exception as e:
            print(f"Error reading metadata: {e}")
        return metadata_list

    def get_latest_metadata(self):
        """Get metadata terakhir"""
        all_metadata = self.get_all_metadata()
        return all_metadata[-1] if all_metadata else None

    def get_metadata_comparison(self):
        """Get perbandingan antara retraining terakhir"""
        all_metadata = self.get_all_metadata()
        if len(all_metadata) < 2:
            return None

        prev = all_metadata[-2]  # Sebelum retraining
        curr = all_metadata[-1]  # Sesudah retraining

        return {
            "before": prev,
            "after": curr,
            "improvements": {
                "accuracy_change": (
                    curr["model_metrics"]["accuracy"]
                    - prev["model_metrics"]["accuracy"]
                ),
                "f1_score_change": (
                    curr["model_metrics"]["f1_score"]
                    - prev["model_metrics"]["f1_score"]
                ),
                "training_samples_added": (
                    curr["training_data"]["total_samples"]
                    - prev["training_data"]["total_samples"]
                ),
            },
        }


class AutoRetrainingManager:
    """Manages automatic model retraining based on correction dataset"""

    def __init__(
        self,
        training_data_file="data/training_data.json",  # Changed to .json
        model_path="models/expense_classifier.pkl",
        metadata_file="data/retraining_metadata.jsonl",
    ):
        self.training_data_file = training_data_file
        self.model_path = model_path
        self.is_retraining = False
        self.last_retraining_time = None
        self.retraining_logs = []
        self.metadata_tracker = MetadataTracker(metadata_file)

        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(training_data_file), exist_ok=True)

    def _add_log(self, message, level="info"):
        """Tambah log retraining"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        self.retraining_logs.append(log_entry)

        # Print ke console
        prefix = "ℹ️" if level == "info" else "⚠️" if level == "warning" else "❌"
        print(f"{prefix} {message}")

    def _get_model_metrics_before(self):
        """Get metrics model sebelum retraining"""
        try:
            # Load JSON training data
            with open(self.training_data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            categories = df["category"].value_counts().to_dict()

            return {
                "total_samples": len(df),
                "categories": categories,
                "timestamp": datetime.now().isoformat(),
            }
        except FileNotFoundError:
            self._add_log(
                f"Training data file tidak ditemukan: {self.training_data_file}",
                level="warning",
            )
            return {
                "total_samples": 0,
                "categories": {},
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self._add_log(f"Error getting metrics before: {e}", level="warning")
            return {
                "total_samples": 0,
                "categories": {},
                "timestamp": datetime.now().isoformat(),
            }

    def _get_model_metrics_after(self, merged_df, y_pred, y_test):
        """Get metrics model setelah retraining"""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        try:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

            categories = merged_df["category"].value_counts().to_dict()

            return {
                "total_samples": len(merged_df),
                "categories": categories,
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self._add_log(f"Error getting metrics after: {e}", level="warning")
            return {}

    def trigger_retraining(self, corrections_data):
        """
        Trigger retraining dengan data koreksi dari backend

        Args:
            corrections_data: List of dicts
        """
        if self.is_retraining:
            return {
                "success": False,
                "error": "Retraining sedang berjalan",
                "message": "Tunggu retraining sebelumnya selesai",
            }

        if not corrections_data or len(corrections_data) == 0:
            return {
                "success": False,
                "error": "Data koreksi kosong",
                "message": "Minimal ada 1 data koreksi",
            }

        # Validate data format
        valid_corrections = []
        for item in corrections_data:
            if "activity" in item and "category" in item:
                valid_corrections.append(item)

        if len(valid_corrections) == 0:
            return {
                "success": False,
                "error": "Format data tidak valid",
                "message": "Setiap item harus punya 'activity' dan 'category'",
            }

        print("\n" + "=" * 60)
        print(f"🔄 RETRAINING TRIGGERED DARI BACKEND")
        print(f"Total koreksi: {len(valid_corrections)}")
        print("=" * 60 + "\n")

        retraining_thread = Thread(
            target=self._perform_retraining, args=(valid_corrections,)
        )
        retraining_thread.daemon = True
        retraining_thread.start()

        return {
            "success": True,
            "message": "Retraining dimulai di background",
            "corrections_count": len(valid_corrections),
            "is_retraining": True,
        }

    def _merge_datasets(self, new_corrections):
        """Merge original training data dengan corrections baru"""
        original_data = []
        if os.path.exists(self.training_data_file):
            try:
                # Load JSON file
                with open(self.training_data_file, "r", encoding="utf-8") as f:
                    original_data = json.load(f)

                self._add_log(
                    f"Loaded {len(original_data)} samples dari {self.training_data_file}"
                )
            except Exception as e:
                self._add_log(f"Error loading original data: {e}", level="warning")

        # Merge data
        merged_data = original_data + new_corrections
        self._add_log(f"Total samples untuk training: {len(merged_data)}")

        # Save updated training data back to JSON
        try:
            with open(self.training_data_file, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            self._add_log(f"Updated training data saved to {self.training_data_file}")
        except Exception as e:
            self._add_log(f"Error saving merged data: {e}", level="error")

        return pd.DataFrame(merged_data)

    def _perform_retraining(self, corrections_data):
        """Perform actual model retraining (background process)"""
        try:
            self.is_retraining = True

            # Get metrics SEBELUM retraining
            metrics_before = self._get_model_metrics_before()

            self._add_log("=" * 60)
            self._add_log("MEMULAI RETRAINING PROCESS")
            self._add_log("=" * 60)

            # Merge datasets
            self._add_log("Merging training data dengan corrections baru...")
            merged_df = self._merge_datasets(corrections_data)

            # Show distribution
            self._add_log(f"Distribusi kategori:")
            dist = merged_df["category"].value_counts().to_dict()
            for category, count in dist.items():
                self._add_log(f"  - {category}: {count}")

            # Prepare training data
            X = merged_df["activity"].values
            y = merged_df["category"].values

            # Train new model (function ini return y_pred dan y_test)
            self._add_log("Training model baru...")
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train dan evaluate
            new_model = train_model(X, y, output_path=self.model_path)

            # Get predictions untuk metrics
            from sklearn.pipeline import Pipeline

            if isinstance(new_model, Pipeline):
                y_pred = new_model.predict(X_test)
            else:
                # Jika wrapped dengan RuleAwareClassifier
                y_pred = new_model.predict(X_test)

            # Get metrics SETELAH retraining
            metrics_after = self._get_model_metrics_after(merged_df, y_pred, y_test)

            # Update timestamp
            self.last_retraining_time = datetime.now().isoformat()

            # Save metadata
            metadata = {
                "retraining_id": self._generate_id(),
                "timestamp": self.last_retraining_time,
                "status": "success",
                "corrections_applied": len(corrections_data),
                "training_data": {
                    "total_samples": len(merged_df),
                    "categories": dist,
                },
                "model_metrics": metrics_after,
                "metrics_before": metrics_before,
                "model_path": self.model_path,
                "test_set_size": len(X_test),
            }

            self.metadata_tracker.save_retraining_metadata(metadata)
            self._add_log(f"Metadata disimpan")

            self._add_log("=" * 60)
            self._add_log("✅ RETRAINING BERHASIL!")
            self._add_log(f"Accuracy: {metrics_after.get('accuracy', 'N/A'):.4f}")
            self._add_log(f"F1-Score: {metrics_after.get('f1_score', 'N/A'):.4f}")
            self._add_log(f"Waktu: {self.last_retraining_time}")
            self._add_log("=" * 60)

        except Exception as e:
            self._add_log(f"ERROR saat retraining: {e}", level="error")
            import traceback

            traceback.print_exc()

        finally:
            self.is_retraining = False

    def _generate_id(self):
        """Generate unique ID untuk retraining"""
        from datetime import datetime

        return f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_status(self):
        """Get current retraining status"""
        return {
            "is_retraining": self.is_retraining,
            "last_retraining_time": self.last_retraining_time,
            "logs": self.retraining_logs[-10:],
        }

    def get_all_logs(self):
        """Get semua retraining logs"""
        return self.retraining_logs

    def get_metadata_history(self):
        """Get history semua retraining metadata"""
        return self.metadata_tracker.get_all_metadata()

    def get_metadata_comparison(self):
        """Get perbandingan retraining terakhir"""
        return self.metadata_tracker.get_metadata_comparison()

    def get_latest_metadata(self):
        """Get metadata retraining terakhir"""
        return self.metadata_tracker.get_latest_metadata()
