import json
import os
from datetime import datetime
from typing import Any


class ModelTracker:
    def __init__(self, tracking_file: str = "model_tracking.json") -> None:
        self.tracking_file = tracking_file
        self.models: dict[str, dict[str, Any]] = self._load_tracking_data()

    def _load_tracking_data(self) -> dict[str, dict[str, Any]]:
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file) as f:
                return json.load(f)
        return {}

    def _save_tracking_data(self) -> None:
        with open(self.tracking_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def add_model(
        self,
        model_path: str,
        parent_models: list[str],
        merge_techniques: list[str],
        config: dict[str, Any],
        score: float,
    ) -> None:
        model_id = os.path.basename(model_path)
        self.models[model_id] = {
            "path": model_path,
            "parent_models": parent_models,
            "merge_techniques": merge_techniques,
            "config": config,
            "score": score,
            "creation_time": datetime.now().isoformat(),
        }
        self._save_tracking_data()

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        return self.models.get(model_id, {})

    def update_model_score(self, model_id: str, new_score: float) -> None:
        if model_id in self.models:
            self.models[model_id]["score"] = new_score
            self._save_tracking_data()

    def remove_model(self, model_id: str) -> None:
        if model_id in self.models:
            del self.models[model_id]
            self._save_tracking_data()

    def get_all_models(self) -> dict[str, dict[str, Any]]:
        return self.models


model_tracker = ModelTracker()
