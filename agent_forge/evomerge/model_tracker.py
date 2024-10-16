import json
import os
from typing import Dict, List, Any
from datetime import datetime

class ModelTracker:
    def __init__(self, tracking_file: str = "model_tracking.json"):
        self.tracking_file = tracking_file
        self.models: Dict[str, Dict[str, Any]] = self._load_tracking_data()

    def _load_tracking_data(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_tracking_data(self):
        with open(self.tracking_file, 'w') as f:
            json.dump(self.models, f, indent=2)

    def add_model(self, model_path: str, parent_models: List[str], merge_techniques: List[str], 
                  config: Dict[str, Any], score: float):
        model_id = os.path.basename(model_path)
        self.models[model_id] = {
            "path": model_path,
            "parent_models": parent_models,
            "merge_techniques": merge_techniques,
            "config": config,
            "score": score,
            "creation_time": datetime.now().isoformat()
        }
        self._save_tracking_data()

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        return self.models.get(model_id, {})

    def update_model_score(self, model_id: str, new_score: float):
        if model_id in self.models:
            self.models[model_id]["score"] = new_score
            self._save_tracking_data()

    def remove_model(self, model_id: str):
        if model_id in self.models:
            del self.models[model_id]
            self._save_tracking_data()

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        return self.models

model_tracker = ModelTracker()
