import json
from typing import Any, Dict

class UnifiedConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnifiedConfig, cls).__new__(cls)
            cls._instance._config = {}
        return cls._instance

    def load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self._config = json.load(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value

    def save_config(self, config_path: str):
        with open(config_path, 'w') as config_file:
            json.dump(self._config, config_file, indent=2)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

# Global instance
unified_config = UnifiedConfig()
