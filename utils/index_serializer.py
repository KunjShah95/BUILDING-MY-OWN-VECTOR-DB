import pickle
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict


class IndexSerializer:
    @staticmethod
    def save(data: Dict[str, Any], filepath: str, format: str = "json"):
        filepath = Path(filepath)
        if format == "binary":
            with open(filepath.with_suffix('.bin'), 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(data, f)

    @staticmethod
    def load(filepath: str, format: str = "json") -> Dict[str, Any]:
        filepath = Path(filepath)
        if format == "binary":
            with open(filepath.with_suffix('.bin'), 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath.with_suffix('.json'), 'r') as f:
                return json.load(f)

    @staticmethod
    def estimate_size(data: Dict[str, Any], format: str = "json") -> int:
        if format == "binary":
            return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        return len(json.dumps(data))
