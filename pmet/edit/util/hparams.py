import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)


def get_model_path(base_path: str, model_name: str) -> str:
    """Return a local model path for `model_name` using `base_path`.

    Checks several possibilities and returns the first existing path. If
    none exist, returns `model_name` unchanged so `transformers` can
    interpret it (e.g., a hub identifier).
    """
    # If model_name itself is an existing path, use it
    p = Path(model_name)
    if p.exists():
        return str(p)

    base = Path(base_path)
    candidate = base / model_name
    if candidate.exists():
        return str(candidate)

    # try replacing path separators in model_name (e.g., repo/name -> repo_name)
    alt = base / model_name.replace("/", "_")
    if alt.exists():
        return str(alt)

    # fallback: return model_name unchanged
    return model_name
