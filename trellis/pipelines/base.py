from __future__ import annotations

import re
from pathlib import Path
from typing import *

import torch
import torch.nn as nn

from .. import models


def _is_hub_model_id(s: str) -> bool:
    """`org/model` style id, not a filesystem path."""
    t = s.strip().replace("\\", "/")
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", t))


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def _resolve_pretrained_root(path: str) -> str:
        """Local directory with ``pipeline.yaml``, or Hub ``org/model`` (full snapshot cache)."""
        import os

        path = path.strip()
        root = Path(path).expanduser()
        cfg = root / "pipeline.yaml"
        if cfg.is_file():
            return str(root.resolve())

        if root.is_dir():
            raise FileNotFoundError(
                f"Missing pipeline.yaml under {root.resolve()}. "
                "Add checkpoints or pass a Hub repo id (e.g. luh0502/NeAR)."
            )

        if _is_hub_model_id(path):
            from huggingface_hub import snapshot_download

            print(f"[Pipeline] Downloading Hub snapshot {path!r} ...", flush=True)
            out = snapshot_download(repo_id=path, token=os.environ.get("HF_TOKEN"))
            print("[Pipeline] Hub snapshot ready.", flush=True)
            return out

        raise FileNotFoundError(
            f"Not a local checkpoint directory and not a Hub repo id: {path!r}"
        )

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained pipeline from a local directory or Hugging Face Hub ``org/model``.

        Local: folder containing ``pipeline.yaml`` (and ``ckpts/``, ``weights/``, ...).
        Hub: ``snapshot_download`` into the HF cache, same layout as the model repo.
        """
        import json
        import os

        import yaml

        def _load_config(config_file: str) -> dict:
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f)["args"]
                return json.load(f)["args"]

        root = Pipeline._resolve_pretrained_root(path)
        config_file = os.path.join(root, "pipeline.yaml")
        print(f"[Pipeline] Using config {config_file}", flush=True)
        args = _load_config(config_file)

        _models = {}
        for k, v in args["models"].items():
            try:
                _models[k] = models.from_pretrained(os.path.join(root, v))
            except Exception:
                _models[k] = models.from_pretrained(v)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)
        rembg = getattr(self, "rembg_model", None)
        if rembg is not None and hasattr(rembg, "to"):
            rembg.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
