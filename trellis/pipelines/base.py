from typing import *
import torch
import torch.nn as nn
from .. import models


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
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        import yaml

        def _load_config(config_file: str) -> dict:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)['args']
                return json.load(f)['args']

        config_file = f"{path}/pipeline.yaml"

        if config_file is None:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.yaml")
            print(f"Downloaded config file from {config_file}")
        else:
            print(f"Using local config file from {config_file}")
        args = _load_config(config_file)

        _models = {}
        for k, v in args['models'].items():
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except:
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

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
