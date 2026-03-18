from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from .trellis_text_to_3d import TrellisTextTo3DPipeline
from .near_image_to_relightable_3d import NeARImageToRelightable3DPipeline


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.yaml")

    if is_local:
        config_file = f"{path}/pipeline.yaml"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.yaml")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
