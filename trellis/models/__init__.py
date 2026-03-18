import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatGaussianDecoderOri': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'SLatGaussianRenderer': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoderOri': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatGaussianRenderer': 'structured_latent_vae',
    'Hdri_Encoder': 'structured_latent_vae',
    'NeuralBasis': 'structured_latent_vae',

    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',

    'SLatFlowS2EModelLoRA': 'structured_latent_flow_s2e',
    'ElasticSLatFlowS2ELORAModel': 'structured_latent_flow_s2e',

}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    import torch
    is_local = os.path.exists(f"{path}.json") and (os.path.exists(f"{path}.safetensors") or os.path.exists(f"{path}.pt"))

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors" if os.path.exists(f"{path}.safetensors") else f"{path}.pt"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    
    # check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_file.endswith('.safetensors'):
        model.load_state_dict(load_file(model_file))
    else:
        model.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        SLatGaussianRenderer,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
        ElasticSLatGaussianRenderer,
        Hdri_Encoder,
        SLatGaussianDecoderOri,
        ElasticSLatGaussianDecoderOri,
        NeuralBasis,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )

    from .structured_latent_flow_s2e import (
        SLatFlowS2EModelLoRA,
        ElasticSLatFlowS2ELORAModel,
    )
