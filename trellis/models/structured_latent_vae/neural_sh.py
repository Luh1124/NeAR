import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .nerf_encoding import NeRFEncoding

class SineActivation(nn.Module):
    """SIREN-style sine activation function"""
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)

class NeuralBasis(nn.Module):
    """
    Simplified neural MLP model
    
    Features:
    - input 48-dimensional feature vector
    - multi-layer MLP structure
    - support normal input (optional)
    - small model, high computational efficiency
    - support SIREN-style sine activation function
    """
    
    def __init__(self, 
                 input_dim=48,           # input feature dimension
                 output_dim=4,           # output dimension (e.g. RGB color + brightness)
                 hidden_dim=64,          # hidden layer dimension
                 num_layers=3,           # number of layers
                 num_frequencies=6,
                 activation='sine',      # activation function: 'relu', 'sine', 'silu', 'gelu'
                 w0=30.0,               # SIREN frequency parameter
                 dropout_rate=0.0,       # dropout rate
                 use_normal=True,       # whether to use normal input
                 normal_dim=3,           # normal dimension
                 use_residual=False,     # whether to use residual connection
                 pretrained_path=None):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_normal = use_normal
        self.use_residual = use_residual
        self.activation_type = activation
        self.w0 = w0
        
        # calculate actual input dimension
        actual_input_dim = input_dim
        self.nerf_pe = NeRFEncoding(
            in_dim=normal_dim,
            num_frequencies=6,
            include_input=True
        )
        if use_normal:
            actual_input_dim += self.nerf_pe.get_out_dim()
            
        # activation function selection
        if activation == 'sine':
            self.activation = SineActivation(w0=w0)
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # build MLP network
        layers = []
        
        # input layer
        layers.append(nn.Linear(actual_input_dim, hidden_dim))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # hidden layer
        for i in range(num_layers - 2):
            if use_residual and i > 0:
                # residual connection requires dimension matching
                layers.append(ResidualBlock(hidden_dim, self.activation, dropout_rate))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # output activation function
        self.rgb_activation = nn.ELU(1e-3)
        self.brightness_activation = nn.Sigmoid()
        
        # initialize weights (important for SIREN)
        self._initialize_weights()
    
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            self.load_state_dict(state_dict, strict=True)
            print(f"SimpleNeuralMLP: Loaded pretrained model from {pretrained_path}")

    def _initialize_weights(self):
        """weight initialization - special initialization for SIREN"""
        with torch.no_grad():
            if self.activation_type == 'sine':
                # SIREN weight initialization
                is_first = True
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        num_input = module.weight.size(-1)
                        if is_first:
                            # first layer using uniform distribution initialization
                            module.weight.uniform_(-1 / num_input, 1 / num_input)
                            is_first = False
                        else:
                            # other layers using SIREN special initialization
                            bound = math.sqrt(6 / num_input) / self.w0
                            module.weight.uniform_(-bound, bound)
                        
                        if module.bias is not None:
                            module.bias.uniform_(-1/num_input, 1/num_input)
            else:
                # standard weight initialization
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
    
    def convert_to_fp16(self):
        """convert to half precision"""
        pass
        
    def convert_to_fp32(self):
        """convert to single precision"""
        pass
    
    def forward(self, features, normals=None):
        """
        forward propagation
        
        Args:
            features: input features [..., input_dim]
            normals: normal vector [..., 3] (optional)
        Returns:
            output [..., output_dim]
        """
        # prepare input
        inputs = features
        
        if self.use_normal and normals is not None:
            # normalize normal vector
            normals = F.normalize(normals, p=2, dim=-1)
            inputs = torch.cat([features, self.nerf_pe(normals)], dim=-1)
        
        # through MLP
        output = self.mlp(inputs)
        
        # apply output activation function
        output = torch.cat([self.rgb_activation(output[:, :3]), self.brightness_activation(output[:, 3:])], dim=-1)
        
        return output


class ResidualBlock(nn.Module):
    """residual block - support different activation functions"""
    def __init__(self, dim, activation, dropout_rate=0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.linear2(out)
        out = out + residual  # residual connection
        out = self.activation(out)
        return out


def test_different_activations():
    """test different activation functions"""
    activations = ['relu', 'sine', 'silu', 'gelu']
    batch_size = 10
    features = torch.randn(batch_size, 48)
    normals = torch.randn(batch_size, 3)
    
    for activation in activations:
        print(f"\ntest activation function: {activation}")
        model = NeuralBasis(
            input_dim=48, 
            output_dim=4, 
            hidden_dim=64, 
            num_layers=3, 
            activation=activation,
            w0=30.0 if activation == 'sine' else 1.0
        )
        
        output = model(features, normals)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  number of parameters: {num_params}")
        print(f"  output shape: {output.shape}")
        print(f"  output statistics: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        print(f"  output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


def generate_data(num_samples=1000, input_dim=48, output_dim=4):
    # generate random data and labels
    features = torch.randn(num_samples, input_dim)
    normals = torch.randn(num_samples, 3)
    # generate target output (e.g. random generation)
    targets = torch.randn(num_samples, output_dim)
    return features, targets, normals

def train_model(model, data, targets, normals, optimizer, num_epochs=100):
    """train model and record loss"""
    model.train()
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data, normals)
        loss = F.mse_loss(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses

def test_training_speed():
    """compare different activation functions"""
    # 准备数据
    input_dim = 48
    output_dim = 4
    features, targets, normals = generate_data(num_samples=1000, input_dim=input_dim, output_dim=output_dim)

    activations_to_test = ['relu', 'sine', 'silu', 'gelu']
    all_losses = {}
    
    import time
    
    for activation in activations_to_test:
        print(f"\ntrain activation function: {activation}")
        
        # 初始化模型
        model = NeuralBasis(
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_dim=64,
            num_layers=3, 
            activation=activation,
            w0=30.0 if activation == 'sine' else 1.0
        )
        
        # for SIREN, use smaller learning rate
        lr = 0.0001 if activation == 'sine' else 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        start_time = time.time()
        losses = train_model(model, features, targets, normals, optimizer, num_epochs=100)
        training_time = time.time() - start_time
        
        all_losses[activation] = losses
        print(f"  training time: {training_time:.2f} seconds")
        print(f"  final loss: {losses[-1]:.6f}")

    # plot loss comparison
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (activation, losses) in enumerate(all_losses.items()):
        plt.plot(losses, label=f'{activation.upper()} activation function', color=colors[i])
    
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.title('loss comparison of different activation functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # use log scale to better display differences
    plt.savefig('activation_comparison.png', dpi=300, bbox_inches='tight')
    print("\nloss comparison plot saved as activation_comparison.png")


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import torch.optim as optim
    
    # set random seed for reproducibility
    torch.manual_seed(42)
    
    # test different activation functions
    test_different_activations()
    
    # test training effect comparison
    test_training_speed()