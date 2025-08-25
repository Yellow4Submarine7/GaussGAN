"""Direct coordinate generators without distributional assumptions"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BoundedDirectGenerator(nn.Module):
    """Direct coordinate generator with bounded outputs for stability.
    
    This generator directly outputs coordinates without any mean+std parameterization,
    making it truly distribution-agnostic. Uses bounded activation functions
    to ensure training stability.
    """
    
    def __init__(self, z_dim=4, hidden_dims=[256, 256], output_dim=2, 
                 output_bound=10.0, bound_type='tanh'):
        super().__init__()
        self.z_dim = z_dim
        self.output_bound = output_bound
        self.bound_type = bound_type
        
        # Create a dummy parameter to track the device
        self.register_buffer("dummy", torch.zeros(1))
        
        # Build network with LayerNorm for stability
        layers = []
        current_dim = z_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.LayerNorm(hdim))  # More stable than BatchNorm
            layers.append(nn.GELU())  # Smoother gradients than ReLU
            current_dim = hdim
        
        self.features = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        
        # Initialize output layer with small weights for stability
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, batch_size):
        # Generate z internally
        device = self.dummy.device
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        features = self.features(z)
        raw_output = self.output_layer(features)
        
        # Apply bounding for stability
        if self.bound_type == 'tanh':
            # Smooth bounding with tanh: output in [-bound, bound]
            return self.output_bound * torch.tanh(raw_output)
        elif self.bound_type == 'soft_clip':
            # Soft clipping with smooth transition
            return self.output_bound * torch.tanh(raw_output / self.output_bound)
        elif self.bound_type == 'sigmoid':
            # Map to [-bound, bound] using sigmoid
            return self.output_bound * (2 * torch.sigmoid(raw_output) - 1)
        else:
            # No bounding (not recommended)
            return raw_output


class ResidualBlock(nn.Module):
    """Residual block with optional spectral normalization"""
    
    def __init__(self, in_dim, out_dim, use_spectral_norm=False):
        super().__init__()
        
        # Apply spectral norm if requested
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x
        
        self.fc1 = norm_fn(nn.Linear(in_dim, out_dim))
        self.fc2 = norm_fn(nn.Linear(out_dim, out_dim))
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Projection for dimension matching
        if in_dim != out_dim:
            self.proj = norm_fn(nn.Linear(in_dim, out_dim))
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        residual = self.proj(x)
        
        h = self.fc1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.norm2(h)
        
        return F.gelu(h + residual)


class ResidualDirectGenerator(nn.Module):
    """Direct generator with residual connections for gradient stability.
    
    Uses residual blocks and skip connections to ensure good gradient flow,
    making training more stable for direct coordinate generation.
    """
    
    def __init__(self, z_dim=4, hidden_dims=[256, 256], output_dim=2,
                 output_bound=10.0, use_spectral_norm=False):
        super().__init__()
        self.z_dim = z_dim
        self.output_bound = output_bound
        
        # Create a dummy parameter to track the device
        self.register_buffer("dummy", torch.zeros(1))
        
        # Initial projection
        self.input_proj = nn.Linear(z_dim, hidden_dims[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], use_spectral_norm)
            )
        
        # Output with bounded activation
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
        
        # Skip connection from input for stability
        self.skip_proj = nn.Linear(z_dim, output_dim)
        
        # Learnable mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights appropriately
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Small weights for output layers
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.xavier_uniform_(self.skip_proj.weight, gain=0.5)
        nn.init.zeros_(self.skip_proj.bias)
    
    def forward(self, batch_size):
        # Generate z internally
        device = self.dummy.device
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        # Main path through residual blocks
        h = F.gelu(self.input_proj(z))
        for block in self.res_blocks:
            h = block(h)
        main_output = self.output_proj(h)
        
        # Skip connection from input
        skip_output = self.skip_proj(z)
        
        # Weighted combination with learnable alpha
        alpha = torch.sigmoid(self.alpha)
        combined = alpha * main_output + (1 - alpha) * skip_output
        
        # Apply bounding for stability
        return self.output_bound * torch.tanh(combined)


class ProgressiveDirectGenerator(nn.Module):
    """Generator with progressive complexity increase during training.
    
    Starts with simple architecture and gradually increases complexity,
    allowing for more stable training of complex distributions.
    """
    
    def __init__(self, z_dim=4, hidden_dims=[128, 256, 256], output_dim=2,
                 output_bound=10.0):
        super().__init__()
        self.z_dim = z_dim
        self.output_bound = output_bound
        
        # Create a dummy parameter to track the device
        self.register_buffer("dummy", torch.zeros(1))
        self.current_depth = 0
        self.max_depth = len(hidden_dims)
        
        # Build progressive layers
        self.layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        current_dim = z_dim
        for i, hdim in enumerate(hidden_dims):
            # Main layer block
            layer = nn.Sequential(
                nn.Linear(current_dim, hdim),
                nn.LayerNorm(hdim),
                nn.LeakyReLU(0.2)
            )
            self.layers.append(layer)
            
            # Output layer for this depth
            out_layer = nn.Linear(hdim, output_dim)
            nn.init.xavier_uniform_(out_layer.weight, gain=0.5)
            nn.init.zeros_(out_layer.bias)
            self.output_layers.append(out_layer)
            
            current_dim = hdim
        
        # Fade-in parameter for smooth transitions
        self.alpha = 1.0
    
    def set_depth(self, depth, alpha=1.0):
        """Set current depth and fade-in parameter for progressive training"""
        self.current_depth = min(depth, self.max_depth - 1)
        self.alpha = alpha
    
    def forward(self, batch_size):
        # Generate z internally
        device = self.dummy.device
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        h = z
        
        # Process up to current depth
        for i in range(self.current_depth + 1):
            h = self.layers[i](h)
        
        # Get output from current depth
        output = self.output_layers[self.current_depth](h)
        
        # Apply fade-in if transitioning between depths
        if self.alpha < 1.0 and self.current_depth > 0:
            # Get output from previous depth
            h_prev = z
            for i in range(self.current_depth):
                h_prev = self.layers[i](h_prev)
            prev_output = self.output_layers[self.current_depth - 1](h_prev)
            
            # Blend outputs
            output = self.alpha * output + (1 - self.alpha) * prev_output
        
        # Apply bounding
        return self.output_bound * torch.tanh(output)


class StableDirectGenerator(nn.Module):
    """Combines multiple stability techniques for robust direct generation.
    
    This is the recommended generator for distribution-agnostic generation,
    combining bounded outputs, residual connections, and optional spectral
    normalization for maximum stability.
    """
    
    def __init__(self, z_dim=4, hidden_dims=[256, 256], output_dim=2,
                 output_bound=10.0, use_residual=True, use_spectral_norm=True,
                 use_skip_connection=True):
        super().__init__()
        self.z_dim = z_dim
        self.output_bound = output_bound
        
        # Create a dummy parameter to track the device
        self.register_buffer("dummy", torch.zeros(1))
        self.use_skip_connection = use_skip_connection
        
        # Apply spectral norm if requested
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x
        
        # Build network
        if use_residual:
            # Use residual architecture
            self.input_proj = norm_fn(nn.Linear(z_dim, hidden_dims[0]))
            self.blocks = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                self.blocks.append(
                    ResidualBlock(hidden_dims[i], hidden_dims[i+1], use_spectral_norm)
                )
        else:
            # Standard feedforward architecture
            layers = []
            current_dim = z_dim
            for hdim in hidden_dims:
                layers.extend([
                    norm_fn(nn.Linear(current_dim, hdim)),
                    nn.LayerNorm(hdim),
                    nn.GELU()
                ])
                current_dim = hdim
            self.features = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = norm_fn(nn.Linear(hidden_dims[-1], output_dim))
        
        # Optional skip connection
        if use_skip_connection:
            self.skip_proj = norm_fn(nn.Linear(z_dim, output_dim))
            self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Learnable temperature for output scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Conservative initialization for stability
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.zeros_(self.output_layer.bias)
        
        if self.use_skip_connection:
            nn.init.xavier_uniform_(self.skip_proj.weight, gain=0.5)
            nn.init.zeros_(self.skip_proj.bias)
    
    def forward(self, batch_size):
        # Generate z internally
        device = self.dummy.device
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        if hasattr(self, 'blocks'):
            # Residual path
            h = F.gelu(self.input_proj(z))
            for block in self.blocks:
                h = block(h)
        else:
            # Standard path
            h = self.features(z)
        
        # Main output
        output = self.output_layer(h)
        
        # Add skip connection if enabled
        if self.use_skip_connection:
            skip_output = self.skip_proj(z)
            alpha = torch.sigmoid(self.alpha)
            output = alpha * output + (1 - alpha) * skip_output
        
        # Apply temperature-controlled bounding
        temperature = F.softplus(self.temperature) + 0.1  # Ensure positive
        return self.output_bound * torch.tanh(output / temperature)


class GradientControlledDirectGenerator(nn.Module):
    """Direct generator with explicit gradient control mechanisms.
    
    Uses gradient clipping and scaling to ensure stable training,
    particularly useful for challenging distributions.
    """
    
    def __init__(self, z_dim=4, hidden_dims=[256, 256], output_dim=2,
                 output_bound=10.0, grad_clip_value=1.0):
        super().__init__()
        self.z_dim = z_dim
        self.output_bound = output_bound
        
        # Create a dummy parameter to track the device
        self.register_buffer("dummy", torch.zeros(1))
        self.grad_clip_value = grad_clip_value
        
        # Build network with spectral normalization
        layers = []
        current_dim = z_dim
        
        for hdim in hidden_dims:
            layers.extend([
                spectral_norm(nn.Linear(current_dim, hdim)),
                nn.LayerNorm(hdim),
                nn.LeakyReLU(0.2)
            ])
            current_dim = hdim
        
        self.features = nn.Sequential(*layers)
        self.output_layer = spectral_norm(nn.Linear(current_dim, output_dim))
        
        # Gradient scaling parameter
        self.grad_scale = nn.Parameter(torch.tensor(1.0))
        
        # Initialize conservatively
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.2)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, batch_size):
        # Generate z internally
        device = self.dummy.device
        z = torch.randn(batch_size, self.z_dim, device=device)
        
        features = self.features(z)
        output = self.output_layer(features)
        
        # Apply gradient scaling in backward pass
        if self.training:
            output = GradientScaling.apply(output, self.grad_scale, self.grad_clip_value)
        
        # Bounded output
        return self.output_bound * torch.tanh(output)


class GradientScaling(torch.autograd.Function):
    """Custom autograd function for gradient control"""
    
    @staticmethod
    def forward(ctx, input, scale, clip_value):
        ctx.save_for_backward(scale)
        ctx.clip_value = clip_value
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        
        # Clip gradients
        grad_output = torch.clamp(grad_output, -ctx.clip_value, ctx.clip_value)
        
        # Scale gradients
        grad_output = grad_output * torch.sigmoid(scale)
        
        return grad_output, None, None


# Utility function for choosing the best generator
def create_stable_generator(generator_type='stable', **kwargs):
    """Factory function to create different types of stable generators.
    
    Args:
        generator_type: One of 'bounded', 'residual', 'progressive', 'stable', 'gradient'
        **kwargs: Arguments passed to the generator constructor
    
    Returns:
        A generator instance
    """
    generators = {
        'bounded': BoundedDirectGenerator,
        'residual': ResidualDirectGenerator,
        'progressive': ProgressiveDirectGenerator,
        'stable': StableDirectGenerator,
        'gradient': GradientControlledDirectGenerator
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. "
                        f"Choose from {list(generators.keys())}")
    
    return generators[generator_type](**kwargs)