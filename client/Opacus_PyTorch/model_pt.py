import torch
import torch.nn as nn
import torch.nn.functional as F

class IoTModelPyTorch:
    def __init__(self):
        """Initialize the PyTorch IoT model class"""
        pass

    def create_mlp_model(self, input_dim, num_classes, architecture=[128, 128, 64]):
        """
        Create an MLP model with comprehensive regularization techniques.
        This mirrors the TensorFlow implementation but uses PyTorch layers.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        architecture : list of int
            List specifying the number of units in each hidden layer
        
        Returns:
        --------
        model : IoTMLPNet
            PyTorch model with regularization
        """
        model = IoTMLPNet(input_dim, num_classes, architecture)
        return model

    def load_model(self, model_path, input_dim, num_classes, architecture):
        """
        Load a saved model from disk

        Parameters:
        -----------
        model_path : str
            Path to the saved model
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        architecture : list
            Model architecture

        Returns:
        --------
        model : IoTMLPNet
            Loaded model
        """
        model = IoTMLPNet(input_dim, num_classes, architecture)
        model.load_state_dict(torch.load(model_path))
        return model

    def load_model_weights(self, model, weights_path):
        """
        Load model weights from disk

        Parameters:
        -----------
        model : IoTMLPNet
            Model to load weights into
        weights_path : str
            Path to the saved weights

        Returns:
        --------
        model : IoTMLPNet
            Model with loaded weights
        """
        model.load_state_dict(torch.load(weights_path))
        return model

class IoTMLPNet(nn.Module):
    """
    PyTorch MLP Network for IoT data classification
    Mirrors the TensorFlow implementation architecture
    """
    
    def __init__(self, input_dim, num_classes, architecture=[256, 128, 64]):
        """
        Initialize the MLP network
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        architecture : list of int
            List specifying the number of units in each hidden layer
        """
        super(IoTMLPNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        # First hidden layer
        layers.append(nn.Linear(prev_dim, architecture[0]))
        layers.append(nn.LeakyReLU(negative_slope=0.1))  # Matches TF LeakyReLU(alpha=0.1)
        layers.append(nn.BatchNorm1d(architecture[0]))
        prev_dim = architecture[0]
        
        # Subsequent hidden layers
        for i in range(1, len(architecture)):
            layers.append(nn.Linear(prev_dim, architecture[i]))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.BatchNorm1d(architecture[i]))
            prev_dim = architecture[i]
        
        # Output layer (no activation, will use CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Create the sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (similar to TensorFlow's default initialization)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights similar to TensorFlow's default initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization (similar to TF default)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def get_num_parameters(self):
        """
        Get the total number of parameters in the model
        
        Returns:
        --------
        int
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self):
        """
        Get the number of trainable parameters
        
        Returns:
        --------
        int
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """
        Print model summary similar to TensorFlow's model.summary()
        """
        print("=" * 70)
        print("PyTorch IoT MLP Model Summary")
        print("=" * 70)
        print(f"Input dimension: {self.input_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Architecture: {self.architecture}")
        print("-" * 70)
        
        total_params = 0
        trainable_params = 0
        
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
        print("-" * 70)
        
        x = torch.randn(1, self.input_dim)
        layer_idx = 0
        
        for name, module in self.network.named_children():
            if isinstance(module, nn.Linear):
                layer_name = f"Linear-{layer_idx}"
                x = module(x)
                params = sum(p.numel() for p in module.parameters())
                print(f"{layer_name:<25} {str(tuple(x.shape)):<20} {params:<15}")
                total_params += params
                trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
                layer_idx += 1
            elif isinstance(module, nn.LeakyReLU):
                layer_name = f"LeakyReLU-{layer_idx-1}"
                x = module(x)
                print(f"{layer_name:<25} {str(tuple(x.shape)):<20} {'0':<15}")
            elif isinstance(module, nn.BatchNorm1d):
                layer_name = f"BatchNorm1d-{layer_idx-1}"
                x = module(x)
                params = sum(p.numel() for p in module.parameters())
                print(f"{layer_name:<25} {str(tuple(x.shape)):<20} {params:<15}")
                total_params += params
                trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 70)

class IoTQuantizedMLPNet(nn.Module):
    """
    Quantized version of the IoT MLP Network
    For future use or comparison studies
    """
    
    def __init__(self, input_dim, num_classes, architecture=[256, 128, 128, 64]):
        super(IoTQuantizedMLPNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for units in architecture:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Prepare for quantization
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.network(x)
        x = self.dequant(x)
        return x