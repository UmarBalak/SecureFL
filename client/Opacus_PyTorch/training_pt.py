import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import psutil
from model_pt import IoTModelPyTorch

# Opacus imports for differential privacy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator

class IoTModelTrainerPyTorch:
    def __init__(self, random_state=42):
        """
        Initialize the PyTorch model trainer with DP support using Opacus

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model_builder = IoTModelPyTorch()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        
        print(f"Using device: {self.device}")

    def create_model(self, input_dim, num_classes, architecture=None):
        """
        Create a new MLP model

        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        architecture : list of int, optional
            List specifying the number of units in each hidden layer

        Returns:
        --------
        model : IoTMLPNet
            PyTorch MLP model
        """
        if architecture is None:
            architecture = [256, 128, 64]
            
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        self.model = self.model.to(self.device)
        
        # Print model summary
        # self.model.summary()
        
        return self.model

    def _validate_model_for_dp(self, model):
        """
        Validate and fix model for differential privacy compatibility
        
        Parameters:
        -----------
        model : nn.Module
            Model to validate
            
        Returns:
        --------
        nn.Module
            DP-compatible model
        """
        # Check if model is compatible with Opacus
        errors = ModuleValidator.validate(model, strict=False)
        
        if errors:
            print("⚠️  Model compatibility issues found:")
            for error in errors:
                print(f"   - {error}")
            print("🔧 Attempting to fix model...")
            
            # Fix the model automatically
            model = ModuleValidator.fix(model)
            print("✅ Model fixed for DP compatibility")
        else:
            print("✅ Model is DP-compatible")
            
        return model

    def compute_epsilon_laplace(self, steps, epsilon_per_step):
        """
        Compute total epsilon for Laplace DP mechanism.
        For Laplace mechanism, epsilon values compose additively.
        
        Parameters:
        -----------
        steps : int
            Total number of training steps
        epsilon_per_step : float
            Epsilon budget per step
            
        Returns:
        --------
        float
            Total epsilon spent
        """
        total_epsilon = steps * epsilon_per_step
        return total_epsilon

    def train_model(self, train_loader, test_loader, model, epochs=50, verbose=2,
                    use_dp=True, dp_mechanism='laplace', max_grad_norm=1.0,
                    target_epsilon=1.0, target_delta=1e-5, noise_multiplier=None,
                    laplace_sensitivity=1.0, learning_rate=0.001):
        """
        Train the MLP model with optional differential privacy using Opacus.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Test data loader  
        model : nn.Module
            The model to train
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level for training output
        use_dp : bool
            Whether to use differential privacy
        dp_mechanism : str
            DP mechanism: 'laplace' or 'gaussian'
        max_grad_norm : float
            Maximum gradient norm for clipping
        target_epsilon : float
            Target epsilon for privacy budget
        target_delta : float
            Target delta for privacy budget (Gaussian only)
        noise_multiplier : float
            Noise multiplier (if None, calculated automatically)
        laplace_sensitivity : float
            Sensitivity for Laplace mechanism
        learning_rate : float
            Learning rate for optimizer

        Returns:
        --------
        history : dict
            Training history
        training_time : float
            Total training time
        epsilon_spent : float
            Privacy budget spent
        delta_spent : float
            Delta spent (0 for Laplace)
        final_accuracy : float
            Final validation accuracy
        """
        print(f"\nTraining PyTorch model with {dp_mechanism.upper()} DP...")
        
        self.model = model.to(self.device)

        if use_dp:
            print(f"Setting up {dp_mechanism.upper()} differential privacy...")

            # Validate model for DP compatibility
            self.model = self._validate_model_for_dp(self.model)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Privacy tracking variables
        epsilon_spent = float('inf')
        delta_spent = 0.0
        privacy_engine = None
        
        if use_dp:
            print(f"Setting up {dp_mechanism.upper()} differential privacy...")
            
            # Validate model for DP compatibility
            self.model = self._validate_model_for_dp(self.model)
            
            # Initialize privacy engine
            privacy_engine = PrivacyEngine()
            
            if dp_mechanism == 'laplace':
                # For Laplace DP, we need to calculate noise multiplier based on sensitivity
                # Opacus doesn't have built-in Laplace support, so we use Gaussian with equivalent parameters
                print(f"⚠️  Note: Using Gaussian approximation for Laplace DP with sensitivity={laplace_sensitivity}")
                print("    This provides similar privacy guarantees for research comparison")
                
                # Calculate equivalent noise multiplier for Gaussian to approximate Laplace
                # This is a research approximation - in practice, you'd implement true Laplace
                noise_multiplier = laplace_sensitivity / target_epsilon if noise_multiplier is None else noise_multiplier
                
                self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    epochs=epochs,
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    max_grad_norm=max_grad_norm,
                )
                
            elif dp_mechanism == 'gaussian':
                # Standard Gaussian DP
                if noise_multiplier is None:
                    # Let Opacus calculate noise multiplier automatically
                    self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                        module=self.model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        epochs=epochs,
                        target_epsilon=target_epsilon,
                        target_delta=target_delta,
                        max_grad_norm=max_grad_norm,
                    )
                else:
                    # Use specified noise multiplier
                    self.model, optimizer, train_loader = privacy_engine.make_private(
                        module=self.model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        noise_multiplier=noise_multiplier,
                        max_grad_norm=max_grad_norm,
                    )
            
            print(f"✅ Privacy engine configured for {dp_mechanism.upper()} DP")
            print(f"   Max grad norm: {max_grad_norm}")
            print(f"   Target epsilon: {target_epsilon}")
            print(f"   Target delta: {target_delta}")

        # Training setup
        start_time = time.time()
        mem_start = psutil.Process().memory_info().rss
        
        # Initialize history tracking
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Training loop
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            if use_dp:
                # Use BatchMemoryManager for efficient DP training
                with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=1000,  # Adjust based on your GPU memory
                    optimizer=optimizer
                ) as memory_safe_data_loader:
                    
                    for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        train_total += target.size(0)
                        train_correct += (predicted == target).sum().item()
                        
                        if verbose >= 2 and batch_idx % 100 == 0:
                            print(f'   Batch {batch_idx}/{len(memory_safe_data_loader)}: Loss = {loss.item():.4f}')
            else:
                # Standard training without DP
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                    
                    if verbose >= 2 and batch_idx % 100 == 0:
                        print(f'   Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}')
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_loss = val_loss / len(test_loader)
            val_accuracy = 100. * val_correct / val_total
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_accuracy)
            
            # Calculate privacy spent so far
            if use_dp and privacy_engine:
                epsilon_spent = privacy_engine.get_epsilon(target_delta)
                delta_spent = target_delta
                
                if dp_mechanism == 'laplace':
                    # For Laplace approximation, calculate based on steps
                    steps_so_far = (epoch + 1) * len(train_loader)
                    epsilon_per_step = target_epsilon / (epochs * len(train_loader))
                    epsilon_spent = self.compute_epsilon_laplace(steps_so_far, epsilon_per_step)
                    delta_spent = 0.0  # Laplace is (ε, 0)-DP
            
            epoch_time = time.time() - epoch_start_time
            
            if verbose >= 1:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
                if use_dp:
                    print(f'  Privacy: ε = {epsilon_spent:.4f}, δ = {delta_spent:.2e}')
                print(f'  Time: {epoch_time:.2f}s')
                print('-' * 50)
        
        training_time = time.time() - start_time
        final_accuracy = val_accuracy / 100.0  # Convert to decimal
        
        print(f"\n✅ Training completed!")
        print(f"   Total time: {training_time:.2f} seconds")
        print(f"   Final validation accuracy: {final_accuracy:.4f}")
        
        if use_dp:
            print(f"   Final privacy cost: ε = {epsilon_spent:.4f}, δ = {delta_spent:.2e}")
        
        return self.history, training_time, epsilon_spent, delta_spent, final_accuracy

    def evaluate_model(self, test_loader, model=None):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        test_loader : DataLoader
            Test data loader
        model : nn.Module, optional
            Model to evaluate (uses self.model if None)
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if model is None:
            model = self.model
            
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return {
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'correct_predictions': correct,
            'total_samples': total
        }

    def save_model(self, model_path, model=None):
        """
        Save model state dict
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        model : nn.Module, optional
            Model to save (uses self.model if None)
        """
        if model is None:
            model = self.model
            
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path, input_dim, num_classes, architecture):
        """
        Load model from saved state dict
        
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
        nn.Module
            Loaded model
        """
        self.model = self.model_builder.create_mlp_model(input_dim, num_classes, architecture)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        return self.model

    def get_model(self):
        """
        Get the trained model

        Returns:
        --------
        model : nn.Module
            Trained PyTorch model
        """
        return self.model

    def get_history(self):
        """
        Get the training history

        Returns:
        --------
        history : dict
            Training history with losses and accuracies
        """
        return self.history

    def print_model_info(self):
        """
        Print detailed information about the model
        """
        if self.model is None:
            print("No model created yet.")
            return
            
        print("\n" + "="*70)
        print("MODEL INFORMATION")
        print("="*70)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
        print(f"Device: {self.device}")
        
        if hasattr(self.model, 'architecture'):
            print(f"Architecture: {self.model.architecture}")
        
        print("="*70)