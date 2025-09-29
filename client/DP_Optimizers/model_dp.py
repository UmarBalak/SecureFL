import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

class IoTModel:
    def __init__(self):
        """Initialize the IoT model class with DP-compatible architecture"""
        pass
    
    def create_mlp_model(self, input_dim, num_classes, architecture=[256, 256]):
        """
        Create DP-compatible MLP model with proper layer naming for aggregation
        
        Args:
            input_dim (int): Input feature dimension
            num_classes (int): Number of output classes
            architecture (list): Hidden layer sizes
            
        Returns:
            tf.keras.Model: DP-compatible sequential model
        """
        model = Sequential(name='dp_compatible_model')
        
        # Build architecture with consistent naming for DP weight aggregation
        for i, units in enumerate(architecture):
            if i == 0:
                model.add(Dense(
                    units,
                    input_dim=input_dim,
                    kernel_regularizer=l2(0.001),
                    activation='relu',
                    name=f'dense_{i}'
                ))
            else:
                model.add(Dense(
                    units,
                    kernel_regularizer=l2(0.001),
                    activation='relu',
                    name=f'dense_{i}'
                ))
        
        # Output layer with softmax activation
        model.add(Dense(
            num_classes, 
            activation='softmax', 
            name='output_dense'
        ))
        
        return model
    
    def load_model(self, model_path):
        """Load saved model from disk"""
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def load_model_weights(self, model, weights_path):
        """Load model weights from disk"""
        try:
            model.load_weights(weights_path)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {e}")
