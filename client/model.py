import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
class IoTModel:
    def __init__(self):
        """Initialize the IoT model class"""
        pass

    def create_mlp_model(self, input_dim, num_classes, architecture=[128, 128, 64]):
        model = Sequential()
        model.add(Dense(architecture[0], input_dim=input_dim, kernel_regularizer=l2(0.001)))
        model.add(LayerNormalization())
        model.add(tf.keras.layers.ReLU())

        for units in architecture[1:]:
            model.add(Dense(units, kernel_regularizer=l2(0.001)))
            model.add(LayerNormalization())
            model.add(tf.keras.layers.ReLU())

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy']
        )

        return model

    
    def create_quantized_mlp_model(self, input_dim, num_classes, architecture=[256, 128, 128, 64]):
        model = Sequential()
        model.add(Dense(architecture[0], input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.0005)))
        for units in architecture[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.0005)))
 
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy']
        )

        return model

    def load_model(self, model_path):
        """
        Load a saved model from disk

        Parameters:
        -----------
        model_path : str
            Path to the saved model

        Returns:
        --------
        model : tf.keras.models.Sequential
            Loaded model
        """
        return tf.keras.models.load_model(model_path)

    def load_model_weights(self, model, weights_path):
        """
        Load model weights from disk

        Parameters:
        -----------
        model : tf.keras.models.Sequential
            Model to load weights into
        weights_path : str
            Path to the saved weights

        Returns:
        --------
        model : tf.keras.models.Sequential
            Model with loaded weights
        """
        model.load_weights(weights_path)
        return model 