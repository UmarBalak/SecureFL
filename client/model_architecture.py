
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_model(input_shapes):
    """
    FL-compatible model with L2 regularization and BatchNorm
    """
    # RGB branch
    rgb_input = layers.Input(shape=input_shapes['rgb'], name="rgb")
    rgb_branch = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(rgb_input)
    rgb_branch = layers.BatchNormalization()(rgb_branch)
    rgb_branch = layers.MaxPooling2D((2, 2))(rgb_branch)
    rgb_branch = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(rgb_branch)
    rgb_branch = layers.BatchNormalization()(rgb_branch)
    rgb_branch = layers.MaxPooling2D((2, 2))(rgb_branch)
    rgb_branch = layers.Flatten()(rgb_branch)
    rgb_branch = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(rgb_branch)
    
    # Segmentation branch
    segmentation_input = layers.Input(shape=input_shapes['segmentation'], name="segmentation")
    segmentation_branch = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(segmentation_input)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.MaxPooling2D((2, 2))(segmentation_branch)
    segmentation_branch = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(segmentation_branch)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.MaxPooling2D((2, 2))(segmentation_branch)
    segmentation_branch = layers.Flatten()(segmentation_branch)
    segmentation_branch = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(segmentation_branch)

    # Other inputs
    hlc_input = layers.Input(shape=input_shapes['hlc'], name="hlc")
    hlc_branch = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hlc_input)

    light_input = layers.Input(shape=input_shapes['light'], name="light")
    light_branch = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(light_input)

    measurements_input = layers.Input(shape=input_shapes['measurements'], name="measurements")
    measurements_branch = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(measurements_input)

    # Final layers
    concatenated = layers.concatenate([rgb_branch, segmentation_branch, hlc_branch, light_branch, measurements_branch])
    concatenated = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concatenated)

    # Outputs
    throttle_output = layers.Dense(1, activation='sigmoid', name="throttle")(concatenated)
    steering_output = layers.Dense(1, activation='tanh', name="steering")(concatenated)
    brake_output = layers.Dense(1, activation='sigmoid', name="brake")(concatenated)

    return models.Model(
        inputs=[rgb_input, segmentation_input, hlc_input, light_input, measurements_input],
        outputs=[throttle_output, steering_output, brake_output]
    )