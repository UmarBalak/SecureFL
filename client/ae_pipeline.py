import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import resample
import os
import joblib
from model import IoTModel
from preprocessing import IoTDataPreprocessor
from tensorflow.keras.utils import to_categorical


def get_model_weights(model):
    """Extract and flatten model weights."""
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        for w in layer_weights:
            weights.append(w.flatten())
    return np.concatenate(weights)

from sklearn.model_selection import train_test_split

def simulate_trained_models(data_file, num_models=500, sample_fraction=0.1):
    """Train models on random 10% samples while preserving data distribution."""
    preprocessor = IoTDataPreprocessor()

    X_train, X_test, y_train_type, y_test_type, y_type, num_classes = preprocessor.preprocess_data(data_file)
    
    input_dim = X_train.shape[1]
    num_classes = num_classes
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    model_generator = IoTModel()
    
    weights_list = []

    for i in range(num_models):
        # Stratified random 30% sampling
        X_sub, _, y_sub, _ = train_test_split(
            X_train,
            y_train_type,
            train_size=sample_fraction,
            stratify=y_train_type,
            random_state=i
        )
        y_sub = to_categorical(y_sub, num_classes=num_classes)

        model = model_generator.create_mlp_model(input_dim, num_classes)
        model.fit(X_sub, y_sub, epochs=5, batch_size=32, verbose=0)

        weights = get_model_weights(model)
        weights_list.append(weights)

        print(f"Trained model {i+1}/{num_models} on 10% stratified sample")
    
    return np.array(weights_list), input_dim, num_classes

def main():
    # Parameters
    data_file = './DATA/ML-EdgeIIoT-dataset.csv'
    num_models = 1500
    
    # Simulate trained models
    print("Simulating training of 1500 models...")
    weights_data, input_dim, num_classes = simulate_trained_models(data_file, num_models)
    
    # Calculate weight dimension
    model = IoTModel().create_mlp_model(input_dim, num_classes)
    weight_dim = get_model_weights(model).shape[0]
    print(f"Weight vector dimension: {weight_dim}")

if __name__ == '__main__':
    main()