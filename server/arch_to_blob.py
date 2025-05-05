import os
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.models import Model
import keras
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_model_architecture(model: Model, blob_service_client, container_name: str):
    """
    Upload model architecture to Azure Blob Storage.
    
    Args:
        model: Keras model to upload
        connection_string: Azure storage connection string
        container_name: Name of the container to upload to
    """
    try:
        # Save model architecture temporarily
        temp_path = "temp_model_architecture.h5"
        model.save(temp_path)
        
        # Upload to blob storage
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob="model_architecture.h5"
        )
        
        with open(temp_path, "rb") as file:
            blob_client.upload_blob(file, overwrite=True)
            
        logging.info(f"Successfully uploaded model architecture to {container_name}/model_architecture.h5")
        
    except Exception as e:
        logging.error(f"Error uploading model architecture: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Example usage:
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env.server')
    
    # Azure Blob Storage configuration
    CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
    CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

    try:
        blob_service_client = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
    except Exception as e:
        logging.error(f"Failed to initialize Azure Blob Service: {e}")
        raise

    model = tf.keras.saving.load_model("D:\FL\client\models\model_arch.h5")
    
    # Upload the architecture
    upload_model_architecture(model, blob_service_client, CLIENT_CONTAINER_NAME)