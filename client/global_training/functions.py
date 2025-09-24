import os
import time
import glob
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env.server')
BLOB_SERVICE_CLIENT = None

if 'SERVER_ACCOUNT_URL' in os.environ:
    SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
    SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
    if not SERVER_ACCOUNT_URL:
        raise ValueError("Missing required environment variable: Account url")
    try:
        BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
    except Exception as e:
        print(f"Failed to initialize Azure Blob Service: {e}")
        raise

def find_csv_file(file_pattern):
    matching_files = glob.glob(file_pattern)
    if matching_files:
        print(f"Found dataset: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"No dataset found matching pattern: {file_pattern}")
        return None

def wait_for_csv(file_pattern, wait_time=300):
    print(f"Checking for dataset matching pattern: {file_pattern}")
    while True:
        csv_file = find_csv_file(file_pattern)
        if csv_file:
            return csv_file
        print(f"Dataset not found. Waiting for {wait_time // 60} minutes...")
        time.sleep(wait_time)
        print(f"Rechecking for dataset matching pattern: {file_pattern}")

def upload_file(file_path, container_name, metadata):
    filename = os.path.basename(file_path)
    print(f"Uploading weights ({filename}) to Azure Blob Storage...")
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=filename)
        with open(file_path, "rb") as file:
            blob_client.upload_blob(file.read(), overwrite=True, metadata=metadata)
        print(f"Weights ({filename}) uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        print(f"Error uploading weights ({filename}): {e}")

def save_weights(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, filename)
    try:
        model.save_weights(weights_path)
        print(f"Weights saved at {weights_path}")
    except Exception as e:
        print(f"Failed to save weights: {e}")
    return weights_path