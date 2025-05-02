import os
import logging
import numpy as np
import h5py
from skimage.transform import resize

# --- Utility functions ---

def setup_logger(log_dir):
    """
    Set up a client-specific logger.
    """
    log_dir = os.path.join(log_dir, "logs")
    logging.basicConfig(
        filename=os.path.join(log_dir, "preprocessing.log"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# --- Preprocessing functions ---

def load_hdf5_file(file_path):
    """
    Load a single HDF5 file from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file {file_path} does not exist.")
    
    return h5py.File(file_path, "r")

def preprocess_rgb_images(file, image_size):
    """
    Preprocess the RGB images from a single HDF5 file.
    Resize and normalize pixel values.
    """
    rgb_data = file["rgb"][:]
    resized_images = np.array([resize(img, image_size) for img in rgb_data])
    normalized_images = resized_images / 255.0
    return normalized_images



def preprocess_segmentation_masks(file, image_size):
    """
    Preprocess segmentation masks from a single HDF5 file.
    Resize masks to the target size.
    """
    seg_data = file["segmentation"][:]
    resized_masks = np.array([resize(mask, image_size) for mask in seg_data])
    return resized_masks

# def preprocess_controls(file):
#     """
#     Preprocess control data from a single HDF5 file.
#     """
#     controls = file["controls"][:]
#     return controls

def preprocess_controls(file):
    """
    Preprocess control data from a single HDF5 file.
    Normalize throttle, steering, and brake.
    """
    controls = file["controls"][:]
    throttle = controls[:, 0]  # Assuming throttle is the first column
    steering = controls[:, 1]  # Assuming steering is the second column
    brake = controls[:, 2]     # Assuming brake is the third column

    # Normalize throttle and brake to [0, 1] range
    throttle = np.clip(throttle, 0, 1)
    brake = np.clip(brake, 0, 1)

    # Normalize steering to [-1, 1] range
    steering = np.clip(steering, -1, 1)

    return np.column_stack((throttle, steering, brake))



def preprocess_frames(file):
    """
    Preprocess frame indices from a single HDF5 file.
    """
    frames = file["frame"][:]
    return frames

def preprocess_hlc(file):
    """
    Preprocess high-level commands (HLC) from a single HDF5 file.
    """
    hlc = file["hlc"][:]
    return hlc

def preprocess_light(file):
    """
    Preprocess traffic light status from a single HDF5 file.
    """
    light = file["light"][:]
    return light

def preprocess_measurements(file):
    """
    Preprocess speed measurements from a single HDF5 file.
    """
    measurements = file["measurements"][:]
    return measurements

def save_preprocessed_data(output_path, data):
    """
    Save preprocessed data to a NumPy file.
    """
    output_file = os.path.join(output_path, "preprocessed_data.npz")
    np.savez_compressed(output_file, **data)
    logging.info(f"Preprocessed data saved to {output_file}")

# --- Main preprocessing function ---

def preprocess_client_data(data_path, output_path, image_size=(128, 128)):
    """
    Preprocess data for the client..

    Parameters:
    - data_path: Path to the raw data (single HDF5 file).
    - output_path: Path to store preprocessed data.
    - image_size: Target size for resizing images.

    Saves:
    - Preprocessed data into the output_path folder for the respective client.
    """
    script_directory = os.path.dirname(os.path.realpath(__file__))

    setup_logger(script_directory)  # Set up client-specific logging
    logging.info(f"Starting preprocessing...")
    print("preprocessing started")

    try:
        # Load the single HDF5 file
        print(data_path)
        hdf5_file = load_hdf5_file(data_path)
        print("file loaded")

        # Preprocess the data
        preprocessed_data = {
            "rgb": preprocess_rgb_images(hdf5_file, image_size),
            "segmentation": preprocess_segmentation_masks(hdf5_file, image_size),
            "controls": preprocess_controls(hdf5_file),
            "frames": preprocess_frames(hdf5_file),
            "hlc": preprocess_hlc(hdf5_file),
            "light": preprocess_light(hdf5_file),
            "measurements": preprocess_measurements(hdf5_file),
        }

        # Save preprocessed data into a single file
        save_preprocessed_data(output_path, preprocessed_data)

        logging.info(f"Preprocessing completed successfully.")

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
