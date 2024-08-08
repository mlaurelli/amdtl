import os
import urllib.request
import tarfile
import zipfile
import cv2
import numpy as np

# Define paths
dataset_dir = 'datasets/office31'
amazon_dir = os.path.join(dataset_dir, 'amazon')
dslr_dir = os.path.join(dataset_dir, 'dslr')
webcam_dir = os.path.join(dataset_dir, 'webcam')

# Create directories if they don't exist
os.makedirs(amazon_dir, exist_ok=True)
os.makedirs(dslr_dir, exist_ok=True)
os.makedirs(webcam_dir, exist_ok=True)

# URLs for downloading the Office-31 dataset
urls = {
    'amazon': 'https://drive.google.com/uc?export=download&id=1O6cRyWYtKJ_Rp_t0v4Rs1UZV8vYE0rI5',
    'dslr': 'https://drive.google.com/uc?export=download&id=1vlpKO0mnP2OYH3yoSoz_JmPIoiL5sNGk',
    'webcam': 'https://drive.google.com/uc?export=download&id=1vjvGCBpjVDLnCnEg-m-fusMA8-tGEOJZ'
}

# Function to download and extract dataset
def download_and_extract(url, download_path, extract_path):
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, download_path)
    print(f'Extracting {download_path}...')
    if download_path.endswith('.tar.gz'):
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    elif download_path.endswith('.zip'):
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    print(f'{download_path} extracted.')

# Download and extract each domain
download_and_extract(urls['amazon'], os.path.join(dataset_dir, 'amazon.zip'), amazon_dir)
download_and_extract(urls['dslr'], os.path.join(dataset_dir, 'dslr.zip'), dslr_dir)
download_and_extract(urls['webcam'], os.path.join(dataset_dir, 'webcam.zip'), webcam_dir)

# Function to preprocess images
def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, img_size)
                img_normalized = img_resized / 255.0
                save_path = os.path.join(output_dir, file)
                cv2.imwrite(save_path, img_normalized * 255)
    print(f'Preprocessed images saved to {output_dir}.')

# Define output directories for preprocessed images
amazon_preprocessed = os.path.join(dataset_dir, 'amazon_preprocessed')
dslr_preprocessed = os.path.join(dataset_dir, 'dslr_preprocessed')
webcam_preprocessed = os.path.join(dataset_dir, 'webcam_preprocessed')

# Preprocess images for each domain
preprocess_images(amazon_dir, amazon_preprocessed)
preprocess_images(dslr_dir, dslr_preprocessed)
preprocess_images(webcam_dir, webcam_preprocessed)