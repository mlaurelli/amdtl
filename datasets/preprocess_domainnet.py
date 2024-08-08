import os
import urllib.request
import tarfile
import zipfile
import cv2
import numpy as np

# Define paths
dataset_dir = 'datasets/domainnet'
domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

# Create directories if they don't exist
os.makedirs(dataset_dir, exist_ok=True)
for domain in domains:
    os.makedirs(os.path.join(dataset_dir, domain), exist_ok=True)

# URLs for downloading the DomainNet dataset (example links, need valid ones)
urls = {
    'clipart': 'https://domainnet/path/to/clipart.zip',
    'infograph': 'https://domainnet/path/to/infograph.zip',
    'painting': 'https://domainnet/path/to/painting.zip',
    'quickdraw': 'https://domainnet/path/to/quickdraw.zip',
    'real': 'https://domainnet/path/to/real.zip',
    'sketch': 'https://domainnet/path/to/sketch.zip'
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
for domain in domains:
    download_and_extract(urls[domain], os.path.join(dataset_dir, f'{domain}.zip'), os.path.join(dataset_dir, domain))

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

# Preprocess images for each domain
for domain in domains:
    input_dir = os.path.join(dataset_dir, domain)
    output_dir = os.path.join(dataset_dir, f'{domain}_preprocessed')
    preprocess_images(input_dir, output_dir)