import os
import urllib.request
import tarfile
import zipfile
import librosa
import numpy as np
import soundfile as sf

# Define paths
dataset_dir = 'datasets/librispeech'
train_clean_dir = os.path.join(dataset_dir, 'train-clean-100')
train_other_dir = os.path.join(dataset_dir, 'train-other-500')
test_clean_dir = os.path.join(dataset_dir, 'test-clean')

# Create directories if they don't exist
os.makedirs(train_clean_dir, exist_ok=True)
os.makedirs(train_other_dir, exist_ok=True)
os.makedirs(test_clean_dir, exist_ok=True)

# URLs for downloading the LibriSpeech dataset
urls = {
    'train-clean-100': 'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'train-other-500': 'http://www.openslr.org/resources/12/train-other-500.tar.gz',
    'test-clean': 'http://www.openslr.org/resources/12/test-clean.tar.gz'
}

# Function to download and extract dataset
def download_and_extract(url, download_path, extract_path):
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, download_path)
    print(f'Extracting {download_path}...')
    with tarfile.open(download_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f'{download_path} extracted.')

# Download and extract each part of the dataset
download_and_extract(urls['train-clean-100'], os.path.join(dataset_dir, 'train-clean-100.tar.gz'), train_clean_dir)
download_and_extract(urls['train-other-500'], os.path.join(dataset_dir, 'train-other-500.tar.gz'), train_other_dir)
download_and_extract(urls['test-clean'], os.path.join(dataset_dir, 'test-clean.tar.gz'), test_clean_dir)

# Function to preprocess audio files
def preprocess_audio(input_dir, output_dir, sample_rate=16000):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac') or file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                audio, sr = librosa.load(audio_path, sr=sample_rate)
                save_path = os.path.join(output_dir, file.replace('.flac', '.wav'))
                sf.write(save_path, audio, sample_rate)
    print(f'Preprocessed audio files saved to {output_dir}.')

# Define output directories for preprocessed audio files
train_clean_preprocessed = os.path.join(dataset_dir, 'train_clean_preprocessed')
train_other_preprocessed = os.path.join(dataset_dir, 'train_other_preprocessed')
test_clean_preprocessed = os.path.join(dataset_dir, 'test_clean_preprocessed')

# Preprocess audio files for each part of the dataset
preprocess_audio(train_clean_dir, train_clean_preprocessed)
preprocess_audio(train_other_dir, train_other_preprocessed)
preprocess_audio(test_clean_dir, test_clean_preprocessed)