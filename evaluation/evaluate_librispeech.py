import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
from sklearn.metrics import classification_report, accuracy_score

# Define paths
dataset_dir = 'datasets/librispeech'
train_clean_preprocessed = os.path.join(dataset_dir, 'train_clean_preprocessed')
train_other_preprocessed = os.path.join(dataset_dir, 'train_other_preprocessed')
test_clean_preprocessed = os.path.join(dataset_dir, 'test_clean_preprocessed')
model_path = 'models/librispeech_resnet18.pth'

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class for LibriSpeech
class LibriSpeechDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        label = int(file_path.split('-')[0].split('/')[-1])
        return mel_spec_db, label

# Get list of files for each dataset
def get_file_list(dataset_path):
    file_list = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_list.append(os.path.join(root, file))
    return file_list

test_clean_files = get_file_list(test_clean_preprocessed)

# Create Datasets and Dataloaders
test_clean_dataset = LibriSpeechDataset(test_clean_files)

batch_size = 32
test_clean_loader = DataLoader(test_clean_dataset, batch_size=batch_size, shuffle=False)

# Define the model (ResNet-18 for this example)
class AudioResNet(nn.Module):
    def __init__(self, num_classes):
        super(AudioResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

model = AudioResNet(num_classes=len(test_clean_dataset)).to(device)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluation function
def evaluate_model(dataloader, model):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Evaluate on Test Clean dataset
print('Evaluating on Test Clean dataset...')
test_clean_labels, test_clean_preds = evaluate_model(test_clean_loader, model)
print('Test Clean Classification Report:')
print(classification_report(test_clean_labels, test_clean_preds))
print('Test Clean Accuracy:', accuracy_score(test_clean_labels, test_clean_preds))