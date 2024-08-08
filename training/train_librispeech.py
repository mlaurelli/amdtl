import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
from tqdm import tqdm

# Define paths
dataset_dir = 'datasets/librispeech'
train_clean_preprocessed = os.path.join(dataset_dir, 'train_clean_preprocessed')
train_other_preprocessed = os.path.join(dataset_dir, 'train_other_preprocessed')
test_clean_preprocessed = os.path.join(dataset_dir, 'test_clean_preprocessed')
model_save_path = 'models/librispeech_resnet18.pth'

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

train_clean_files = get_file_list(train_clean_preprocessed)
train_other_files = get_file_list(train_other_preprocessed)
test_clean_files = get_file_list(test_clean_preprocessed)

# Create Datasets and Dataloaders
train_clean_dataset = LibriSpeechDataset(train_clean_files)
train_other_dataset = LibriSpeechDataset(train_other_files)
test_clean_dataset = LibriSpeechDataset(test_clean_files)

batch_size = 32
train_clean_loader = DataLoader(train_clean_dataset, batch_size=batch_size, shuffle=True)
train_other_loader = DataLoader(train_other_dataset, batch_size=batch_size, shuffle=True)
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

model = AudioResNet(num_classes=len(train_clean_dataset)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(dataloaders, model, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train_clean', 'train_other']:
            if phase == 'train_clean':
                model.train()
                dataloader = dataloaders['train_clean']
            else:
                model.train()
                dataloader = dataloaders['train_other']

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train_clean' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best train Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# Train the model on LibriSpeech dataset
print('Training on LibriSpeech dataset...')
dataloaders = {'train_clean': train_clean_loader, 'train_other': train_other_loader}
model = train_model(dataloaders, model, criterion, optimizer, num_epochs=25)

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')