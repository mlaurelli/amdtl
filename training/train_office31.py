import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define paths
dataset_dir = 'datasets/office31'
amazon_preprocessed = os.path.join(dataset_dir, 'amazon_preprocessed')
dslr_preprocessed = os.path.join(dataset_dir, 'dslr_preprocessed')
webcam_preprocessed = os.path.join(dataset_dir, 'webcam_preprocessed')
model_save_path = 'models/office31_resnet50.pth'

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
amazon_dataset = datasets.ImageFolder(amazon_preprocessed, transform=data_transforms)
dslr_dataset = datasets.ImageFolder(dslr_preprocessed, transform=data_transforms)
webcam_dataset = datasets.ImageFolder(webcam_preprocessed, transform=data_transforms)

# Create dataloaders
batch_size = 32
amazon_loader = DataLoader(amazon_dataset, batch_size=batch_size, shuffle=True)
dslr_loader = DataLoader(dslr_dataset, batch_size=batch_size, shuffle=True)
webcam_loader = DataLoader(webcam_dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(amazon_dataset.classes))
model = model.to(device)

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

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = running_corrects.double() / len(dataloaders.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model on Amazon dataset
print('Training on Amazon dataset...')
model = train_model(amazon_loader, model, criterion, optimizer, num_epochs=25)

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')