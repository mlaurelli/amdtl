import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Define paths
dataset_dir = 'datasets/office31'
amazon_preprocessed = os.path.join(dataset_dir, 'amazon_preprocessed')
dslr_preprocessed = os.path.join(dataset_dir, 'dslr_preprocessed')
webcam_preprocessed = os.path.join(dataset_dir, 'webcam_preprocessed')
model_save_path = 'models/ablation_office31_resnet50.pth'

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

# Function to perform ablation study
def train_ablation_model(dataloader, model, criterion, optimizer, num_epochs=25, ablation_layer=None):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    if ablation_layer:
        # Freeze layers up to the ablation layer
        for name, param in model.named_parameters():
            if ablation_layer not in name:
                param.requires_grad = False

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
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

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print(f'Best Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# Perform ablation study by freezing different layers and training the model
ablation_layers = ['layer1', 'layer2', 'layer3', 'layer4']

for layer in ablation_layers:
    print(f'Performing ablation study, freezing {layer}...')
    ablation_model = train_ablation_model(amazon_loader, model, criterion, optimizer, num_epochs=25, ablation_layer=layer)
    torch.save(ablation_model.state_dict(), f'models/ablation_office31_{layer}_resnet50.pth')
    print(f'Model with {layer} frozen saved.')

# Evaluate the ablation models on the test sets
def evaluate_model(dataloader, model):
    model.eval()
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

for layer in ablation_layers:
    print(f'Evaluating model with {layer} frozen on Amazon dataset...')
    ablation_model.load_state_dict(torch.load(f'models/ablation_office31_{layer}_resnet50.pth'))
    amazon_labels, amazon_preds = evaluate_model(amazon_loader, ablation_model)
    print(f'{layer.capitalize()} Frozen Amazon Classification Report:')
    print(classification_report(amazon_labels, amazon_preds))
    print(f'{layer.capitalize()} Frozen Amazon Accuracy:', accuracy_score(amazon_labels, amazon_preds))
