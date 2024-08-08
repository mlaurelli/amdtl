import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Define paths
dataset_dir = 'datasets/office31'
amazon_preprocessed = os.path.join(dataset_dir, 'amazon_preprocessed')
dslr_preprocessed = os.path.join(dataset_dir, 'dslr_preprocessed')
webcam_preprocessed = os.path.join(dataset_dir, 'webcam_preprocessed')
model_path = 'models/office31_resnet50.pth'

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
amazon_loader = DataLoader(amazon_dataset, batch_size=batch_size, shuffle=False)
dslr_loader = DataLoader(dslr_dataset, batch_size=batch_size, shuffle=False)
webcam_loader = DataLoader(webcam_dataset, batch_size=batch_size, shuffle=False)

# Load the model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(amazon_dataset.classes))
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

# Evaluate on Amazon dataset
print('Evaluating on Amazon dataset...')
amazon_labels, amazon_preds = evaluate_model(amazon_loader, model)
print('Amazon Classification Report:')
print(classification_report(amazon_labels, amazon_preds))
print('Amazon Accuracy:', accuracy_score(amazon_labels, amazon_preds))

# Evaluate on DSLR dataset
print('Evaluating on DSLR dataset...')
dslr_labels, dslr_preds = evaluate_model(dslr_loader, model)
print('DSLR Classification Report:')
print(classification_report(dslr_labels, dslr_preds))
print('DSLR Accuracy:', accuracy_score(dslr_labels, dslr_preds))

# Evaluate on Webcam dataset
print('Evaluating on Webcam dataset...')
webcam_labels, webcam_preds = evaluate_model(webcam_loader, model)
print('Webcam Classification Report:')
print(classification_report(webcam_labels, webcam_preds))
print('Webcam Accuracy:', accuracy_score(webcam_labels, webcam_preds))
