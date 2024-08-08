import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Define paths
dataset_dir = 'datasets/domainnet'
domains = ['clipart_preprocessed', 'infograph_preprocessed', 'painting_preprocessed', 'quickdraw_preprocessed', 'real_preprocessed', 'sketch_preprocessed']
model_path = 'models/domainnet_resnet50.pth'

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets and create dataloaders
dataloaders = {}
for domain in domains:
    domain_dir = os.path.join(dataset_dir, domain)
    domain_dataset = datasets.ImageFolder(domain_dir, transform=data_transforms)
    dataloaders[domain] = DataLoader(domain_dataset, batch_size=32, shuffle=False)

# Load the model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataloaders[domains[0]].dataset.classes))
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

# Evaluate on each domain
for domain in domains:
    print(f'Evaluating on {domain} dataset...')
    labels, preds = evaluate_model(dataloaders[domain], model)
    print(f'{domain.capitalize()} Classification Report:')
    print(classification_report(labels, preds))
    print(f'{domain.capitalize()} Accuracy:', accuracy_score(labels, preds))