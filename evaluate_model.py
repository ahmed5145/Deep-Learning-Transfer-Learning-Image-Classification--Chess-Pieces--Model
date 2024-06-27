import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Define the transformation and dataset
transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder('Chess', transform=transform)
dataset_len = len(dataset)

# Train - Test Splitting
train_len = int(dataset_len * 0.8)
test_len = dataset_len - train_len

_, test_set = random_split(dataset, [train_len, test_len])
batch_size = 200

# Test dataloader
test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size)

# Use cuda, if not use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = 'Chess_Pieces.pt'
labels = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
model = models.vgg16_bn(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(labels))

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

# Test the Model
with torch.no_grad():
    total_correct = 0.0
    total_loss = 0.0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
    accuracy = (total_correct / test_len) * 100
    avg_loss = total_loss / test_len
    print(f'Accuracy: {accuracy}% Loss: {avg_loss}')
