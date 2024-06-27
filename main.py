import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Data Processing

path = 'Chess'  # Root Folder Name

# Check if the path exists
if not os.path.exists(path):
    raise FileNotFoundError(f"The specified path '{path}' does not exist.")

for folder in os.listdir(path):  # Get all folder names
    for img_file in os.listdir(os.path.join(path, folder)):  # Loop each folder to get all image files
        img_file = os.path.join(path, folder, img_file)  # Create full path for each image
        
        try:
            img = Image.open(img_file)
            if img.mode != 'RGB':
                os.remove(img_file)  # Removing Gray Scale Images
        except:
            os.remove(img_file)  # Removing file type None Images

transform = transforms.Compose([
    transforms.Resize(225),  # Resize image to 225px square image
    transforms.CenterCrop(224),  # Center Crop by 224px
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize([0.5], [0.5])  # Normalize Tensors
])

dataset = datasets.ImageFolder('Chess', transform=transform)
dataset_len = len(dataset)  # Total number of images in all folders

# Check the classes in your dataset
print(dataset.class_to_idx)

# Check the length of dataset classes
print(len(dataset.classes))

# Train - Test Splitting
train_len = int(dataset_len * 0.8)
test_len = dataset_len - train_len

train_set, test_set = random_split(dataset, [train_len, test_len])
batch_size = 200

# Train & Test dataloader
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size)

# Use cuda, if not use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Downloading vgg16 pretrained image classification model
vgg16 = models.vgg16_bn(pretrained=True)

# Freeze the pretrained layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# Modify the classifier to match the number of classes in your dataset
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features, len(dataset.classes))

model = vgg16.to(device)

# Optimizer & Loss Function
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification problem
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # weight_decay & lr are hyperparameters

# Train Model
model.train()

for epoch in range(3):
    total_correct = 0.0
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch} Loss: {running_loss/train_len} Accuracy: {(total_correct/train_len)*100}%')
print("Training Finished")

# Test the Model
model.eval()
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
    print(f'Accuracy: {(total_correct/test_len)*100}% Loss: {total_loss/test_len}')

# Save the Model
torch.save(model.state_dict(), 'Chess_Pieces.pt')

# Use on Local Computer
img_file_path = ''  # Define the path to your image file

labels = dataset.classes  # Use the actual dataset classes

transform = transforms.Compose([transforms.CenterCrop(360), transforms.ToTensor()])

# Load the trained model
model_path = 'Chess_Pieces.pt'
model = models.vgg16_bn(pretrained=False)  # Load the base model without pretrained weights
model.classifier[6] = nn.Linear(num_features, len(labels))  # Adjust the final layer

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load pretrained model weights
model.eval()  # Set to evaluation mode

if img_file_path:
    img = Image.open(img_file_path)  # Load an image file
    img = transform(img)  # Transform the image into a tensor
    img = img.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        print(f'This is a {labels[predicted.item()]}')
else:
    print("No image file path provided.")
