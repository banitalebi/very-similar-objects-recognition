import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import transforms, models


# Hyperparameters
epochs = 9
batch_size = 24
learning_rate = 0.0001
height = 224
width = 224
rotation = 0.1
brightness = 0.1
contrast = 0.1
saturation = 0.1
hue = 0.1

train_path = "chihuahua-muffin/dataset/train"
validation_path = "chihuahua-muffin/dataset/validation"
test_path = "chihuahua-muffin/dataset/test"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ChihuahuaMuffin(nn.Module):
    def __init__(self, num_classes):
        super(ChihuahuaMuffin, self).__init__()        
        self.mobilenet = models.mobilenet_v3_small(weights='DEFAULT')      
        for param in self.mobilenet.parameters():
            param.requires_grad = False        
        # Get the output features from the last layer of the original classifier
        num_features = self.mobilenet.classifier[3].in_features        
        # Modify the classifier to match the number of classes
        self.mobilenet.classifier = nn.Sequential(
            # Keep all but the last layer
            *self.mobilenet.classifier[:-1],
            # New classifier layer
            nn.Linear(num_features, num_classes)  
        )        
        # Unfreeze the parameters of the classifier
        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.mobilenet(x)

def train_one_epoch(model, train_dataloader, criterion, optimizer):
    model.train()    
    total_loss = 0
    correct = 0
    for images, labels in train_dataloader:            
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / len(train_dataloader.dataset) 
    average_loss = total_loss / len(train_dataloader)
    return average_loss, accuracy

def test(model, test_dataloader, criterion):
    model.eval()    
    with torch.no_grad():
        total_loss = 0
        correct = 0
        for images, labels in test_dataloader:            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / len(test_dataloader.dataset)
        average_loss = total_loss / len(test_dataloader)        
    return average_loss, accuracy

def train(model, train_dataloader, validation_dataloader, criterion, optimizer, epochs):    
    for epoch in range(epochs):        
        train_loss, train_accuracy  = train_one_epoch(model, train_dataloader, criterion, optimizer)
        validation_loss, validation_accuracy = test(model, validation_dataloader, criterion)
        print(f'Epoch: {epoch+1}/{epochs}:{" "*6} Train loss: {train_loss:.4f}  |      Train accuracy: {train_accuracy:.2f}%')
        print(f'{" "*12} validation loss: {validation_loss:.4f}  | validation accuracy: {validation_accuracy:.2f}%')
        print('_'*70)

# Train dataset preparation
train_transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ColorJitter(
        brightness=brightness, 
        contrast=contrast, 
        saturation=saturation, 
        hue=hue),
    transforms.RandomHorizontalFlip(),   
    transforms.RandomRotation(rotation),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])        
])
train_dataset = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset preparation
validation_transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),  
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])    
])
validation_dataset = torchvision.datasets.ImageFolder(validation_path, transform=validation_transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

# The model
model = ChihuahuaMuffin(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
summary(model, input_size=(1, 3, height, width))

# Training phase
train(model, train_dataloader, validation_dataloader, criterion, optimizer, epochs)

# Evaluation phase
test_dataset = torchvision.datasets.ImageFolder(test_path, transform=validation_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
test_loss, test_accuracy = test(model, test_dataloader, criterion)
print(f'Test results:{" "*5} Test loss: {test_loss:.4f}  | Test accuracy: {test_accuracy:.2f}%')
print('_'*70)
