import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

simple_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_path = "flowers_augmented/train/"
valid_path = "flowers_augmented/valid/"

train = ImageFolder(train_path, transform=simple_transform)
valid = ImageFolder(valid_path, transform=simple_transform)

print("Class-to-Index Mapping:", train.class_to_idx)
print("Classes:", train.classes)

# DataLoaders
train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False, num_workers=0)

class FlowerCNN(nn.Module):
    def __init__(self):
        super(FlowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()


        self._to_linear = None
        self._calculate_flattened_size()

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 3)  # happy, sad, angry

    def _calculate_flattened_size(self):

        dummy_input = torch.zeros(1, 1, 256, 256)
        x = F.relu(F.max_pool2d(self.conv1(dummy_input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        self._to_linear = x.numel()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = FlowerCNN().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def fit(epoch, model, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        if phase == 'training':
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if phase == 'training':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * data.size(0)
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).sum().item()
        total += data.size(0)

    loss = running_loss / total
    accuracy = 100. * running_correct / total

    print(f"{phase.capitalize()} - Loss: {loss:.4f}, Accuracy: {running_correct}/{total} ({accuracy:.2f}%)")
    return loss, accuracy

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total += target.size(0)
    print(f'Validation Accuracy: {100. * correct / total:.2f}%')


if __name__ == "__main__":
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss, train_accuracy = fit(epoch, model, train_loader, phase='training')
        val_loss, val_accuracy = fit(epoch, model, valid_loader, phase='validation')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

#performance plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    #plt.show()
    evaluate_model(model, valid_loader)
    # Save the model
    model_save_path = "flower_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


