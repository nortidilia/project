import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

class FlowerCNN(nn.Module):
    def __init__(self):
        super(FlowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self._to_linear = None
        self._calculate_flattened_size()

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 3)

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


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = FlowerCNN().to(device)
model.load_state_dict(torch.load("flower_model.pth", map_location=device))
model.eval()


def preprocess_image(image_path):
    """Preprocess the input image: resize, grayscale, normalize."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict_image(image_path):
    """Predict the class of a given image."""
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
    class_labels = ['angry', 'happy', 'sad']
    predicted_label = class_labels[predicted_idx.item()]
    return predicted_label


image_path = ""  # Replace with the path to your test image
predicted_label = predict_image(image_path)
print(f"The predicted label is: {predicted_label}")