from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64
import io
import torch
from torchvision import transforms


app = Flask(__name__)

class FlowerCNN(torch.nn.Module):
    def __init__(self):
        super(FlowerCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(64 * 61 * 61, 128)
        self.fc2 = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

model = FlowerCNN()
model.load_state_dict(torch.load("flower_model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class_names = ["angry", "happy", "sad"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":

        data_url = request.json.get("image")
        if not data_url:
            return jsonify({"error": "No image received"}), 400


        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]

        return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True, port = 5050)