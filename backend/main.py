from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import io
import os

app = FastAPI(title="Plant Disease Classifier API (Image Only)")

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend se connect karne ke liye ise "*" kar diya hai
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("recommendations.json", "r") as f:
    recommendations = json.load(f)
normalized_recommendations = {k.lower().strip(): v for k, v in recommendations.items()}

IMG_MODEL_PATH = "plant_cnn.pth"
CLASS_MAPPING_PATH = "class_mapping.json"

with open(CLASS_MAPPING_PATH, "r") as f:
    idx_to_class = json.load(f)

# CNN Model Architecture
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_model = PlantCNN(len(idx_to_class)).to(device)
img_model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location=device))
img_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict_image")
def predict_image(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = img_model(img_tensor)
        probs = F.softmax(outputs, dim=1)

    top_prob, top_idx = torch.topk(probs, 1)

    cls_idx = top_idx[0].item()
    confidence = top_prob[0].item()
    class_name = idx_to_class[str(cls_idx)].strip()

    rec = normalized_recommendations.get(class_name.lower(), "No recommendation available.")

    return {
        "predicted_class": class_name,
        "confidence": round(confidence * 100, 2),
        "recommendation": rec
    }