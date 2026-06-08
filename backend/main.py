from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import io
import os
import zipfile
import gdown


#  FASTAPI APP CONFIGURATION

app = FastAPI(title="Plant Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# TEXT CLASSIFIER SETUP & AUTO-DOWNLOAD FROM DRIVE

MODEL_PATH = "best_plant_text_classifier"
ZIP_FILE = "best_plant_text_classifier.zip"


DRIVE_FILE_ID = "1rYemMnyjMKfadvdlWRBBiCT-D8xQhui-"

if not os.path.exists(MODEL_PATH):
    print("Downloading trained model from Google Drive...")
    url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
    gdown.download(url, ZIP_FILE, quiet=False)
    
    print("Unzipping model folder...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(ZIP_FILE)  
    print("Model folder is ready!")

ENCODER_PATH = f"{MODEL_PATH}/encoder_classes.npy"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
text_model.eval()

encoder = LabelEncoder()
encoder.classes_ = np.load(ENCODER_PATH, allow_pickle=True)

with open("recommendations.json", "r") as f:
    recommendations = json.load(f)

normalized_recommendations = {k.lower().strip(): v for k, v in recommendations.items()}

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict_text")
def predict_text(input_data: TextInput):

    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_idx].item()

    pred_label = encoder.inverse_transform([pred_idx])[0].strip()
    rec = normalized_recommendations.get(pred_label.lower(), "No recommendation available.")

    return {
        "predicted_class": pred_label,
        "confidence": round(confidence * 100, 2),
        "recommendation": rec
    }


#  IMAGE CLASSIFIER SETUP

IMG_MODEL_PATH = "plant_cnn.pth"
CLASS_MAPPING_PATH = "class_mapping.json"

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
        self.fc2 = nn.Linear(512, 38)

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

with open(CLASS_MAPPING_PATH, "r") as f:
    idx_to_class = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_model = PlantCNN(len(idx_to_class)).to(device)
img_model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location=device))
img_model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#IMAGE PREDICTION — ONLY TOP-1 OUTPUT
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