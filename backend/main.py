# main.py

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import os
from models.PlantDiseaseModel import PlantDiseaseModel
from transformers import pipeline
import io
from PIL import Image
from torchvision import transforms
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware

# ✅ Text Input Schema
class TextRequest(BaseModel):
    input: str


# ✅ Base Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load Class Names
with open(os.path.join(BASE_DIR, "image_class_names.json"), "r") as f:
    CLASS_NAMES = json.load(f)

# ✅ Load Text Model
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "disease_detection_model")
text_classifier = pipeline("text-classification", model=TEXT_MODEL_PATH)

# ✅ Load Text Label Encoder
TEXT_ENCODER_PATH = os.path.join(BASE_DIR, "text_label_encoder.joblib")
text_encoder = joblib.load(TEXT_ENCODER_PATH)

# ✅ Load CNN Image Model
image_model = PlantDiseaseModel()
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_cnn.pth")

with open(IMAGE_MODEL_PATH, "rb") as f:
    weights = torch.load(f, map_location=torch.device("cpu"))
image_model.load_state_dict(weights)
image_model.eval()

# ✅ Load Recommendations JSON
with open(os.path.join(BASE_DIR, "recommendations.json"), "r") as f:
    RECOMMENDATIONS = json.load(f)

# ✅ FastAPI Initialize
app = FastAPI(title="Plant Disease Detection API")

# ✅ Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Health Check
@app.get("/health-check")
def health_check():
    return {"status": "OK"}


# ✅ IMAGE Prediction
@app.post("/image-prediction")
async def image_prediction(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4759, 0.5003, 0.4266],
            std=[0.2102, 0.1888, 0.2262]
        )
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

    recommendation = RECOMMENDATIONS.get(predicted_class, "No recommendation available.")

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": f"{confidence_score:.4f}",
        "recommendation": recommendation,
    }


# ✅ TEXT Prediction
@app.post("/text-prediction")
async def text_prediction(data: TextRequest):
    raw_pred = text_classifier(data.input)[0]

    label_id = int(raw_pred["label"].split("_")[-1])
    disease_name = text_encoder.inverse_transform([label_id])[0]

    recommendation = RECOMMENDATIONS.get(disease_name, "No recommendation available.")

    return {
        "input_text": data.input,
        "predicted_disease": disease_name,
        "confidence": f"{raw_pred['score']:.4f}",
        "recommendation": recommendation,
    }


# ✅ Server Start (for manual run)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)