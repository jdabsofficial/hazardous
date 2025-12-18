from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
import os

app = FastAPI(title="WasteHub ML API")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
print("Loading models...")
try:
    # Try to load best models first, fallback to regular models
    if os.path.exists("wastehub_hazard_model_best.h5"):
        hazard_model = tf.keras.models.load_model("wastehub_hazard_model_best.h5")
        print("✅ Loaded best hazard model")
    else:
        hazard_model = tf.keras.models.load_model("wastehub_hazard_model.h5")
        print("✅ Loaded hazard model")
    
    if os.path.exists("wastehub_hazard_type_model_best.h5"):
        type_model = tf.keras.models.load_model("wastehub_hazard_type_model_best.h5")
        print("✅ Loaded best hazard type model")
    else:
        type_model = tf.keras.models.load_model("wastehub_hazard_type_model.h5")
        print("✅ Loaded hazard type model")
    
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Load class names from saved file if it exists, otherwise use default
if os.path.exists("hazard_type_classes.json"):
    with open("hazard_type_classes.json", "r") as f:
        class_names = json.load(f)
    # Map to proper labels (capitalize first letter)
    LABELS = [name.capitalize() for name in class_names]
else:
    # Default labels (should match training order)
    LABELS = ["Oil", "Paints", "Pesticides", "Septics"]

# Threshold for hazardous classification (0.5 is standard, but can be adjusted)
HAZARD_THRESHOLD = 0.5

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {
        "message": "WasteHub ML API is running!",
        "endpoints": {
            "predict": "/predict (POST) - Upload an image to classify waste"
        },
        "models_loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess(img_bytes)

    # Get hazard prediction
    hazard_prediction = hazard_model.predict(img, verbose=0)[0][0]
    
    # Note: In binary classification with ImageDataGenerator:
    # - Class 0 (hazardous) = values close to 0
    # - Class 1 (non_hazardous) = values close to 1
    # So we check if prediction is BELOW threshold for hazardous
    
    is_hazardous = hazard_prediction < HAZARD_THRESHOLD
    
    if not is_hazardous:
        return {
            "hazardous": False,
            "label": "Non-Hazardous",
            "confidence": float(hazard_prediction),  # High value = high confidence for non-hazardous
            "hazard_score": float(hazard_prediction)
        }

    # If hazardous, predict the type
    type_scores = type_model.predict(img, verbose=0)[0]
    type_idx = int(np.argmax(type_scores))
    type_confidence = float(np.max(type_scores))
    hazard_type = LABELS[type_idx] if type_idx < len(LABELS) else f"Class_{type_idx}"

    return {
        "hazardous": True,
        "label": f"Hazardous - {hazard_type}",
        "confidence": float(type_confidence),
        "hazard_score": float(hazard_prediction),
        "type_breakdown": {
            LABELS[i] if i < len(LABELS) else f"Class_{i}": float(score)
            for i, score in enumerate(type_scores)
        }
    }
