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

def load_model_with_compat(model_path):
    """
    Load a Keras model with compatibility handling for quantization_config.
    This function filters out quantization_config from layer configs if present.
    """
    # Create custom layer classes that ignore quantization_config
    # This is needed for compatibility with models saved in newer Keras versions
    def make_compatible_layer(base_class):
        """Create a compatible layer class that ignores quantization_config"""
        class CompatibleLayer(base_class):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)
            
            @classmethod
            def from_config(cls, config):
                config = config.copy() if isinstance(config, dict) else config
                if isinstance(config, dict):
                    config.pop('quantization_config', None)
                return super().from_config(config)
        
        CompatibleLayer.__name__ = f'Compatible{base_class.__name__}'
        return CompatibleLayer
    
    CompatibleDense = make_compatible_layer(tf.keras.layers.Dense)
    CompatibleGlobalAveragePooling2D = make_compatible_layer(tf.keras.layers.GlobalAveragePooling2D)
    CompatibleDropout = make_compatible_layer(tf.keras.layers.Dropout)
    
    # Register custom objects for all layer types that might have quantization_config
    custom_objects = {
        'Dense': CompatibleDense,
        'GlobalAveragePooling2D': CompatibleGlobalAveragePooling2D,
        'Dropout': CompatibleDropout,
    }
    
    try:
        # Try loading with custom objects that ignore quantization_config
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except (ValueError, TypeError) as e:
        if 'quantization_config' in str(e):
            # If still failing, try a more comprehensive approach
            print(f"⚠️  Compatibility issue detected, using advanced workaround...")
            import h5py
            
            with h5py.File(model_path, 'r') as f:
                model_config_json = f.attrs.get('model_config')
                if not model_config_json:
                    raise ValueError("Model config not found in HDF5 file")
                
                model_config = json.loads(model_config_json.decode('utf-8'))
                
                # Recursively remove quantization_config from all configs
                def clean_config(obj):
                    if isinstance(obj, dict):
                        obj.pop('quantization_config', None)
                        for v in obj.values():
                            clean_config(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            clean_config(item)
                
                clean_config(model_config)
                
                # Reconstruct model from cleaned config
                model = tf.keras.models.model_from_config(model_config, custom_objects=custom_objects)
                
                # Load weights using model.load_weights with by_name=True
                # Create a temporary weights file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_weights:
                    tmp_weights_path = tmp_weights.name
                
                try:
                    # Extract and save weights
                    with h5py.File(model_path, 'r') as f_in:
                        with h5py.File(tmp_weights_path, 'w') as f_out:
                            if 'model_weights' in f_in:
                                f_in.copy('model_weights', f_out, 'model_weights')
                    
                    # Load weights
                    model.load_weights(tmp_weights_path, by_name=True, skip_mismatch=True)
                    
                    # Clean up
                    os.unlink(tmp_weights_path)
                    
                    return model
                except Exception as cleanup_error:
                    if os.path.exists(tmp_weights_path):
                        os.unlink(tmp_weights_path)
                    raise cleanup_error
        else:
            raise

# Load models once at startup
print("Loading models...")
try:
    # Try to load best models first, fallback to regular models
    if os.path.exists("wastehub_hazard_model_best.h5"):
        hazard_model = load_model_with_compat("wastehub_hazard_model_best.h5")
        print("✅ Loaded best hazard model")
    else:
        hazard_model = load_model_with_compat("wastehub_hazard_model.h5")
        print("✅ Loaded hazard model")
    
    if os.path.exists("wastehub_hazard_type_model_best.h5"):
        type_model = load_model_with_compat("wastehub_hazard_type_model_best.h5")
        print("✅ Loaded best hazard type model")
    else:
        type_model = load_model_with_compat("wastehub_hazard_type_model.h5")
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
