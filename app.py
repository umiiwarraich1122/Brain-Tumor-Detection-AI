import io
import os
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
app = FastAPI(title="AI Brain Tumor Detection API")

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from tensorflow.keras.models import load_model


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "brain_tumor_vgg16.h5")
print(f"Loading VGG16 model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_image(image_bytes):
    # Open image using PIL
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to 224x224 as required by VGG16
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Expand dimensions to create batch of size 1 (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use standard VGG16 preprocessing (subtracts ImageNet mean, converts RGB to BGR)
    # This is almost always required for VGG16 base models unless custom normalization was strictly applied.
    img_array = preprocess_input(img_array)
    
    return img_array

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)
        
        # Run inference
        predictions = model.predict(processed_img)
        
        # Assuming binary classification where output is a single probability
        # or categorical with 2 classes [p_no_tumor, p_tumor]
        # Let's handle both common cases based on the output shape:
        if predictions.shape[-1] == 1:
            tumor_prob = float(predictions[0][0])
            tumor_detected = tumor_prob > 0.5
        else:
            # Assumes class 1 is tumor, class 0 is no tumor
            tumor_prob = float(predictions[0][1])
            tumor_detected = np.argmax(predictions[0]) == 1
            
        if tumor_detected:
            conf_val = tumor_prob * 100.0
        else:
            conf_val = (1.0 - tumor_prob) * 100.0
            
        return {
            "tumor_detected": bool(tumor_detected),
            "confidence": float(f"{conf_val:.2f}")
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
