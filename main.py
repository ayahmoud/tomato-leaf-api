from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load your updated TFLite model
interpreter = tf.lite.Interpreter(model_path="eff_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Use your actual class labels from training
class_names = [
    "Healthy",
    "Leaf Mold",
    "Septoria Leaf Spot"
    # Add more if needed
]

# 🧠 Preprocess to match EfficientNetB0 (300x300, normalize 0–1)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((300, 300))
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalized
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)

    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    predicted_label = class_names[prediction_index]

    return {
        "prediction": prediction_index,
        "label": predicted_label,
        "confidence": round(confidence * 100, 2)
    }

# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
