from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Your original class names (replace with actual ones from training)
class_names = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mite",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Healthy"
]

# 🧠 Match training preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)  # match training
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()

    # Preprocess
    input_data = preprocess_image(image_bytes)

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_index = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    predicted_label = class_names[prediction_index]

    # Respond with results
    return {
        "prediction": prediction_index,
        "label": predicted_label,
        "confidence": round(confidence * 100, 2)
    }

# Uncomment this block to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
