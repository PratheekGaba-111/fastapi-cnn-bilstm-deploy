from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import cv2

IMG_WIDTH = 128
IMG_HEIGHT = 128

# Load your CNN + BiLSTM model
model = tf.keras.models.load_model("cnn-bilstm-model.keras")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "CNN-BiLSTM Osteoporosis Model API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    img_bytes = await file.read()

    # Convert bytes → NumPy array
    img_array = np.frombuffer(img_bytes, np.uint8)

    # Decode image using OpenCV
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize to 128×128
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize
    img = img / 255.0

    # Expand dims → (1, 128, 128, 3)
    img = np.expand_dims(img, axis=0)

    # Predict (binary classifier)
    pred = model.predict(img)[0][0]

    label = "osteoporosis" if pred > 0.5 else "normal"

    return {
        "prediction_raw": float(pred),
        "label": label
    }

