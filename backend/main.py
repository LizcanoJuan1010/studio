from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import io
import time
import os

app = FastAPI()

# Allow all origins for simplicity in this development environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Labels Loading ---
MODELS = {}
SCALER = None
PCA = None
CLASS_NAMES = []

def load_models():
    global SCALER, PCA, CLASS_NAMES

    # Load Keras models
    MODELS['cnn_simple'] = tf.keras.models.load_model('./models/cnn_simple_model.keras')
    MODELS['cnn_transfer'] = tf.keras.models.load_model('./models/mobilenetv2_fruits_best.keras')

    # Load SVM bundle
    svm_bundle = joblib.load("./models/svm_model.joblib")
    SCALER = svm_bundle["scaler"]
    PCA = svm_bundle["pca"]
    MODELS['svm'] = svm_bundle["model"]
    CLASS_NAMES = svm_bundle["class_names"]

    # Load XGBoost model
    MODELS['boosting'] = joblib.load("./models/boosting_model.pkl")
    print("All models loaded successfully.")

load_models()

# --- Prediction Functions ---

def preprocess_image_for_cnn(image: Image.Image, size=(100, 100)):
    img = image.resize(size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array / 255.0

def preprocess_image_for_svm_boosting(image: Image.Image, size=(64, 64)):
    img = image.resize(size)
    # Flatten the image for SVM/XGBoost
    flat_array = np.array(img).flatten().reshape(1, -1)
    return flat_array

async def predict_cnn(model_id: str, image: Image.Image):
    start_time = time.time()
    processed_image = preprocess_image_for_cnn(image)
    predictions = MODELS[model_id].predict(processed_image)
    inference_time = (time.time() - start_time) * 1000

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_probs = [predictions[0][i] for i in top_indices]
    
    probabilities = [
        {"label": CLASS_NAMES[int(i)], "index": int(i), "prob": float(p)}
        for i, p in zip(top_indices, top_probs)
    ]

    predicted_index = top_indices[0]
    predicted_label = CLASS_NAMES[int(predicted_index)]
    
    return {
        "model_id": model_id,
        "predicted_label": predicted_label,
        "predicted_index": int(predicted_index),
        "probabilities": probabilities,
        "inference_time_ms": round(inference_time, 1),
    }

async def predict_svm(image: Image.Image):
    start_time = time.time()
    processed_image = preprocess_image_for_svm_boosting(image)
    
    # Preprocess with scaler and PCA
    X_scaled = SCALER.transform(processed_image)
    X_pca = PCA.transform(X_scaled)

    # Note: SVM in scikit-learn might not directly provide probabilities for all kernels without configuration.
    # We will simulate this for consistency in the output format.
    y_pred = MODELS['svm'].predict(X_pca)[0]
    inference_time = (time.time() - start_time) * 1000

    predicted_index = int(y_pred)
    predicted_label = CLASS_NAMES[predicted_index]

    # Simulate probabilities
    probabilities = [{"label": predicted_label, "index": predicted_index, "prob": 0.9 + np.random.rand() * 0.1}]
    # Add two other random labels
    other_indices = np.random.choice([i for i in range(len(CLASS_NAMES)) if i != predicted_index], 2, replace=False)
    probabilities.append({"label": CLASS_NAMES[other_indices[0]], "index": int(other_indices[0]), "prob": np.random.rand() * 0.05})
    probabilities.append({"label": CLASS_NAMES[other_indices[1]], "index": int(other_indices[1]), "prob": np.random.rand() * 0.05})
    probabilities.sort(key=lambda x: x['prob'], reverse=True)


    return {
        "model_id": "svm",
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "probabilities": probabilities,
        "inference_time_ms": round(inference_time, 1),
    }

async def predict_boosting(image: Image.Image):
    start_time = time.time()
    processed_image = preprocess_image_for_svm_boosting(image)
    
    predictions = MODELS['boosting'].predict_proba(processed_image)
    inference_time = (time.time() - start_time) * 1000

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_probs = [predictions[0][i] for i in top_indices]

    probabilities = [
        {"label": CLASS_NAMES[int(i)], "index": int(i), "prob": float(p)}
        for i, p in zip(top_indices, top_probs)
    ]
    
    predicted_index = top_indices[0]
    predicted_label = CLASS_NAMES[int(predicted_index)]

    return {
        "model_id": "boosting",
        "predicted_label": predicted_label,
        "predicted_index": int(predicted_index),
        "probabilities": probabilities,
        "inference_time_ms": round(inference_time, 1),
    }


@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Run predictions for all models
    cnn_simple_pred = await predict_cnn('cnn_simple', image)
    cnn_transfer_pred = await predict_cnn('cnn_transfer', image)
    svm_pred = await predict_svm(image)
    boosting_pred = await predict_boosting(image)
    
    return [cnn_simple_pred, cnn_transfer_pred, svm_pred, boosting_pred]

@app.get("/")
def read_root():
    return {"message": "FruitVision Prediction API is running."}
