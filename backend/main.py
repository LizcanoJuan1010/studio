from fastapi import FastAPI, File, UploadFile, HTTPException
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

    model_dir = './models'
    print(f"Loading models from: {os.path.abspath(model_dir)}")
    
    loaded_count = 0

    # Load Keras models
    try:
        cnn_simple_path = os.path.join(model_dir, 'cnn_simple_model.keras')
        if os.path.exists(cnn_simple_path):
            MODELS['cnn_simple'] = tf.keras.models.load_model(cnn_simple_path)
            print(f"✓ Loaded: cnn_simple_model.keras")
            loaded_count += 1
        else:
            print(f"⚠ Skipped: cnn_simple_model.keras (not found)")
    except Exception as e:
        print(f"✗ Error loading cnn_simple_model.keras: {e}")

    try:
        cnn_transfer_path = os.path.join(model_dir, 'mobilenetv2_fruits_best.keras')
        if os.path.exists(cnn_transfer_path):
            MODELS['cnn_transfer'] = tf.keras.models.load_model(cnn_transfer_path)
            print(f"✓ Loaded: mobilenetv2_fruits_best.keras")
            loaded_count += 1
        else:
            print(f"⚠ Skipped: mobilenetv2_fruits_best.keras (not found)")
    except Exception as e:
        print(f"✗ Error loading mobilenetv2_fruits_best.keras: {e}")

    # Load SVM bundle
    try:
        svm_bundle_path = os.path.join(model_dir, "svm_model.joblib")
        if os.path.exists(svm_bundle_path):
            svm_bundle = joblib.load(svm_bundle_path)
            SCALER = svm_bundle["scaler"]
            PCA = svm_bundle["pca"]
            MODELS['svm'] = svm_bundle["model"]
            CLASS_NAMES = svm_bundle["class_names"]
            print(f"✓ Loaded: svm_model.joblib")
            loaded_count += 1
        else:
            print(f"⚠ Skipped: svm_model.joblib (not found)")
    except Exception as e:
        print(f"✗ Error loading svm_model.joblib: {e}")
        print(f"  Note: SVM model may require cuML or other dependencies")

    # Load XGBoost model
    try:
        xgboost_path = os.path.join(model_dir, "boosting_model.pkl")
        if os.path.exists(xgboost_path):
            MODELS['boosting'] = joblib.load(xgboost_path)
            print(f"✓ Loaded: boosting_model.pkl")
            loaded_count += 1
        else:
            print(f"⚠ Skipped: boosting_model.pkl (not found)")
    except Exception as e:
        print(f"✗ Error loading boosting_model.pkl: {e}")

    if loaded_count == 0:
        raise RuntimeError("No models could be loaded! Please check your models directory.")
    
    print(f"✓ Successfully loaded {loaded_count} model(s).")

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
    
    y_pred_proba = MODELS['svm'].predict_proba(X_pca)
    inference_time = (time.time() - start_time) * 1000

    top_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top_probs = [y_pred_proba[0][i] for i in top_indices]

    probabilities = [
        {"label": CLASS_NAMES[int(i)], "index": int(i), "prob": float(p)}
        for i, p in zip(top_indices, top_probs)
    ]
    
    predicted_index = top_indices[0]
    predicted_label = CLASS_NAMES[int(predicted_index)]


    return {
        "model_id": "svm",
        "predicted_label": predicted_label,
        "predicted_index": int(predicted_index),
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
    
    results = []
    
    # Run predictions only for available models
    if 'cnn_simple' in MODELS:
        try:
            results.append(await predict_cnn('cnn_simple', image.copy()))
        except Exception as e:
            print(f"Error predicting with cnn_simple: {e}")
    
    if 'cnn_transfer' in MODELS:
        try:
            results.append(await predict_cnn('cnn_transfer', image.copy()))
        except Exception as e:
            print(f"Error predicting with cnn_transfer: {e}")
    
    if 'svm' in MODELS and SCALER is not None and PCA is not None:
        try:
            results.append(await predict_svm(image.copy()))
        except Exception as e:
            print(f"Error predicting with svm: {e}")
    
    if 'boosting' in MODELS:
        try:
            results.append(await predict_boosting(image.copy()))
        except Exception as e:
            print(f"Error predicting with boosting: {e}")
    
    if not results:
        raise HTTPException(status_code=503, detail="No models available for prediction")
    
    return results

@app.get("/")
def read_root():
    return {"message": "FruitVision Prediction API is running."}
