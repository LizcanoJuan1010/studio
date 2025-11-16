from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import pickle
import numpy as np
from PIL import Image
import io
import time
import os
from pathlib import Path

app = FastAPI()

# Allow all origins for simplicity in this development environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model state ---
MODELS = {}
SCALER = None
PCA = None
CLASS_NAMES = []

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Alias opcionales para mantener IDs bonitos en frontend
KERAS_ALIASES = {
    "cnn_simple_model": "cnn_simple",
    "mobilenetv2_fruits_best": "cnn_transfer",
}


def _get_label(idx: int) -> str:
    """
    Devuelve el nombre de la clase si está disponible.
    Si CLASS_NAMES está vacío o el índice se sale de rango,
    devuelve el índice como string para no romper la API.
    """
    try:
        if CLASS_NAMES:
            return str(CLASS_NAMES[int(idx)])
    except (IndexError, TypeError, ValueError):
        pass
    return str(int(idx))


def load_models():
    """
    Carga todos los modelos encontrados en la carpeta /models
    relativa a este archivo:
        - Todos los *.keras -> modelos Keras (CNN)
        - svm_model.joblib -> scaler, PCA, modelo SVM + CLASS_NAMES
        - boosting_model.pkl -> modelo de boosting (XGBoost / sklearn)
    """
    global SCALER, PCA, CLASS_NAMES

    print("=" * 60)
    print("🚀 INICIANDO CARGA DE MODELOS")
    print("=" * 60)
    print(f"📁 Directorio de modelos: {MODEL_DIR}")
    print(f"📁 Ruta absoluta: {MODEL_DIR.absolute()}")
    
    if not MODEL_DIR.exists():
        print("⚠ ADVERTENCIA: El directorio 'models' no existe, creándolo vacío.")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directorio creado: {MODEL_DIR.absolute()}")
    else:
        print(f"✓ Directorio 'models' encontrado")
        # Listar archivos en el directorio
        files = list(MODEL_DIR.iterdir())
        print(f"📋 Archivos encontrados en el directorio: {len(files)}")
        for f in files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   - {f.name} ({size_mb:.2f} MB)")

    loaded_count = 0
    loaded_models = []
    failed_models = []

    print("\n" + "-" * 60)
    print("📦 CARGANDO MODELOS KERAS (.keras)")
    print("-" * 60)
    
    # ---- Keras models (.keras) ----
    keras_files = list(MODEL_DIR.glob("*.keras"))
    if not keras_files:
        print("⚠ No se encontraron archivos .keras en el directorio")
    else:
        for keras_path in keras_files:
            try:
                stem = keras_path.stem
                model_id = KERAS_ALIASES.get(stem, stem)  # alias opcional
                print(f"   🔄 Cargando: {keras_path.name}...")
                MODELS[model_id] = tf.keras.models.load_model(keras_path)
                print(f"   ✅ ÉXITO: Modelo '{model_id}' cargado correctamente")
                print(f"      - Archivo: {keras_path.name}")
                print(f"      - ID interno: {model_id}")
                loaded_count += 1
                loaded_models.append(f"Keras: {model_id} ({keras_path.name})")
            except Exception as e:
                print(f"   ❌ ERROR al cargar {keras_path.name}: {e}")
                failed_models.append(f"Keras: {keras_path.name} - {str(e)}")

    print("\n" + "-" * 60)
    print("📦 CARGANDO MODELO SVM (.pkl)")
    print("-" * 60)
    
    # ---- SVM model (.pkl) ----
    try:
        svm_path = MODEL_DIR / "svm_final_model.pkl"
        if svm_path.exists():
            print(f"   🔄 Cargando: svm_final_model.pkl...")
            print(f"      - Tamaño del archivo: {svm_path.stat().st_size / (1024*1024):.2f} MB")
            
            try:
                # Intentar primero con joblib (formato estándar)
                print(f"      - Intentando cargar con joblib...")
                svm_data = joblib.load(svm_path)
                
                # El archivo puede ser un diccionario con scaler, PCA, model, class_names
                # o solo el modelo directamente
                if isinstance(svm_data, dict):
                    SCALER = svm_data.get("scaler")
                    PCA = svm_data.get("pca")
                    MODELS["svm"] = svm_data.get("model")
                    CLASS_NAMES = svm_data.get("class_names", CLASS_NAMES)
                    print(f"   ✅ ÉXITO: Modelo SVM cargado correctamente (formato bundle)")
                    print(f"      - Scaler: {'✓' if SCALER is not None else '✗'}")
                    print(f"      - PCA: {'✓' if PCA is not None else '✗'}")
                    print(f"      - Modelo: {'✓' if 'svm' in MODELS else '✗'}")
                    print(f"      - Clases: {len(CLASS_NAMES) if CLASS_NAMES else 0} clases")
                else:
                    # Si es solo el modelo, lo cargamos directamente
                    MODELS["svm"] = svm_data
                    print(f"   ✅ ÉXITO: Modelo SVM cargado correctamente (solo modelo)")
                    print(f"      - Tipo de modelo: {type(svm_data).__name__}")
                    print(f"      - Modelo: {'✓' if 'svm' in MODELS else '✗'}")
                    print(f"      - Nota: Scaler y PCA no encontrados en el archivo")
                    print(f"      - ADVERTENCIA: El modelo puede requerir preprocesamiento adicional")
                
                loaded_count += 1
                loaded_models.append("SVM: svm_final_model.pkl")
                
            except ModuleNotFoundError as e:
                if "cuml" in str(e).lower():
                    print(f"      - Error con joblib (requiere cuML), intentando con pickle estándar...")
                    try:
                        # Intentar con pickle estándar
                        with open(svm_path, 'rb') as f:
                            svm_data = pickle.load(f)
                        
                        if isinstance(svm_data, dict):
                            SCALER = svm_data.get("scaler")
                            PCA = svm_data.get("pca")
                            MODELS["svm"] = svm_data.get("model")
                            CLASS_NAMES = svm_data.get("class_names", CLASS_NAMES)
                            print(f"   ✅ ÉXITO: Modelo SVM cargado con pickle (formato bundle)")
                            print(f"      - Scaler: {'✓' if SCALER is not None else '✗'}")
                            print(f"      - PCA: {'✓' if PCA is not None else '✗'}")
                            print(f"      - Modelo: {'✓' if 'svm' in MODELS else '✗'}")
                            print(f"      - Clases: {len(CLASS_NAMES) if CLASS_NAMES else 0} clases")
                        else:
                            MODELS["svm"] = svm_data
                            print(f"   ✅ ÉXITO: Modelo SVM cargado con pickle (solo modelo)")
                            print(f"      - Tipo: {type(svm_data).__name__}")
                            print(f"      - Modelo: {'✓' if 'svm' in MODELS else '✗'}")
                        
                        loaded_count += 1
                        loaded_models.append("SVM: svm_final_model.pkl (pickle)")
                    except Exception as e2:
                        print(f"   ❌ ERROR: No se pudo cargar con pickle tampoco")
                        print(f"      - Error pickle: {type(e2).__name__}: {str(e2)}")
                        print(f"   ⚠ ADVERTENCIA: El modelo requiere cuML (RAPIDS)")
                        print(f"      - Error original: {e}")
                        print(f"      - El modelo fue entrenado con cuML pero no está disponible")
                        print(f"      - Opciones:")
                        print(f"        1. Instalar cuML en el contenedor (requiere GPU)")
                        print(f"        2. Re-entrenar el modelo con scikit-learn")
                        print(f"        3. Usar solo los modelos Keras disponibles")
                        failed_models.append(f"SVM: svm_final_model.pkl - Requiere cuML: {str(e)}")
                else:
                    raise
        else:
            print(f"   ⚠ Archivo no encontrado: svm_final_model.pkl")
            # Intentar con el nombre anterior por compatibilidad
            svm_old_path = MODEL_DIR / "svm_model.joblib"
            if svm_old_path.exists():
                print(f"   🔄 Intentando con formato antiguo: svm_model.joblib...")
                try:
                    svm_bundle = joblib.load(svm_old_path)
                    SCALER = svm_bundle.get("scaler")
                    PCA = svm_bundle.get("pca")
                    MODELS["svm"] = svm_bundle.get("model")
                    CLASS_NAMES = svm_bundle.get("class_names", CLASS_NAMES)
                    print(f"   ✅ ÉXITO: Modelo SVM cargado desde formato antiguo")
                    loaded_count += 1
                    loaded_models.append("SVM: svm_model.joblib (formato antiguo)")
                except Exception as e2:
                    print(f"   ❌ ERROR al cargar svm_model.joblib: {e2}")
                    failed_models.append(f"SVM: svm_model.joblib - {str(e2)}")
    except Exception as e:
        print(f"   ❌ ERROR al cargar svm_final_model.pkl: {e}")
        print(f"      Detalles del error: {type(e).__name__}: {str(e)}")
        failed_models.append(f"SVM: svm_final_model.pkl - {str(e)}")

    print("\n" + "-" * 60)
    print("📦 CARGANDO MODELO BOOSTING (.pkl)")
    print("-" * 60)
    
    # ---- Boosting model ----
    try:
        xgboost_path = MODEL_DIR / "boosting_model.pkl"
        if xgboost_path.exists():
            print(f"   🔄 Cargando: boosting_model.pkl...")
            MODELS["boosting"] = joblib.load(xgboost_path)
            print(f"   ✅ ÉXITO: Modelo Boosting cargado correctamente")
            loaded_count += 1
            loaded_models.append("Boosting: boosting_model.pkl")
        else:
            print(f"   ⚠ Archivo no encontrado: boosting_model.pkl")
    except Exception as e:
        print(f"   ❌ ERROR al cargar boosting_model.pkl: {e}")
        failed_models.append(f"Boosting: boosting_model.pkl - {str(e)}")

    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE CARGA DE MODELOS")
    print("=" * 60)
    print(f"✅ Modelos cargados exitosamente: {loaded_count}")
    
    if loaded_models:
        print("\n📋 Modelos disponibles:")
        for i, model in enumerate(loaded_models, 1):
            print(f"   {i}. {model}")
    
    if failed_models:
        print(f"\n❌ Modelos con errores: {len(failed_models)}")
        for i, model in enumerate(failed_models, 1):
            print(f"   {i}. {model}")
    
    if not CLASS_NAMES:
        print("\n⚠ ADVERTENCIA: CLASS_NAMES está vacío.")
        print("   Las etiquetas en las predicciones serán índices numéricos.")
    else:
        print(f"\n✓ CLASS_NAMES cargado: {len(CLASS_NAMES)} clases disponibles")
        print(f"   Primeras 5 clases: {CLASS_NAMES[:5]}")
    
    print("\n" + "=" * 60)
    print(f"🎯 ESTADO FINAL: {loaded_count} modelo(s) listo(s) para usar")
    print("=" * 60 + "\n")

    if loaded_count == 0:
        raise RuntimeError("❌ CRÍTICO: No se pudo cargar ningún modelo! Verifica el directorio 'models'.")


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
        {"label": _get_label(i), "index": int(i), "prob": float(p)}
        for i, p in zip(top_indices, top_probs)
    ]

    predicted_index = int(top_indices[0])
    predicted_label = _get_label(predicted_index)

    return {
        "model_id": model_id,
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "probabilities": probabilities,
        "inference_time_ms": round(inference_time, 1),
    }


async def predict_svm(image: Image.Image):
    if "svm" not in MODELS or SCALER is None or PCA is None:
        raise RuntimeError("SVM model or preprocessing objects not loaded.")

    start_time = time.time()
    processed_image = preprocess_image_for_svm_boosting(image)

    # Preprocess with scaler and PCA
    X_scaled = SCALER.transform(processed_image)
    X_pca = PCA.transform(X_scaled)

    y_pred_proba = MODELS["svm"].predict_proba(X_pca)
    inference_time = (time.time() - start_time) * 1000

    top_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top_probs = [y_pred_proba[0][i] for i in top_indices]

    probabilities = [
        {"label": _get_label(i), "index": int(i), "prob": float(p)}
        for i, p in zip(top_indices, top_probs)
    ]

    predicted_index = int(top_indices[0])
    predicted_label = _get_label(predicted_index)

    return {
        "model_id": "svm",
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "probabilities": probabilities,
        "inference_time_ms": round(inference_time, 1),
    }


async def predict_boosting(image: Image.Image):
    if "boosting" not in MODELS:
        raise RuntimeError("Boosting model not loaded.")

    start_time = time.time()
    processed_image = preprocess_image_for_svm_boosting(image)

    predictions = MODELS["boosting"].predict_proba(processed_image)
    inference_time = (time.time() - start_time) * 1000

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_probs = [predictions[0][i] for i in top_indices]

    probabilities = [
        {"label": _get_label(i), "index": int(i), "prob": float(p)}
        for i, p in zip(top_indices, top_probs)
    ]

    predicted_index = int(top_indices[0])
    predicted_label = _get_label(predicted_index)

    return {
        "model_id": "boosting",
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "probabilities": probabilities,
        "inference_time_ms": round(inference_time, 1),
    }


@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = []

    # 1) Todos los modelos Keras (*.keras) cargados
    for model_id, model in MODELS.items():
        if isinstance(model, tf.keras.Model):
            try:
                results.append(await predict_cnn(model_id, image.copy()))
            except Exception as e:
                print(f"Error predicting with {model_id}: {e}")

    # 2) SVM (si está disponible)
    if "svm" in MODELS and SCALER is not None and PCA is not None:
        try:
            results.append(await predict_svm(image.copy()))
        except Exception as e:
            print(f"Error predicting with svm: {e}")

    # 3) Boosting (si está disponible)
    if "boosting" in MODELS:
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


@app.get("/status")
def get_status():
    """
    Endpoint para verificar el estado de los modelos cargados.
    Muestra qué modelos están disponibles y listos para usar.
    """
    model_status = {}
    
    # Verificar modelos Keras
    keras_models = {}
    for model_id, model in MODELS.items():
        if isinstance(model, tf.keras.Model):
            keras_models[model_id] = {
                "type": "Keras/CNN",
                "status": "loaded",
                "input_shape": str(model.input_shape) if hasattr(model, 'input_shape') else "unknown"
            }
    
    # Verificar SVM
    svm_status = {
        "status": "loaded" if "svm" in MODELS and SCALER is not None and PCA is not None else "not_available",
        "model": "loaded" if "svm" in MODELS else "not_loaded",
        "scaler": "loaded" if SCALER is not None else "not_loaded",
        "pca": "loaded" if PCA is not None else "not_loaded"
    }
    
    # Verificar Boosting
    boosting_status = {
        "status": "loaded" if "boosting" in MODELS else "not_available"
    }
    
    return {
        "api_status": "running",
        "models_directory": str(MODEL_DIR.absolute()),
        "total_models_loaded": len(MODELS),
        "class_names_count": len(CLASS_NAMES),
        "models": {
            "keras": keras_models,
            "svm": svm_status,
            "boosting": boosting_status
        },
        "summary": {
            "available_models": list(MODELS.keys()),
            "ready_for_prediction": len([m for m in MODELS.values() if isinstance(m, tf.keras.Model)]) + 
                                   (1 if svm_status["status"] == "loaded" else 0) +
                                   (1 if boosting_status["status"] == "loaded" else 0)
        }
    }
