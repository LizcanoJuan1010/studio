#!/bin/bash

# Script para copiar los modelos al directorio correcto del backend
# Los modelos deben estar en backend/models/ con los nombres correctos

echo "Configurando modelos para el backend..."

BACKEND_MODELS_DIR="./backend/models"
FRONTEND_MODELS_DIR="./src/app/(main)/models"

# Crear directorio si no existe
mkdir -p "$BACKEND_MODELS_DIR"

# Copiar y renombrar modelos si existen
if [ -f "$FRONTEND_MODELS_DIR/cnn.keras" ]; then
    echo "Copiando cnn.keras -> cnn_simple_model.keras"
    cp "$FRONTEND_MODELS_DIR/cnn.keras" "$BACKEND_MODELS_DIR/cnn_simple_model.keras"
fi

if [ -f "$FRONTEND_MODELS_DIR/cnn_transfer.keras" ]; then
    echo "Copiando cnn_transfer.keras -> mobilenetv2_fruits_best.keras"
    cp "$FRONTEND_MODELS_DIR/cnn_transfer.keras" "$BACKEND_MODELS_DIR/mobilenetv2_fruits_best.keras"
fi

if [ -f "$FRONTEND_MODELS_DIR/svm.pkl" ]; then
    echo "ADVERTENCIA: El archivo svm.pkl existe pero el backend espera svm_model.joblib"
    echo "Por favor, verifica si necesitas convertir el archivo o renombrarlo."
    echo "Copiando svm.pkl -> svm_model.joblib (asumiendo que es compatible)"
    cp "$FRONTEND_MODELS_DIR/svm.pkl" "$BACKEND_MODELS_DIR/svm_model.joblib"
fi

echo ""
echo "Modelos configurados en: $BACKEND_MODELS_DIR"
echo ""
echo "NOTA: Asegúrate de que todos los modelos requeridos estén presentes:"
echo "  - cnn_simple_model.keras"
echo "  - mobilenetv2_fruits_best.keras"
echo "  - svm_model.joblib"
echo "  - boosting_model.pkl"

