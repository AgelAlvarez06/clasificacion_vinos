import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("=" * 80)
print("PRUEBA DEL MODELO - CLASIFICACIÓN DE VINOS")
print("=" * 80)

# Cargar modelo y escalador
model_dir = Path(__file__).parent.parent / "models"

with open(model_dir / "best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(model_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("\n✓ Modelo cargado correctamente")

# Cargar datos de prueba
data_path = Path(__file__).parent.parent / "WineQT.csv"
df = pd.read_csv(data_path)

# Usar últimas 10 muestras para prueba
X_test_sample = df.drop('quality', axis=1).iloc[-10:]
y_test_sample = df['quality'].iloc[-10:]

# Normalizar
X_test_scaled = scaler.transform(X_test_sample)

# Predicciones
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

print("\n" + "=" * 80)
print("RESULTADOS DE PREDICCIÓN")
print("=" * 80)

results_df = pd.DataFrame({
    'Valor Real': y_test_sample.values,
    'Predicción': predictions,
    'Confianza (%)': probabilities.max(axis=1) * 100
})

print("\n" + results_df.to_string(index=True))

# Calcular exactitud en muestra de prueba
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_sample, predictions)

print(f"\n{'=' * 80}")
print(f"Exactitud en muestra de prueba: {accuracy:.2%}")
print(f"{'=' * 80}")

# Función para hacer predicciones personalizadas
def predict_wine_quality(features_dict):
    """
    Predice la calidad del vino basado en características.
    
    Args:
        features_dict: Diccionario con características del vino
    
    Returns:
        tuple: (predicción, confianza)
    """
    # Obtener nombres de características del modelo
    feature_names = df.drop('quality', axis=1).columns
    
    # Crear array con características en el orden correcto
    features_array = np.array([features_dict[feat] for feat in feature_names]).reshape(1, -1)
    
    # Normalizar
    features_scaled = scaler.transform(features_array)
    
    # Predecir
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled).max() * 100
    
    return prediction, confidence

# Ejemplo de predicción personalizada
print("\n" + "=" * 80)
print("PREDICCIÓN PERSONALIZADA")
print("=" * 80)

example_wine = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

pred, conf = predict_wine_quality(example_wine)
print(f"\nCalidad Predicha: {pred}")
print(f"Confianza: {conf:.2f}%")