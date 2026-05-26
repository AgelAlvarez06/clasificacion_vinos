import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
data_path = Path(__file__).parent.parent / "WineQT.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("ENTRENAMIENTO DEL MODELO - CLASIFICACIÓN DE VINOS")
print("=" * 80)

# Separar características y objetivo
X = df.drop('quality', axis=1)
y = df['quality']

# Dividir en train y test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamaño conjunto entrenamiento: {X_train.shape}")
print(f"Tamaño conjunto prueba: {X_test.shape}")

# Normalizar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Datos normalizados")

# Entrenar múltiples modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n" + "=" * 80)
print("EVALUACIÓN DE MODELOS")
print("=" * 80)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Entrenar
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Seleccionar mejor modelo
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print("\n" + "=" * 80)
print(f"MEJOR MODELO: {best_model_name}")
print("=" * 80)

y_pred_best = best_model.predict(X_test_scaled)

print(f"\nAccuracy Final: {accuracy_score(y_test, y_pred_best):.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_best))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_best)
print("\nMatriz de Confusión:")
print(cm)

# Guardar modelo y escalador
model_dir = Path(__file__).parent.parent / "models"
model_dir.mkdir(exist_ok=True)

with open(model_dir / "best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(model_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✓ Modelo guardado en 'models/best_model.pkl'")
print(f"✓ Escalador guardado en 'models/scaler.pkl'")