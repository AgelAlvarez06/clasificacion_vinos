# Clasificación de Vinos con Machine Learning

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema de clasificación de calidad de vinos** utilizando técnicas de Machine Learning. El objetivo es predecir la calidad de un vino (en una escala de 0-10) basándose en sus características físico-químicas.

### Características Analizadas:
- Acidez fija
- Acidez volátil
- Ácido cítrico
- Azúcar residual
- Cloruros
- Dióxido de azufre libre
- Dióxido de azufre total
- Densidad
- pH
- Sulfatos
- Alcohol

---

## 👤 Alumno

**Ángel Gael Álvarez López**  
Semestre 4 - Arquitectura de IA

---

## 🏗️ Estructura del Proyecto

```
clasificacion_vinos/
├── src/
│   ├── eda.py              # Análisis exploratorio de datos
│   ├── entrenamiento.py    # Entrenamiento de modelos
│   └── prueba.py           # Prueba y predicciones
├── models/                 # Modelos entrenados guardados
│   ├── best_model.pkl
│   └── scaler.pkl
├── WineQT.csv             # Dataset de entrada
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

---

## 🚀 Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone 
```

### 2. Crear un entorno virtual (recomendado)

```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 📊 Uso del Proyecto

### Paso 1: Análisis Exploratorio de Datos (EDA)

Ejecuta el análisis exploratorio para entender la estructura y distribución de los datos:

```bash
python src/eda.py
```

**Salida:**
- Información general del dataset (dimensiones, tipos de datos, valores faltantes)
- Estadísticas descriptivas
- Distribución de la variable objetivo (quality)
- Visualizaciones:
  - Histograma de distribución de calidad
  - Top 10 características correlacionadas con quality
  - Box plot de alcohol vs calidad
  - Matriz de correlación completa
- Archivo `eda_analysis.png` con los gráficos

### Paso 2: Entrenamiento del Modelo

Entrena y evalúa múltiples modelos de ML:

```bash
python src/entrenamiento.py
```

**Modelos evaluados:**
1. **Logistic Regression** - Regresión logística
2. **Random Forest** - Bosque aleatorio (100 árboles)
3. **Gradient Boosting** - Gradient Boosting Classifier (100 estimadores)

**Salida:**
- Comparación de precisión de cada modelo
- Puntuaciones de validación cruzada (5-fold CV)
- Identificación del mejor modelo
- Reporte detallado de clasificación
- Matriz de confusión
- Modelos guardados en `models/`

### Paso 3: Prueba del Modelo

Realiza predicciones con el modelo entrenado:

```bash
python src/prueba.py
```

**Salida:**
- Predicciones en 10 muestras de prueba
- Nivel de confianza para cada predicción
- Exactitud en la muestra de prueba
- Ejemplo de predicción personalizada

---

## 📈 Detalles Técnicos

### Preprocesamiento de Datos
- **División train-test:** 80% entrenamiento, 20% prueba
- **Normalización:** StandardScaler para escalar características
- **Estratificación:** Mantiene la distribución de clases

### Evaluación de Modelos
- **Métrica principal:** Accuracy (exactitud)
- **Validación cruzada:** 5-fold cross-validation
- **Reportes:** Precisión, Recall, F1-Score por clase

### Persistencia
- Modelos guardados con `pickle`
- Escalador guardado para predicciones futuras
- Permite reutilizar modelos sin reentrenamiento

---

## 📁 Archivos Principales

### `src/eda.py`
Realiza análisis exploratorio:
- Carga el dataset WineQT.csv
- Genera estadísticas descriptivas
- Crea 4 visualizaciones principales
- Analiza correlaciones

### `src/entrenamiento.py`
Entrena y compara modelos:
- Preprocesa los datos
- Entrena 3 modelos diferentes
- Compara rendimiento
- Guarda el mejor modelo

### `src/prueba.py`
Utiliza el modelo entrenado:
- Carga el modelo guardado
- Realiza predicciones
- Calcula exactitud
- Incluye función para predicciones personalizadas

---

## 🔧 Requisitos

- **Python:** >= 3.7
- **pandas:** >= 1.3.0 (manipulación de datos)
- **numpy:** >= 1.21.0 (operaciones numéricas)
- **scikit-learn:** >= 0.24.0 (Machine Learning)
- **matplotlib:** >= 3.4.0 (visualización)
- **seaborn:** >= 0.11.0 (gráficos estadísticos)

---

## 💡 Ejemplo de Salida

```
================================================================================
ANÁLISIS EXPLORATORIO DE DATOS - CLASIFICACIÓN DE VINOS
================================================================================

1. INFORMACIÓN GENERAL DEL DATASET
Dimensiones: (1599, 12)

Tipos de datos:
fixed acidity           float64
volatile acidity        float64
...

================================================================================
ENTRENAMIENTO DEL MODELO - CLASIFICACIÓN DE VINOS
================================================================================

Random Forest:
Accuracy: 0.7125
CV Score: 0.7089 (+/- 0.0234)

Gradient Boosting:
Accuracy: 0.7281
CV Score: 0.7156 (+/- 0.0198)

================================================================================
MEJOR MODELO: Gradient Boosting
================================================================================

Accuracy Final: 0.7281
```

---

## 🎯 Conclusiones

El proyecto demuestra:
- ✅ Implementación completa de pipeline ML
- ✅ Comparación de múltiples algoritmos
- ✅ Preprocesamiento y normalización de datos
- ✅ Validación cruzada para evaluación robusta
- ✅ Persistencia de modelos para producción

---

## 📝 Notas

- El dataset debe estar en la raíz del proyecto con nombre `WineQT.csv`
- Los modelos se guardan en la carpeta `models/` automáticamente
- Las predicciones pueden personalizarse modificando el diccionario de características
- Se recomienda ejecutar los archivos en orden: eda.py → entrenamiento.py → prueba.py

---

## 📧 Contacto

**Alumno:** Ángel Gael Álvarez López  
**Curso:** Arquitectura de IA - Semestre 4  
**Institución:** LIACD

---

*Último actualizado: 25 de mayo de 2026*