import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos
data_path = Path(__file__).parent.parent / "WineQT.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("ANÁLISIS EXPLORATORIO DE DATOS - CLASIFICACIÓN DE VINOS")
print("=" * 80)

# 1. Información general
print("\n1. INFORMACIÓN GENERAL DEL DATASET")
print(f"Dimensiones: {df.shape}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nValores faltantes:\n{df.isnull().sum()}")

# 2. Estadísticas descriptivas
print("\n2. ESTADÍSTICAS DESCRIPTIVAS")
print(df.describe())

# 3. Distribución de la variable objetivo
print("\n3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (Quality)")
print(df['quality'].value_counts().sort_index())

# 4. Visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribución de quality
axes[0, 0].hist(df['quality'], bins=10, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribución de Quality')
axes[0, 0].set_xlabel('Quality')
axes[0, 0].set_ylabel('Frecuencia')

# Correlación con quality
correlations = df.corr()['quality'].sort_values(ascending=False)
axes[0, 1].barh(correlations.index[:10], correlations.values[:10])
axes[0, 1].set_title('Top 10 Características Correlacionadas con Quality')
axes[0, 1].set_xlabel('Correlación')

# Box plot de alcohol vs quality
df.boxplot(column='alcohol', by='quality', ax=axes[1, 0])
axes[1, 0].set_title('Alcohol por Nivel de Calidad')
axes[1, 0].set_xlabel('Quality')
axes[1, 0].set_ylabel('Alcohol (%)')

# Matriz de correlación
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Matriz de Correlación')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficas guardadas en 'eda_analysis.png'")
plt.show()

# 5. Análisis de características
print("\n4. ANÁLISIS DE CARACTERÍSTICAS")
print(f"Número de características: {len(df.columns) - 1}")
print(f"Características:\n{', '.join(df.columns[:-1])}")