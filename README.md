# 💳 PropensityNet: Scoring de Propensión para Créditos Personales

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2B-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-F7931E.svg)](https://scikit-learn.org/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

Sistema integral de Machine Learning y Deep Learning enfocado en predecir la propensión de clientes bancarios a aceptar una oferta de **crédito personal**. Diseñado bajo un enfoque end-to-end: desde el análisis exploratorio y la ingeniería de características hasta el entrenamiento con redes neuronales profundas, optimización de umbrales en datos desbalanceados y el empaquetado del artefacto para servicio de inferencia en producción.

---

## 📌 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Arquitectura del Pipeline](#-arquitectura-del-pipeline)
- [Ingeniería de Características](#-ingeniería-de-características)
- [Modelado y Deep Learning](#-modelado-y-deep-learning)
- [Optimización de Umbral de Decisión](#-optimización-de-umbral-de-decisión)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Guía de Inferencia en Producción](#-guía-de-inferencia-en-producción)

---

## 🎯 Descripción General

El objetivo de este proyecto es optimizar las campañas de colocación de crédito personal en instituciones financieras, identificando proactivamente a aquellos clientes con mayor probabilidad de conversión mientras se minimizan los costos operativos y el impacto por falsos positivos (campañas invasivas hacia clientes no interesados).

### Puntos Destacados:
- **Tratamiento robusto del desbalance**: Ponderación dinámica de clases (`pos_weight`) en la función de coste.
- **Pipeline de preprocesamiento desacoplado**: Escalado robusto ante valores atípicos y codificación suave de alta cardinalidad (`TargetEncoder` con validación cruzada).
- **Bundle todo-en-uno**: Empaquetado en un único archivo (`propensity_net_produccion.pt`) que incluye pesos del modelo, arquitectura, transformador de características y umbral óptimo.
- **Servicio de inferencia desacoplado**: Clase orientada a objetos `PropensityPredictor` lista para ser consumida por APIs REST (FastAPI, Flask) o microservicios en batch/tiempo real.

---

## 🏗 Arquitectura del Pipeline

```
  [Datos Crudos del Cliente]
             │
             ▼
 ┌──────────────────────────────────────────────┐
 │    Preprocesamiento (ColumnTransformer)      │
 │  ├─ RobustScaler (Variables numéricas/sesgadas)│
 │  ├─ TargetEncoder (Código Postal con CV)     │
 │  └─ OrdinalEncoder (Rangos de Edad/Década)   │
 └──────────────────────┬───────────────────────┘
                        │ Tensores
                        ▼
 ┌──────────────────────────────────────────────┐
 │        Red Neuronal (PropensityNet)          │
 │  ├─ Linear(d_in, 128) + BatchNorm + Dropout  │
 │  ├─ Linear(128, 64)   + BatchNorm + Dropout  │
 │  └─ Linear(64, 1)     --> Logits             │
 └──────────────────────┬───────────────────────┘
                        │ Sigmoid
                        ▼
 ┌──────────────────────────────────────────────┐
 │         Decisión con Umbral Óptimo           │
 │      Probabilidad >= Umbral F1-Score         │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
          {'ofrecer_prestamo': True / False}
```

---

## 🧪 Ingeniería de Características

El pipeline implementa un preprocesador centralizado usando `sklearn.compose.ColumnTransformer`:

1. **Variables Numéricas Continuas**: (`Edad`, `Experiencia`, `Ingreso_Anual`, `Valor_hipoteca`, `Gasto_mensual_promedio`).
   - Se escalan utilizando **`RobustScaler`**, que sustrae la mediana y escala según el Rango Intercuartílico (IQR), resistiendo la influencia de clientes con ingresos o hipotecas atípicas.
2. **Variables de Alta Cardinalidad**: (`Codigo postal`).
   - Transformadas mediante **`TargetEncoder(smooth='auto', cv=5)`**, capturando la tasa esperada de aceptación por zona geográfica sin sufrir fuga de datos ni dimensionalidad excesiva.
3. **Agrupación Categórica Ordinal**: (`Decada`).
   - Codificación ordenada por rangos de edad (`20s`, `30s`, `40s`, `50s`, `60s`) mediante **`OrdinalEncoder`**.

---

## 🧠 Modelado y Deep Learning

### Arquitectura de `PropensityNet`
Implementada en PyTorch (`torch.nn`):

```python
class ModeloPredictivo(nn.Module):
    def __init__(self, caracteristicas: int):
        super().__init__()
        self.secuencia = nn.Sequential(
            nn.Linear(caracteristicas, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )
```

### Estrategia de Entrenamiento
- **Función de Pérdida**: `nn.BCEWithLogitsLoss(pos_weight=...)` para compensar la baja proporción de clientes que aceptan el crédito frente a los que lo rechazan.
- **Optimizador**: `AdamW` con decaimiento de peso para regularización L2.
- **Scheduler**: `ReduceLROnPlateau` para reducir dinámicamente la tasa de aprendizaje cuando la pérdida de validación se estanca.

---

## 🎯 Optimización de Umbral de Decisión

En problemas de propensión bancaria con fuerte desbalance, el umbral por defecto de **0.5** resulta ineficiente:
- A través de la curva **Precision-Recall (PR-Curve)**, se evalúa exhaustivamente el F1-Score en todo el espectro continuo de umbrales.
- Se selecciona el **umbral óptimo** que maximiza el equilibrio entre precisión (evitar contactar clientes que dirán que no) y cobertura/recall (capturar la mayor cantidad de clientes interesados).
- El umbral óptimo queda fijado y persistido dentro del bundle de producción.

---

## 📁 Estructura del Proyecto

```plaintext
Credito_bancario/
├── analisis_datos.ipynb           # EDA exhaustivo, benchmarking de modelos y entrenamiento
├── propensity-net.ipynb           # Construcción de la clase de inferencia y pruebas
├── propensity_net_produccion.pt   # Bundle de producción (pesos, preprocesador, umbral)
├── datos_entreno.csv              # Matriz de entrenamiento preprocesada
├── datos_etiquetas.csv            # Target binario de aceptación
├── CreditoPersonal_procesados.csv # Dataset con ingeniería de atributos aplicada
├── Model_creditoPersonal.xls      # Dataset tabular base
├── main.py                        # Script de entrada
├── pyproject.toml                 # Gestión de dependencias (uv / pip)
├── uv.lock                        # Lockfile de dependencias exactas
├── .gitignore                     # Configuración de exclusiones de Git
└── README.md                      # Documentación técnica
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone git@github.com:AbiWorkspace/credito-bancario..git
cd credito-bancario.
```

### 2. Crear y activar el entorno virtual
Puedes utilizar `uv` (recomendado por velocidad y reproducibilidad) o `venv` tradicional:

**Con uv:**
```bash
uv sync
```

**Con pip tradicional:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch pandas scikit-learn seaborn xgboost ipykernel
```

---

## 💻 Guía de Inferencia en Producción

Puedes consumir el modelo directamente en cualquier script o servicio web utilizando el bundle generado:

```python
from propensity_predictor import PropensityPredictor

# 1. Cargar el servicio de inferencia (carga pesos, arquitectura y transformador)
predictor = PropensityPredictor("propensity_net_produccion.pt")

# 2. Datos de un nuevo prospecto
nuevo_cliente = {
    "Edad": 38,
    "Experiencia": 14,
    "Ingreso_Anual": 145,
    "Codigo postal": "94720",
    "Familia": 3,
    "Gasto_mensual_promedio": 4.2,
    "Nivel_educativo": 3,
    "Valor_hipoteca": 0,
    "Banca_internet": 1
}

# 3. Predicción en tiempo real
resultado = predictor.predict_one(nuevo_cliente)
print(resultado)
# Salida:
# {
#     'probabilidad': 0.9812,
#     'ofrecer_prestamo': True,
#     'umbral_aplicado': 0.9149
# }
```

---

## 👥 Autor

- **Abimael Guadarrama** - [GitHub @AbiGuadarrama](https://github.com/AbiGuadarrama)
