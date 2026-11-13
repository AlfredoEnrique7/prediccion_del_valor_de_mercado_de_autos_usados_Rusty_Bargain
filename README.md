# Predicción del Valor de Mercado de Autos Usados — Rusty Bargain

![Python](https://shields.io)
![Scikit-Learn](https://shields.io)
![LightGBM](https://shields.io)
![Status](https://shields.io)

## Descripción del Proyecto

El servicio de venta de autos usados **Rusty Bargain** está desarrollando una aplicación móvil y web para permitir a sus clientes conocer de forma rápida y confiable el valor de mercado de su vehículo. 

Este proyecto desarrolla, valida y selecciona un motor de tasación basado en *Machine Learning* entrenado sobre más de **354,000 registros históricos**, optimizando el equilibrio entre tres pilares clave definidos por el negocio:
1. **Calidad de la predicción:** Minimizar el error de estimación utilizando la **Raíz del Error Cuadrático Medio (RECM / RMSE)**.
2. **Velocidad de predicción:** Garantizar tiempos de inferencia mínimos para ofrecer respuestas en tiempo real dentro de la aplicación.
3. **Tiempo de entrenamiento:** Mantener tiempos de ajuste razonables para facilitar el reentrenamiento periódico en producción.

---

## Estructura del Repositorio

```text
├── notebooks/
│   └── prediccion_del_valor_de_mercado_de_autos_usados_Rusty_Bargain.ipynb   # Cuaderno principal con análisis, modelado y evaluación
├── data/
│   └── car_data.csv                               # Dataset histórico (no incluido si supera límite de tamaño)
├── requirements.txt                               # Dependencias del entorno
└── README.md                                      # Documentación del proyecto
```

---

## Metodología y Flujo de Trabajo

### 1. Limpieza y Preprocesamiento sin Fuga de Datos (*Data Leakage*)
* **Saneamiento inicial:** Eliminación de **262 duplicados exactos** originales antes de descartar seis variables administrativas sin valor técnico (`DateCrawled`, `DateCreated`, `LastSeen`, `PostalCode`, `RegistrationMonth`, `NumberOfPictures`), preservando anuncios legítimos de vehículos populares.
* **Filtrado de anomalías:** Conservación de vehículos con precios reales (`Price >= 100 €`) y años de matriculación coherentes con la fecha de extracción (`1950 <= RegistrationYear <= 2016`), obteniendo **326,965 registros limpios**.
* **Partición `60% / 20% / 20%`:** División estratificada temporal/aleatoria (`random_state=42`) en **Entrenamiento (196,179)**, **Validación (65,393)** para selección de hiperparámetros y **Prueba (65,393)** reservado exclusivamente para la evaluación final imparcial.
* **Imputación y codificación aislada:** Imputación de nulos categóricos como `'unknown'` y aprendizaje de medianas de potencia (`Power`) por `Brand` y `Model` y transformadores (`OneHotEncoder`, `StandardScaler`, `OrdinalEncoder`) **exclusivamente sobre el conjunto de entrenamiento**.

### 2. Modelos Evaluados y Ajuste de Hiperparámetros
Se implementó una función de evaluación unificada (`evaluar_modelo`) auditando tiempos con `%%time` y `time.perf_counter()`:
* **Regresión Lineal (Prueba de cordura / *Baseline*):** Con `OneHotEncoder` + `StandardScaler`.
* **Árbol de Decisión (`DecisionTreeRegressor`):** Exploración de profundidades (`max_depth` de `5` a `25`) para diagnosticar *underfitting* y *overfitting*.
* **Bosque Aleatorio (`RandomForestRegressor`):** Evaluación de tres configuraciones (`n=30, depth=12`, `n=50, depth=15`, `n=100, depth=20`).
* **LightGBM (`LGBMRegressor`):** Aprovechando soporte nativo de variables `category` mediante un experimento controlado variando un hiperparámetro a la vez (`leaves=31, n=200` → `leaves=63, n=200` → `leaves=63, n=500`).

---

## Resultados Comparativos (Conjunto de Validación — 20%)

| Modelo | RECM Validación (€) | Tiempo Entrenamiento (s) | Tiempo Predicción Lote (s) |
| :--- | :---: | :---: | :---: |
| **LightGBM (`leaves=63, n=500`)** | **1,541.82 €** | 7.097 s | 0.110 s |
| LightGBM (`leaves=63, n=200`) | 1,562.01 € | 2.904 s | 0.062 s |
| LightGBM (`leaves=31, n=200`) | 1,597.15 € | 1.545 s | 0.043 s |
| Bosque Aleatorio (`n=100, depth=20`) | 1,617.76 € | 2.037 s | 0.125 s |
| Bosque Aleatorio (`n=50, depth=15`) | 1,657.95 € | 0.883 s | 0.033 s |
| Bosque Aleatorio (`n=30, depth=12`) | 1,768.99 € | 0.544 s | 0.027 s |
| Árbol de Decisión (`depth=15`) | 1,839.97 € | 0.225 s | 0.006 s |
| Regresión Lineal (*Baseline*) | 2,618.10 € | 0.780 s | 0.001 s |

---

## Evaluación Final en Prueba (Test — 20%) y Conclusiones

* **Modelo ganador seleccionado:** **`LightGBM (num_leaves=63, learning_rate=0.1, n_estimators=500)`**
* **RECM Final en Prueba (65,393 vehículos):** **`1,516.38 €`** (reducción de más de **1,100 €** frente a la línea base lineal).
* **Rendimiento por lotes (*Throughput*):** **`0.116 s`** para predecir los 65,393 registros de prueba.
* **Latencia de consulta individual (1 vehículo):** Mediana de **`0.81 ms`** (`P95: 1.03 ms`) tras calentamiento (*warm-up*), garantizando una experiencia instantánea en la aplicación móvil.
* **Alternativa económica:** En escenarios de alta concurrencia con recursos de cómputo muy restringidos, **`LightGBM (leaves=63, n=200)`** ofrece un excelente compromiso con un RECM de `1,562.01 €` en menos de la mitad de tiempo de entrenamiento e inferencia.

---

## Cómo Ejecutar el Proyecto

1. Clona este repositorio:
   ```bash
   git clone https://github.com
   cd rusty-bargain-car-price-prediction
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install pandas numpy matplotlib scikit-learn lightgbm jupyter
   ```
3. Asegúrate de colocar el archivo `car_data.csv` en el directorio raíz (o en `/datasets/car_data.csv`) y abre el cuaderno:
   ```bash
   jupyter notebook
   ```
---

### Ver el análisis completo
👉 [Haz clic aquí para ver el código y los gráficos interactivos](prediccion_del_valor_de_mercado_de_autos_usados_Rusty_Bargain.ipynb)
