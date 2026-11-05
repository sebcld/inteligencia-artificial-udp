# Tarea 3 — Clasificación con redes neuronales MLP

En esta tarea diseñamos y comparamos redes neuronales *Multi-Layer Perceptron* para clasificar posturas de la mano. Trabajamos con dos versiones del mismo dataset, una **balanceada** y otra **desbalanceada**. El objetivo es estudiar cómo el desbalance de clases afecta el desempeño y cómo corregirlo.

**Herramientas:** `scikit-learn` (`MLPClassifier`) · `pandas` · `numpy` · `matplotlib`


---

## Datos

El dataset contiene posturas de la mano capturadas con marcadores ópticos: 50.000 registros, 5 clases y 21 variables, que son las coordenadas X, Y, Z de hasta 7 marcadores.

- **Versión balanceada:** 10.000 registros por clase.
- **Versión desbalanceada:** la clase menos frecuente representa solo el 2 % de los registros.

## Procedimiento

1. **Preprocesamiento:**
   - Eliminamos la columna `User`.
   - Imputamos con 0 los marcadores que no se detectaron.
   - Dividimos los datos en 70/15/15, estratificando por clase.
   - Ajustamos el `StandardScaler` solo con los datos de entrenamiento.
2. **Cuatro arquitecturas** entrenadas en ambos datasets:

   | Arquitectura | Capas ocultas | Activación | Learning rate | Batch |
   |---|---|---|---|---|
   | A1 simple | (64) | ReLU | 0,001 | 128 |
   | A2 profunda regularizada | (128, 64) | ReLU | 0,0005 | 256 |
   | A3 tanh profunda | (128, 64, 32) | tanh | 0,001 | 128 |
   | A4 ancha con LR alto | (256, 128) | ReLU | 0,01 | 512 |

3. **Selección y evaluación:** elegimos la mejor arquitectura según la accuracy en validación y la evaluamos en test con accuracy, precision, recall y F1 macro, además de matrices de confusión. Registramos curvas de pérdida y accuracy por época.
4. **Mejora del modelo más débil:** aplicamos oversampling aleatorio de las clases minoritarias.

## Resultados en test

| Modelo | Accuracy | F1 macro |
|---|---|---|
| A4 — dataset balanceado | 0,958 | 0,958 |
| A4 — dataset desbalanceado | 0,958 | 0,918 |
| A4 — desbalanceado + oversampling | 0,966 | 0,935 |

Con datos desbalanceados, la accuracy se mantiene igual mientras el F1 macro cae. Esto muestra que la accuracy no basta cuando las clases están desbalanceadas. Con oversampling, el recall de las clases minoritarias sube de 0,82 a 0,91 y a 0,87.

## Limitaciones y posibles mejoras

- **Sobreajuste:** desde la epoch 5 aproximadamente, la pérdida de validación deja de bajar mientras la de entrenamiento sigue descendiendo. Usar *early stopping* o más regularización ayudaría.
- **Comparación entre arquitecturas:** A1, A2 y A3 seguían mejorando al terminar las épocas fijadas. Una comparación más justa entrenaría cada arquitectura hasta que converja.
- **Efecto del oversampling:** el oversampling también aumenta el número de actualizaciones por época. Para aislar el efecto del balanceo habría que comparar con el mismo número de pasos de entrenamiento y con varias semillas.
- **Valores faltantes:** agregar columnas que indiquen si un marcador falta sería mejor que imputar con 0, porque 0 también es una coordenada válida.

## Ejecución

Los datasets se descargan automáticamente desde el repositorio [`sebcld/dataset_tarea3IA`](https://github.com/sebcld/dataset_tarea3IA).
