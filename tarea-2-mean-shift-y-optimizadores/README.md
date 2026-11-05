# Tarea 2 — Segmentación con Mean Shift y comparación de optimizadores

Esta tarea tiene dos partes. La primera es de visión por computador: segmentamos imágenes de complejidad creciente. La segunda es de aprendizaje supervisado: comparamos cinco algoritmos de optimización entrenando regresiones lineal y logística.

**Herramientas:** `OpenCV` · `TensorFlow / Keras` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `ucimlrepo`

---

## Parte 1 — Segmentación de imágenes con Mean Shift

Aplicamos `cv2.pyrMeanShiftFiltering` a tres imágenes de complejidad creciente: un paisaje simple, un retrato con fondo y un mercado lleno de personas y objetos.

**Procedimiento:**
- Probamos **18 combinaciones de parámetros por imagen**:
  - `sp` (radio espacial): 10, 20 y 40.
  - `sr` (radio de color): 10, 40 y 60.
  - `maxLevel` (niveles de pirámide): 0 y 1.
- Medimos el grado de simplificación contando los **colores únicos** de cada resultado y registramos el tiempo de ejecución.

**Resultados principales:**
- `sp` es el parámetro con más impacto. Pasar de 10 a 40 reduce los colores en 68 %, 57 % y 45 % en cada imagen.
- `sr` tiene un efecto secundario y no siempre en la misma dirección.
- `maxLevel = 1` cambia poco el resultado visual, pero filtra entre 2 y 3,6 veces más rápido.

Mejores configuraciones encontradas:

| Imagen | sp | sr | maxLevel |
|---|---|---|---|
| Simple | 40 | 40 | 0 |
| Media | 20 | 40 | 0 |
| Compleja | 20 | 40 | 0 |

La imagen simple tolera parámetros agresivos. La compleja pierde objetos pequeños con facilidad.

## Parte 2 — Regresión lineal y logística con cinco optimizadores

**Datos:** [*Default of Credit Card Clients*](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI), con 30.000 clientes.

**Procedimiento:**
- **Variables:**
  - Y1 continua: `LIMIT_BAL`, el cupo de crédito, que predecimos con regresión lineal.
  - Y2 binaria: `default`, el incumplimiento de pago, que predecimos con regresión logística.
  - Las mismas 10 variables predictoras para ambos modelos.
- **División y escalado:** 70/15/15 estratificado, con el `StandardScaler` ajustado solo en entrenamiento.
- **Modelos:** cada uno se implementa como una red de una sola neurona (`Dense(1)`) y se entrena con GD, SGD, SGD con momentum, RMSProp y Adam, usando los mismos hiperparámetros.
- **Selección:** elegimos los 2 mejores optimizadores de cada tarea según validación y los evaluamos en test.

**Resultados en test:**

| Tarea | Mejor optimizador | Métricas |
|---|---|---|
| Regresión lineal | Adam | R² = 0,32 · RMSE ≈ 107.000 NT$ |
| Regresión logística | RMSProp | F1 = 0,39 · ROC-AUC = 0,71 |

## Limitaciones y posibles mejoras

- Con 100 epochs, GD solo hace 100 actualizaciones, mientras que los métodos por minibatch hacen unas 65.000. Parte de la diferencia se debe a esa diferencia en el número de actualizaciones y no al algoritmo en sí. Una comparación más justa igualaría el número de actualizaciones o entrenaría cada optimizador hasta que converja.
- SGD con momentum diverge en la regresión con learning rate 0,01. Valdría la pena probar tasas de aprendizaje más bajas.
- Medimos la segmentación con el número de colores únicos. Una métrica que compare con la imagen original, como SSIM, permitiría cuantificar mejor la estabilidad de cada imagen.
