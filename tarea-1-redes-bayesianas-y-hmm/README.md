# Tarea 1 — Redes Bayesianas y Modelos Ocultos de Markov

Tarea de modelos probabilísticos en dos partes. En la primera aprendemos una red bayesiana a partir de datos reales. En la segunda modelamos actividad humana con un HMM.

**Herramientas:** `pgmpy` · `hmmlearn` · `pandas` · `numpy` · `networkx` · `matplotlib` · `kagglehub`

---

## Parte 1 — Red Bayesiana

**Datos:** [*Student Depression Dataset*](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Kaggle). Tiene 27.898 registros después de limpiar los datos y usamos 8 variables categóricas: presión académica, hábitos alimenticios, duración del sueño, estrés financiero, historial familiar, satisfacción con los estudios, pensamientos suicidas y depresión.

**Procedimiento:**
1. **Aprendizaje de estructura** con dos métodos:
   - `HillClimbSearch`, con puntaje BIC y como máximo 3 padres por nodo.
   - `ExhaustiveSearch`, con 5 variables. Con 8 variables habría que evaluar unos 7,8 × 10¹¹ grafos posibles, lo que no es viable.
2. **Estimación de las tablas de probabilidad condicional (CPDs)** por máxima verosimilitud.
3. **Inferencia diagnóstica** con `VariableElimination`. Por ejemplo, P(pensamientos suicidas | depresión = 1) ≈ 0,85.
4. **Datos sintéticos:** generamos registros con *forward sampling* desde la red aprendida para aumentar el dataset en +10 %, +20 % y +40 %. También probamos reducirlo en −50 % y −71 %.
5. **Comparación** de estructuras, CPDs e inferencias entre todas las versiones del dataset.

**Resultados principales:**
- En la red de Hill-Climbing, `Depression` queda como nodo central.
- La búsqueda exhaustiva encuentra la cadena `Academic Pressure → Depression → {Financial Stress, Sleep, Suicidal Thoughts}`.
- Al reducir los datos se pierde la arista `Study Satisfaction → Academic Pressure`. Las inferencias, en cambio, casi no cambian.

## Parte 2 — Modelo Oculto de Markov (HMM)

**Datos:** [*Human Activity Recognition Using Smartphones*](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (UCI). Son 10.299 ventanas de 30 sujetos realizando 6 actividades.

**Procedimiento:**
1. **Estados ocultos:** las 6 actividades.
2. **Observaciones:** calculamos un índice del acelerómetro y otro del giroscopio, y discretizamos cada uno en terciles (bajo, medio y alto). Al combinarlos salen 9 símbolos.
3. **Parámetros π, A y B** estimados a mano, contando sobre una secuencia por sujeto, con suavizado de Laplace. Luego los cargamos en `CategoricalHMM` sin reentrenar.
4. **Forward-Backward**, para obtener la probabilidad de cada estado en cada instante, y **Viterbi**, para el camino de estados más probable. Ambos sobre una subsecuencia real de 15 pasos.

**Resultados principales:** la matriz de transición está dominada por la diagonal, porque permanecer en la misma actividad tiene probabilidad cercana a 0,96. Las caminatas emiten sobre todo el símbolo `alto_alto`. Las actividades estáticas (sentado y de pie) quedan casi indistinguibles con estos dos índices.

## Limitaciones y posibles mejoras

- Las dos inferencias de la Parte 1 se reducen a frecuencias observadas en los datos, así que no dependen de la estructura aprendida. Consultas que atraviesen varias aristas compararían mejor las redes.
- En la subsecuencia analizada, Viterbi no detecta el cambio de STANDING a SITTING y acierta 5 de 15 pasos. Con más niveles de discretización, o con un `GaussianHMM` sobre los índices continuos, se podría separar mejor las actividades estáticas.

## Ejecución

- **Parte 1:** el dataset se descarga automáticamente con `kagglehub`.
- **Parte 2:** descarga el zip desde [UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) y déjalo **en esta carpeta** con el nombre `human+activity+recognition+using+smartphones.zip`. El notebook lo descomprime solo.
