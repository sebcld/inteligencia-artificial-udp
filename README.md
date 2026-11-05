# Inteligencia Artificial — Tareas (UDP, 2026)

Tres proyectos del curso **Inteligencia Artificial** de Ingeniería Civil en Informática y Telecomunicaciones. Cada uno aplica una familia distinta de técnicas a datos reales, en Python y Jupyter Notebook:

- modelos probabilísticos,
- visión por computador,
- aprendizaje supervisado,
- redes neuronales.

**Autores:** Sebastián Díaz · Julián Zavala

---

## Proyectos

| # | Proyecto | Técnicas | Herramientas |
|---|---|---|---|
| 1 | [Redes Bayesianas y Modelos Ocultos de Markov](tarea-1-redes-bayesianas-y-hmm/) | Aprendizaje de estructura (Hill-Climbing, búsqueda exhaustiva), estimación de parámetros, inferencia, datos sintéticos, HMM (Forward-Backward, Viterbi) | `pgmpy`, `hmmlearn`, `pandas`, `networkx` |
| 2 | [Segmentación con Mean Shift y comparación de optimizadores](tarea-2-mean-shift-y-optimizadores/) | Mean Shift sobre imágenes, regresión lineal y logística, GD, SGD, Momentum, RMSProp, Adam | `OpenCV`, `TensorFlow / Keras`, `scikit-learn` |
| 3 | [Clasificación con redes neuronales MLP](tarea-3-redes-neuronales-mlp/) | Diseño de arquitecturas MLP, curvas de aprendizaje, métricas para clases desbalanceadas, oversampling | `scikit-learn`, `pandas`, `matplotlib` |

Cada carpeta tiene su propio README con el objetivo, el procedimiento, los resultados principales y las limitaciones.

## Estructura del repositorio

```
inteligencia-artificial-udp/
├── README.md
├── tarea-1-redes-bayesianas-y-hmm/
│   ├── README.md
│   └── Sebastian_Diaz_Julian_Zavala_Tarea1.ipynb
├── tarea-2-mean-shift-y-optimizadores/
│   ├── README.md
│   └── Sebastian_Diaz_Julian_Zavala_Tarea2.ipynb
└── tarea-3-redes-neuronales-mlp/
    ├── README.md
    └── Sebastian_Diaz_Julian_Zavala_Tarea3.ipynb
```

## Cómo ejecutar

Los notebooks se guardaron con sus salidas, así que los resultados y gráficos se pueden revisar directamente, sin ejecutar nada.

La ejecución local requiere Python 3.10 o superior y Jupyter:

```bash
pip install notebook numpy pandas matplotlib scikit-learn
pip install pgmpy hmmlearn kagglehub networkx   # Tarea 1
pip install opencv-python tensorflow ucimlrepo  # Tarea 2
jupyter notebook
```

