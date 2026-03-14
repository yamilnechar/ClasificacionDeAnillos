Clasificación de Anillos en Galaxias – Pipeline basado en Zoobot

Este repositorio contiene el pipeline completo desarrollado para el proyecto de clasificación de anillos en galaxias, donde se entrena un modelo de deep learning basado en la arquitectura Zoobot para detectar estructuras de anillo utilizando imágenes astronómicas en formato FITS.

El proyecto integra las etapas de preparación de datos, transformación de imágenes astronómicas, entrenamiento del modelo, evaluación de desempeño y análisis de interpretabilidad.

Durante el desarrollo se realizaron múltiples experimentos y pruebas. Posteriormente, el pipeline fue unificado en un notebook principal, con el objetivo de garantizar consistencia en el preprocesamiento y reproducibilidad en los resultados.

Estructura del Proyecto

Project Root
│
├── Images (no incluida en el repositorio)
│   Archivos FITS de galaxias que contienen las bandas fotométricas g, r y z.
│   Estos archivos representan los datos astronómicos originales que se
│   transforman posteriormente a representaciones RGB para el entrenamiento.
│
├── Data
│   Archivos CSV y metadatos del dataset procesado.
│   Incluye tablas con identificadores de galaxias, coordenadas,
│   etiquetas de anillos, origen del dataset y rutas a los archivos FITS.
│
├── outputs
│   Resultados generados durante el entrenamiento del modelo:
│   - modelos entrenados (.pt)
│   - métricas de entrenamiento
│   - gráficas de desempeño
│   - matrices de confusión y evaluaciones del modelo
│
├── Notebooks
│   Notebooks principales que implementan el pipeline del proyecto:
│   - carga y validación del dataset
│   - transformación de FITS a representaciones RGB
│   - aumento de datos (data augmentation)
│   - entrenamiento del modelo Zoobot
│   - evaluación y análisis de resultados
│
└── Auxiliares
    Notebooks experimentales y pruebas intermedias desarrolladas durante
    las primeras fases del proyecto. Contienen prototipos y exploraciones
    alternativas que no forman parte del pipeline final consolidado.


Descripción del Pipeline
El pipeline final de entrenamiento sigue las siguientes etapas:

1. Carga del Dataset
  Se carga el dataset final desde archivos CSV previamente curados y se validan las rutas a los archivos FITS asociados a cada galaxia.
2. Transformación de Imágenes
  Las imágenes FITS que contienen las bandas g, r, z se transforman a representaciones RGB mediante distintas transformaciones astronómicas, como:
    normalización logarítmica
    apilamiento de bandas
    realce de estructuras
  Estas transformaciones permiten resaltar características morfológicas relevantes para el modelo.
3. Aumento de Datos (Data Augmentation)
  Se aplican transformaciones aleatorias para mejorar la capacidad de generalización del modelo:
    rotaciones
    reflejos horizontales y verticales
    transformaciones afines
    desenfoque gaussiano
    variaciones de brillo y contraste
4. Entrenamiento del Modelo
  Se entrena un modelo basado en Zoobot utilizando un esquema de entrenamiento en múltiples fases.
  Para mitigar el desbalance entre clases se utiliza muestreo ponderado (WeightedRandomSampler), lo que permite que el modelo observe con mayor frecuencia ejemplos de clases minoritarias.
5. Evaluación del Modelo
  El desempeño del modelo se evalúa mediante:
  Accuracy
  Macro F1 Score
  matrices de confusión
  análisis de probabilidades de predicción
6. Interpretabilidad
  Se incluyen visualizaciones que muestran ejemplos de galaxias junto con las probabilidades de clasificación del modelo para cada clase.
  Dataset
  El dataset contiene galaxias etiquetadas según la presencia de estructuras de anillo:
    inner – galaxias con anillo interno
    outer – galaxias con anillo externo
    inner+outer – galaxias que contienen ambos tipos de anillo
    none – galaxias sin estructuras de anillo (cuando está presente en el dataset)
Cada entrada del dataset incluye la ruta al archivo FITS correspondiente que contiene la información fotométrica de la galaxia.
Resultados
  Durante el entrenamiento se generan diferentes artefactos que se almacenan en la carpeta outputs/, incluyendo:
    checkpoints del mejor modelo por fase de entrenamiento
    modelo final entrenado
    gráficas del historial de entrenamiento
    visualizaciones de predicciones y probabilidades por clase
