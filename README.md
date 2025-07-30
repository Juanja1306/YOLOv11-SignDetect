# YOLOv11-SignDetect

## Descripción
Proyecto de detección de lenguaje de señas americano (ASL) usando un modelo YOLOv11 personalizado. Incluye entrenamientos con variantes Nano y Small, scripts de inferencia y resultados organizados.

## Estructura del proyecto
```
YOLOv11-SignDetect/
 ├─ .git/                 
 ├─ .gitattributes
 ├─ .gitignore
 ├─ .venv/                 # Entorno virtual (Python 3.11)
 ├─ data/                  # Dataset y configuración
 │   ├─ American Sign Language Letters.v1-v1.yolov11.zip  # Dataset original
 │   ├─ train/              # Imágenes y etiquetas de entrenamiento
 │   │   ├─ images/
 │   │   └─ labels/
 │   ├─ valid/              # Imágenes y etiquetas de validación
 │   │   ├─ images/
 │   │   └─ labels/
 │   ├─ test/               # Imágenes y etiquetas de prueba
 │   │   ├─ images/
 │   │   └─ labels/
 │   ├─ README.dataset.txt  # Información del dataset
 │   ├─ README.roboflow.txt # Detalles de exportación Roboflow
 │   └─ data.yaml           # Configuración para entrenamiento
 ├─ models/                # Pesos pre-entrenados originales
 │   ├─ yolo11n.pt         # Modelo Nano base
 │   └─ yolo11s.pt         # Modelo Small base
 ├─ runs/                  # Resultados de entrenamiento e inferencia
 │   ├─ train/             # Resultados de cada experimento de entrenamiento
 │   │   ├─ sign_detect_nano/
 │   │   │   ├─ args.yaml
 │   │   │   ├─ *.png       # Curvas, matrices de confusión, etiquetas vs predicciones
 │   │   │   ├─ results.csv
 │   │   │   └─ weights/{best.pt, last.pt}
 │   │   └─ sign_detect_small/
 │   │       ├─ args.yaml
 │   │       ├─ *.png
 │   │       ├─ results.csv
 │   │       └─ weights/{best.pt, last.pt}
 │   └─ detect/            # Salida de inferencia sobre imágenes de prueba
 │       ├─ sign_preds/    # Con modelo Nano
 │       └─ small/         # Con modelo Small
 ├─ SignDetect.ipynb       # Notebook de entrenamiento y evaluación interactiva
 ├─ predict_signs.py       # Script de inferencia (modelo Small por defecto)
 ├─ predict_signs_small.py # Script de inferencia optimizado (modelo Nano)
 ├─ requirements.txt       # Lista de dependencias Python
 ├─ README.md              # Documentación del proyecto
 ├─ foto1.png              # Ejemplo de imagen de entrada
 ├─ foto2.png
 ├─ MetricasGrafica.png    # Gráficas resumen de métricas
 └─ OtrasMetricas.png      # Gráficas adicionales
```

## Requisitos
- Python 3.11
- (Opcional) CUDA/cuDNN para aceleración GPU
- Dependencias especificadas en `requirements.txt`

## Instalación
```bash
# Clonar repositorio
git clone https://github.com/Juanja1306/YOLOv11-SignDetect
cd YOLOv11-SignDetect

# Crear y activar entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Preparación de datos

> Descargar de `https://public.roboflow.com/object-detection/american-sign-language-letters`

1. Descomprimir el dataset:
   ```bash
   unzip data/"American Sign Language Letters.v1-v1.yolov11.zip" -d data
   ```
2. Verificar rutas en `data/data.yaml` para `train/`, `valid/` y `test/`.

## Entrenamiento

Puede ejecutar las celdas de entrenamiento en `SignDetect.ipynb`.

## Inferencia
```bash
# Usando predict_signs.py (Small)
python predict_signs.py --source foto1.png \
                         --weights runs/train/sign_detect_small/weights/best.pt \
                         --output runs/detect/small

# Usando predict_signs_small.py (Nano)
python predict_signs_small.py --source foto2.png \
                               --weights runs/train/sign_detect_nano/weights/best.pt \
                               --output runs/detect/sign_preds
```
Los resultados se guardan en la carpeta indicada por `--output`.

## Métricas y resultados
- Curvas de P/R/F1, matrices de confusión y CSV en `runs/train/...`
