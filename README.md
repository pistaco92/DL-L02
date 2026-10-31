# DL-2026-02-L02

Proyecto base del **Laboratorio 02 de Deep Learning** (segundo semestre 2026):
redes con una **capa convolucional 1D** para predecir la etapa de deterioro
cognitivo (escala GDS) a partir de 15 respuestas binarias de un test de
orientacion.

Es la continuacion del Laboratorio 01: se mantiene el dataset, los seis
experimentos (`GDS`, `GDS_R1` ... `GDS_R5`), la validacion cruzada anidada y
las metricas multiclase y ordinales. El cambio principal es la arquitectura:
cada fila de 15 atributos se trata como una secuencia de largo 15 y se procesa
con `nn.Conv1d` antes del MLP.

## Experimentos

| Algoritmo | Arquitectura | Perdida | Prediccion | Estado |
|---|---|---|---|---|
| `conv1d_softmax` | Conv1D + MLP + `Linear(K)` | `CrossEntropyLoss` | `argmax` | Implementado |
| `conv1d_coral` | Conv1D + MLP + `CoralLayer(K-1)` | BCE sobre umbrales | conteo de umbrales | **TODO(alumno)** |

Ambos comparten el mismo tronco (`Conv1DBackbone`). Solo cambia la cabeza y la
estrategia de entrenamiento (`ClassificationTask`).

## Que esta implementado

- Carga de datos `csv` / `sav` y recodificacion del objetivo a `0 .. K-1`.
- `Dataset` que entrega cada fila como tensor `(1, 15)` para `nn.Conv1d`.
- Bloques `Conv1DBlock` (Conv1d + BatchNorm1d + ReLU), extractor convolucional,
  MLP y la red completa `Conv1DSoftmaxNet`.
- Fabricas `ModelFactory` y `TaskFactory` para registrar algoritmos.
- `Trainer` con Adam, `GridSearch` interna y `NestedCrossValidator` (5x3).
- **Busqueda de hiperparametros en el loop interno**: recorre
  `HYPERPARAMETER_GRID`, elige por menor MAE (empate: mayor QWK), reentrena en
  todo el entrenamiento externo y evalua una sola vez en el test externo.
- Metricas: accuracy, balanced accuracy, precision/recall/F1 macro, MAE, QWK,
  accuracy ±1 y errores graves.
- Reportes: CSV, Markdown, rankings, graficos de barras, matrices de confusion
  y la configuracion elegida en cada fold (`seleccion_hp.csv`).

## Que queda como TODO (alumno)

- `CoralLayer` y `Conv1DCoralNet` en `src/models/coral.py`.
- `labels_to_levels`, `coral_loss`, `effective_number_weights` y
  `logits_to_ordinal_predictions` en `src/tasks/ordinal_task.py`.
- Comparar `conv1d_softmax` y `conv1d_coral` (con y sin pesos) en los seis
  objetivos y completar el informe.

Detalle en [docs/04_extension_ordinal.md](docs/04_extension_ordinal.md).

## Inicio rapido

```bash
conda activate lab_pytorch
cp "/ruta/al/archivo/15 atributos R0-R5.sav" dataset/

# Un experimento rapido (sin grid) para verificar el entorno
python main.py --data-path "dataset/15 atributos R0-R5.sav" --epochs 5 --no-grid

# Experimento completo en GDS_R2 (grid interno + 5x3 folds)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS_R2

# Los seis objetivos con tablas, rankings y graficos en results/
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets
```

## Estructura

```text
DL-2026-02-L02/
|-- main.py                  # CLI: arma ExperimentConfig y llama a ExperimentRunner
|-- src/
|   |-- config.py            # constantes, dataclasses de configuracion y HYPERPARAMETER_GRID
|   |-- utils.py             # semillas y dispositivo
|   |-- experiment.py        # ExperimentRunner: algoritmos x objetivos
|   |-- data/                # DataFrameLoader, TargetEncoder, StratifiedSplitter, Dataset (1, 15)
|   |-- models/              # Conv1DBlock, Conv1DBackbone, Conv1DSoftmaxNet, CORAL (TODO), ModelFactory
|   |-- tasks/               # ClassificationTask, SoftmaxTask, CoralTask (TODO), TaskFactory
|   |-- training/            # Trainer, GridSearch, NestedCrossValidator
|   |-- evaluation/          # MetricsCalculator
|   `-- reporting/           # ReportWriter, rankings y graficos
|-- docs/                    # documentacion por partes
|-- dataset/                 # datos locales (no versionados)
|-- results/                 # salidas de main.py (no versionadas)
|-- environment.yml
`-- README.md
```

## Documentacion

1. [El problema](docs/01_problema.md): GDS, reagrupaciones, desbalance y por que
   usar Conv1D sobre los 15 items.
2. [Arquitectura](docs/02_arquitectura.md): convolucion 1D, trazado de formas,
   clases y responsabilidades.
3. [Uso](docs/03_uso.md): entorno, argumentos, ejemplos de comandos y salidas.
4. [Extension ordinal](docs/04_extension_ordinal.md): esqueletos CORAL y orden
   sugerido de implementacion.

## Plan sugerido para alumnos

1. Leer [docs/01_problema.md](docs/01_problema.md) y revisar el dataset.
2. Ejecutar `conv1d_softmax` en `GDS_R2` con `--no-grid` y luego con grid.
3. Correr `--all-targets` y revisar `results/` (tablas, rankings, `seleccion_hp.csv`).
4. Implementar los esqueletos CORAL (ver [docs/04_extension_ordinal.md](docs/04_extension_ordinal.md)).
5. Ejecutar `--algorithm all` con y sin `--use-weights`.
6. Completar el informe comparando Softmax y CORAL en los seis objetivos.
