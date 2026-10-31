"""
Configuracion central del laboratorio.

Contiene:
- constantes del dataset (columnas de entrada y objetivos),
- nombres de los algoritmos disponibles,
- valores por defecto de la linea de comandos,
- dataclasses que agrupan la configuracion de modelo, entrenamiento,
  validacion y experimento,
- el grid de hiperparametros que recorre la busqueda interna.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# El orden importa: la Conv1D recorre estas columnas como una secuencia.
# Grupos: orientacion temporal (4), orientacion espacial (4), toponimos (7).
FEATURE_COLUMNS = [
    "Día",
    "Mes",
    "Año",
    "Estación",
    "País",
    "Ciudad",
    "CalleLugar",
    "NumeroPiso",
    "Miguel2",
    "González2",
    "Avenida2",
    "Imperial2",
    "A682",
    "Caldera2",
    "Copiapo2",
]

TARGET_COLUMNS = [
    "GDS",
    "GDS_R1",
    "GDS_R2",
    "GDS_R3",
    "GDS_R4",
    "GDS_R5",
]

ID_COLUMN = "ID"

# ---------------------------------------------------------------------------
# Algoritmos
# ---------------------------------------------------------------------------

ALGORITHM_SOFTMAX = "conv1d_softmax"
ALGORITHM_CORAL = "conv1d_coral"
ALGORITHMS = (ALGORITHM_SOFTMAX, ALGORITHM_CORAL)

# ---------------------------------------------------------------------------
# Valores por defecto
# ---------------------------------------------------------------------------

DEFAULT_TARGET = "GDS_R2"
DEFAULT_ALGORITHM = ALGORITHM_SOFTMAX
DEFAULT_CONV_CHANNELS = (16, 32)
DEFAULT_KERNEL_SIZE = 3
DEFAULT_HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.15
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_BETA = 0.99
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_GDS_OUTER_FOLDS = 2
DEFAULT_GDS_INNER_FOLDS = 2
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_RANK_METRIC = "f1_macro"
DEFAULT_RANK_MODE = "auto"
MINIMIZE_METRICS = ("mae_ordinal", "errores_graves")

# ---------------------------------------------------------------------------
# Grid de hiperparametros (busqueda interna)
# ---------------------------------------------------------------------------

# Cada diccionario sobrescribe la configuracion base. Las claves deben ser
# campos de ModelConfig o TrainingConfig. La busqueda interna elige la
# configuracion con menor MAE promedio (empate: mayor QWK).
_BASE_GRID = [
    {"conv_channels": (16, 32), "kernel_size": 3, "hidden_dim": 32, "dropout": 0.15},
    {"conv_channels": (16, 32), "kernel_size": 5, "hidden_dim": 32, "dropout": 0.15},
    {"conv_channels": (8, 16), "kernel_size": 3, "hidden_dim": 32, "dropout": 0.15},
    {"conv_channels": (16, 32), "kernel_size": 3, "hidden_dim": 64, "dropout": 0.30},
]

HYPERPARAMETER_GRID: dict[str, list[dict[str, Any]]] = {
    ALGORITHM_SOFTMAX: [dict(entry) for entry in _BASE_GRID],
    # Para CORAL, beta solo tiene efecto si se activa --use-weights.
    ALGORITHM_CORAL: [dict(entry, beta=DEFAULT_BETA) for entry in _BASE_GRID],
}


# ---------------------------------------------------------------------------
# Dataclasses de configuracion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """
    Hiperparametros de la arquitectura Conv1D + MLP.

    Atributos:
        conv_channels: canales de salida de cada bloque convolucional.
            (16, 32) crea dos bloques: 1 -> 16 -> 32.
        kernel_size: tamano del kernel (impar, para conservar el largo 15).
        hidden_dim: neuronas de la capa oculta del MLP.
        dropout: probabilidad de dropout del MLP.
    """

    conv_channels: tuple[int, ...] = DEFAULT_CONV_CHANNELS
    kernel_size: int = DEFAULT_KERNEL_SIZE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    dropout: float = DEFAULT_DROPOUT

    def __post_init__(self) -> None:
        if not self.conv_channels or any(c < 1 for c in self.conv_channels):
            raise ValueError("conv_channels debe tener al menos un entero positivo.")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser un entero impar positivo.")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim debe ser mayor o igual que 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1).")


@dataclass(frozen=True)
class TrainingConfig:
    """
    Hiperparametros del optimizador y del ciclo de entrenamiento.

    Atributos:
        learning_rate, weight_decay: parametros de Adam.
        batch_size: tamano de batch.
        epochs: epocas por cada entrenamiento (fold).
        use_class_weights: activa pesos por numero efectivo (solo CORAL).
        beta: parametro del numero efectivo de muestras.
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    use_class_weights: bool = False
    beta: float = DEFAULT_BETA

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size debe ser mayor o igual que 1.")
        if self.epochs < 1:
            raise ValueError("epochs debe ser mayor o igual que 1.")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta debe estar en (0, 1).")


@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuracion de la validacion cruzada anidada.

    Atributos:
        outer_folds, inner_folds: folds externos (reporte) e internos (seleccion).
        gds_outer_folds, gds_inner_folds: folds de GDS con --all-targets
            (la clase 7 de GDS solo tiene 2 muestras).
        use_grid: si es False, la busqueda interna evalua solo la
            configuracion fija entregada por linea de comandos.
    """

    outer_folds: int = DEFAULT_OUTER_FOLDS
    inner_folds: int = DEFAULT_INNER_FOLDS
    gds_outer_folds: int = DEFAULT_GDS_OUTER_FOLDS
    gds_inner_folds: int = DEFAULT_GDS_INNER_FOLDS
    use_grid: bool = True

    def __post_init__(self) -> None:
        values = (
            self.outer_folds,
            self.inner_folds,
            self.gds_outer_folds,
            self.gds_inner_folds,
        )
        if any(value < 2 for value in values):
            raise ValueError("Todos los folds deben ser al menos 2.")

    def folds_for(self, target_name: str, all_targets: bool) -> tuple[int, int]:
        """Devuelve (outer, inner). Con --all-targets, GDS usa folds reducidos."""

        if all_targets and target_name == "GDS":
            return self.gds_outer_folds, self.gds_inner_folds
        return self.outer_folds, self.inner_folds


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuracion completa de una ejecucion de main.py."""

    data_path: Path
    targets: tuple[str, ...] = (DEFAULT_TARGET,)
    algorithms: tuple[str, ...] = (DEFAULT_ALGORITHM,)
    all_targets: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    seed: int = DEFAULT_RANDOM_SEED
    device: str = "cpu"
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)
    rank_metric: str = DEFAULT_RANK_METRIC
    rank_mode: str = DEFAULT_RANK_MODE

    def __post_init__(self) -> None:
        unknown_targets = [t for t in self.targets if t not in TARGET_COLUMNS]
        if unknown_targets:
            raise ValueError(f"Objetivos invalidos: {unknown_targets}.")
        unknown_algorithms = [a for a in self.algorithms if a not in ALGORITHMS]
        if unknown_algorithms:
            raise ValueError(f"Algoritmos invalidos: {unknown_algorithms}.")


# ---------------------------------------------------------------------------
# Utilidades sobre configuraciones
# ---------------------------------------------------------------------------

_MODEL_FIELDS = {f.name for f in fields(ModelConfig)}
_TRAINING_FIELDS = {f.name for f in fields(TrainingConfig)}


def apply_overrides(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    overrides: dict[str, Any],
) -> tuple[ModelConfig, TrainingConfig]:
    """
    Aplica un diccionario del grid sobre las configuraciones base.

    Cada clave se envia a ModelConfig o TrainingConfig segun su nombre.
    Una clave desconocida es un error (evita typos silenciosos en el grid).
    """

    model_updates = {k: v for k, v in overrides.items() if k in _MODEL_FIELDS}
    training_updates = {k: v for k, v in overrides.items() if k in _TRAINING_FIELDS}
    unknown = set(overrides) - _MODEL_FIELDS - _TRAINING_FIELDS
    if unknown:
        raise KeyError(f"Claves desconocidas en el grid: {sorted(unknown)}.")

    if "conv_channels" in model_updates:
        model_updates["conv_channels"] = tuple(model_updates["conv_channels"])

    return (
        replace(model_config, **model_updates),
        replace(training_config, **training_updates),
    )


def describe_configs(model_config: ModelConfig, training_config: TrainingConfig) -> str:
    """Texto corto para imprimir una configuracion en consola."""

    channels = "-".join(str(c) for c in model_config.conv_channels)
    return (
        f"conv={channels} k={model_config.kernel_size} "
        f"hidden={model_config.hidden_dim} dropout={model_config.dropout} "
        f"lr={training_config.learning_rate} wd={training_config.weight_decay}"
    )


def configs_to_dict(
    model_config: ModelConfig, training_config: TrainingConfig
) -> dict[str, Any]:
    """Une ambas configuraciones en un diccionario plano."""

    merged = asdict(model_config)
    merged.update(asdict(training_config))
    return merged
