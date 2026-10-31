"""Registro de tareas (perdida + prediccion) por nombre de algoritmo."""

from __future__ import annotations

from src.config import ALGORITHM_CORAL, ALGORITHM_SOFTMAX
from src.tasks.base import ClassificationTask
from src.tasks.ordinal_task import CoralTask
from src.tasks.softmax_task import SoftmaxTask


class TaskFactory:
    """
    Crea la tarea asociada a cada algoritmo.

    Igual que ModelFactory, permite sustituir una implementacion:

        TaskFactory.register("conv1d_coral", MiCoralTask)
    """

    _registry: dict[str, type[ClassificationTask]] = {
        ALGORITHM_SOFTMAX: SoftmaxTask,
        ALGORITHM_CORAL: CoralTask,
    }

    @classmethod
    def register(cls, algorithm: str, task_class: type[ClassificationTask]) -> None:
        cls._registry[algorithm] = task_class

    @classmethod
    def create(cls, algorithm: str, num_classes: int) -> ClassificationTask:
        if algorithm not in cls._registry:
            raise KeyError(
                f"Algoritmo sin tarea registrada: {algorithm}. "
                f"Disponibles: {sorted(cls._registry)}."
            )
        return cls._registry[algorithm](num_classes)
