"""Tareas de clasificacion: definen perdida y regla de prediccion."""

from src.tasks.base import ClassificationTask
from src.tasks.factory import TaskFactory
from src.tasks.ordinal_task import CoralTask
from src.tasks.softmax_task import SoftmaxTask

__all__ = ["ClassificationTask", "CoralTask", "SoftmaxTask", "TaskFactory"]
