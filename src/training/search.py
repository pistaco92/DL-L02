"""Busqueda de hiperparametros en el loop interno de la validacion anidada."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig, apply_overrides
from src.data.preprocessing import StratifiedSplitter
from src.training.trainer import Trainer


@dataclass
class CandidateScore:
    """MAE y QWK promedio de una configuracion sobre los folds internos."""

    overrides: dict[str, Any]
    model_config: ModelConfig
    training_config: TrainingConfig
    mae_mean: float = float("nan")
    mae_std: float = float("nan")
    qwk_mean: float = float("nan")
    qwk_std: float = float("nan")

    def selection_key(self) -> tuple[float, float]:
        """Menor MAE primero; empate: mayor QWK. NaN queda al final."""

        mae = math.inf if math.isnan(self.mae_mean) else round(self.mae_mean, 10)
        qwk = -math.inf if math.isnan(self.qwk_mean) else self.qwk_mean
        return mae, -qwk


@dataclass
class SearchResult:
    """Configuracion ganadora y puntajes de todas las candidatas."""

    best: CandidateScore
    candidates: list[CandidateScore] = field(default_factory=list)
    skipped: bool = False


class GridSearch:
    """
    Recorre un grid de configuraciones sobre folds internos estratificados.

    Reglas del laboratorio:
    - Solo se usan datos del entrenamiento externo.
    - Se elige la configuracion con menor MAE interno promedio.
    - Empate: mayor QWK interno promedio.
    - Nunca se mira el fold externo de prueba.

    Si candidates es [{}], se evalua solo la configuracion base (--no-grid):
    sirve para leer el MAE interno sin buscar.
    """

    def __init__(
        self,
        algorithm: str,
        candidates: list[dict[str, Any]],
        base_model_config: ModelConfig,
        base_training_config: TrainingConfig,
        inner_folds: int,
        device: torch.device,
    ) -> None:
        if not candidates:
            raise ValueError("El grid debe tener al menos una configuracion.")
        self.algorithm = algorithm
        self.candidates = candidates
        self.base_model_config = base_model_config
        self.base_training_config = base_training_config
        self.inner_folds = inner_folds
        self.device = device

    def _candidate(self, overrides: dict[str, Any]) -> CandidateScore:
        model_config, training_config = apply_overrides(
            self.base_model_config, self.base_training_config, overrides
        )
        return CandidateScore(overrides, model_config, training_config)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        seed: int,
    ) -> SearchResult:
        """Evalua cada candidata y devuelve la ganadora."""

        splitter = StratifiedSplitter(self.inner_folds, random_state=seed)
        if not splitter.can_split(y):
            # Clase demasiado rara: se usa la configuracion base sin buscar.
            return SearchResult(best=self._candidate({}), skipped=True)

        splits = splitter.split(y)
        scores = []
        for candidate_index, overrides in enumerate(self.candidates):
            candidate = self._candidate(overrides)
            mae_values, qwk_values = [], []
            for fold_index, (train_idx, val_idx) in enumerate(splits):
                run_seed = seed + candidate_index * 10 + fold_index
                trainer = Trainer.build(
                    self.algorithm,
                    candidate.model_config,
                    candidate.training_config,
                    num_features=X.shape[1],
                    num_classes=num_classes,
                    device=self.device,
                    seed=run_seed,
                )
                result = trainer.fit_and_evaluate(
                    X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed=run_seed
                )
                mae_values.append(result.metrics["mae_ordinal"])
                qwk_values.append(result.metrics["qwk"])

            candidate.mae_mean = float(np.mean(mae_values))
            candidate.mae_std = float(np.std(mae_values))
            candidate.qwk_mean = float(np.mean(qwk_values))
            candidate.qwk_std = float(np.std(qwk_values))
            scores.append(candidate)

        best = min(scores, key=CandidateScore.selection_key)
        return SearchResult(best=best, candidates=scores)
