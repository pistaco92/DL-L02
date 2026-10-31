"""Orquestacion de experimentos: objetivos x algoritmos."""

from __future__ import annotations

from src.config import HYPERPARAMETER_GRID, ExperimentConfig, describe_configs
from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data
from src.evaluation.metrics import METRIC_KEYS
from src.reporting.reports import Ranking, ReportWriter, ResultRows
from src.training.nested_cv import ExperimentResult, NestedCrossValidator
from src.utils import resolve_device


class ConsoleReporter:
    """Impresion en consola de cada experimento y de los rankings."""

    @staticmethod
    def experiment(result: ExperimentResult) -> None:
        print(f"Algoritmo: {result.algorithm} | Objetivo: {result.target_name}")
        print(
            f"Muestras: {result.num_samples} | Clases: {result.classes} | "
            f"Folds: outer={result.outer_folds}, inner={result.inner_folds} | "
            f"HP: {'grid' if result.use_grid else 'fija'} | Dispositivo: {result.device}"
        )
        print("Resumen externo (media +/- std):")
        for metric in METRIC_KEYS:
            mean_value = result.summary[f"mean_{metric}"]
            std_value = result.summary[f"std_{metric}"]
            print(f"  {metric}: {mean_value:.4f} +/- {std_value:.4f}")
        selected = result.most_selected()
        print(
            "Configuracion mas elegida: "
            + describe_configs(selected.model_config, selected.training_config)
        )
        print("Reporte del ultimo fold externo:")
        print(result.last_fold_report())
        print("Matriz de confusion del ultimo fold externo:")
        print(result.last_fold_confusion())

    @staticmethod
    def rankings(results: list[ExperimentResult], rank_metric: str, rank_mode: str) -> None:
        rows = [ResultRows.summary_row(r) for r in results]
        mode = Ranking.resolve_mode(rank_metric, rank_mode)
        print()
        print(Ranking.format_console(Ranking.rank(rows, "f1_macro", "max"), "f1_macro", "max"))
        print(Ranking.format_console(Ranking.rank(rows, "mae_ordinal", "min"), "mae_ordinal", "min"))
        print(Ranking.format_console(Ranking.rank(rows, rank_metric, mode), rank_metric, mode))


class ExperimentRunner:
    """
    Ejecuta todos los experimentos pedidos y escribe los reportes.

    Para cada algoritmo y cada objetivo:
    1. Prepara X (N, 15) e y (N,) con indices 0 .. K-1.
    2. Ejecuta NestedCrossValidator (grid interno + evaluacion externa).
    3. Acumula el resultado para tablas, rankings y graficos.

    Si un algoritmo lanza NotImplementedError (esqueletos CORAL sin
    completar) se informa y se continua con el resto.
    """

    def __init__(self, config: ExperimentConfig, loader: DataFrameLoader | None = None) -> None:
        self.config = config
        self.loader = loader or DataFrameLoader()
        self.device = resolve_device(config.device)
        self.console = ConsoleReporter()

    def _grid_for(self, algorithm: str) -> list[dict] | None:
        if not self.config.validation.use_grid:
            return None
        return HYPERPARAMETER_GRID[algorithm]

    def run_one(self, dataframe, algorithm: str, target_name: str) -> ExperimentResult:
        """Ejecuta un experimento (algoritmo x objetivo)."""

        outer_folds, inner_folds = self.config.validation.folds_for(
            target_name, self.config.all_targets
        )
        data = prepare_experiment_data(dataframe, target_name, self.loader)
        validator = NestedCrossValidator(
            algorithm=algorithm,
            base_model_config=self.config.model,
            base_training_config=self.config.training,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            grid=self._grid_for(algorithm),
            seed=self.config.seed,
            device=self.device,
        )
        return validator.run(data)

    def run(self) -> list[ExperimentResult]:
        """Ejecuta todos los experimentos y escribe los reportes."""

        dataframe = self.loader.load(self.config.data_path)
        results: list[ExperimentResult] = []
        pending_algorithms: list[tuple[str, str]] = []

        for algorithm in self.config.algorithms:
            for target_name in self.config.targets:
                print(f"\n=== {algorithm} / {target_name} ===")
                try:
                    result = self.run_one(dataframe, algorithm, target_name)
                except NotImplementedError as error:
                    print(f"Se omite {algorithm}: {error}")
                    pending_algorithms.append((algorithm, str(error)))
                    break
                self.console.experiment(result)
                results.append(result)

        if not results:
            print("\nNo hay resultados para reportar.")
            return results

        self.console.rankings(results, self.config.rank_metric, self.config.rank_mode)
        paths = ReportWriter(self.config.output_dir).save(
            results,
            rank_metric=self.config.rank_metric,
            rank_mode=self.config.rank_mode,
        )
        print(f"\nReportes escritos en {self.config.output_dir}/")
        for name, path in paths.items():
            print(f"  {name}: {path}")

        for algorithm, message in pending_algorithms:
            print(f"\nPendiente ({algorithm}): {message}")
        return results
