"""Tablas, rankings y graficos para comparar experimentos."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DEFAULT_RANK_METRIC, MINIMIZE_METRICS  # noqa: E402
from src.evaluation.metrics import METRIC_KEYS  # noqa: E402
from src.training.nested_cv import ExperimentResult  # noqa: E402

DISPLAY_NAMES = {
    "accuracy": "Acc",
    "balanced_accuracy": "BalAcc",
    "precision_macro": "Prec_m",
    "recall_macro": "Rec_m",
    "f1_macro": "F1_m",
    "mae_ordinal": "MAE",
    "qwk": "QWK",
    "accuracy_pm1": "Acc±1",
    "errores_graves": "Err≥2",
}

CONFIG_FIELDS = [
    "algorithm",
    "target",
    "hp_mode",
    "conv_channels",
    "kernel_size",
    "hidden_dim",
    "dropout",
    "learning_rate",
    "weight_decay",
    "use_class_weights",
    "beta",
    "epochs",
    "outer_folds",
    "inner_folds",
]

CSV_FIELDS = [
    *CONFIG_FIELDS,
    *METRIC_KEYS,
    *[f"{key}_std" for key in METRIC_KEYS],
]

SELECTION_FIELDS = [
    "algorithm",
    "target",
    "outer_fold",
    "inner_skipped",
    "conv_channels",
    "kernel_size",
    "hidden_dim",
    "dropout",
    "learning_rate",
    "weight_decay",
    "beta",
    "inner_mae",
    "inner_qwk",
    "test_f1_macro",
    "test_mae_ordinal",
]


def _channels_label(channels: tuple[int, ...]) -> str:
    return "-".join(str(c) for c in channels)


class Ranking:
    """Ordena filas de resultados por una metrica."""

    @staticmethod
    def resolve_mode(metric: str, mode: str) -> str:
        """auto minimiza MAE y errores graves; maximiza el resto."""

        if mode == "auto":
            return "min" if metric in MINIMIZE_METRICS else "max"
        if mode not in {"max", "min"}:
            raise ValueError("rank-mode debe ser max, min o auto.")
        return mode

    @staticmethod
    def rank(rows: list[dict], metric: str, mode: str) -> list[dict]:
        if metric not in METRIC_KEYS:
            raise ValueError(f"Metrica invalida: {metric}. Use una de {METRIC_KEYS}.")
        return sorted(rows, key=lambda row: row[metric], reverse=(mode == "max"))

    @staticmethod
    def format_console(ranked_rows: list[dict], metric: str, mode: str) -> str:
        lines = [f"Ranking {DISPLAY_NAMES.get(metric, metric)} ({mode}):"]
        for index, row in enumerate(ranked_rows, start=1):
            marker = " <- mejor" if index == 1 else ""
            lines.append(
                f"  {index}. {row['algorithm']} / {row['target']}: "
                f"{row[metric]:.4f} ± {row[f'{metric}_std']:.4f}{marker}"
            )
        return "\n".join(lines)


class ResultRows:
    """Convierte ExperimentResult en filas planas para CSV y Markdown."""

    @staticmethod
    def summary_row(result: ExperimentResult) -> dict:
        """
        Una fila por experimento con media y std externa.

        Con grid activo, los hiperparametros mostrados son los elegidos con
        mas frecuencia entre los folds externos (detalle en seleccion_hp.csv).
        """

        selected = result.most_selected()
        model, training = selected.model_config, selected.training_config
        row = {
            "algorithm": result.algorithm,
            "target": result.target_name,
            "hp_mode": "grid" if result.use_grid else "fija",
            "conv_channels": _channels_label(model.conv_channels),
            "kernel_size": model.kernel_size,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
            "use_class_weights": training.use_class_weights,
            "beta": training.beta,
            "epochs": training.epochs,
            "outer_folds": result.outer_folds,
            "inner_folds": result.inner_folds,
        }
        for metric in METRIC_KEYS:
            row[metric] = result.summary[f"mean_{metric}"]
            row[f"{metric}_std"] = result.summary[f"std_{metric}"]
        return row

    @staticmethod
    def selection_rows(result: ExperimentResult) -> list[dict]:
        """Una fila por fold externo con la configuracion elegida."""

        rows = []
        for fold in result.folds:
            model = fold.selected.model_config
            training = fold.selected.training_config
            rows.append(
                {
                    "algorithm": result.algorithm,
                    "target": result.target_name,
                    "outer_fold": fold.outer_fold,
                    "inner_skipped": fold.inner_skipped,
                    "conv_channels": _channels_label(model.conv_channels),
                    "kernel_size": model.kernel_size,
                    "hidden_dim": model.hidden_dim,
                    "dropout": model.dropout,
                    "learning_rate": training.learning_rate,
                    "weight_decay": training.weight_decay,
                    "beta": training.beta,
                    "inner_mae": fold.selected.mae_mean,
                    "inner_qwk": fold.selected.qwk_mean,
                    "test_f1_macro": fold.metrics["f1_macro"],
                    "test_mae_ordinal": fold.metrics["mae_ordinal"],
                }
            )
        return rows


class MarkdownTables:
    """Tablas Markdown de resultados y rankings."""

    @staticmethod
    def _mean_std(mean_value: float, std_value: float) -> str:
        return f"{mean_value:.4f} ± {std_value:.4f}"

    @classmethod
    def results(cls, rows: list[dict]) -> str:
        headers = ["Algoritmo", "Objetivo", "HP", "Conv", "k"] + [
            DISPLAY_NAMES[key] for key in METRIC_KEYS
        ]
        lines = [
            "# Resultados (media ± std de los folds externos)\n",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            cells = [
                str(row["algorithm"]),
                str(row["target"]),
                str(row["hp_mode"]),
                str(row["conv_channels"]),
                str(row["kernel_size"]),
            ]
            cells += [cls._mean_std(row[k], row[f"{k}_std"]) for k in METRIC_KEYS]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    @classmethod
    def ranking(cls, ranked_rows: list[dict], metric: str, mode: str) -> str:
        direction = "maximizar" if mode == "max" else "minimizar"
        name = DISPLAY_NAMES.get(metric, metric)
        headers = ["Puesto", "Algoritmo", "Objetivo", name, "Acc", "F1_m", "MAE", "QWK", "Err≥2"]
        lines = [
            f"# Ranking por {name} ({direction})\n",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for index, row in enumerate(ranked_rows, start=1):
            algorithm, target = row["algorithm"], row["target"]
            metric_cell = cls._mean_std(row[metric], row[f"{metric}_std"])
            if index == 1:
                algorithm, target = f"**{algorithm}**", f"**{target}**"
                metric_cell = f"**{metric_cell}**"
            cells = [
                str(index),
                algorithm,
                target,
                metric_cell,
                f"{row['accuracy']:.4f}",
                f"{row['f1_macro']:.4f}",
                f"{row['mae_ordinal']:.4f}",
                f"{row['qwk']:.4f}",
                f"{row['errores_graves']:.4f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if ranked_rows:
            best = ranked_rows[0]
            lines.append("")
            lines.append(
                f"Mejor resultado: {best['algorithm']} en {best['target']} "
                f"({name} = {best[metric]:.4f})."
            )
        return "\n".join(lines) + "\n"


class Plotter:
    """Graficos PNG de barras y matrices de confusion."""

    @staticmethod
    def _save(figure, output_path: Path) -> None:
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    @classmethod
    def metric_bars(cls, rows: list[dict], metric: str, output_path: Path, ylabel: str) -> None:
        """Barras agrupadas por objetivo, una serie por algoritmo."""

        targets = list(dict.fromkeys(row["target"] for row in rows))
        algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
        x = np.arange(len(targets))
        width = 0.8 / max(len(algorithms), 1)

        figure, axis = plt.subplots(figsize=(10, 4.5))
        for index, algorithm in enumerate(algorithms):
            lookup = {r["target"]: r for r in rows if r["algorithm"] == algorithm}
            values = [lookup[t][metric] if t in lookup else 0.0 for t in targets]
            errors = [lookup[t][f"{metric}_std"] if t in lookup else 0.0 for t in targets]
            axis.bar(x + index * width, values, width, yerr=errors, capsize=3, label=algorithm)

        axis.set_xticks(x + width * (len(algorithms) - 1) / 2)
        axis.set_xticklabels(targets, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.legend()
        axis.grid(axis="y", linestyle=":", alpha=0.4)
        cls._save(figure, output_path)

    @classmethod
    def ranking_bars(
        cls, ranked_rows: list[dict], metric: str, mode: str, output_path: Path
    ) -> None:
        labels = [f"{row['algorithm']} / {row['target']}" for row in ranked_rows]
        values = [row[metric] for row in ranked_rows]
        figure, axis = plt.subplots(figsize=(8, max(3.0, 0.45 * len(ranked_rows) + 1.2)))
        colors = ["#B51700" if i == 0 else "#0076BA" for i in range(len(values))]
        axis.barh(range(len(values)), values, color=colors)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        direction = "mayor es mejor" if mode == "max" else "menor es mejor"
        name = DISPLAY_NAMES.get(metric, metric)
        axis.set_xlabel(f"{name} ({direction})")
        axis.set_title(f"Ranking por {name}")
        cls._save(figure, output_path)

    @classmethod
    def confusion_matrix(cls, matrix: np.ndarray, title: str, output_path: Path) -> None:
        figure, axis = plt.subplots(figsize=(4.8, 4.2))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xlabel("Predicho")
        axis.set_ylabel("Real")
        axis.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
        cls._save(figure, output_path)


class ReportWriter:
    """
    Escribe todos los artefactos de una ejecucion en output_dir.

    Uso:
        writer = ReportWriter("results")
        paths = writer.save(results, rank_metric="f1_macro", rank_mode="auto")
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def save(
        self,
        results: list[ExperimentResult],
        rank_metric: str = DEFAULT_RANK_METRIC,
        rank_mode: str = "auto",
    ) -> dict[str, Path]:
        """Escribe CSV, Markdown y PNG. Devuelve las rutas generadas."""

        if not results:
            raise ValueError("No hay resultados para reportar.")

        rows = [ResultRows.summary_row(r) for r in results]
        selections = [s for r in results for s in ResultRows.selection_rows(r)]
        mode = Ranking.resolve_mode(rank_metric, rank_mode)
        ranking_f1 = Ranking.rank(rows, "f1_macro", "max")
        ranking_mae = Ranking.rank(rows, "mae_ordinal", "min")
        ranking_custom = Ranking.rank(rows, rank_metric, mode)

        out = self.output_dir
        paths = {
            "resultados_csv": out / "resultados.csv",
            "resultados_md": out / "resultados.md",
            "seleccion_hp_csv": out / "seleccion_hp.csv",
            "ranking_f1_csv": out / "ranking_f1.csv",
            "ranking_f1_md": out / "ranking_f1.md",
            "ranking_mae_csv": out / "ranking_mae.csv",
            "ranking_mae_md": out / "ranking_mae.md",
            "ranking_custom_csv": out / f"ranking_{rank_metric}.csv",
            "ranking_custom_md": out / f"ranking_{rank_metric}.md",
            "plot_f1": out / "barras_f1.png",
            "plot_mae": out / "barras_mae.png",
            "plot_ranking": out / f"ranking_{rank_metric}.png",
        }

        self._write_csv(paths["resultados_csv"], rows, CSV_FIELDS)
        self._write_text(paths["resultados_md"], MarkdownTables.results(rows))
        self._write_csv(paths["seleccion_hp_csv"], selections, SELECTION_FIELDS)
        self._write_csv(paths["ranking_f1_csv"], ranking_f1, CSV_FIELDS)
        self._write_text(paths["ranking_f1_md"], MarkdownTables.ranking(ranking_f1, "f1_macro", "max"))
        self._write_csv(paths["ranking_mae_csv"], ranking_mae, CSV_FIELDS)
        self._write_text(
            paths["ranking_mae_md"], MarkdownTables.ranking(ranking_mae, "mae_ordinal", "min")
        )
        self._write_csv(paths["ranking_custom_csv"], ranking_custom, CSV_FIELDS)
        self._write_text(
            paths["ranking_custom_md"], MarkdownTables.ranking(ranking_custom, rank_metric, mode)
        )
        Plotter.metric_bars(rows, "f1_macro", paths["plot_f1"], "F1 macro")
        Plotter.metric_bars(rows, "mae_ordinal", paths["plot_mae"], "MAE ordinal")
        Plotter.ranking_bars(ranking_custom, rank_metric, mode, paths["plot_ranking"])

        for result in results:
            key = f"confusion_{result.algorithm}_{result.target_name}"
            path = out / f"{key}.png"
            Plotter.confusion_matrix(
                np.asarray(result.last_fold_confusion()),
                f"{result.algorithm} / {result.target_name}",
                path,
            )
            paths[key] = path
        return paths
