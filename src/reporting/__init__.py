"""Reportes: tablas CSV/Markdown, rankings y graficos."""

from src.reporting.latex_report import generar_informe_latex
from src.reporting.reports import (
    DISPLAY_NAMES,
    MarkdownTables,
    Plotter,
    Ranking,
    ReportWriter,
    ResultRows,
)

__all__ = [
    "DISPLAY_NAMES",
    "MarkdownTables",
    "Plotter",
    "Ranking",
    "ReportWriter",
    "ResultRows",
    "generar_informe_latex",
]
