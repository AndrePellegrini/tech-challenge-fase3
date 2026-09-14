"""Primitivas de visualização compartilhadas pelos scripts de modelagem.

Antes deste módulo, cada script de `src/modeling` montava suas próprias figuras com
chamadas diretas a `matplotlib`, repetindo curva ROC, curva precision-recall,
histograma sobreposto e barra horizontal em arquivos diferentes. As funções aqui
concentram esse desenho, mantendo eixos, títulos e resolução consistentes entre todos
os gráficos do projeto.

Todas recebem vetores ou estruturas simples, nunca o dataset de modelagem, e por isso
são testáveis com dados sintéticos sem depender do parquet ou de credenciais.

O backend `Agg` é definido aqui porque estes scripts rodam sem interface gráfica.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

DPI = 150
REFERENCE_COLOR = "#808080"
PRIMARY_COLOR = "#4c72b0"
ALERT_COLOR = "#c44e52"


def save_figure(path: Path, *, tight: bool = True, bbox_inches: str | None = None) -> Path:
    """Grava a figura corrente, criando o diretório de destino se necessário."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches=bbox_inches)
    plt.close()
    return path


def roc_curves(curves: Mapping[str, tuple[Sequence, Sequence]], path: Path,
               *, title: str, show_auc: bool = True) -> Path:
    """Curvas ROC sobrepostas, com a diagonal de referência.

    `curves` mapeia rótulo para o par (verdadeiro, score da classe positiva).
    """
    if not curves:
        raise ValueError("Nenhuma curva informada para o gráfico ROC.")
    plt.figure(figsize=(7, 6))
    for label, (y_true, score) in curves.items():
        false_positive, true_positive, _ = roc_curve(y_true, score)
        legend = label
        if show_auc:
            legend = f"{label} ({roc_auc_score(y_true, score):.4f})"
        plt.plot(false_positive, true_positive, label=legend)
    plt.plot([0, 1], [0, 1], "--", color=REFERENCE_COLOR)
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title(title)
    plt.legend(fontsize=8)
    return save_figure(path)


def precision_recall_curves(curves: Mapping[str, tuple[Sequence, Sequence]], path: Path,
                            *, title: str, xlabel: str = "Recall",
                            ylabel: str = "Precision", color: str | None = None) -> Path:
    """Curvas precision-recall sobrepostas."""
    if not curves:
        raise ValueError("Nenhuma curva informada para o gráfico precision-recall.")
    plt.figure(figsize=(7, 6))
    show_legend = len(curves) > 1
    for label, (y_true, score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, score)
        plt.plot(recall, precision, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_legend:
        plt.legend(fontsize=8)
    return save_figure(path)


def overlapping_histograms(series: Mapping[str, Sequence], path: Path, *, title: str,
                           xlabel: str, ylabel: str = "Densidade", bins: int = 40,
                           density: bool = True) -> Path:
    """Histogramas sobrepostos, para comparar distribuições lado a lado."""
    if not series:
        raise ValueError("Nenhuma série informada para o histograma.")
    plt.figure(figsize=(8, 5))
    for label, values in series.items():
        plt.hist(np.asarray(values), bins=bins, alpha=.55, density=density, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    return save_figure(path)


def histogram_with_marker(values: Sequence, path: Path, *, title: str, xlabel: str,
                          ylabel: str = "Frequência", marker: float | None = None,
                          bins: int = 40) -> Path:
    """Histograma simples com uma linha vertical de referência opcional."""
    plt.figure(figsize=(8, 5))
    plt.hist(np.asarray(values), bins=bins, color=PRIMARY_COLOR, alpha=.8)
    if marker is not None:
        plt.axvline(marker, color=ALERT_COLOR, linestyle="--", linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return save_figure(path)


def horizontal_bar(labels: Iterable[str], values: Iterable[float], path: Path,
                   *, title: str, xlabel: str) -> Path:
    """Barras horizontais, usada para ranking de importância de features."""
    labels, values = list(labels), list(values)
    if not labels:
        raise ValueError("Nenhuma barra informada para o gráfico horizontal.")
    plt.figure(figsize=(8, 6))
    plt.barh(labels, values, color=PRIMARY_COLOR)
    plt.title(title)
    plt.xlabel(xlabel)
    return save_figure(path, bbox_inches="tight")


def grouped_bar(frame, columns: Sequence[str], path: Path, *, title: str,
                ylabel: str, ylim: tuple[float, float] | None = (0, 1),
                rot: int = 20) -> Path:
    """Barras agrupadas a partir de um DataFrame já indexado pelo rótulo."""
    frame[list(columns)].plot.bar(figsize=(9, 5), ylim=ylim, rot=rot, title=title)
    plt.ylabel(ylabel)
    return save_figure(path, bbox_inches="tight")
