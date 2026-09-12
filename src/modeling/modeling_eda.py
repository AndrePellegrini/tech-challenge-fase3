"""EDA compacta orientada às decisões de modelagem.

Execute: python -m src.modeling.modeling_eda
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.split import get_modeling_columns
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "modeling_eda_summary.json"
IMAGE_DIR = ROOT / "images" / "modeling"


def _save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def _safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _group_target(dataset: pd.DataFrame, column: str) -> list[dict]:
    table = dataset.groupby(column, dropna=False)["alfabetizado"].agg(["size", "mean"]).reset_index()
    return [
        {column: _safe_value(row[column]), "students": int(row["size"]), "target_mean": float(row["mean"])}
        for _, row in table.iterrows()
    ]


def _binned_target(dataset: pd.DataFrame, column: str) -> list[dict]:
    complete = dataset[[column, "alfabetizado"]].dropna()
    try:
        bins = pd.qcut(complete[column], q=5, duplicates="drop")
    except ValueError:
        return []
    table = complete.groupby(bins, observed=True)["alfabetizado"].agg(["size", "mean"])
    return [
        {"bin": str(index), "students": int(row["size"]), "target_mean": float(row["mean"])}
        for index, row in table.iterrows()
    ]


def build_eda_report(dataset: pd.DataFrame) -> dict:
    columns = get_modeling_columns()
    X_columns = columns["X"]
    numeric = [*columns["numeric"], *columns["availability"]]
    missing_by_availability = (
        dataset.groupby("gold_historico_disponivel", dropna=False)[X_columns]
        .agg(lambda series: float(series.isna().mean()))
    )
    quantiles = dataset[numeric].quantile([0, .01, .25, .5, .75, .99, 1]).T
    outliers = {}
    for column in numeric:
        series = dataset[column].dropna()
        q1, q3 = series.quantile([.25, .75]) if len(series) else (np.nan, np.nan)
        iqr = q3 - q1
        outside = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        outliers[column] = {"iqr": _safe_value(iqr), "outside_1_5_iqr": int(outside)}
    binned_columns = [
        "taxa_alfabetizacao_municipio_2023",
        "media_portugues_municipio_2023",
        "idhm_educacao",
        "gap_para_meta_municipio_2024",
    ]
    correlation_columns = [
        "taxa_alfabetizacao_municipio_2023", "pct_alfabetizados_municipio_2023",
        "media_portugues_municipio_2023", "proficiencia_media_ponderada_2023",
        "idhm", "idhm_renda", "idhm_educacao",
    ]
    return {
        "scope": "EDA descritiva e associativa; não suporta inferência causal.",
        "rows": len(dataset),
        "target": {str(k): int(v) for k, v in dataset["alfabetizado"].value_counts().sort_index().items()},
        "target_rate": float(dataset["alfabetizado"].mean()),
        "missing": {column: {"count": int(dataset[column].isna().sum()), "rate": float(dataset[column].isna().mean())} for column in X_columns},
        "missing_by_gold_availability": {
            str(index): {column: float(value) for column, value in row.items()}
            for index, row in missing_by_availability.iterrows()
        },
        "categorical_cardinality": {
            column: int(dataset[column].nunique(dropna=True))
            for column in [*columns["categorical"], *columns["boolean"]]
        },
        "numeric_quantiles": {
            column: {str(q): _safe_value(value) for q, value in row.items()}
            for column, row in quantiles.iterrows()
        },
        "outliers": outliers,
        "correlations": dataset[correlation_columns].corr().to_dict(),
        "target_by_rede": _group_target(dataset, "rede"),
        "target_by_uf": _group_target(dataset, "sigla_uf"),
        "target_by_gold_availability": _group_target(dataset, "gold_historico_disponivel"),
        "target_by_quantile": {column: _binned_target(dataset, column) for column in binned_columns},
    }


def create_figures(dataset: pd.DataFrame, report: dict) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    target = pd.Series(report["target"], dtype=int)
    target.plot.bar(color=["#c44e52", "#4c72b0"], rot=0, title="Distribuição do target")
    plt.ylabel("Alunos")
    _save_figure("01_target_distribution.png")

    missing = pd.Series({k: v["rate"] for k, v in report["missing"].items()}).sort_values()
    missing.plot.barh(figsize=(8, 6), color="#8172b3", title="Valores ausentes nas features")
    plt.xlabel("Proporção ausente")
    _save_figure("02_missing_features.png")

    corr = pd.DataFrame(report["correlations"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), [name.replace("_municipio_2023", "").replace("proficiencia_media_ponderada_2023", "prof_ponderada") for name in corr], rotation=75, ha="right")
    ax.set_yticks(range(len(corr)), [name.replace("_municipio_2023", "").replace("proficiencia_media_ponderada_2023", "prof_ponderada") for name in corr])
    fig.colorbar(image, ax=ax, label="Correlação de Pearson")
    ax.set_title("Redundância entre features selecionadas")
    _save_figure("03_feature_correlations.png")

    uf = pd.DataFrame(report["target_by_uf"]).sort_values("target_mean")
    plt.figure(figsize=(10, 6))
    plt.barh(uf["sigla_uf"].astype(str), uf["target_mean"], color="#55a868")
    plt.xlabel("Proporção observada de alfabetizados")
    plt.title("Target médio por UF na população analisada")
    _save_figure("04_target_by_uf.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, rows) in zip(axes.flat, report["target_by_quantile"].items()):
        frame = pd.DataFrame(rows)
        ax.plot(range(1, len(frame) + 1), frame["target_mean"], marker="o")
        ax.set_title(column.replace("_municipio_2023", "").replace("_municipio_2024", ""))
        ax.set_xlabel("Faixa por quantil (menor → maior)")
        ax.set_ylabel("Target médio")
        ax.set_xticks(range(1, len(frame) + 1))
    _save_figure("05_target_by_feature_quantiles.png")


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)
    validation = validate_dataset(dataset, strict_counts=True)
    if not validation["valido"]:
        raise ValueError(validation["errors"])
    report = build_eda_report(dataset)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    create_figures(dataset, report)
    print(json.dumps({
        "report": str(REPORT_PATH), "images": str(IMAGE_DIR),
        "rows": report["rows"], "target_rate": report["target_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
