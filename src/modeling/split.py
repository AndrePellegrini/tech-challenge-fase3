"""Definição de X/y/groups e split municipal, sem transformações ou modelos.

Execute: python -m src.modeling.split
Índices retornados são posições para uso com iloc, não rótulos do índice pandas.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing.dataset_contract import COLUMN_GROUPS, TARGET_COLUMN
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

GROUP_COLUMN = "id_municipio"
WEIGHT_COLUMN = "peso_aluno"
RANDOM_STATE = 42
SPLIT_NAMES = ("train", "validation", "test")
# Tolerâncias diagnósticas em proporções absolutas (5 e 3 pontos percentuais).
ROW_SHARE_TOLERANCE = 0.05
CLASS_SHARE_TOLERANCE = 0.03


def get_modeling_columns() -> dict[str, object]:
    """Candidatas iniciais; não representa seleção por desempenho/correlação."""
    categorical = list(COLUMN_GROUPS["contexto_categorico"]["colunas"])
    numeric = [
        *COLUMN_GROUPS["historico_2023"]["colunas"],
        "meta_alfabetizacao_municipio_2024", "gap_para_meta_municipio_2024",
        *COLUMN_GROUPS["socioeconomicas"]["colunas"],
    ]
    boolean = ["atingiu_meta_municipio_2024"]
    availability = ["gold_historico_disponivel"]
    return {
        "X": categorical + numeric + boolean + availability,
        "categorical": categorical, "numeric": numeric,
        "boolean": boolean, "availability": availability,
        "y": TARGET_COLUMN, "groups": GROUP_COLUMN, "sample_weight": WEIGHT_COLUMN,
        "excluded_from_X": [
            *COLUMN_GROUPS["identificacao"]["colunas"], WEIGHT_COLUMN, TARGET_COLUMN,
        ],
    }


def _check_dataset(dataset: pd.DataFrame) -> None:
    report = validate_dataset(dataset, strict_counts=False)
    if not report["valido"]:
        raise ValueError(f"Dataset fora do contrato: {report['errors']}")
    if dataset.empty or dataset[GROUP_COLUMN].isna().any():
        raise ValueError("Dataset vazio ou município nulo.")
    if not dataset[GROUP_COLUMN].map(lambda value: isinstance(value, str)).all():
        raise ValueError("id_municipio deve conter códigos textuais, como no parquet vigente.")


def get_modeling_data(dataset: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Separa dados brutos alinhados; peso apenas reservado, nunca aplicado."""
    _check_dataset(dataset)
    columns = get_modeling_columns()
    return {
        "X": dataset.loc[:, columns["X"]].copy(),
        "y": dataset[TARGET_COLUMN].copy(),
        "groups": dataset[GROUP_COLUMN].copy(),
        "sample_weight": dataset[WEIGHT_COLUMN].copy(),
    }


def split_by_municipality(
    dataset: pd.DataFrame, random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """70/15/15 dos municípios, aproximadamente; não estratifica alunos.

    Dois GroupShuffleSplit, com uma única realização e a mesma semente.
    O target não participa da escolha. Retorna posições disjuntas e completas.
    """
    _check_dataset(dataset)
    if not isinstance(random_state, int):
        raise ValueError("random_state deve ser inteiro para reprodução.")
    if dataset[GROUP_COLUMN].nunique() < 4:
        raise ValueError("São necessários pelo menos quatro municípios para três conjuntos.")
    dummy = np.empty((len(dataset), 0))
    train, temporary = next(GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=random_state,
    ).split(dummy, groups=dataset[GROUP_COLUMN]))
    validation, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state,
    ).split(dummy[temporary], groups=dataset.iloc[temporary][GROUP_COLUMN]))
    return {"train": train, "validation": temporary[validation], "test": temporary[test]}


def validate_split(dataset: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict:
    """Verifica cobertura exata, IDs/grupos disjuntos e resume classes/tamanhos.

    Desvios de proporção são avisos explícitos: a unidade sorteada é município.
    Conjuntos vazios ou sem ambas as classes são erros para esta tarefa binária.
    """
    _check_dataset(dataset)
    errors, warnings, summary = [], [], {}
    if set(splits) != set(SPLIT_NAMES):
        return {"valid": False, "errors": ["Conjuntos devem ser train, validation e test."], "warnings": [], "summary": {}}
    arrays = {name: np.asarray(splits[name]) for name in SPLIT_NAMES}
    for name, positions in arrays.items():
        if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
                or (positions < 0).any() or (positions >= len(dataset)).any()):
            errors.append(f"Índices posicionais inválidos: {name}.")
    if errors:
        return {"valid": False, "errors": errors, "warnings": [], "summary": {}}
    combined = np.concatenate(list(arrays.values()))
    complete = len(combined) == len(dataset) and np.array_equal(
        np.sort(combined), np.arange(len(dataset)),
    )
    if not complete:
        errors.append("As posições não cobrem todas as linhas exatamente uma vez.")
    groups, students = {}, {}
    global_rate = float(dataset[TARGET_COLUMN].mean())
    for name, expected_share in zip(SPLIT_NAMES, (0.70, 0.15, 0.15)):
        part = dataset.iloc[arrays[name]]
        groups[name] = set(part[GROUP_COLUMN])
        students[name] = set(part["id_aluno"])
        counts = {str(c): int(part[TARGET_COLUMN].eq(c).sum()) for c in (0, 1)}
        rate = float(part[TARGET_COLUMN].mean()) if len(part) else None
        share = len(part) / len(dataset)
        summary[name] = {
            "rows": len(part), "row_share": share, "municipalities": len(groups[name]),
            "class_counts": counts,
            "class_shares": {str(c): counts[str(c)] / len(part) if len(part) else None for c in (0, 1)},
            "class_1_delta_pp": (rate - global_rate) * 100 if rate is not None else None,
        }
        if not len(part) or min(counts.values()) == 0:
            errors.append(f"{name} está vazio ou não contém ambas as classes.")
        if abs(share - expected_share) > ROW_SHARE_TOLERANCE:
            warnings.append(f"{name}: proporção de linhas distante mais de 5 p.p. da referência.")
        if rate is not None and abs(rate - global_rate) > CLASS_SHARE_TOLERANCE:
            warnings.append(f"{name}: proporção da classe 1 distante mais de 3 p.p. do global.")
    group_overlap, student_overlap = {}, {}
    for left, right in combinations(SPLIT_NAMES, 2):
        pair = f"{left}/{right}"
        group_overlap[pair] = len(groups[left] & groups[right])
        student_overlap[pair] = len(students[left] & students[right])
        if group_overlap[pair] or student_overlap[pair]:
            errors.append(f"Sobreposição de municípios ou alunos em {pair}.")
    return {
        "valid": not errors, "errors": errors, "warnings": warnings, "summary": summary,
        "global_class_1_share": global_rate, "rows_preserved": bool(complete),
        "group_overlap": group_overlap, "student_overlap": student_overlap,
    }


def audit_context_features(dataset: pd.DataFrame) -> dict:
    """Auditoria descritiva; não exclui features nem ajusta transformações."""
    rate = dataset["taxa_alfabetizacao_municipio_2023"].astype(float)
    target = dataset["meta_alfabetizacao_municipio_2024"].astype(float)
    gap = dataset["gap_para_meta_municipio_2024"]
    reached = dataset["atingiu_meta_municipio_2024"]
    gap_mask = rate.notna() & target.notna() & gap.notna()
    reached_mask = rate.notna() & target.notna() & reached.notna()
    differences = (gap[gap_mask] - (target - rate)[gap_mask]).abs()
    pairs = [
        ("taxa_alfabetizacao_municipio_2023", "pct_alfabetizados_municipio_2023"),
        ("media_portugues_municipio_2023", "proficiencia_media_ponderada_2023"),
    ]
    return {
        "gap_rows_checked": int(gap_mask.sum()),
        "gap_max_absolute_difference": float(differences.max()),
        "gap_mismatches_atol_1e_4": int((differences > 1e-4).sum()),
        "reached_rows_checked": int(reached_mask.sum()),
        "reached_mismatches": int((~reached[reached_mask].eq((rate >= target)[reached_mask])).sum()),
        "correlations": [
            {"columns": [a, b], "pearson": float(dataset[[a, b]].corr().iloc[0, 1]),
             "complete_pairs": int(dataset[[a, b]].notna().all(axis=1).sum())}
            for a, b in pairs
        ],
    }


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(dataset, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    splits = split_by_municipality(dataset)
    report = validate_split(dataset, splits)
    if not report["valid"]:
        raise ValueError(report["errors"])
    municipal = dataset.groupby(GROUP_COLUMN)[TARGET_COLUMN].agg(rows="size", class_1="sum")
    municipal["class_0"] = municipal["rows"] - municipal["class_1"]
    municipal["class_1_share"] = municipal["class_1"] / municipal["rows"]
    for name, positions in splits.items():
        municipal.loc[dataset.iloc[positions][GROUP_COLUMN].unique(), "split"] = name
    import sklearn
    report.update({
        "random_state": RANDOM_STATE, "sklearn_version": sklearn.__version__,
        "method": "GroupShuffleSplit 70/30 seguido de 50/50; mesma semente; sem seleção pelo target",
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "modeling_columns": get_modeling_columns(),
        "municipalities": len(municipal),
        "single_class_municipalities": int(((municipal.class_0 == 0) | (municipal.class_1 == 0)).sum()),
        "only_class_0_municipalities": int((municipal.class_1 == 0).sum()),
        "only_class_1_municipalities": int((municipal.class_0 == 0).sum()),
        "municipal_summary": municipal[["rows", "class_1_share"]].describe(
            percentiles=[.01, .10, .25, .50, .75, .90, .99],
        ).to_dict(),
        "feature_audit": audit_context_features(dataset),
    })
    root = Path(__file__).resolve().parents[2]
    report_path = root / "reports" / "modeling_split_summary.json"
    manifest_path = root / "data" / "processed" / "modeling_split_2024.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Manifesto municipal permite recuperar o split mesmo se a ordem das linhas mudar.
    manifest = {
        "dataset_sha256": report["dataset_sha256"], "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "municipalities": municipal.reset_index().to_dict(orient="records"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Manifesto local: {manifest_path}")


if __name__ == "__main__":
    main()
