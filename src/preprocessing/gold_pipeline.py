"""Fluxo oficial: alunos Silver + features Gold -> parquet Gold validado.

Execute na raiz: python -m src.preprocessing.gold_pipeline
O pipeline.py anterior permanece como alternativa legada de BigQuery.
"""
from src.preprocessing.silver_reader import read_silver_students
from src.preprocessing.gold_reader import read_gold_table
from src.preprocessing.gold_features import build_consolidated_gold_features
from src.preprocessing.build_modeling_dataset import build_modeling_dataset_from_gold
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset


def main() -> None:
    print("Lendo população Silver de 2024...", flush=True)
    students = read_silver_students()
    print(f"Alunos elegíveis: {len(students):,}", flush=True)
    print("Lendo e consolidando features Gold...", flush=True)
    features = build_consolidated_gold_features(
        indicador_gold=read_gold_table("indicador_alfabetizacao_municipio"),
        desempenho_gold=read_gold_table("desempenho_alunos_municipio"),
        metas_gold=read_gold_table("comparativo_metas_resultados"),
    )
    dataset = build_modeling_dataset_from_gold(students, features)
    report = validate_dataset(dataset, strict_counts=True)
    if not report["valido"]:
        raise ValueError(f"Dataset Gold inválido: {report['errors']}")
    print(f"Shape validado: {dataset.shape}", flush=True)
    print(f"Classes: {dataset['alfabetizado'].value_counts().to_dict()}", flush=True)
    print(f"Cobertura Gold: {dataset['gold_historico_disponivel'].value_counts().to_dict()}", flush=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"Parquet salvo: {DATASET_PATH}", flush=True)


if __name__ == "__main__":
    main()
