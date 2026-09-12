"""Contrato do artefato modeling_dataset_2024_gold.parquet da Fase 3.

Schema confirmado nos builders Gold e nos notebooks de integração e EDA.
As contagens são referências do artefato auditado pelo grupo, não requisitos FIAP.
"""
from __future__ import annotations


# ------------------------------------------------------------
# Grupos de colunas do dataset de modelagem
# ------------------------------------------------------------

COLUMN_GROUPS: dict[str, dict[str, object]] = {
    "identificacao": {
        "colunas": [
            "ano", "id_aluno", "id_municipio", "id_escola",
            "id_municipio_nome", "sigla_uf_nome",
        ],
        "uso": "Rastreabilidade; não define features finais de ML",
    },
    "target": {
        "colunas": ["alfabetizado"],
        "uso": "Classificação individual em 2024",
    },
    "informacao_amostral": {
        "colunas": ["peso_aluno"],
        "uso": "Peso da observação",
    },
    "contexto_categorico": {
        "colunas": ["rede", "sigla_uf"],
        "uso": "Contexto territorial e rede",
    },
    "historico_2023": {
        "colunas": [
            "taxa_alfabetizacao_municipio_2023",
            "media_portugues_municipio_2023",
            "percentual_participacao_municipio_2023",
            "total_alunos_municipio_2023",
            "pct_alfabetizados_municipio_2023",
            "proficiencia_media_ponderada_2023",
        ],
        "uso": "Indicadores históricos Gold por município e rede",
    },
    "metas_contexto": {
        "colunas": [
            "meta_alfabetizacao_municipio_2024",
            "gap_para_meta_municipio_2024",
            "atingiu_meta_municipio_2024",
        ],
        "uso": "Metas de 2024 comparadas ao histórico de 2023",
    },
    "socioeconomicas": {
        "colunas": ["idhm", "idhm_educacao", "idhm_renda", "idhm_longevidade"],
        "uso": "Contexto socioeconômico estadual da Gold",
    },
    "controle_ausencia": {
        "colunas": ["gold_historico_disponivel"],
        "uso": "Correspondência no join com a Gold; não garante completude",
    },
}

TARGET_COLUMN = "alfabetizado"
HISTORY_FLAG_COLUMN = "gold_historico_disponivel"
UNIQUE_KEY_COLUMN = "id_aluno"
DATASET_FILENAME = "modeling_dataset_2024_gold.parquet"

# Núcleo sem nulos nos registros com Gold, conforme EDA (células 10 e 15).
# As demais features admitem ausência mesmo quando houve correspondência.
HISTORY_FEATURE_COLUMNS = [
    "taxa_alfabetizacao_municipio_2023",
    "media_portugues_municipio_2023",
    "id_municipio_nome", "sigla_uf", "sigla_uf_nome",
    "idhm", "idhm_educacao", "idhm_renda", "idhm_longevidade",
]

# Colunas trazidas pelo lado direito do left join em build_modeling_dataset_from_gold.
# Todas devem ser nulas quando não existe correspondência na Gold.
GOLD_FEATURE_COLUMNS = [
    "id_municipio_nome", "sigla_uf", "sigla_uf_nome",
    *COLUMN_GROUPS["historico_2023"]["colunas"],
    *COLUMN_GROUPS["metas_contexto"]["colunas"],
    *COLUMN_GROUPS["socioeconomicas"]["colunas"],
]

EXPECTED_ROWS = 1_851_828
EXPECTED_COLUMNS = 24
EXPECTED_HISTORY_AVAILABLE = 1_816_270
EXPECTED_HISTORY_MISSING = 35_558
EXPECTED_TARGET_VALUES = {0, 1}


def get_expected_columns() -> list[str]:
    """
    Retorna a lista ordenada de colunas esperadas
    no dataset de modelagem.

    Returns
    -------
    list[str]
        Colunas de todos os grupos do contrato.
    """

    columns: list[str] = []

    for group in COLUMN_GROUPS.values():
        columns.extend(group["colunas"])

    return columns


def describe_contract() -> str:
    """
    Retorna uma descrição textual do contrato,
    no formato da tabela do documento do grupo.

    Returns
    -------
    str
        Descrição legível dos grupos de colunas.
    """

    lines = [
        "Contrato do dataset de modelagem (Fase 3):",
        "",
    ]

    for group_name, group in COLUMN_GROUPS.items():
        columns = ", ".join(group["colunas"])

        lines.append(
            f"- {group_name}: {columns} "
            f"({group['uso']})"
        )

    lines.extend(
        [
            "",
            f"Linhas esperadas: {EXPECTED_ROWS:,}",
            f"Colunas esperadas: {EXPECTED_COLUMNS}",
            (
                "Histórico disponível esperado: "
                f"{EXPECTED_HISTORY_AVAILABLE:,}"
            ),
            (
                "Histórico indisponível esperado: "
                f"{EXPECTED_HISTORY_MISSING:,}"
            ),
        ]
    )

    return "\n".join(lines)
