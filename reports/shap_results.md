# Interpretabilidade por SHAP

Fecha a lacuna da DEC-013.

## Método

`shap.TreeExplainer` aplicado ao modelo congelado `random_forest_controlled`, carregado
do `.joblib` sem refit, sobre amostra de 8.000 linhas da partição de
**validação**. O conjunto de teste não foi materializado.

As 43 colunas codificadas foram reagregadas para as
16 features de origem pela mesma regra da importância nativa,
somando as categorias one-hot de `rede` e `sigla_uf` de volta à sua feature.

## Ranking comparado

| # SHAP | Feature | SHAP médio absoluto | # Nativa | Importância nativa | Deslocamento |
|---:|---|---:|---:|---:|---:|
| 1 | `media_portugues_municipio_2023` | 0.0215 | 1 | 0.1578 | +0 |
| 2 | `sigla_uf` | 0.0171 | 8 | 0.0628 | +6 |
| 3 | `idhm_educacao` | 0.0166 | 7 | 0.0700 | +4 |
| 4 | `meta_alfabetizacao_municipio_2024` | 0.0122 | 5 | 0.1097 | +1 |
| 5 | `taxa_alfabetizacao_municipio_2023` | 0.0120 | 4 | 0.1155 | -1 |
| 6 | `pct_alfabetizados_municipio_2023` | 0.0120 | 3 | 0.1235 | -3 |
| 7 | `proficiencia_media_ponderada_2023` | 0.0108 | 2 | 0.1263 | -5 |
| 8 | `gap_para_meta_municipio_2024` | 0.0096 | 6 | 0.0892 | -2 |
| 9 | `percentual_participacao_municipio_2023` | 0.0070 | 9 | 0.0289 | +0 |
| 10 | `atingiu_meta_municipio_2024` | 0.0058 | 12 | 0.0220 | +2 |
| 11 | `idhm_longevidade` | 0.0052 | 10 | 0.0287 | -1 |
| 12 | `idhm` | 0.0050 | 11 | 0.0256 | -1 |
| 13 | `idhm_renda` | 0.0043 | 13 | 0.0206 | +0 |
| 14 | `total_alunos_municipio_2023` | 0.0029 | 14 | 0.0136 | +0 |
| 15 | `rede` | 0.0019 | 15 | 0.0055 | +0 |
| 16 | `gold_historico_disponivel` | 0.0004 | 16 | 0.0003 | +0 |

**Onde os dois métodos divergem** em duas ou mais posições:

- `sigla_uf`: 8º pela importância nativa contra 2º pelo SHAP
- `idhm_educacao`: 7º pela importância nativa contra 3º pelo SHAP
- `pct_alfabetizados_municipio_2023`: 3º pela importância nativa contra 6º pelo SHAP
- `proficiencia_media_ponderada_2023`: 2º pela importância nativa contra 7º pelo SHAP
- `gap_para_meta_municipio_2024`: 6º pela importância nativa contra 8º pelo SHAP
- `atingiu_meta_municipio_2024`: 12º pela importância nativa contra 10º pelo SHAP

Divergência é esperada e informativa. A importância nativa distribui o crédito entre preditores correlacionados de forma sensível à ordem em que as árvores os utilizam; o SHAP atribui contribuição marginal por predição. Como as seis features educacionais de 2023 são fortemente correlacionadas entre si, é justamente entre elas que a diferença aparece.

### O caso de `sigla_uf`

A divergência mais informativa é a da única feature puramente territorial: `sigla_uf` aparece em 8º lugar pela importância nativa e em 2º pelo SHAP, um salto de 6 posições.

A explicação é mecânica. A importância nativa é calculada sobre as colunas codificadas: cada UF vira uma coluna one-hot que isoladamente reduz pouca impureza, e a soma dessas parcelas subestima o quanto o território importa. O SHAP mede contribuição marginal por predição, então captura o efeito conjunto de saber em que UF o aluno está.

A consequência analítica é relevante: **um segundo método, independente, confirma a conclusão da auditoria de granularidade**. O modelo se apoia mais no território do que a importância nativa sugeria, o que reforça a leitura de que o produto é um instrumento de priorização territorial e não um diagnóstico individual.

## Correlação entre os rankings

A correlação de Spearman entre as duas ordenações é de
**0.8559**.

## Leitura

Nenhum dos dois rankings é causal. Ambos descrevem como o modelo usa as features, não
como a alfabetização é produzida no mundo. Uma feature com contribuição alta indica que
o modelo se apoia nela, e não que intervir sobre ela mude o desfecho.

Vale lembrar a auditoria de granularidade: alunos do mesmo município e rede recebem o
mesmo vetor de entrada, então as contribuições aqui descrevem o que separa **contextos
territoriais**, não o que separa crianças dentro de um mesmo contexto.
