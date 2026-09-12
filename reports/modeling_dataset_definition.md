# Dataset de modelagem Gold vigente

## Origem e escopo

**Requisito do Tech Challenge:** o PDF oficial da Fase 3 (páginas 2 a 5)
exige uso dos dados da camada Gold da Fase 2, predição supervisionada da
alfabetização individual e tratamento de data leakage.

**Decisão específica do grupo:** integrar features Gold históricas de 2023
por município + rede aos alunos avaliados em 2024, nas redes Estadual e
Municipal. O artefato vigente é
`data/processed/modeling_dataset_2024_gold.parquet`.
O schema e as contagens abaixo são decisões/referências do projeto, não
quantidades prescritas pela FIAP.

Fluxo atual:

```text
Silver alunos (S3, 2026-07-09) -> silver_reader.py -> população 2024
Gold Fase 2 (S3, 2026-07-10)
  -> gold_reader.py
  -> gold_features.py / build_consolidated_gold_features
  -> build_modeling_dataset.py / build_modeling_dataset_from_gold + alunos 2024
  -> modeling_dataset_2024_gold.parquet
  -> dataset_contract.py -> validate_dataset.py -> dataset validado
```

O fluxo oficial é `python -m src.preprocessing.gold_pipeline`. A população
individual vem de `silver/alunos/processing_date=2026-07-09/alunos.parquet`
no bucket `tc-fase-dois-alfabetizacao-bronze`, lida por `silver_reader.py`.
São aplicados os filtros `ano == 2024`, `presenca == "Presente"`,
`preenchimento_caderno == "Prova preenchida"` e rede Estadual/Municipal,
antes de selecionar ano, IDs de aluno/município/escola, rede, peso e target.
O target é convertido pelo builder Gold existente, sem novas regras.
As features continuam vindo da Gold de `processing_date=2026-07-10`.

O notebook `00_test_gold_integration.ipynb` também usa o reader Silver.
`modeling_dataset_2024.parquet` e `src.preprocessing.pipeline` são legados
opcionais; não participam da geração oficial. O código BigQuery foi mantido.
Não são necessários GCP, OAuth ou BILLING_PROJECT_ID no fluxo oficial.
Configure as credenciais AWS de leitura em `.env` ou `env` na raiz
(variáveis já presentes no ambiente têm prioridade).

A Silver preserva o peso amostral em float32, conforme a transformação da
Fase 2. Não se afirma igualdade bit a bit com o antigo artefato BigQuery.

## Schema confirmado

Confirmado contra `gold_features.py`, `build_modeling_dataset_from_gold`,
a saída da célula 37 do notebook de integração e `df.info()` da célula 6 de
`notebooks/01_eda.ipynb` (índices de células iniciando em zero).
São **24 colunas**, sem divergência entre essas fontes.
`dataset_contract.py` é a definição formal consumida pelo validador.
A ordem física das colunas é livre; nomes ausentes, extras ou repetidos falham.
Os dtypes específicos de uma versão de pandas/Parquet não são fixados.

| Grupo | Colunas | Papel |
|---|---|---|
| Identificação | `ano`, `id_aluno`, `id_municipio`, `id_escola`, `id_municipio_nome`, `sigla_uf_nome` | Rastreabilidade |
| Target | `alfabetizado` | Classificação individual de 2024, domínio {0, 1} |
| Informação amostral | `peso_aluno` | Peso da observação |
| Contexto categórico | `rede`, `sigla_uf` | Rede e território |
| Histórico 2023 | `taxa_alfabetizacao_municipio_2023`, `media_portugues_municipio_2023`, `percentual_participacao_municipio_2023`, `total_alunos_municipio_2023`, `pct_alfabetizados_municipio_2023`, `proficiencia_media_ponderada_2023` | Indicadores educacionais Gold |
| Metas/contexto | `meta_alfabetizacao_municipio_2024`, `gap_para_meta_municipio_2024`, `atingiu_meta_municipio_2024` | Meta de 2024 comparada ao histórico de 2023 |
| Socioeconômicas | `idhm`, `idhm_educacao`, `idhm_renda`, `idhm_longevidade` | Indicadores estaduais Gold |
| Disponibilidade | `gold_historico_disponivel` | Correspondência no join Gold |

A presença no contrato não define a seleção final de features. A definição
inicial de X, y, groups, peso reservado e split foi formalizada em
[Definição de X/y/groups e split](modeling_split_definition.md), sem treinar modelos.

## Referências do artefato auditado

- 1.851.828 linhas e 24 colunas; uma linha por aluno avaliado em 2024.
- Zero IDs de aluno duplicados ou nulos; target sem nulos, apenas {0, 1}.
- 1.816.270 alunos com Gold (98,08%) e 35.558 sem Gold (1,92%).
- Target: 1.107.103 na classe 1 (59,78%) e 744.725 na classe 0 (40,22%).

`strict_counts=True` exige o total de linhas e as duas contagens de cobertura.
`strict_counts=False` dispensa somente essas contagens, mantendo schema,
quantidade de colunas, unicidade, target e coerência de missingness.
A proporção de classes é descritiva, não uma restrição de validação.

## Missingness e leakage

A flag é binária e não nula, obtida do indicador de correspondência do
left join, e não da completude de todas as features.

- Flag 0: todas as 16 colunas trazidas pela Gold devem estar nulas, inclusive
  nomes geográficos, UF, metas e IDHM. As chaves e os atributos individuais
  vêm do lado dos alunos e não são incluídos nessa regra.
- Flag 1: o núcleo auditado deve estar preenchido: taxa de alfabetização,
  média de Português, nome do município, sigla/nome da UF e os quatro IDHMs.
  Essa restrição formaliza o artefato auditado na EDA (células 10 e 15),
  e não uma garantia universal de qualquer fonte Gold futura.
- Participação, desempenho agregado e metas podem ser nulos com flag 1.
  A EDA registra 87.097 nulos em participação/metas e 429.286 em desempenho;
  essas contagens descritivas não são impostas como alinhamentos entre fontes.

Nenhuma imputação ou alteração de dados é executada pelo validador.
O schema exato rejeita também a inclusão de atributos individuais contemporâneos
além dos sete selecionados pelo builder. Isso não substitui uma auditoria da
origem dos valores. Conforme a EDA, gap e atingimento comparam a meta de 2024
à taxa de 2023; o contrato não inicia seleção definitiva ou treinamento.

## Validação local e testes

Com o parquet Gold disponível na pasta esperada, execute na raiz:

```bash
python -m src.preprocessing.validate_dataset
```

Esse comando apenas lê o parquet local e não precisa de credenciais AWS/GCP
nem reexecuta a origem. Ausência de arquivo ou violação do contrato produz
falha explícita, sem fallback para o legado. A reprodução exige somente
acesso autorizado de leitura às partições Silver e Gold no S3:

```bash
python -m src.preprocessing.gold_pipeline
python -m src.preprocessing.validate_dataset
python -m unittest tests.test_silver_reader
```

Os testes usam exclusivamente dados sintéticos em memória, inclusive a
integração entre os builders Gold existentes e o contrato:

```bash
python -m tests.test_dataset_contract
# Alternativamente, se pytest estiver instalado:
python -m pytest
```

Cobrem schema incompleto/extra/repetido, ordem livre, cada feature Gold,
ID duplicado/nulo, target inválido/nulo, flag inválida (sem coerção),
missingness nas duas direções, nulos parciais permitidos, preservação dos
dados e contagens estritas. O teste positivo estrito usa referências pequenas
substituídas apenas durante o teste; não fabrica o artefato real.

## Reprodução local verificada em 2026-09-08

O fluxo `gold_pipeline` gerou o parquet a partir da Silver e da Gold no S3,
sem GCP e sem o intermediário legado. A releitura pelo validador estrito
passou em schema, shape, unicidade, target e missingness Gold.
Foram confirmadas 1.851.828 linhas, 24 colunas, classes 1/0 com
1.107.103/744.725 registros e cobertura Gold 1/0 com 1.816.270/35.558.
Nenhuma divergência nas contagens do contrato. O peso permanece float32
proveniente da Silver; igualdade bit a bit com o parquet antigo não foi testada.

O parquet é local e não versionado. Em outro ambiente, execute a geração
e o validador; testes sintéticos sozinhos não validam um artefato real ausente.
