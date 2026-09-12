# Definição de X, y, groups e split municipal

## Escopo e classificação das decisões

**Requisito do Tech Challenge:** uso da Gold da Fase 2, predição supervisionada
da alfabetização, atenção a leakage, separação adequada entre treino,
validação e teste e avaliação de generalização (PDF oficial, páginas 2–5,
lido na inspeção anterior do projeto).

**Decisão metodológica do grupo:** aluno avaliado em 2024 como unidade,
`id_municipio` como grupo, exclusões abaixo, X inicial com 16 candidatas,
reserva de peso amostral, proporções 70/15/15 e semente 42. O PDF não
prescreve município, GroupShuffleSplit ou essas proporções.

**Implementação apoiada em documentação externa:** GroupShuffleSplit do
Scikit-learn. Suas frações se referem a grupos, não a alunos. Não foi atribuída
à FIAP uma exigência de algoritmo ou biblioteca para o split.
Fonte: [documentação oficial do GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
Não se invocou material curricular adicional como justificativa.

Nenhum modelo, encoding, imputação, seleção estatística de features ou
pré-processamento aprendido foi implementado nesta etapa.

## Definições formais e classificação das 24 colunas

`y = alfabetizado`; `groups = id_municipio`;
`sample_weight = peso_aluno` apenas reservado, sem aplicação a estimador/métrica.
X contém as 16 candidatas explicitamente enumeradas em `get_modeling_columns()`.
Uma coluna pode ter papel principal e ressalva adicional (redundância/temporalidade).

| Coluna | Classificação principal | Uso e ressalva |
|---|---|---|
| `ano` | Identificador/rastreabilidade | Fora de X; constante 2024 |
| `id_aluno` | Identificador/rastreabilidade | Fora de X; chave individual única |
| `id_municipio` | Group | Fora de X; todas as redes de um município ficam juntas |
| `id_escola` | Identificador/rastreabilidade | Fora de X; evita memorizar entidades |
| `rede` | Feature categórica candidata | Em X |
| `peso_aluno` | Sample weight | Fora de X; reservado, sem reponderação nesta etapa |
| `alfabetizado` | Target | y; nunca em X |
| `id_municipio_nome` | Identificador/rastreabilidade | Fora de X |
| `taxa_alfabetizacao_municipio_2023` | Feature numérica candidata | Em X; alta correlação com percentual de alfabetizados |
| `media_portugues_municipio_2023` | Feature numérica candidata | Em X; alta correlação com proficiência ponderada |
| `percentual_participacao_municipio_2023` | Feature numérica candidata | Em X; missingness preservada |
| `total_alunos_municipio_2023` | Feature numérica candidata | Em X; tamanho do contexto histórico |
| `pct_alfabetizados_municipio_2023` | Feature potencialmente redundante | Em X, numérica; histórico válido, não target individual de 2024 |
| `proficiencia_media_ponderada_2023` | Feature potencialmente redundante | Em X, numérica; quase colinear com média de Português |
| `meta_alfabetizacao_municipio_2024` | Feature com possível risco de leakage | Em X, numérica; depende da disponibilidade da meta antes da previsão |
| `gap_para_meta_municipio_2024` | Feature potencialmente redundante | Em X, numérica derivada; herda a ressalva temporal da meta |
| `atingiu_meta_municipio_2024` | Feature potencialmente redundante | Em X, booleana com nulos; compara histórico de 2023 com meta de 2024 |
| `sigla_uf` | Feature categórica candidata | Em X; valores compartilhados entre municípios |
| `sigla_uf_nome` | Identificador/rastreabilidade | Fora de X; representação nominal redundante da UF |
| `idhm` | Feature numérica candidata | Em X; contexto estadual, correlação com componentes |
| `idhm_educacao` | Feature numérica candidata | Em X; avaliar redundância futuramente |
| `idhm_renda` | Feature numérica candidata | Em X; avaliar redundância futuramente |
| `idhm_longevidade` | Feature numérica candidata | Em X; avaliar redundância futuramente |
| `gold_historico_disponivel` | Feature de disponibilidade/qualidade | Em X; flag binária de correspondência Gold |

Portanto X tem 2 categóricas, 12 numéricas, 1 booleana e 1 flag de disponibilidade.
As categorias são semânticas: nenhum dtype, valor ou nulo foi transformado.
O peso permanece float32 da Silver. Sua adequação como peso de treinamento ou
avaliação ainda precisa ser discutida; os resultados deste relatório não são ponderados.

## Auditoria do target e dos municípios

O parquet real tem 1.851.828 linhas e 24 colunas, IDs individuais únicos,
zero target nulo e domínio {0, 1}. Distribuição: 1.107.103 classe 1
(59,7843%) e 744.725 classe 0 (40,2157%). Há 5.517 municípios, sem group nulo.

Alunos por município: mínimo 7; P25 53; mediana 110; P75 246;
P90 568; P99 3.757,32; máximo 94.373; média 335,66.
O município maior sozinho representa aproximadamente 5,10% dos alunos,
o que explica por que fracionar municípios não garante as mesmas frações de linhas.

Proporção de classe 1 por município: mínimo 0%; P25 49,16%; mediana 64,37%;
P75 77,91%; máximo 100%. A média não ponderada entre municípios é 63,28%,
diferente dos 59,78% calculados sobre alunos. Existem 58 municípios com
uma única classe: 1 somente classe 0 e 57 somente classe 1; todos preservados.

O manifesto local `data/processed/modeling_split_2024.json` lista, para cada
município, número de alunos, contagens 0/1, proporção de classe 1 e conjunto
atribuído. O relatório agregado está em `reports/modeling_split_summary.json`.

## Auditoria de leakage e redundância

A. **Leakage temporal real:** não foi detectado uso do resultado individual de
2024 nas features históricas examinadas. Target e identificadores não entram
em X. O fato de usar a Gold não garante sozinho ausência de vazamento:
revisões posteriores e data de publicação das fontes também importam.

B. **Derivação/redundância:** nas 1.764.731 linhas completas verificáveis,
`gap = meta_2024 - taxa_2023` coincide com tolerância absoluta 0,0001
(diferença máxima 0,0000073242); `atingiu_meta = taxa_2023 >= meta_2024`
coincide em 100%. Essas colunas não representam o atingimento observado com
o resultado de 2024. Permanecem candidatas; são derivadas de informações
já presentes em X. As 87.097 linhas sem valores completos não entram na comparação.

C. **Alta correlação:** Pearson em pares completos, ponderado implicitamente
pela repetição do contexto em cada aluno: taxa × percentual = 0,971168;
média Português × proficiência ponderada = 0,999321, ambos com 1.422.542 pares.
Isso sinaliza redundância e possível instabilidade em modelos futuros,
não prova leakage nem justifica remoção automática. IDHM e componentes também
permanecem candidatos conforme a EDA; nenhuma seleção foi realizada.

D. **Informação anterior ao target:** os builders usam indicadores educacionais
de 2023, enquanto y é de 2024. Metas são mantidas sob a hipótese documentada
do grupo de que eram conhecidas antes do momento da previsão. O snapshot S3
é de 2026; as fórmulas acima não comprovam a data original de publicação ou
a ausência de revisões históricas. Essa limitação também vale para a origem
socioeconômica. Antes de afirmar validade prospectiva, confirmar essas datas.

A flag Gold pode capturar padrões de cobertura da fonte. Não é derivada de y,
mas sua estabilidade fora da população atual precisa ser avaliada futuramente.

## Estratégia oficial e resultados

Dois GroupShuffleSplit com `n_splits=1`, `random_state=42` em ambos:
primeiro 70% treino / 30% temporário de municípios; depois metade do temporário
para validação e metade para teste. A ordem de linhas não muda a atribuição
dos códigos municipais. A validação sintética cobre essa propriedade.

Não se buscou a melhor semente, não se escolheu split pelo target e não se
usou split aleatório estratificado de alunos. O target foi consultado somente
para auditoria descritiva e validação de composição. Metas de tamanho são
aproximadas; critérios diagnósticos são desvio de até 5 p.p. no tamanho e
3 p.p. na classe 1. Violações geram avisos, sem ressorteio automático.

| Conjunto | Linhas | % total | Municípios | Classe 0 | % classe 0 | Classe 1 | % classe 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,311,003 | 70.7951% | 3,861 | 530,155 | 40.4389% | 780,848 | 59.5611% |
| validation | 297,079 | 16.0425% | 828 | 117,959 | 39.7063% | 179,120 | 60.2937% |
| test | 243,746 | 13.1625% | 828 | 96,611 | 39.6359% | 147,135 | 60.3641% |

A soma é 1.851.828 linhas. Interseções municipais train/validation,
train/test e validation/test: zero. Interseções de IDs individuais: zero.
Cada posição do dataset aparece uma única vez, sem perdas nem repetições.
Todos os conjuntos contêm ambas as classes. O maior desvio da classe 1 é
0,58 p.p.; os tamanhos são aceitáveis e não houve aviso de desequilíbrio.
As frações observadas são 70,80/16,04/13,16% dos alunos, não 70/15/15% exatos.

O teste mede generalização para municípios não vistos no mesmo ciclo de 2024,
com contexto histórico disponível. Não mede generalização para anos futuros,
UFs inteiramente novas ou cenários sem contexto Gold. Há variáveis estaduais
compartilhadas entre conjuntos; agrupar por município não elimina essa dependência.
O desequilíbrio de tamanho dos municípios também limita a precisão das métricas
futuras. A partir desta definição, reservar o teste para a avaliação final;
comparações e decisões de modelo deverão usar treino/validação.

## Reprodução e uso

Na raiz, usando a .venv existente:

```powershell
.\.venv\Scripts\python.exe -m src.modeling.split
.\.venv\Scripts\python.exe -m unittest tests.test_modeling_split
```

O comando lê exclusivamente o parquet local, valida o contrato, calcula o
split e grava o resumo e o manifesto municipal. Não salva cópias de X/y
nem altera o parquet. SHA-256 do dataset e versão do Scikit-learn ficam
registrados nos dois arquivos para rastreabilidade. O manifesto está na pasta
de dados ignorada pelo Git e permite recuperar a atribuição sem depender da
ordem das linhas. Regerá-lo pressupõe o mesmo artefato e a mesma implementação.

Uso das funções, sem treinamento:

```python
splits = split_by_municipality(dataset)
report = validate_split(dataset, splits)
assert report["valid"], report["errors"]
train = get_modeling_data(dataset.iloc[splits["train"]])
# train["X"], train["y"], train["groups"], train["sample_weight"]
```

Os índices são posições para `iloc`; o índice pandas original é preservado
na separação X/y/groups/peso. Nulos e redundâncias continuam presentes.
O consumidor deve conferir os avisos de composição antes de usar um novo split.

## Verificação executada

10 testes novos de split e separação passaram; 7 de Silver, 4 do reader legado
e 19 do contrato também passaram (40 testes no total). Cobertura inclui
sobreposição deliberada, linhas faltantes/repetidas, índices inválidos,
reprodução, ordem de linhas, independência da atribuição em relação a y,
separação alinhada de pesos e preservação de nulos. O split real foi validado
com sucesso e seu relatório não contém erros ou avisos.
