# Perfis territoriais por K-means

## Classificação curricular

K-means, clustering não supervisionado e Silhouette Score são conteúdos curriculares explícitos FIAP. Features, faixa de k, regras de agregação/seleção e interpretação são decisões metodológicas do grupo. O uso territorial é aplicação analítica do grupo; PCA, estabilidade e relação pós-hoc com risco são extensões.

## Método

Uma linha por município (5,517). Redes foram agregadas por média ponderada pelo número de alunos. Missing foi imputado por mediana e as cinco features foram padronizadas com StandardScaler antes do K-means: `taxa_alfabetizacao_municipio_2023`, `media_portugues_municipio_2023`, `percentual_participacao_municipio_2023`, `meta_alfabetizacao_municipio_2024`, `idhm`. Target, IDs, UF, rede e probabilidades supervisionadas ficaram fora do treinamento.

Foram avaliados k=2..6. Escolheu-se k=2 com Silhouette amostral 0.3591 (3.000 municípios, seed 42), conciliando score, interpretabilidade e tamanho mínimo. O K-means foi ajustado nos 5.517 municípios.

## Perfis encontrados

- Cluster 0 — **melhor contexto educacional e socioeconômico relativo**: 3371 municípios (61.1%); risco médio pós-hoc 0.2936.
- Cluster 1 — **maior vulnerabilidade educacional relativa**: 2146 municípios (38.9%); risco médio pós-hoc 0.4890.

Municípios próximos aos centroides:
- Cluster 0: Coqueiral (MG), Espera Feliz (MG), Resplendor (MG), Rio Manso (MG), Patos de Minas (MG)
- Cluster 1: Galinhos (RN), São Miguel (RN), Tenente Laurentino Cruz (RN), Pendências (RN), Batalha (PI)

Os clusters diferem principalmente em desempenho histórico, Português, participação, meta e IDHM. A relação com risco foi calculada somente depois dos clusters usando as previsões finais já registradas; não entrou no K-means.

## Resposta ao Tech Challenge

A análise identificou 2 perfis territoriais entre os municípios. O primeiro reúne
taxa histórica, Português, participação, meta e IDHM relativamente maiores; o
segundo reúne valores relativamente menores nessas dimensões. Municípios do mesmo
cluster compartilham padrões contextuais semelhantes; isso não implica causalidade
ou equivalência completa.

Como aplicação analítica do grupo, o perfil de maior vulnerabilidade pode apoiar
priorização de acompanhamento e recursos; o perfil relativamente mais favorável
pode apoiar benchmarking cuidadoso de práticas.

PCA explica 82.2% e foi usada apenas para visualização. A estabilidade por sementes está no JSON. O classificador, dataset e protocolo final não foram alterados; o teste não foi reavaliado.
