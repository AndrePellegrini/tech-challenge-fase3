# Validação cruzada agrupada

Fecha a lacuna da DEC-010. O holdout único não permitia dizer se as diferenças de ROC
AUC entre os finalistas eram reais ou ruído de partição.

## Método

`GroupKFold` com 5 folds, agrupado por `id_municipio`, sobre a união de treino
e validação. Nenhum município aparece em dois folds, de modo que cada fold mede
generalização territorial, exatamente como o holdout original.

**O conjunto de teste não foi materializado.** A verificação é explícita em código: os
índices de desenvolvimento são confrontados com os de teste e a execução falha se
houver interseção.

Foram avaliados os 4 finalistas, não os onze candidatos. Onze modelos por
5 folds seriam 55 ajustes, e só a Random Forest custa cerca de 123 segundos
por ajuste. Os finalistas cobrem a faixa de decisão real, que no holdout foi de 0,002
de ROC AUC entre primeiro e segundo colocado.

## Resultado

| Modelo | AUC média | Desvio | Mínimo | Máximo | AUC do holdout |
|---|---:|---:|---:|---:|---:|
| `random_forest_controlled` | 0.6608 | 0.0079 | 0.6523 | 0.6710 | 0.6409 |
| `logistic_baseline` | 0.6560 | 0.0146 | 0.6329 | 0.6705 | 0.6186 |
| `decision_tree_d5_l100` | 0.6512 | 0.0114 | 0.6351 | 0.6637 | 0.6307 |
| `decision_tree_d10_l100` | 0.6504 | 0.0083 | 0.6401 | 0.6625 | 0.6302 |

A diferença de 0.0047 entre `random_forest_controlled` e `logistic_baseline` **não supera** a dispersão combinada entre folds (0.0166). Os dois são estatisticamente indistinguíveis nesta evidência, e a escolha se justifica pelos critérios de desempate da DEC-012, não pela ROC AUC isolada.

A ordem dos finalistas **difere** da do holdout, que foi `random_forest_controlled` > `decision_tree_d5_l100` > `decision_tree_d10_l100` > `logistic_baseline`. A divergência é em si um resultado: indica que o holdout único era sensível à partição, exatamente o risco que motivou esta validação cruzada.

## O que isto explica sobre a generalização

Havia uma ressalva em aberto no projeto: a AUC de teste (0.6631) ficou
0.0222 acima da AUC do holdout de validação
(0.6409), diferença que o próprio protocolo classificou como moderada. A
leitura natural seria suspeitar de uma partição de teste favorável.

A validação cruzada mostra que a explicação é outra.

| Estimativa | ROC AUC |
|---|---:|
| Holdout de validação, partição única | 0.6409 |
| **Validação cruzada, média de 5 folds** | **0.6608** ± 0.0079 |
| Teste, abertura única | 0.6631 |

A média da validação cruzada fica a **0.0023** do teste, dentro de
um desvio-padrão entre folds. Já o holdout isolado ficou 0.0199
abaixo dessa média.

**A conclusão é que a partição de validação era pessimista, não que a de teste fosse
otimista.** A estimativa de generalização do modelo é mais bem representada pela média
da validação cruzada, e o resultado do teste está coerente com ela. Isso reforça, e não
enfraquece, a validade da avaliação final.

Registro metodológico: a validação cruzada foi executada sem qualquer acesso ao
conjunto de teste. A AUC de teste usada aqui vem de `final_test_metrics.json`, artefato
publicado na ocasião da abertura única, e a comparação é estritamente posterior.

## Por fold

| Modelo | Fold | ROC AUC | Alunos na validação | Municípios |
|---|---:|---:|---:|---:|
| `random_forest_controlled` | 1 | 0.6561 | 321.620 | 929 |
| `random_forest_controlled` | 2 | 0.6523 | 321.619 | 939 |
| `random_forest_controlled` | 3 | 0.6710 | 321.619 | 941 |
| `random_forest_controlled` | 4 | 0.6671 | 321.612 | 940 |
| `random_forest_controlled` | 5 | 0.6572 | 321.612 | 940 |
| `decision_tree_d5_l100` | 1 | 0.6351 | 321.620 | 929 |
| `decision_tree_d5_l100` | 2 | 0.6464 | 321.619 | 939 |
| `decision_tree_d5_l100` | 3 | 0.6637 | 321.619 | 941 |
| `decision_tree_d5_l100` | 4 | 0.6603 | 321.612 | 940 |
| `decision_tree_d5_l100` | 5 | 0.6503 | 321.612 | 940 |
| `decision_tree_d10_l100` | 1 | 0.6483 | 321.620 | 929 |
| `decision_tree_d10_l100` | 2 | 0.6401 | 321.619 | 939 |
| `decision_tree_d10_l100` | 3 | 0.6625 | 321.619 | 941 |
| `decision_tree_d10_l100` | 4 | 0.6538 | 321.612 | 940 |
| `decision_tree_d10_l100` | 5 | 0.6474 | 321.612 | 940 |
| `logistic_baseline` | 1 | 0.6540 | 321.620 | 929 |
| `logistic_baseline` | 2 | 0.6329 | 321.619 | 939 |
| `logistic_baseline` | 3 | 0.6705 | 321.619 | 941 |
| `logistic_baseline` | 4 | 0.6665 | 321.612 | 940 |
| `logistic_baseline` | 5 | 0.6563 | 321.612 | 940 |

## Leitura

O desvio entre folds é a informação que faltava. Ele mostra a margem dentro da qual
comparações de ROC AUC não são conclusivas, e justifica a regra de seleção adotada na
DEC-012, que não escolhe pelo maior AUC isolado e sim pelo menor gap entre treino e
validação dentro de uma faixa de equivalência.

Os valores aqui não substituem as métricas oficiais do projeto, que vêm do holdout
municipal e da abertura única do teste. Servem para qualificar a comparação entre
candidatos.
