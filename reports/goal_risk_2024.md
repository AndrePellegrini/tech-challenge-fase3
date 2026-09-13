# Risco de não atingimento da meta municipal de 2024

Responde à pergunta de negócio *"como prever municípios que podem não atingir metas
futuras"*.

## Método

A probabilidade média de alfabetização prevista pelo modelo congelado é usada como
**taxa municipal implícita** para 2024 e comparada com a meta do município. Quando fica
abaixo da meta, o município recebe alerta de risco de não atingimento.

A leitura como taxa depende de as probabilidades estarem calibradas, o que foi
verificado separadamente: viés de -0,0051 e desvio por decil abaixo de 0,0255 no nível
municipal. Ver [Calibração da taxa municipal](calibration_municipal.md).

A base é `reports/final_test_municipal_analysis.csv`, restrita aos
797 municípios do conjunto de teste que possuem meta
publicada. São municípios **inéditos** para o modelo: não aparecem no treino nem na
validação. Dos 828 municípios do teste,
31 foram descartados por não terem meta.

## Resultado da projeção

351 municípios (44.0%)
receberam alerta de não atingimento. O gap projetado mediano entre meta e projeção é
de -0.98 pontos percentuais.

Na realidade observada em 2024, 354 municípios
(44.4%) de fato não atingiram a meta.

## Qualidade do alerta

Como o conjunto de teste carrega a taxa efetivamente observada em 2024, o alerta pode
ser auditado contra o desfecho real. São usadas duas referências: a **classe
majoritária**, que nunca alerta ninguém, e a **ingênua**, que supõe a repetição da taxa
de 2023.

| Estratégia | Acurácia | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Projeção do modelo | **0.7026** | 0.6667 | 0.6610 | 0.6638 |
| Referência: classe majoritária | 0.5558 | 0.0000 | 0.0000 | 0.0000 |
| Referência: repetir 2023 | 0.4780 | 0.4506 | 0.7994 | 0.5764 |

**O ganho real do modelo é de +0.1468 sobre a classe majoritária**, e é esse o
número que deve ser citado. A classe majoritária aposta que todos os municípios atingem
a meta: acerta 55.6% sem modelo algum, e tem precisão e F1 iguais a
zero por construção, porque nunca emite um alerta.

A baseline ingênua merece atenção: com 0.4780 ela é **pior que a classe
majoritária**. Ela sobre-alerta, com recall de 0.7994 e precisão de apenas
0.4506 — sinaliza quase todo mundo e por isso quase não informa.
Comparar o modelo apenas contra ela inflaria o ganho aparente.

Matriz do alerta do modelo: 234 alertas corretos,
117 alarmes falsos, 120 municípios em
risco que passaram despercebidos e 326 corretamente
liberados.

## Municípios prioritários

Os quinze com maior gap projetado para a meta. A última coluna mostra a taxa
efetivamente observada, para conferência.

| # | Município | UF | Alunos | Meta 2024 | Projeção | Gap projetado | Observado |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Palmitinho | RS | 89 | 80.00 | 55.30 | +24.70 | 71.91 |
| 2 | Igrejinha | RS | 375 | 80.00 | 55.35 | +24.65 | 47.73 |
| 3 | Minas do Leão | RS | 85 | 74.31 | 51.91 | +22.40 | 56.47 |
| 4 | Passa Sete | RS | 16 | 80.00 | 58.10 | +21.90 | 75.00 |
| 5 | Erebango | RS | 29 | 74.36 | 52.83 | +21.53 | 44.83 |
| 6 | Três Cachoeiras | RS | 114 | 80.00 | 58.54 | +21.46 | 53.51 |
| 7 | Vale do Sol | RS | 46 | 80.00 | 58.57 | +21.43 | 56.52 |
| 8 | Senador Salgado Filho | RS | 24 | 80.00 | 58.59 | +21.41 | 50.00 |
| 9 | Pinhal da Serra | RS | 17 | 75.04 | 53.89 | +21.15 | 35.29 |
| 10 | Porto Lucena | RS | 32 | 78.78 | 57.64 | +21.14 | 84.38 |
| 11 | Venâncio Aires | RS | 487 | 78.41 | 57.68 | +20.73 | 61.19 |
| 12 | Sananduva | RS | 160 | 75.28 | 55.10 | +20.18 | 53.75 |
| 13 | Sede Nova | RS | 18 | 80.00 | 60.12 | +19.88 | 38.89 |
| 14 | Capitão | RS | 33 | 80.00 | 60.17 | +19.83 | 57.58 |
| 15 | Aratiba | RS | 60 | 80.00 | 60.33 | +19.67 | 40.00 |


**Leitura obrigatória do ranking.** 15 dos 15 municípios do topo são de RS. Isso não indica erro: a meta média em RS é de 71.8%, contra 61.8% no conjunto avaliado. Como o gap é medido em pontos percentuais absolutos, UFs que fixaram metas mais ambiciosas concentram os maiores gaps mesmo quando partem de patamares melhores. Para priorização orçamentária, combine este ranking com o de risco absoluto em `final_test_municipal_analysis.csv`, que ordena por probabilidade de risco e não depende da ambição da meta estadual.

Ranking completo em `goal_risk_ranking.csv`; números consolidados em
`goal_risk_summary.json`.

## Limitações

O alerta é uma projeção associativa, não uma previsão oficial de cumprimento de meta.
É uma análise pós-hoc sobre municípios inéditos, e não um exercício de previsão temporal
para anos futuros. A taxa implícita é a média das probabilidades individuais dentro do
município, e o modelo é predominantemente territorial: alunos do mesmo município e
rede recebem a mesma probabilidade. A projeção herda, portanto, todas as limitações
registradas em `modeling_limitations.md`.

A comparação com a linha de base ingênua mede apenas a capacidade de ordenar e
sinalizar municípios, não de estimar a magnitude exata da taxa futura. Municípios sem
meta publicada ficam fora da análise por construção.
