# Validação final controlada

Foram comparados Dummy, três regressões logísticas, seis árvores e uma Random Forest controlada usando apenas treino e validação. O candidato `random_forest_controlled` foi escolhido entre modelos a até 0,002 da melhor AUC, priorizando menor gap, F1 e custo.

| Modelo | AUC treino | AUC validação | Gap | F1 validação | Recall | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0,5000 | 0,5000 | 0,0000 | 0,7523 | 1,0000 | 5,0 |
| Logística baseline | 0,6661 | 0,6186 | 0,0475 | 0,6819 | 0,7204 | 48,8 |
| Logística `C=0,1` | 0,6661 | 0,6182 | 0,0479 | 0,6820 | 0,7206 | 44,9 |
| Logística `C=0,3` | 0,6661 | 0,6184 | 0,0477 | 0,6819 | 0,7204 | 45,6 |
| Árvore baseline, depth 14 | 0,6817 | 0,6214 | 0,0603 | 0,7298 | 0,8439 | 61,4 |
| Árvore depth 5, folha 100 | 0,6618 | 0,6307 | 0,0310 | 0,7438 | 0,8905 | 10,4 |
| Árvore depth 7, folha 100 | 0,6686 | 0,6294 | 0,0392 | 0,7446 | 0,8919 | 17,0 |
| Árvore depth 10, folha 100 | 0,6751 | 0,6302 | 0,0449 | 0,7395 | 0,8735 | 31,9 |
| Árvore depth 10, folha 500 | 0,6743 | 0,6301 | 0,0442 | 0,7403 | 0,8722 | 30,4 |
| Árvore depth 15, folha 500 | 0,6792 | 0,6285 | 0,0507 | 0,7352 | 0,8541 | 56,2 |
| Random Forest controlada | 0,6752 | 0,6409 | 0,0343 | 0,7462 | 0,8855 | 123,4 |

A interpretação inicial do threshold 0,35 estava incorreta. O score é
`P(alfabetizado)` e as métricas padrão usam classe 1 = alfabetizado. Reduzir o
limiar aumenta previsões de alfabetizado, eleva recall da classe 1 e reduz recall
da classe 0, que representa risco. A análise foi refeita para as duas classes.
O uso de peso foi rejeitado: exige ganho de AUC superior a 0,002 e seu significado
segue **REVISAR COM O GRUPO**.

No limiar 0,35, a classe 0 tem recall 0,0423: 112.968 dos 117.959 casos
de risco são classificados como alfabetizados. Em 0,50, o recall de risco sobe
para 0,2591, ainda com 87.391 erros perigosos. Em 0,60, balanced accuracy chega
a 0,5901, recall de risco a 0,7381 e precision de risco a 0,4656; restam 30.893
casos de risco classificados como alfabetizados e surgem 99.925 falsos alertas.

Decisão corrigida: não congelar threshold binário. Para triagem territorial,
ordenar por `prob_risco = 1 - P(alfabetizado)` é mais transparente e preserva
informação. O limiar 0,60 é apenas referência para um objetivo de equilíbrio ou
alta sensibilidade de risco, sujeito à definição do custo operacional pelo grupo.

O experimento ponderado obteve AUC 0,6405, F1 0,7451, recall 0,8833 e
precision 0,6443, abaixo da versão sem peso. Além disso, o significado e o
desenho amostral de `peso_aluno` não estão documentados o suficiente para
recomendá-lo oficialmente.

A Random Forest foi limitada a 40 árvores, profundidade 10, folha mínima 200 e dois jobs. GroupKFold não foi executado pelo custo em 1,85 milhão de linhas. SHAP não foi executado porque a dependência está ausente; não houve instalação.

Feature importance é associativa, agregada para as 16 colunas originais e não causal. A análise municipal/rede é ferramenta de priorização analítica.

As cinco maiores importâncias da floresta foram média de Português 2023
(0,1578), proficiência média ponderada 2023 (0,1263), percentual alfabetizado
2023 (0,1235), taxa de alfabetização 2023 (0,1155) e meta 2024 (0,1097).
Comparado à árvore baseline, o ranking teve Spearman 0,6265 e duas features em
comum no top 5 (`media_portugues_municipio_2023` e
`pct_alfabetizados_municipio_2023`): estabilidade apenas moderada.

**TEST SET NÃO ACESSADO.** Nenhuma seleção, fit, transformação, predição, SHAP ou métrica usou o teste.
