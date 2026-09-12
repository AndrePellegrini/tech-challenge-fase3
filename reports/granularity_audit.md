# Auditoria de granularidade das features

## Escopo

O target individual é requisito do Tech Challenge. O uso da Gold contextual é
requisito/arquitetura atual. Esta auditoria é uma decisão metodológica do grupo
e uma extensão diagnóstica; não é atribuída como técnica específica ensinada
pela FIAP. Nenhum modelo foi treinado, o teste permaneceu fechado e o X não foi
alterado.

## X oficial bruto (16 features)

| Feature | Granularidade conceitual |
|---|---|
| `rede` | rede |
| `sigla_uf` | UF |
| `taxa_alfabetizacao_municipio_2023` | município + rede |
| `media_portugues_municipio_2023` | município + rede |
| `percentual_participacao_municipio_2023` | município + rede |
| `total_alunos_municipio_2023` | município + rede |
| `pct_alfabetizados_municipio_2023` | município + rede |
| `proficiencia_media_ponderada_2023` | município + rede |
| `meta_alfabetizacao_municipio_2024` | município + rede |
| `gap_para_meta_municipio_2024` | município + rede (derivada) |
| `idhm` | UF |
| `idhm_educacao` | UF |
| `idhm_renda` | UF |
| `idhm_longevidade` | UF |
| `atingiu_meta_municipio_2024` | município + rede (derivada) |
| `gold_historico_disponivel` | disponibilidade do contexto município + rede |

## Perfis idênticos

Há 6,430 perfis de X entre 1,851,828 alunos.
Perfis duplicados representam 100.00% dos
alunos. O tamanho médio é 288.00, mediana
93, p90 505,
p95 975, p99 3181
e máximo 52,622.

| Alunos por perfil | Perfis | Alunos | % alunos |
|---|---:|---:|---:|
| 1 | 0 | 0 | 0.00% |
| 2-5 | 1 | 5 | 0.00% |
| 6-10 | 62 | 575 | 0.03% |
| 11-50 | 1,843 | 55,340 | 2.99% |
| 51-100 | 1,485 | 108,256 | 5.85% |
| 101-500 | 2,392 | 524,619 | 28.33% |
| >500 | 647 | 1,163,033 | 62.80% |

## Conflito do target

| Tipo | Perfis | Alunos | % alunos |
|---|---:|---:|---:|
| mixed | 6,350 | 1,848,507 | 99.82% |
| pure_0 | 2 | 18 | 0.00% |
| pure_1 | 78 | 3,303 | 0.18% |

Perfis mistos contêm 99.82% dos
alunos. Pessoas com X exatamente igual frequentemente possuem targets distintos;
um classificador determinístico baseado somente nesse X não consegue separá-las.

## Município + rede

Existem 6,535 grupos município+rede. 98.71% dos
grupos e 99.82% dos alunos estão em grupos com as
duas classes. 100.00% desses grupos têm um
único perfil X; 0 têm múltiplos
perfis. Portanto, município+rede determina X em 100% dos casos observados. A
relação inversa não é sempre única: 2
perfis são compartilhados por mais de um contexto, pois contextos diferentes
podem ter os mesmos valores agregados, inclusive o padrão sem histórico Gold.

## Previsões na validação

Sem retreino, os pipelines existentes produziram probabilidades idênticas para
todos os alunos do grupo em 100.00%
dos grupos na regressão e 100.00%
na árvore. Isso afeta, respectivamente,
100.00% e
100.00% dos alunos da validação.

## Limite empírico no treino

Escolher a classe majoritária dentro de cada perfil do próprio treino produz
accuracy descritiva de 0.6482.
O erro conflitante mínimo é 461,197
(35.18%). Esse valor é
in-sample e usa o target para descrever ambiguidade; não é estimativa de
generalização nem teto de AUC em municípios inéditos.

## Variabilidade do target

Por município, 9.60% da variância é
entre grupos e 90.40% permanece dentro
dos municípios. Por município+rede, são
10.17% entre grupos e
89.83% dentro dos grupos.

## Interpretação

A tarefa formal continua sendo prever `alfabetizado` por aluno, mas a informação
disponível é quase inteiramente territorial e de rede. A leitura mais adequada é
uma probabilidade individual condicionada ao contexto, útil sobretudo para
identificar territórios/redes de maior risco. Ela não distingue adequadamente
alunos do mesmo contexto e não substitui diagnóstico pedagógico individual.

A auditoria fornece explicação plausível, não prova exclusiva, para AUCs próximas
de 0,62, pequeno ganho da árvore e dificuldade de generalização municipal. A
formulação permanece válida para o desafio se essa limitação ficar explícita.
Recomenda-se manter target e fluxo atuais, ajustar a interpretação e marcar o
enriquecimento futuro com features individuais como **REVISAR COM O GRUPO**.
