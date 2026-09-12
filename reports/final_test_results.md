# Avaliação final única no teste

Abertura registrada em `2026-09-10T01:41:51.457316+00:00`. Foi aplicado o pipeline Random Forest congelado, treinado somente em TRAIN, sem refit. O threshold 0,50 é referência descritiva; não é decisão operacional.

Teste: 243,746 alunos, 828 municípios; classe 0: 96,611 (39.64%); classe 1: 147,135 (60.36%).

AUC teste=0.6631; validação=0.6409; delta=+0.0222, diferença moderada. Em 0,50: accuracy=0.6331, balanced accuracy=0.5832; risco precision/recall/F1=0.5609/0.3426/0.4254; alfabetizado precision/recall/F1=0.6562/0.8239/0.7305. Matriz [linhas verdadeiras, colunas previstas 0/1]: `[[33100, 63511], [25914, 121221]]`.

Overlaps de município teste/treino e teste/validação: 0/0. Overlaps de aluno: 0/0. O resultado mede generalização territorial para municípios inéditos.

## Contextos com maior risco relativo

| Município | UF | Rede | Alunos | Prob. risco | Histórico 2023 | Meta | Gap |
|---|---|---|---:|---:|---:|---:|---:|
| Floresta Azul | BA | Municipal | 78 | 0.7358 | 15.1200 | 21.7400 | 6.6200 |
| Senhor do Bonfim | BA | Municipal | 487 | 0.7095 | 19.9300 | 27.0100 | 7.0800 |
| Araci | BA | Municipal | 331 | 0.7074 | 20.2700 | 27.3700 | 7.1000 |
| Arataca | BA | Municipal | 61 | 0.7046 | 16.8600 | 23.6900 | 6.8300 |
| Nossa Senhora de Lourdes | SE | Municipal | 60 | 0.7029 | 19.1500 | 26.1800 | 7.0300 |
| Pedrão | BA | Municipal | 30 | 0.6995 | 20.7200 | 27.8500 | 7.1300 |
| Pacatuba | SE | Municipal | 106 | 0.6991 | 16.9000 | 23.7400 | 6.8400 |
| Itiruçu | BA | Municipal | 60 | 0.6945 | 32.8600 | 39.7900 | 6.9300 |
| Boquim | SE | Municipal | 142 | 0.6941 | 22.7600 | 29.9600 | 7.2000 |
| Canudos | BA | Municipal | 156 | 0.6940 | 18.8500 | 25.8600 | 7.0100 |

Ranking associativo para priorização analítica; não prevê oficialmente atingimento de meta e não é diagnóstico individual.

## Limitações

O target é individual, mas as features são contextuais; alunos do mesmo município+rede compartilham X. Cerca de 90% da variação observada estava dentro dos contextos. O enriquecimento escolar falhou pela anonimização de `id_escola`; `proficiencia` foi excluída por leakage direto. As associações não são causais. Feature importance permaneceu a do modelo congelado.

**Nenhum retuning ocorreu após a abertura do teste.**
