# Memo de Decisão — Triagem de Reclamações PT-BR (EvalBR)

**Amostra avaliada:** `n = 420` (análise de poder para Δ=5 p.p., α=0,05, poder=80% exige `n ≥ 469`).

## Recomendação de Produção
- **Sistema recomendado:** `gemini-2.5-flash__structured`
- **Macro-F1:** `0.9087` (IC 95%: `[0.8761, 0.9381]`) vs. referência `gemini-2.5-pro__structured` com `0.9328` (IC 95%: `[0.9050, 0.9569]`).
- **Economia & Operação:** custo estimado de **US$ 0.0909 / 1k reqs** (vs. US$ 1.0327 na referência), taxa de JSON válido `100.0%` e alucinação de evidência `3.6%`.
- **Fronteira de Pareto:** `qwen-local-4b__structured, gemini-2.5-flash__structured, gemini-2.5-pro__structured`

## Tabela Consolidada
| Sistema | Macro-F1 (IC 95%) | Acurácia | JSON Válido | Alucinação | Custo / 1k (USD) |
|---|---:|---:|---:|---:|---:|
| `gemini-2.5-pro__structured` | 0.933 [0.905, 0.957] | 0.938 | 100.0% | 2.4% | $1.0327 |
| `gemini-2.5-flash__structured` | 0.909 [0.876, 0.938] | 0.919 | 100.0% | 3.6% | $0.0909 |
| `gemini-2.5-pro__few_shot` | 0.875 [0.839, 0.906] | 0.883 | 100.0% | 1.9% | $0.9265 |
| `qwen-local-4b__structured` | 0.874 [0.837, 0.906] | 0.883 | 100.0% | 2.1% | $0.0000 |
| `gemini-2.5-flash__few_shot` | 0.851 [0.814, 0.885] | 0.862 | 100.0% | 1.4% | $0.0782 |
| `qwen-local-4b__few_shot` | 0.788 [0.746, 0.827] | 0.798 | 100.0% | 2.4% | $0.0000 |
| `gemini-2.5-pro__zero_shot` | 0.756 [0.708, 0.797] | 0.769 | 100.0% | 7.9% | $0.7900 |
| `gemini-2.5-flash__zero_shot` | 0.695 [0.647, 0.736] | 0.702 | 100.0% | 8.8% | $0.0616 |
| `qwen-local-4b__zero_shot` | 0.653 [0.601, 0.697] | 0.681 | 100.0% | 9.5% | $0.0000 |
