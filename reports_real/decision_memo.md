# Memo de Decisão — Triagem de Reclamações PT-BR (EvalBR)

**Amostra avaliada:** `n = 24` (poder para Δ=5 p.p., α=0,05, 80% exige `n ≥ 469`).

## Recomendação de Produção
- **Sistema recomendado:** `dolphin3-8b__structured`
- **Macro-F1:** `0.8333` (IC 95%: `[0.6667, 0.8333]`) vs. referência `llama3.1-8b__structured` com `0.8333` (IC 95%: `[0.6667, 0.8333]`).
- **Economia & Operação:** custo estimado de **US$ 0.0000 / 1k reqs** (vs. US$ 0.0000 na referência), JSON válido `100.0%` e alucinação `16.7%`.
- **Fronteira de Pareto:** `llama3.1-8b__structured, dolphin3-8b__structured, qwen3-vl-4b__structured`

## Tabela Consolidada
| Sistema | Macro-F1 (IC 95%) | Acurácia | JSON Válido | Alucinação | Custo / 1k (USD) |
|---|---:|---:|---:|---:|---:|
| `llama3.1-8b__structured` | 0.833 [0.667, 0.833] | 1.000 | 100.0% | 0.0% | $0.0000 |
| `dolphin3-8b__structured` | 0.833 [0.667, 0.833] | 1.000 | 100.0% | 16.7% | $0.0000 |
| `qwen3-vl-4b__structured` | 0.833 [0.667, 0.833] | 1.000 | 100.0% | 0.0% | $0.0000 |
| `llama3.1-8b__zero_shot` | 0.574 [0.345, 0.643] | 0.667 | 66.7% | 33.3% | $0.0000 |
