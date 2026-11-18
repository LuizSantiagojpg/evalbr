# EvalBR — Avaliação Estatística de LLMs em Português

Benchmark rigoroso de LLMs para **triagem automática de reclamações de e-commerce e delivery em PT-BR**, combinando **engenharia de avaliação** (schemas Pydantic, cache determinístico, retries, checagem determinística de alucinação, custo e latência) com **rigor estatístico** (análise de poder, intervalos de confiança por bootstrap, teste pareado de McNemar, correção de Holm-Bonferroni, κ de Cohen e fronteira de Pareto).

## Arquitetura

```text
evalbr/
├── docs/annotation_guide.md      # Codebook formal de rotulagem + protocolo de κ
├── src/evalbr/
│   ├── schema.py                 # Modelos Pydantic v2 (ReviewLabel, SystemSpec, EvalRecord)
│   ├── prompts.py                # Templates versionados (zero_shot, few_shot, structured) + ruído PT-BR
│   ├── cache.py                  # Cache SQLite determinístico por hash de prompt e entrada
│   ├── runners/
│   │   ├── base.py               # Retries, extração JSON, validação Pydantic, custo e latência
│   │   ├── openai_compat.py      # OpenAI API e LM Studio local (ex.: Qwen)
│   │   ├── gemini.py             # Google Gemini (google-genai)
│   │   └── mock.py               # Simulador determinístico para CI e testes offline
│   ├── metrics.py                # Acurácia, Macro-F1, JSON válido, alucinação literal, p50/p95, $/1k
│   ├── stats.py                  # Poder amostral, Bootstrap CI, McNemar, Holm-Bonferroni, κ, Pareto
│   ├── data.py                   # Loader estratificado da Olist + gerador sintético PT-BR
│   ├── pipeline.py               # Matriz de experimentos, gráficos e memo de decisão
│   └── cli.py                    # CLI (`evalbr power`, `evalbr demo`)
├── app/dashboard.py              # Dashboard Streamlit para inspeção de resultados
└── tests/                        # Suíte de testes unitários e de integração
```

## Decisões Estatísticas (Resumo Técnico)

1. **Por que McNemar e não teste *t*?** Os modelos respondem exatamente aos mesmos itens (*design pareado*) e a saída por item é binária (acertou/errou). O teste de McNemar compara apenas os pares discordantes; com `< 25` discordantes usa teste binomial exato.
2. **Por que Holm-Bonferroni?** Ao comparar `k` configurações contra uma referência, o erro Tipo I infla. Holm-Bonferroni controla o FWER com mais poder estatístico que Bonferroni simples.
3. **Por que alucinação determinística?** O schema exige `evidencia` como **substring literal** do comentário original, permitindo auditar fidelidade sem depender de outro LLM.

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Calcular tamanho de amostra (análise de poder)
evalbr power --discordant-rate 0.15 --delta 0.05

# 2. Rodar o benchmark completo offline (gera JSON, gráficos e memo de decisão)
evalbr demo --samples 420 --output-dir reports

# 3. Abrir o dashboard interativo
streamlit run app/dashboard.py

# 4. Rodar os testes
pytest
```
