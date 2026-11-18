# Guia de Anotação — EvalBR (Triagem de Reclamações PT-BR)

Este guia define as regras de rotulagem para construir o conjunto *ground truth* a partir de avaliações reais em português (ex.: Olist `olist_order_reviews_dataset.csv`).

## 1. Categorias (`categoria`)

Escolha **exatamente uma** categoria principal (a que melhor explica o motivo central do comentário):

| Categoria | Definição | Exemplos |
|---|---|---|
| `atraso_entrega` | Reclamação sobre demora, prazo estourado, pedido não entregue ou rastreio parado. | *"Comprei dia 10 e até hoje nada."* / *"Chegou depois do prazo."* |
| `produto_errado_ou_faltando` | Item diferente do pedido, cor/tamanho errado, pedido incompleto ou caixa vazia. | *"Pedi tamanho M e veio G."* / *"Faltou um item do kit."* |
| `defeito_ou_qualidade` | Produto quebrado, danificado, com defeito de funcionamento ou qualidade muito abaixo do esperado. | *"Veio rachado."* / *"Não liga, material frágil."* |
| `atendimento` | Reclamação sobre suporte, pós-venda, falta de resposta ou dificuldade de cancelamento/estorno. | *"Mandei mensagem e ninguém respondeu."* |
| `elogio` | Avaliação positiva sem reclamação relevante. | *"Tudo certo, chegou rápido e bem embalado."* |
| `outro` | Comentário genérico, ambíguo ou fora das categorias acima. | *"Ok."* / *"Nada a declarar."* |

**Regra de desempate:** se o cliente reclamar de *atraso* **e** *produto quebrado*, priorize o problema de **maior impacto operacional** na triagem (`produto_errado_ou_faltando` > `defeito_ou_qualidade` > `atraso_entrega` > `atendimento`).

## 2. Sentimento (`sentimento`)
- `negativo`: insatisfação explícita ou implícita.
- `neutro`: comentário puramente descritivo ou misto equilibrado.
- `positivo`: satisfação clara.

## 3. Menção a Prazo (`menciona_prazo`)
- `true` se o texto mencionar prazo, entrega rápida/atrasada, data prevista ou demora; `false` caso contrário.

## 4. Ação Desejada (`acao_desejada`)
- `reembolso`: pede dinheiro de volta, estorno ou cancelamento.
- `troca`: pede substituição do item ou reenvio do item correto.
- `entrega`: cobra o recebimento de algo ainda não entregue.
- `nenhuma`: elogio ou relato sem pedido de ação.
- `nao_informado`: reclamação clara, porém sem explicitar qual resolução o cliente espera.

## 5. Evidência Literal (`evidencia`)
- Substring **exata** copiada do texto original que justifica a `categoria` escolhida.
- Qualquer saída cujo campo `evidencia` não seja substring literal do comentário (após normalização simples de espaços) é contabilizada como **alucinação**.

## 6. Protocolo de Concordância (κ de Cohen)
- Re-rotular às cegas **100 itens** após 7 dias (intra-anotador) ou usar um segundo anotador independente.
- Meta mínima de qualidade do *ground truth*: **κ ≥ 0,75**.
