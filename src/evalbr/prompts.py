"""Versioned prompt templates, hashing, and PT-BR noise injection for robustness eval."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata

from evalbr.schema import PromptStrategy, ReviewLabel

PROMPT_VERSION = "v1.0.0"

_SYSTEM_INSTRUCTIONS = """Você é um especialista em triagem de atendimento de e-commerce e delivery.
Classifique o comentário do cliente exatamente no JSON especificado.

Categorias válidas:
- atraso_entrega: demora, prazo estourado, pedido não entregue, rastreio parado.
- produto_errado_ou_faltando: item diferente do pedido, tamanho/cor errada, item faltando.
- defeito_ou_qualidade: produto quebrado, danificado, com defeito ou má qualidade.
- atendimento: suporte ruim, falta de resposta, dificuldade em cancelar/estornar.
- elogio: avaliação positiva sem reclamação relevante.
- outro: comentário genérico ou ambíguo.

Regras:
1. O campo `evidencia` DEVE ser uma cópia LITERAL (substring exata) de um trecho do comentário.
2. Retorne APENAS um objeto JSON válido, sem markdown ou texto extra."""

_FEW_SHOT_EXAMPLES: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "Comprei dia 5 e até hoje não chegou, já passou muito do prazo.",
        {
            "categoria": "atraso_entrega",
            "sentimento": "negativo",
            "menciona_prazo": True,
            "acao_desejada": "entrega",
            "evidencia": "até hoje não chegou, já passou muito do prazo",
        },
    ),
    (
        "Veio faltando o carregador dentro da caixa, quero meu dinheiro de volta.",
        {
            "categoria": "produto_errado_ou_faltando",
            "sentimento": "negativo",
            "menciona_prazo": False,
            "acao_desejada": "reembolso",
            "evidencia": "Veio faltando o carregador",
        },
    ),
    (
        "O fone veio com o lado direito mudo, preciso trocar.",
        {
            "categoria": "defeito_ou_qualidade",
            "sentimento": "negativo",
            "menciona_prazo": False,
            "acao_desejada": "troca",
            "evidencia": "veio com o lado direito mudo",
        },
    ),
    (
        "Tentei falar no chat três vezes e ninguém respondeu.",
        {
            "categoria": "atendimento",
            "sentimento": "negativo",
            "menciona_prazo": False,
            "acao_desejada": "nao_informado",
            "evidencia": "ninguém respondeu",
        },
    ),
    (
        "Chegou antes do prazo e muito bem embalado, recomendo!",
        {
            "categoria": "elogio",
            "sentimento": "positivo",
            "menciona_prazo": True,
            "acao_desejada": "nenhuma",
            "evidencia": "Chegou antes do prazo e muito bem embalado",
        },
    ),
)


def build_messages(review_text: str, strategy: PromptStrategy) -> list[dict[str, str]]:
    """Build chat messages for a given prompting strategy."""
    system_parts = [_SYSTEM_INSTRUCTIONS]

    if strategy == "structured":
        schema_json = json.dumps(ReviewLabel.model_json_schema(), ensure_ascii=False)
        system_parts.append(f"\nJSON Schema obrigatório:\n{schema_json}")

    if strategy in ("few_shot", "structured"):
        examples_block = "\n\nExemplos:\n" + "\n".join(
            f'Comentário: "{text}"\nJSON: {json.dumps(label, ensure_ascii=False)}'
            for text, label in _FEW_SHOT_EXAMPLES
        )
        system_parts.append(examples_block)

    return [
        {"role": "system", "content": "\n".join(system_parts)},
        {"role": "user", "content": f'Classifique o comentário:\n"{review_text}"'},
    ]


def hash_payload(payload: object) -> str:
    """Return a stable SHA-256 hex digest for cache keying."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def inject_ptbr_noise(text: str, seed: int = 42, typo_prob: float = 0.08) -> str:
    """Apply realistic PT-BR user noise (accent drop, informal contractions, light typos).

    Used to evaluate robustness degradation under real-world noisy inputs.
    """
    rng = random.Random(seed)
    noisy = _strip_accents(text)
    replacements = {
        "nao": "n",
        "voce": "vc",
        "tambem": "tbm",
        "porque": "pq",
        "muito": "mt",
        "chegou": "xegou",
        "produto": "produt",
    }
    words = noisy.split()
    out_words: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in replacements and rng.random() < 0.6:
            out_words.append(replacements[lower])
            continue
        if len(word) > 4 and rng.random() < typo_prob:
            idx = rng.randint(1, len(word) - 2)
            chars = list(word)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            out_words.append("".join(chars))
        else:
            out_words.append(word)
    return " ".join(out_words)
