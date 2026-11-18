"""Pydantic v2 domain models and evaluation records for EvalBR."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal[
    "atraso_entrega",
    "produto_errado_ou_faltando",
    "defeito_ou_qualidade",
    "atendimento",
    "elogio",
    "outro",
]

Sentiment = Literal["negativo", "neutro", "positivo"]

DesiredAction = Literal[
    "reembolso",
    "troca",
    "entrega",
    "nenhuma",
    "nao_informado",
]

PromptStrategy = Literal["zero_shot", "few_shot", "structured"]

CATEGORIES: tuple[Category, ...] = (
    "atraso_entrega",
    "produto_errado_ou_faltando",
    "defeito_ou_qualidade",
    "atendimento",
    "elogio",
    "outro",
)


class ReviewLabel(BaseModel):
    """Structured triage label predicted by an LLM or annotated by a human."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    categoria: Category
    sentimento: Sentiment
    menciona_prazo: bool
    acao_desejada: DesiredAction
    evidencia: str = Field(
        min_length=1,
        description="Literal substring from the original review justifying the category.",
    )


class ReviewSample(BaseModel):
    """Unlabeled review record ready for inference."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    review_id: str
    score: int = Field(ge=1, le=5)
    text: str = Field(min_length=1)


class LabeledReview(ReviewSample):
    """Ground-truth review record with human annotation."""

    label: ReviewLabel


class SystemSpec(BaseModel):
    """Configuration for a single (provider, model, prompt_strategy) experiment cell."""

    model_config = ConfigDict(extra="forbid")

    system_id: str
    provider: Literal["openai_compat", "gemini", "mock"]
    model_name: str
    strategy: PromptStrategy
    input_cost_per_1m_usd: float = Field(default=0.0, ge=0.0)
    output_cost_per_1m_usd: float = Field(default=0.0, ge=0.0)
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @field_validator("system_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not value or " " in value:
            raise ValueError("system_id must be non-empty and contain no spaces.")
        return value


class EvalRecord(BaseModel):
    """Result of running one system on one review sample."""

    model_config = ConfigDict(extra="forbid")

    system_id: str
    review_id: str
    variant: Literal["clean", "noisy"] = "clean"
    raw_output: str
    parsed_label: ReviewLabel | None = None
    json_valid: bool
    retries: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    from_cache: bool = False
    error: str | None = None
