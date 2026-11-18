"""Dataset loaders, stratified Olist sampler, and realistic PT-BR demo generator."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from evalbr.schema import (
    Category,
    DesiredAction,
    LabeledReview,
    ReviewLabel,
    ReviewSample,
    Sentiment,
)

_TEMPLATES: dict[Category, list[tuple[int, Sentiment, bool, DesiredAction, str, str]]] = {
    "atraso_entrega": [
        (
            1,
            "negativo",
            True,
            "entrega",
            "Comprei há três semanas e até hoje não chegou, já passou muito do prazo.",
            "até hoje não chegou, já passou muito do prazo",
        ),
        (
            2,
            "negativo",
            True,
            "nao_informado",
            "O rastreio está parado na transportadora faz 12 dias, entrega muito atrasada.",
            "entrega muito atrasada",
        ),
        (
            1,
            "negativo",
            True,
            "reembolso",
            "Passou da data prevista e ninguém resolve a entrega, quero meu dinheiro de volta.",
            "Passou da data prevista",
        ),
    ],
    "produto_errado_ou_faltando": [
        (
            1,
            "negativo",
            False,
            "troca",
            "Pedi tamanho M na cor preta e veio G branco, preciso da troca urgente.",
            "Pedi tamanho M na cor preta e veio G branco",
        ),
        (
            1,
            "negativo",
            False,
            "reembolso",
            "A caixa chegou lacrada mas veio faltando o cabo de energia, quero estorno.",
            "veio faltando o cabo de energia",
        ),
        (
            2,
            "negativo",
            False,
            "entrega",
            "Comprei um kit com três unidades e entregaram apenas duas.",
            "entregaram apenas duas",
        ),
    ],
    "defeito_ou_qualidade": [
        (
            1,
            "negativo",
            False,
            "troca",
            "O liquidificador chegou com o copo rachado e não liga na tomada.",
            "chegou com o copo rachado e não liga",
        ),
        (
            2,
            "negativo",
            False,
            "nao_informado",
            "Material bem frágil pelo preço cobrado, soltou a costura no primeiro uso.",
            "soltou a costura no primeiro uso",
        ),
        (
            1,
            "negativo",
            False,
            "reembolso",
            "Fone veio com chiado forte no lado esquerdo, solicito devolução e reembolso.",
            "veio com chiado forte no lado esquerdo",
        ),
    ],
    "atendimento": [
        (
            1,
            "negativo",
            False,
            "reembolso",
            "Tentei cancelar pelo chat três vezes e o suporte não responde nenhuma mensagem.",
            "o suporte não responde nenhuma mensagem",
        ),
        (
            2,
            "negativo",
            False,
            "nao_informado",
            "Atendimento muito demorado para emitir a nota fiscal e resolver o protocolo.",
            "Atendimento muito demorado",
        ),
        (
            1,
            "negativo",
            False,
            "nao_informado",
            "Fui mal atendido no pós-venda e encerraram meu chamado sem solução.",
            "encerraram meu chamado sem solução",
        ),
    ],
    "elogio": [
        (
            5,
            "positivo",
            True,
            "nenhuma",
            "Chegou antes do prazo e muito bem embalado, excelente compra!",
            "Chegou antes do prazo e muito bem embalado",
        ),
        (
            5,
            "positivo",
            False,
            "nenhuma",
            "Produto original, acabamento perfeito e ótimo custo-benefício.",
            "acabamento perfeito e ótimo custo-benefício",
        ),
        (
            4,
            "positivo",
            True,
            "nenhuma",
            "Entrega rápida e dentro do combinado, recomendo a loja.",
            "Entrega rápida e dentro do combinado",
        ),
    ],
    "outro": [
        (
            3,
            "neutro",
            False,
            "nenhuma",
            "Ainda não abri o pacote para testar, avaliando apenas o recebimento.",
            "Ainda não abri o pacote para testar",
        ),
        (
            3,
            "neutro",
            False,
            "nenhuma",
            "Compra ok, dentro do normal sem nada de especial.",
            "Compra ok, dentro do normal",
        ),
    ],
}


def load_olist_reviews_csv(
    csv_path: Path | str,
    n_samples: int = 420,
    seed: int = 42,
) -> list[ReviewSample]:
    """Load Olist `olist_order_reviews_dataset.csv` and return a stratified sample."""
    df = pd.read_csv(csv_path)
    required = {"review_id", "review_score", "review_comment_message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required Olist columns: {sorted(missing)}")

    filtered = df.dropna(subset=["review_comment_message"]).copy()
    filtered["review_comment_message"] = filtered["review_comment_message"].astype(str).str.strip()
    filtered = filtered[filtered["review_comment_message"].str.len() >= 10]
    filtered = filtered.drop_duplicates(subset=["review_id"])

    if len(filtered) <= n_samples:
        sampled = filtered
    else:
        sampled = (
            filtered.groupby("review_score", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, round(n_samples * len(g) / len(filtered)))),
                    random_state=seed,
                ),
                include_groups=False,
            )
            .head(n_samples)
        )
        # Re-attach review_score if needed after include_groups=False
        if "review_score" not in sampled.columns:
            sampled = filtered.loc[sampled.index]

    return [
        ReviewSample(
            review_id=str(row["review_id"]),
            score=int(row["review_score"]),
            text=str(row["review_comment_message"]),
        )
        for _, row in sampled.iterrows()
    ]


def download_olist_reviews(output_dir: Path | str = "data") -> Path:
    """Download `olist_order_reviews_dataset.csv` from Kaggle (`olistbr/brazilian-ecommerce`)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "olist_order_reviews_dataset.csv"
    if csv_path.exists():
        return csv_path

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        "olistbr/brazilian-ecommerce",
        file_name="olist_order_reviews_dataset.csv",
        path=str(out_path),
    )
    zip_path = out_path / "olist_order_reviews_dataset.csv.zip"
    if zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        zip_path.unlink()
    return csv_path


def save_samples_jsonl(samples: list[ReviewSample], path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(item.model_dump_json() + "\n")


def load_samples_jsonl(path: Path | str) -> list[ReviewSample]:
    in_path = Path(path)
    items: list[ReviewSample] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                items.append(ReviewSample.model_validate_json(stripped))
    return items


def save_labeled_jsonl(samples: list[LabeledReview], path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(item.model_dump_json() + "\n")


def load_labeled_jsonl(path: Path | str) -> list[LabeledReview]:
    in_path = Path(path)
    items: list[LabeledReview] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                items.append(LabeledReview.model_validate_json(stripped))
    return items


def generate_synthetic_ptbr_dataset(
    n_samples: int = 420,
    seed: int = 42,
) -> list[LabeledReview]:
    """Generate a realistic stratified PT-BR e-commerce dataset for offline execution."""
    rng = random.Random(seed)
    categories = list(_TEMPLATES.keys())
    weights = [0.26, 0.18, 0.22, 0.12, 0.14, 0.08]
    prefixes = ["", "Olha, ", "Sinceramente: ", "Infelizmente ", ""]
    suffixes = ["", " Aguardo retorno.", " Muito insatisfeito.", " Obrigado.", ""]

    samples: list[LabeledReview] = []
    for i in range(n_samples):
        cat = rng.choices(categories, weights=weights, k=1)[0]
        score, sentiment, mentions_deadline, action, base_text, evidence = rng.choice(
            _TEMPLATES[cat]
        )
        prefix = rng.choice(prefixes) if cat != "elogio" else ""
        suffix = rng.choice(suffixes) if cat != "elogio" else ""
        full_text = f"{prefix}{base_text}{suffix}".strip()
        samples.append(
            LabeledReview(
                review_id=f"rev_{i:04d}",
                score=score,
                text=full_text,
                label=ReviewLabel(
                    categoria=cat,
                    sentimento=sentiment,
                    menciona_prazo=mentions_deadline,
                    acao_desejada=action,
                    evidencia=evidence,
                ),
            )
        )
    return samples
