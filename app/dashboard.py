"""Streamlit dashboard for exploring EvalBR benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="EvalBR — Benchmark Estatístico de LLMs", layout="wide")

st.title("EvalBR — Avaliação Estatística de LLMs em PT-BR")
st.caption(
    "Benchmark de triagem de reclamações com ICs por bootstrap, McNemar + Holm-Bonferroni e fronteira de Pareto."
)

summary_path = Path(st.sidebar.text_input("Caminho do summary.json", value="reports/summary.json"))

if not summary_path.exists():
    st.warning(
        f"Arquivo `{summary_path}` não encontrado. Execute `evalbr demo` primeiro para gerar os resultados."
    )
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Amostra (n)", summary["n_samples"])
col2.metric("Poder Mínimo (N)", summary["power_analysis_min_n"])
col3.metric("Referência (Maior F1)", summary["reference_system"])
col4.metric("Recomendado (Pareto)", summary["recommended_system"])

st.subheader("Métricas Consolidadas por Sistema")
metrics_df = pd.DataFrame(summary["metrics"]).sort_values("macro_f1", ascending=False)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

c_left, c_right = st.columns(2)
pareto_img = summary_path.parent / "pareto_f1_cost.png"
ci_img = summary_path.parent / "macro_f1_ci.png"
if pareto_img.exists():
    c_left.image(str(pareto_img), caption="Fronteira de Pareto: Custo vs. Macro-F1")
if ci_img.exists():
    c_right.image(str(ci_img), caption="Macro-F1 com Intervalos de Confiança (95%)")

cm_img = summary_path.parent / "confusion_matrix.png"
if cm_img.exists():
    st.subheader("Matriz de Confusão do Sistema Recomendado")
    st.image(str(cm_img), width=620)

if summary.get("judge_calibration"):
    st.subheader("Calibração LLM-como-Juiz (acao_desejada)")
    st.json(summary["judge_calibration"])

st.subheader("Comparações Pareadas vs. Referência (McNemar & Bootstrap + Holm-Bonferroni)")
paired_df = pd.DataFrame(summary["paired_vs_reference"])
st.dataframe(paired_df, use_container_width=True, hide_index=True)

st.subheader("Robustez sob Ruído PT-BR (sem acentos, gírias e typos)")
robust_df = pd.DataFrame.from_dict(summary["robustness_noisy_ptbr"], orient="index")
st.dataframe(robust_df, use_container_width=True)
