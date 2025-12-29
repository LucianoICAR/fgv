# streamlit_app.py
# App: Priorização Estratégica de Portfólio de Transformação Digital (com apoio de IA)
# Linguagem: 100% estratégica (executivos)

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Priorização Estratégica de Portfólio (IA)",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Configurações e constantes
# -----------------------------
CRITERIA_COLS = {
    "impacto_estrategico": "Impacto Estratégico (1–5)",
    "alinhamento_estrategico": "Alinhamento Estratégico (1–5)",
    "risco": "Risco (1–5)",
    "complexidade": "Complexidade (1–5)",
    "investimento": "Investimento (R$)",
}

DEFAULT_SCENARIOS = {
    "Balanceado (padrão de comitê)": {
        "w_impacto": 0.30,
        "w_alinhamento": 0.30,
        "w_risco": 0.20,
        "w_complexidade": 0.20,
    },
    "Crescimento (agressivo em valor)": {
        "w_impacto": 0.40,
        "w_alinhamento": 0.35,
        "w_risco": 0.15,
        "w_complexidade": 0.10,
    },
    "Crise (defensivo, preservação)": {
        "w_impacto": 0.20,
        "w_alinhamento": 0.25,
        "w_risco": 0.30,
        "w_complexidade": 0.25,
    },
    "Eficiência (redução de custo/entrega rápida)": {
        "w_impacto": 0.25,
        "w_alinhamento": 0.25,
        "w_risco": 0.20,
        "w_complexidade": 0.30,
    },
}

# Limiares para classificação executiva (pode ajustar)
CLASS_THRESHOLDS = {
    "executar_agora": 0.75,   # top tier
    "avaliar_melhor": 0.55,   # mid tier
    # abaixo disso -> postergar
}

# -----------------------------
# Funções utilitárias
# -----------------------------
def _currency_br(value: float) -> str:
    try:
        return f"R$ {value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"R$ {value}"

def _safe_minmax(series: pd.Series) -> pd.Series:
    """Min-max scaling robusto para séries constantes."""
    s = series.astype(float)
    mn, mx = float(np.nanmin(s)), float(np.nanmax(s))
    if np.isfinite(mn) and np.isfinite(mx) and mx != mn:
        return (s - mn) / (mx - mn)
    # se constante, devolve 0.5 para não zerar a influência
    return pd.Series([0.5] * len(s), index=series.index)

def normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza critérios em [0,1] com semântica executiva:
    - Impacto e Alinhamento: quanto maior, melhor.
    - Risco, Complexidade e Investimento: quanto menor, melhor (invertido).
    """
    out = df.copy()

    out["n_impacto"] = _safe_minmax(out["impacto_estrategico"])
    out["n_alinhamento"] = _safe_minmax(out["alinhamento_estrategico"])

    # Para risco/complexidade/investimento, menor é melhor -> inverte após normalizar
    out["n_risco"] = 1.0 - _safe_minmax(out["risco"])
    out["n_complexidade"] = 1.0 - _safe_minmax(out["complexidade"])
    out["n_investimento"] = 1.0 - _safe_minmax(out["investimento"])

    return out

def compute_score(df_norm: pd.DataFrame, weights: Dict[str, float], include_investment: bool) -> pd.DataFrame:
    """
    Score estratégico explicável:
    Valor = (Impacto, Alinhamento) - (Risco, Complexidade) e opcionalmente Investimento.
    Todos em [0,1]. Score final também em [0,1] por construção (soma de pesos).
    """
    w_imp = weights["w_impacto"]
    w_ali = weights["w_alinhamento"]
    w_ris = weights["w_risco"]
    w_com = weights["w_complexidade"]

    # base score
    score = (
        df_norm["n_impacto"] * w_imp
        + df_norm["n_alinhamento"] * w_ali
        + df_norm["n_risco"] * w_ris
        + df_norm["n_complexidade"] * w_com
    )

    if include_investment:
        # Distribui peso de investimento sem distorcer (reescala pesos)
        # Estratégia: aplica penalização/benefício como um "ajuste" com peso leve.
        # Para manter governança: o usuário controla isso via slider separado.
        w_inv = weights.get("w_investimento", 0.0)
        score = score + df_norm["n_investimento"] * w_inv

    out = df_norm.copy()
    out["score_estrategico"] = score

    # Contribuições para explicabilidade
    out["c_impacto"] = df_norm["n_impacto"] * w_imp
    out["c_alinhamento"] = df_norm["n_alinhamento"] * w_ali
    out["c_risco"] = df_norm["n_risco"] * w_ris
    out["c_complexidade"] = df_norm["n_complexidade"] * w_com
    if include_investment:
        out["c_investimento"] = df_norm["n_investimento"] * weights.get("w_investimento", 0.0)
    else:
        out["c_investimento"] = 0.0

    return out

def classify_portfolio(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Classificação executiva baseada em score.
    """
    out = df_scored.copy()

    def _class(s: float) -> str:
        if s >= CLASS_THRESHOLDS["executar_agora"]:
            return "Executar agora"
        if s >= CLASS_THRESHOLDS["avaliar_melhor"]:
            return "Avaliar melhor"
        return "Postergar"

    out["decisao_executiva"] = out["score_estrategico"].apply(_class)
    return out

def apply_budget_cut(df: pd.DataFrame, budget_cut_pct: float) -> pd.DataFrame:
    """
    Simula restrição orçamentária: marca iniciativas que cabem no orçamento pós-corte,
    alocando na ordem do ranking (alto score primeiro).
    """
    out = df.sort_values("score_estrategico", ascending=False).copy()
    total = float(out["investimento"].sum())
    cap = total * (1.0 - budget_cut_pct / 100.0)

    running = 0.0
    within = []
    for v in out["investimento"].astype(float).tolist():
        if running + v <= cap:
            within.append(True)
            running += v
        else:
            within.append(False)

    out["cabe_no_orcamento_simulado"] = within
    out["orcamento_total_base"] = total
    out["orcamento_cap_simulado"] = cap
    out["orcamento_usado_simulado"] = running
    return out

def example_dataset() -> pd.DataFrame:
    """
    Base exemplo com linguagem executiva e iniciativas típicas de transformação.
    """
    data = [
        ["IA Antifraude", 500_000, 5, 5, 3, 4],
        ["Automação (RPA) Compras", 150_000, 4, 4, 2, 2],
        ["Analytics de Receita (BI avançado)", 220_000, 4, 4, 2, 3],
        ["Modernização Core Legado (API)", 800_000, 5, 5, 4, 5],
        ["Personalização com IA (CRM)", 300_000, 5, 5, 4, 3],
        ["Data Lake / Plataforma de Dados", 650_000, 5, 5, 3, 4],
        ["Gestão de Identidade (Zero Trust)", 280_000, 4, 4, 3, 3],
        ["Otimização de Logística (IA)", 420_000, 4, 4, 3, 4],
        ["Chatbot Atendimento (N1)", 120_000, 3, 3, 2, 2],
        ["Automação Contábil (OCR + regras)", 180_000, 3, 3, 2, 2],
        ["Squads Produto (Operating Model)", 200_000, 4, 5, 2, 3],
        ["Gestão de Risco de IA (Governança)", 160_000, 4, 5, 2, 2],
    ]
    df = pd.DataFrame(
        data,
        columns=["iniciativa", "investimento", "impacto_estrategico", "alinhamento_estrategico", "risco", "complexidade"],
    )
    return df

def validate_input_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    required = ["iniciativa", "investimento", "impacto_estrategico", "alinhamento_estrategico", "risco", "complexidade"]
    for c in required:
        if c not in df.columns:
            errors.append(f"Coluna obrigatória ausente: '{c}'")

    if errors:
        return False, errors

    # valida escalas básicas
    for c in ["impacto_estrategico", "alinhamento_estrategico", "risco", "complexidade"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"Coluna '{c}' precisa ser numérica (1–5).")
    if not pd.api.types.is_numeric_dtype(df["investimento"]):
        errors.append("Coluna 'investimento' precisa ser numérica (R$).")

    if df["iniciativa"].isna().any():
        errors.append("Há iniciativas sem nome (valores vazios em 'iniciativa').")

    return (len(errors) == 0), errors

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    return bio.getvalue()

# -----------------------------
# UI - Cabeçalho e narrativa
# -----------------------------
st.title("Priorização Estratégica de Portfólio de Transformação Digital (com apoio de IA)")
st.write(
    "Este painel simula uma **decisão de comitê executivo**: dado um conjunto de iniciativas, "
    "o sistema apoia a priorização com **critérios explícitos**, **pesos estratégicos** e **simulação de cenários**. "
    "A IA aqui não “define a estratégia”; ela **executa a estratégia definida pela liderança**, com consistência e rastreabilidade."
)

with st.expander("Como usar no Zoom (roteiro de condução em 30 segundos)", expanded=False):
    st.markdown(
        "- 1) Selecione um **cenário estratégico**.\n"
        "- 2) Ajuste os **pesos** (sliders) conforme a orientação do comitê.\n"
        "- 3) Aplique um **corte de orçamento** e observe o impacto no ranking.\n"
        "- 4) Discuta com a turma: o que mudou? quais trade-offs ficaram explícitos?"
    )

# -----------------------------
# Sidebar - Dados e controles estratégicos
# -----------------------------
st.sidebar.header("Configuração Executiva")

data_mode = st.sidebar.radio(
    "Base de iniciativas",
    ["Usar base exemplo (12 iniciativas)", "Carregar CSV da organização"],
    index=0,
)

if data_mode == "Carregar CSV da organização":
    st.sidebar.caption("Formato esperado (colunas): iniciativa, investimento, impacto_estrategico, alinhamento_estrategico, risco, complexidade")
    uploaded = st.sidebar.file_uploader("Upload do CSV", type=["csv"])
    if uploaded is None:
        st.info("Carregue um CSV para continuar ou selecione a base exemplo na barra lateral.")
        st.stop()
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = example_dataset()

ok, errs = validate_input_df(df_raw)
if not ok:
    st.error("A base carregada não está no formato esperado.")
    st.write("Ajustes necessários:")
    for e in errs:
        st.write(f"- {e}")
    st.stop()

# Cenários e pesos
scenario_name = st.sidebar.selectbox("Cenário estratégico", list(DEFAULT_SCENARIOS.keys()), index=0)
preset = DEFAULT_SCENARIOS[scenario_name]

st.sidebar.subheader("Pesos estratégicos (trade-offs do comitê)")
w_impacto = st.sidebar.slider("Peso: Impacto Estratégico", 0.0, 1.0, float(preset["w_impacto"]), 0.05)
w_alinhamento = st.sidebar.slider("Peso: Alinhamento Estratégico", 0.0, 1.0, float(preset["w_alinhamento"]), 0.05)
w_risco = st.sidebar.slider("Peso: Risco (quanto menor risco, maior score)", 0.0, 1.0, float(preset["w_risco"]), 0.05)
w_complexidade = st.sidebar.slider("Peso: Complexidade (quanto menor complexidade, maior score)", 0.0, 1.0, float(preset["w_complexidade"]), 0.05)

include_investment = st.sidebar.toggle("Considerar investimento no score (menor investimento = maior score)", value=True)
w_investimento = 0.0
if include_investment:
    st.sidebar.caption("Use com parcimônia: investimento é importante, mas não deve ‘matar’ iniciativas estratégicas.")
    w_investimento = st.sidebar.slider("Peso: Investimento", 0.0, 0.40, 0.15, 0.05)

# Reescala pesos para manter governança (soma = 1 quando não inclui investimento como extra)
# Estratégia: manter transparência. Se incluir investimento, ele entra como ajuste adicional.
sum_main = w_impacto + w_alinhamento + w_risco + w_complexidade
if sum_main <= 0:
    st.sidebar.error("Defina pelo menos um peso maior que zero.")
    st.stop()

w_impacto_n = w_impacto / sum_main
w_alinhamento_n = w_alinhamento / sum_main
w_risco_n = w_risco / sum_main
w_complexidade_n = w_complexidade / sum_main

weights = {
    "w_impacto": w_impacto_n,
    "w_alinhamento": w_alinhamento_n,
    "w_risco": w_risco_n,
    "w_complexidade": w_complexidade_n,
    "w_investimento": w_investimento,
}

st.sidebar.subheader("Restrição orçamentária (simulação)")
budget_cut = st.sidebar.slider("Corte de orçamento (%)", 0, 60, 25, 5)

# -----------------------------
# Processamento
# -----------------------------
df = df_raw.copy()
df_norm = normalize_inputs(df)
df_scored = compute_score(df_norm, weights=weights, include_investment=include_investment)
df_class = classify_portfolio(df_scored)
df_budget = apply_budget_cut(df_class, budget_cut_pct=float(budget_cut))

# -----------------------------
# Painel principal
# -----------------------------
colA, colB, colC, colD = st.columns(4)

total_invest = float(df_raw["investimento"].sum())
cap_sim = float(df_budget["orcamento_cap_simulado"].iloc[0])
used_sim = float(df_budget["orcamento_usado_simulado"].iloc[0])

colA.metric("Iniciativas no portfólio", f"{len(df_raw)}")
colB.metric("Orçamento total (base)", _currency_br(total_invest))
colC.metric(f"Cap pós-corte ({budget_cut}%)", _currency_br(cap_sim))
colD.metric("Orçamento alocado (simulação)", _currency_br(used_sim))

st.divider()

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Decisão recomendada (ranking + corte orçamentário)")

    # Monta tabela executiva
    show = df_budget.sort_values("score_estrategico", ascending=False).copy()
    show["Score (0–1)"] = show["score_estrategico"].round(3)
    show["Investimento"] = show["investimento"].apply(_currency_br)
    show["Dentro do orçamento?"] = show["cabe_no_orcamento_simulado"].map({True: "Sim", False: "Não"})
    show["Decisão executiva"] = show["decisao_executiva"]

    executive_table = show[
        [
            "iniciativa",
            "Investimento",
            "Score (0–1)",
            "Decisão executiva",
            "Dentro do orçamento?",
        ]
    ].rename(columns={"iniciativa": "Iniciativa"})

    st.dataframe(executive_table, use_container_width=True, hide_index=True)

    # Download do resultado
    export = show.copy()
    export = export.rename(columns={
        "iniciativa": "iniciativa",
        "decisao_executiva": "decisao_executiva",
        "score_estrategico": "score_estrategico",
        "cabe_no_orcamento_simulado": "cabe_no_orcamento_simulado",
    })
    st.download_button(
        "Baixar decisão (CSV)",
        data=df_to_csv_bytes(export[[
            "iniciativa",
            "investimento",
            "impacto_estrategico",
            "alinhamento_estrategico",
            "risco",
            "complexidade",
            "score_estrategico",
            "decisao_executiva",
            "cabe_no_orcamento_simulado",
        ]]),
        file_name="decisao_portfolio_transformacao_digital.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right:
    st.subheader("Matriz estratégica: Impacto × Risco (bolha = investimento)")
    # Scatter com matplotlib (sem setar cor específica)
    import matplotlib.pyplot as plt

    plot_df = df_budget.copy()
    # Impacto e risco originais (1–5) para leitura executiva
    x = plot_df["impacto_estrategico"].astype(float)
    y = plot_df["risco"].astype(float)
    sizes = (plot_df["investimento"].astype(float) / max(plot_df["investimento"].astype(float).max(), 1.0)) * 1200 + 80

    fig = plt.figure()
    plt.scatter(x, y, s=sizes, alpha=0.6)
    for _, r in plot_df.iterrows():
        plt.text(float(r["impacto_estrategico"]) + 0.03, float(r["risco"]) + 0.03, str(r["iniciativa"])[:18], fontsize=8)

    plt.xlabel("Impacto Estratégico (maior é melhor)")
    plt.ylabel("Risco (maior é pior)")
    plt.title("Trade-off: valor vs risco (tamanho = investimento)")
    plt.xlim(0.5, 5.5)
    plt.ylim(0.5, 5.5)
    plt.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Distribuição do Score Estratégico")
    fig2 = plt.figure()
    plt.hist(df_budget["score_estrategico"].astype(float), bins=10, alpha=0.8)
    plt.xlabel("Score Estratégico (0–1)")
    plt.ylabel("Quantidade de iniciativas")
    plt.title("Concentração de valor priorizado")
    plt.grid(True, alpha=0.25)
    st.pyplot(fig2, use_container_width=True)

st.divider()

# -----------------------------
# Explicabilidade executiva
# -----------------------------
st.subheader("Por que este ranking? (explicabilidade para discussão de comitê)")
st.write(
    "A decomposição abaixo deixa explícito **quais critérios puxaram cada iniciativa para cima ou para baixo**. "
    "Isso reduz ‘caixa-preta’ e melhora a qualidade do debate estratégico."
)

top_n = st.slider("Quantas iniciativas analisar", 3, min(12, len(df_budget)), 5)
focus = df_budget.sort_values("score_estrategico", ascending=False).head(top_n).copy()

# Tabela explicável
explain = focus[[
    "iniciativa",
    "score_estrategico",
    "c_impacto",
    "c_alinhamento",
    "c_risco",
    "c_complexidade",
    "c_investimento",
]].copy()

explain = explain.rename(columns={
    "iniciativa": "Iniciativa",
    "score_estrategico": "Score",
    "c_impacto": "Contrib. Impacto",
    "c_alinhamento": "Contrib. Alinhamento",
    "c_risco": "Contrib. Risco (invertido)",
    "c_complexidade": "Contrib. Complexidade (invertida)",
    "c_investimento": "Contrib. Investimento (invertido)",
})

for c in ["Score", "Contrib. Impacto", "Contrib. Alinhamento", "Contrib. Risco (invertido)", "Contrib. Complexidade (invertida)", "Contrib. Investimento (invertido)"]:
    explain[c] = explain[c].astype(float).round(3)

st.dataframe(explain, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Governança e próximos passos (executivo)
# -----------------------------
st.subheader("Governança: como usar isso no mundo real (checklist executivo)")
st.markdown(
    "- **Critérios e escalas** foram aprovados pelo comitê? (para evitar subjetividade ad hoc)\n"
    "- **Pesos por cenário** estão registrados? (para impedir ‘mudança de regra’ a cada reunião)\n"
    "- Quem é o **dono do modelo** (Estratégia/PMO/Transformação) e quem faz a **curadoria** dos dados?\n"
    "- As decisões ficam **rastreáveis**: ranking + justificativa + exceções deliberadas.\n"
    "- Revisão periódica (trimestral): ajustes de critérios e pesos conforme o contexto."
)

st.info(
    "Mensagem para fechamento: Transformação digital madura não é ‘ter IA’. "
    "É **decidir melhor, mais rápido e com transparência**, usando IA para tornar trade-offs explícitos."
)

# -----------------------------
# Amostra CSV para download (para facilitar adoção)
# -----------------------------
with st.expander("Baixar CSV modelo para sua organização", expanded=False):
    st.write("Use este arquivo como template (colunas e formato já compatíveis).")
    template = example_dataset().head(0)
    st.download_button(
        "Baixar template CSV",
        data=df_to_csv_bytes(template),
        file_name="template_iniciativas_transformacao_digital.csv",
        mime="text/csv",
        use_container_width=True,
    )
