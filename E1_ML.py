"""FGV | Exercício 1: ML supervisionado e decisão sobre retenção.

Aplicativo autossuficiente: a base sintética e o treinamento do modelo
são gerados no próprio código. Não necessita de arquivos CSV ou .joblib
no repositório. Uso exclusivamente didático.
"""

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="FGV | Exercício 1 — Churn", page_icon="📉", layout="wide")

CAMPOS_MODELO = [
    "meses_contrato", "valor_mensal", "reclamacoes_90d", "atrasos_pagamento_6m",
    "uso_dados_gb", "ticket_suporte_90d", "tipo_plano", "regiao",
    "forma_pagamento", "fidelidade",
]
CAMPOS_ARQUIVO = ["cliente_id"] + CAMPOS_MODELO
NUMERICAS = CAMPOS_MODELO[:6]
CATEGORICAS = CAMPOS_MODELO[6:]
OPCOES = {
    "tipo_plano": ["Básico", "Padrão", "Premium"],
    "regiao": ["Norte/Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
    "forma_pagamento": ["Boleto", "Cartão", "Débito automático"],
    "fidelidade": ["Sem fidelidade", "12 meses", "24 meses"],
}


def gerar_clientes(n=1500, seed=42):
    """Gera características artificiais. Não são dados pessoais reais."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "meses_contrato": rng.integers(1, 73, n),
        "valor_mensal": np.round(rng.uniform(59, 349, n), 2),
        "reclamacoes_90d": rng.integers(0, 8, n),
        "atrasos_pagamento_6m": rng.integers(0, 7, n),
        "uso_dados_gb": np.round(rng.uniform(5, 220, n), 1),
        "ticket_suporte_90d": rng.integers(0, 10, n),
        "tipo_plano": rng.choice(OPCOES["tipo_plano"], n, p=[0.38, 0.42, 0.20]),
        "regiao": rng.choice(OPCOES["regiao"], n, p=[0.28, 0.12, 0.42, 0.18]),
        "forma_pagamento": rng.choice(OPCOES["forma_pagamento"], n, p=[0.33, 0.41, 0.26]),
        "fidelidade": rng.choice(OPCOES["fidelidade"], n, p=[0.40, 0.35, 0.25]),
    })


@st.cache_resource(show_spinner="Preparando o modelo didático...")
def carregar_modelo():
    """Treina uma única vez por execução/versão do app; alunos não o treinam."""
    treino = gerar_clientes(1500, 42)
    rng = np.random.default_rng(2026)
    score = (
        -1.4
        + (treino["meses_contrato"] < 6) * 1.15
        + treino["meses_contrato"].between(6, 12) * 0.45
        + (treino["valor_mensal"] > 220) * 0.35
        + (treino["reclamacoes_90d"] >= 3) * 1.10
        + (treino["atrasos_pagamento_6m"] >= 2) * 1.25
        + (treino["uso_dados_gb"] < 18) * 0.45
        + (treino["ticket_suporte_90d"] >= 4) * 0.60
        + (treino["tipo_plano"] == "Básico") * 0.40
        + (treino["forma_pagamento"] == "Boleto") * 0.35
        + (treino["fidelidade"] == "Sem fidelidade") * 0.95
        - (treino["fidelidade"] == "24 meses") * 0.60
        - (treino["forma_pagamento"] == "Débito automático") * 0.25
        - (treino["tipo_plano"] == "Premium") * 0.20
        + rng.normal(0, 0.45, len(treino))
    )
    probabilidade_sintetica = 1 / (1 + np.exp(-score))
    rotulos_historicos = rng.binomial(1, probabilidade_sintetica)

    transformacao = ColumnTransformer([
        ("numericas", "passthrough", NUMERICAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    modelo = Pipeline([
        ("preparacao", transformacao),
        ("classificador", RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_leaf=5, random_state=42
        )),
    ])
    modelo.fit(treino, rotulos_historicos)
    return modelo


def verificar_csv(df):
    if df.empty:
        raise ValueError("A planilha está vazia.")
    if len(df) > 5000:
        raise ValueError("Use até 5.000 clientes por arquivo nesta atividade.")
    faltantes = [c for c in CAMPOS_ARQUIVO if c not in df.columns]
    if faltantes:
        raise ValueError("Colunas ausentes: " + ", ".join(faltantes))
    base = df[CAMPOS_ARQUIVO].copy()
    if base["cliente_id"].isna().any() or (base["cliente_id"].astype(str).str.strip() == "").any():
        raise ValueError("Existem identificadores de cliente vazios.")
    base["cliente_id"] = base["cliente_id"].astype(str).str.strip()
    if base["cliente_id"].duplicated().any():
        raise ValueError("Existem IDs repetidos. Cada cliente deve aparecer apenas uma vez.")

    for coluna in NUMERICAS:
        # Aceita 12.5 e 12,5 nos campos numéricos.
        serie = base[coluna].astype(str).str.replace(",", ".", regex=False)
        base[coluna] = pd.to_numeric(serie, errors="coerce")
        if base[coluna].isna().any() or (base[coluna] < 0).any():
            raise ValueError(f"A coluna '{coluna}' contém valores inválidos ou negativos.")
    for coluna in CATEGORICAS:
        base[coluna] = base[coluna].astype("string").str.strip()
        if base[coluna].isna().any() or (~base[coluna].isin(OPCOES[coluna])).any():
            raise ValueError(
                f"Valores inválidos em '{coluna}'. Use: {', '.join(OPCOES[coluna])}."
            )
        base[coluna] = base[coluna].astype(str)
    return base


def classificar_risco(escore):
    if escore >= 0.70:
        return "Alto"
    if escore >= 0.40:
        return "Médio"
    return "Baixo"


# --------------------------- INTERFACE ----------------------------
st.title("Exercício 1 | Você confiaria nesta IA?")
st.caption("FGV Educação Executiva • Primeira aula • Machine Learning supervisionado")

st.subheader("O problema de negócio — Operadora Conecta+ (fictícia)")
st.write(
    "Uma operadora está perdendo clientes e dispõe de recursos limitados para ações "
    "de retenção. A diretoria recebeu um sistema de **Machine Learning supervisionado**, "
    "**já treinado** com exemplos históricos de clientes que cancelaram ou permaneceram. "
    "Seu papel é analisar os clientes atuais e decidir se a empresa deve confiar nas previsões."
)
st.markdown(
    "**Objetivo do negócio:** reduzir cancelamentos com uso responsável do orçamento.  \n"
    "**Objetivo da IA:** classificar clientes por risco estimado de churn e apoiar a priorização."
)
col1, col2, col3 = st.columns(3)
col1.metric("Orçamento disponível", "R$ 4.000")
col2.metric("Custo ilustrativo por ação", "R$ 200")
col3.metric("Capacidade máxima", "20 clientes")
st.caption("Todos os clientes, valores e resultados desta aplicação são artificiais e didáticos.")

st.divider()
st.subheader("1. Escolha a base de clientes atuais")
st.write(
    "A base de exemplo com **80 clientes fictícios** já está disponível no sistema. "
    "Basta clicar em **Executar previsão**. Opcionalmente, você pode enviar um arquivo CSV."
)
exemplo = gerar_clientes(80, seed=2027)
exemplo.insert(0, "cliente_id", [f"CLI-{i:03d}" for i in range(1, 81)])

origem = st.radio(
    "Origem dos clientes",
    ["Usar os 80 clientes de exemplo (sem upload)", "Enviar meu arquivo CSV"],
    horizontal=True,
)
if origem.startswith("Usar"):
    base = exemplo.copy()
    assinatura = "base_interna_80_v1"
    st.success("Base interna selecionada: 80 clientes fictícios.")
else:
    st.warning("Não envie dados pessoais reais. Use dados fictícios ou anonimizados.")
    st.download_button(
        "📥 Baixar o modelo de CSV com 80 clientes fictícios",
        data=exemplo.to_csv(index=False).encode("utf-8-sig"),
        file_name="clientes_atuais_exemplo.csv",
        mime="text/csv",
    )
    with st.expander("Ver formato exigido para o CSV"):
        st.code(", ".join(CAMPOS_ARQUIVO))
        st.dataframe(pd.DataFrame({"Campo": list(OPCOES), "Opções aceitas": [
            " | ".join(v) for v in OPCOES.values()
        ]}), hide_index=True, use_container_width=True)
    arquivo = st.file_uploader("Escolha o CSV de clientes", type="csv")
    if arquivo is None:
        st.info("Envie um CSV para executar a análise ou selecione a base interna.")
        st.stop()
    assinatura = hashlib.sha256(arquivo.getvalue()).hexdigest()
    try:
        csv = pd.read_csv(io.BytesIO(arquivo.getvalue()), sep=None, engine="python", encoding="utf-8-sig")
        base = verificar_csv(csv)
    except (ValueError, UnicodeError, pd.errors.ParserError) as e:
        st.error(f"Confira o arquivo: {e}")
        st.stop()
    st.success(f"Arquivo validado: {len(base)} clientes.")

if st.button("▶ Executar previsão", type="primary"):
    modelo = carregar_modelo()
    escores = modelo.predict_proba(base[CAMPOS_MODELO])[:, 1]
    resultado = base.copy()
    resultado["risco_estimado"] = escores
    resultado["faixa_de_risco"] = [classificar_risco(p) for p in escores]
    resultado = resultado.sort_values("risco_estimado", ascending=False).reset_index(drop=True)
    st.session_state["analise"] = resultado
    st.session_state["assinatura"] = assinatura

if st.session_state.get("assinatura") != assinatura or "analise" not in st.session_state:
    st.info("Clique em Executar previsão para conhecer os resultados.")
    st.stop()

resultado = st.session_state["analise"]
st.divider()
st.subheader("2. Observe as previsões e o orçamento")
selecionados = min(20, len(resultado))
result_top = resultado.head(selecionados).copy()
quantidade_alertas = int((resultado["risco_estimado"] >= 0.40).sum())

c1, c2, c3 = st.columns(3)
c1.metric("Clientes analisados", len(resultado))
c2.metric("Risco médio ou alto", quantidade_alertas)
c3.metric("Clientes atendíveis", selecionados)
if quantidade_alertas > 20:
    st.warning(
        f"A IA sinalizou {quantidade_alertas} clientes com risco médio ou alto, "
        "mas o orçamento só comporta 20 ações de R$ 200. Quem deveria ser atendido?"
    )

visao = resultado[["cliente_id", "risco_estimado", "faixa_de_risco", "valor_mensal"]].copy()
visao["Risco estimado (%)"] = (visao.pop("risco_estimado") * 100).round(1)
visao.rename(columns={
    "cliente_id": "Cliente", "faixa_de_risco": "Faixa (didática)",
    "valor_mensal": "Mensalidade (R$)"
}, inplace=True)
visao["Prioridade pelo ranking"] = [
    "Dentro do orçamento" if i < selecionados else "Fora do orçamento"
    for i in range(len(visao))
]
st.dataframe(visao, use_container_width=True, hide_index=True)

saida = resultado.copy()
saida["risco_estimado"] = (saida["risco_estimado"] * 100).round(1)
st.download_button(
    "📥 Baixar ranking completo (CSV)",
    data=saida.to_csv(index=False).encode("utf-8-sig"),
    file_name="resultado_churn.csv", mime="text/csv"
)
st.caption(
    "A ordenação por risco é uma proposta de análise, NÃO uma recomendação validada "
    "de gasto. O sistema não informa se as previsões estão corretas, nem se as ações "
    "de retenção funcionarão. Os limiares de 40% e 70% são escolhas didáticas."
)

st.divider()
st.subheader("3. Discussão executiva em equipe")
st.markdown(
    "**1. Você confiaria nessa IA para direcionar o orçamento de retenção?**  \n"
    "**2. Qual o risco de agir sobre previsões erradas?**  \n"
    "**3. Quem deveria aprovar o uso dessa IA no negócio?**"
)
decisao = st.radio(
    "Qual é a decisão da equipe?",
    ["Solicitar mais evidências antes de usar", "Autorizar apenas um piloto controlado", "Aprovar uso imediato"],
    index=0,
)
justificativa = st.text_area("Justifique a decisão (para discussão em sala)", height=100)
st.info(
    "**Ponto de reflexão:** você conhece os riscos estimados, mas não recebeu "
    "métricas de validação nem informações sobre os erros do modelo. "
    "Uma previsão não é uma autorização automática para gastar."
)
st.caption("Exercício didático com dados e modelo sintéticos. Não utilizar para decisões reais.")
