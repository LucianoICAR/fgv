import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="IA aplicada aos negócios - Churn", page_icon="📉", layout="wide")

# ============================================================
# DADOS DE TREINO NATIVOS DO SISTEMA
# O aluno não vê esta base. Ela existe apenas dentro do código.
# ============================================================

def gerar_base_treino(n=1400, seed=42):
    rng = np.random.default_rng(seed)

    meses_contrato = rng.integers(1, 73, n)
    valor_mensal = np.round(rng.uniform(59, 349, n), 2)
    reclamacoes_90d = rng.integers(0, 8, n)
    atrasos_pagamento_6m = rng.integers(0, 7, n)
    uso_dados_gb = np.round(rng.uniform(5, 220, n), 1)
    ticket_suporte_90d = rng.integers(0, 10, n)
    tipo_plano = rng.choice(["Básico", "Padrão", "Premium"], n, p=[0.38, 0.42, 0.20])
    regiao = rng.choice(["Norte/Nordeste", "Centro-Oeste", "Sudeste", "Sul"], n, p=[0.28, 0.12, 0.42, 0.18])
    forma_pagamento = rng.choice(["Boleto", "Cartão", "Débito automático"], n, p=[0.33, 0.41, 0.26])
    fidelidade = rng.choice(["Sem fidelidade", "12 meses", "24 meses"], n, p=[0.40, 0.35, 0.25])

    df = pd.DataFrame({
        "meses_contrato": meses_contrato,
        "valor_mensal": valor_mensal,
        "reclamacoes_90d": reclamacoes_90d,
        "atrasos_pagamento_6m": atrasos_pagamento_6m,
        "uso_dados_gb": uso_dados_gb,
        "ticket_suporte_90d": ticket_suporte_90d,
        "tipo_plano": tipo_plano,
        "regiao": regiao,
        "forma_pagamento": forma_pagamento,
        "fidelidade": fidelidade,
    })

    score = (
        -1.4
        + (df["meses_contrato"] < 6) * 1.15
        + (df["meses_contrato"].between(6, 12)) * 0.45
        + (df["valor_mensal"] > 220) * 0.35
        + (df["reclamacoes_90d"] >= 3) * 1.10
        + (df["atrasos_pagamento_6m"] >= 2) * 1.25
        + (df["uso_dados_gb"] < 18) * 0.45
        + (df["ticket_suporte_90d"] >= 4) * 0.60
        + (df["tipo_plano"] == "Básico") * 0.40
        + (df["forma_pagamento"] == "Boleto") * 0.35
        + (df["fidelidade"] == "Sem fidelidade") * 0.95
        + (df["fidelidade"] == "24 meses") * (-0.60)
        + (df["forma_pagamento"] == "Débito automático") * (-0.25)
        + (df["tipo_plano"] == "Premium") * (-0.20)
        + rng.normal(0, 0.45, len(df))
    )

    prob = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, prob, len(df))
    df["churn"] = churn
    return df

@st.cache_resource
def treinar_pipeline():
    df = gerar_base_treino()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    numericas = [
        "meses_contrato", "valor_mensal", "reclamacoes_90d",
        "atrasos_pagamento_6m", "uso_dados_gb", "ticket_suporte_90d"
    ]
    categoricas = ["tipo_plano", "regiao", "forma_pagamento", "fidelidade"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    pipeline.fit(X, y)
    return pipeline

modelo = treinar_pipeline()

COLUNAS_ESPERADAS = [
    "cliente_id",
    "meses_contrato",
    "valor_mensal",
    "reclamacoes_90d",
    "atrasos_pagamento_6m",
    "uso_dados_gb",
    "ticket_suporte_90d",
    "tipo_plano",
    "regiao",
    "forma_pagamento",
    "fidelidade",
]

def classificar_risco(prob):
    if prob >= 0.70:
        return "Alto risco"
    if prob >= 0.40:
        return "Médio risco"
    return "Baixo risco"

def sugerir_acao(prob):
    if prob >= 0.70:
        return "Ação imediata: contato humano + oferta de retenção"
    if prob >= 0.40:
        return "Monitorar e ofertar benefício leve"
    return "Manter relacionamento padrão"

def validar_entrada(df):
    faltantes = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        return False, f"Colunas ausentes no arquivo: {', '.join(faltantes)}"

    for c in ["meses_contrato", "reclamacoes_90d", "atrasos_pagamento_6m", "ticket_suporte_90d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["valor_mensal", "uso_dados_gb"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["tipo_plano", "regiao", "forma_pagamento", "fidelidade", "cliente_id"]:
        df[c] = df[c].astype(str).fillna("")

    return True, df

st.title("📉 IA aplicada aos negócios")
st.subheader("Exercício executivo de Machine Learning: previsão de cancelamento de clientes")

with st.container():
    st.markdown(
        """
        Este sistema simula uma IA corporativa já treinada.  
        O participante insere dados de clientes e recebe previsões de risco de cancelamento.

        **Intenção pedagógica:** mostrar que Machine Learning pode gerar valor de negócio
        sem ser IA generativa — e também evidenciar os riscos de usar uma IA sem conhecer
        a base de treino, as métricas de avaliação e os limites do modelo.
        """
    )

with st.expander("📌 Como usar o sistema"):
    st.markdown(
        """
        1. Baixe e preencha o arquivo de teste, ou use o exemplo pronto.  
        2. Faça upload do CSV na área abaixo.  
        3. Clique em **Executar previsão**.  
        4. Analise os clientes com maior risco e discuta a decisão de negócio.
        """
    )

arquivo = st.file_uploader("Envie o arquivo CSV com os dados dos clientes", type=["csv"])

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Formato esperado das colunas**")
    st.code(
        "cliente_id, meses_contrato, valor_mensal, reclamacoes_90d, atrasos_pagamento_6m, "
        "uso_dados_gb, ticket_suporte_90d, tipo_plano, regiao, forma_pagamento, fidelidade"
    )

with col2:
    modelo_info = pd.DataFrame({
        "Campo": [
            "tipo_plano", "regiao", "forma_pagamento", "fidelidade"
        ],
        "Valores aceitos": [
            "Básico | Padrão | Premium",
            "Norte/Nordeste | Centro-Oeste | Sudeste | Sul",
            "Boleto | Cartão | Débito automático",
            "Sem fidelidade | 12 meses | 24 meses"
        ]
    })
    st.dataframe(modelo_info, use_container_width=True, hide_index=True)

if arquivo is not None:
    try:
        df_input = pd.read_csv(arquivo)
        ok, resultado_validacao = validar_entrada(df_input.copy())

        if not ok:
            st.error(resultado_validacao)
        else:
            df_input = resultado_validacao
            st.success("Arquivo carregado com sucesso.")
            st.dataframe(df_input, use_container_width=True)

            if st.button("Executar previsão", type="primary", use_container_width=True):
                X_pred = df_input.drop(columns=["cliente_id"])
                probs = modelo.predict_proba(X_pred)[:, 1]

                resultado = df_input.copy()
                resultado["prob_cancelamento"] = np.round(probs, 4)
                resultado["risco"] = resultado["prob_cancelamento"].apply(classificar_risco)
                resultado["acao_sugerida"] = resultado["prob_cancelamento"].apply(sugerir_acao)

                resumo1, resumo2, resumo3 = st.columns(3)
                with resumo1:
                    st.metric("Clientes analisados", len(resultado))
                with resumo2:
                    st.metric("Alto risco", int((resultado["risco"] == "Alto risco").sum()))
                with resumo3:
                    st.metric("Risco médio ou alto", int((resultado["prob_cancelamento"] >= 0.40).sum()))

                st.markdown("### Ranking de clientes mais críticos")
                resultado_exibicao = resultado.sort_values("prob_cancelamento", ascending=False).reset_index(drop=True)
                st.dataframe(resultado_exibicao, use_container_width=True)

                csv_bytes = resultado_exibicao.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Baixar resultado da análise",
                    data=csv_bytes,
                    file_name="resultado_previsao_churn.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.markdown("### Reflexão executiva")
                st.warning(
                    """
                    Você recebeu previsões úteis para a tomada de decisão.  
                    Mas há uma questão central: **você não conhece a base usada no treinamento, não conhece as métricas
                    de avaliação do modelo e não sabe os riscos de erro da IA**.

                    Perguntas para debate:
                    - Você confiaria nessa IA para direcionar orçamento de retenção?
                    - Que indicadores exigiria antes de colocá-la em produção?
                    - Qual o risco de agir sobre previsões erradas?
                    - Quem deveria aprovar o uso dessa IA no negócio?
                    """
                )

    except Exception as e:
        st.error(f"Não foi possível processar o arquivo. Detalhe técnico: {e}")

else:
    st.info("Envie o arquivo CSV de teste para iniciar a análise.")

st.markdown("---")
st.caption("FGV | Exercício inicial de IA aplicada aos negócios com Machine Learning no Streamlit")