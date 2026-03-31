
import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IA aplicada aos negócios | Churn", page_icon="📉", layout="wide")

MODEL_PATH = "artifacts/modelo_churn.joblib"
model = joblib.load(MODEL_PATH)

REQUIRED_COLUMNS = [
    "tempo_contrato_meses",
    "valor_mensal",
    "reclamacoes_6m",
    "atrasos_pagamento_6m",
    "uso_dados_gb_mes",
    "tipo_plano",
    "chamados_suporte_6m",
    "cliente_promocao",
    "engajamento_digital_0_100",
    "regiao",
]

DISPLAY_NAMES = {
    "tempo_contrato_meses": "Tempo de contrato (meses)",
    "valor_mensal": "Valor mensal (R$)",
    "reclamacoes_6m": "Reclamações nos últimos 6 meses",
    "atrasos_pagamento_6m": "Atrasos de pagamento nos últimos 6 meses",
    "uso_dados_gb_mes": "Uso médio de dados (GB/mês)",
    "tipo_plano": "Tipo de plano",
    "chamados_suporte_6m": "Chamados de suporte nos últimos 6 meses",
    "cliente_promocao": "Cliente entrou por promoção?",
    "engajamento_digital_0_100": "Engajamento digital (0 a 100)",
    "regiao": "Região",
}

def classify_risk(prob):
    if prob >= 0.70:
        return "🔴 Alto risco"
    if prob >= 0.40:
        return "🟡 Médio risco"
    return "🟢 Baixo risco"

def action_suggestion(prob):
    if prob >= 0.70:
        return "Contato imediato + oferta de retenção + revisão de relacionamento"
    if prob >= 0.40:
        return "Monitoramento ativo + campanha segmentada"
    return "Acompanhamento normal"

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS and c != "cliente_id"]
    return missing, extra

def prepare_df(df):
    out = df.copy()
    if "cliente_id" not in out.columns:
        out.insert(0, "cliente_id", [f"CLI_{i+1:03d}" for i in range(len(out))])
    return out

st.title("📉 IA aplicada aos negócios | Previsão de cancelamento de clientes")
st.markdown(
    """
    **Contexto:** você é um executivo da operadora fictícia **Connect+** e precisa decidir quais clientes
    devem receber ações de retenção.  
    O sistema já está treinado, mas **você não conhece a base usada, o algoritmo exato nem as métricas de avaliação**.
    """
)

st.warning(
    "Simulação intencional de uma IA 'caixa-preta': use os resultados para decidir, "
    "mas questione a confiabilidade, os riscos e a governança dessa decisão."
)

with st.sidebar:
    st.header("Como usar")
    st.markdown(
        """
        1. Faça upload de um CSV com clientes **ou** preencha manualmente um caso.  
        2. Clique em **Prever risco de cancelamento**.  
        3. Analise quem parece mais crítico.  
        4. Discuta: **você usaria essa IA no negócio sem conhecer métricas e riscos?**
        """
    )
    st.download_button(
        "Baixar CSV de exemplo",
        data=open("data/clientes_exemplo.csv", "rb").read(),
        file_name="clientes_exemplo.csv",
        mime="text/csv",
    )

tab1, tab2 = st.tabs(["Upload de clientes", "Entrada manual"])

with tab1:
    uploaded = st.file_uploader("Envie um arquivo CSV com os dados dos clientes", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df = prepare_df(df)
        missing, extra = validate_columns(df)
        if missing:
            st.error(
                "Seu arquivo não possui todas as colunas obrigatórias. "
                f"Faltando: {', '.join(missing)}"
            )
        else:
            if extra:
                st.info(f"Colunas extras serão ignoradas: {', '.join(extra)}")
            st.subheader("Prévia dos dados")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Prever risco de cancelamento", type="primary"):
                scoring_df = df[["cliente_id"] + REQUIRED_COLUMNS].copy()
                X = scoring_df[REQUIRED_COLUMNS]
                probs = model.predict_proba(X)[:, 1]
                result = scoring_df[["cliente_id"]].copy()
                result["probabilidade_cancelamento"] = probs
                result["classificacao_risco"] = result["probabilidade_cancelamento"].apply(classify_risk)
                result["acao_sugerida"] = result["probabilidade_cancelamento"].apply(action_suggestion)
                result = result.sort_values("probabilidade_cancelamento", ascending=False)

                st.subheader("Resultado")
                st.dataframe(
                    result.assign(
                        probabilidade_cancelamento=lambda d: (d["probabilidade_cancelamento"] * 100).round(1).astype(str) + "%"
                    ),
                    use_container_width=True,
                )

                alto_risco = (probs >= 0.70).sum()
                st.metric("Clientes em alto risco", int(alto_risco))
                st.metric("Maior probabilidade prevista", f"{probs.max()*100:.1f}%")

                csv_out = result.copy()
                csv_out["probabilidade_cancelamento"] = (csv_out["probabilidade_cancelamento"] * 100).round(2)
                csv_bytes = csv_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Baixar resultados",
                    data=csv_bytes,
                    file_name="resultado_previsao_churn.csv",
                    mime="text/csv",
                )

with tab2:
    col1, col2, col3 = st.columns(3)

    with col1:
        tempo_contrato_meses = st.number_input("Tempo de contrato (meses)", min_value=1, max_value=120, value=8)
        valor_mensal = st.number_input("Valor mensal (R$)", min_value=30.0, max_value=500.0, value=149.0, step=1.0)
        reclamacoes_6m = st.number_input("Reclamações nos últimos 6 meses", min_value=0, max_value=20, value=3)
        atrasos_pagamento_6m = st.number_input("Atrasos de pagamento nos últimos 6 meses", min_value=0, max_value=12, value=2)

    with col2:
        uso_dados_gb_mes = st.number_input("Uso médio de dados (GB/mês)", min_value=0.0, max_value=500.0, value=38.0, step=1.0)
        tipo_plano = st.selectbox("Tipo de plano", ["Basico", "Familia", "Premium", "Empresarial"])
        chamados_suporte_6m = st.number_input("Chamados de suporte nos últimos 6 meses", min_value=0, max_value=30, value=4)

    with col3:
        cliente_promocao = st.selectbox("Cliente entrou por promoção?", ["Sim", "Nao"])
        engajamento_digital_0_100 = st.slider("Engajamento digital (0 a 100)", min_value=0, max_value=100, value=35)
        regiao = st.selectbox("Região", ["Capital", "Metropolitana", "Interior"])

    if st.button("Prever risco do cliente", type="primary"):
        manual_df = pd.DataFrame([{
            "tempo_contrato_meses": tempo_contrato_meses,
            "valor_mensal": valor_mensal,
            "reclamacoes_6m": reclamacoes_6m,
            "atrasos_pagamento_6m": atrasos_pagamento_6m,
            "uso_dados_gb_mes": uso_dados_gb_mes,
            "tipo_plano": tipo_plano,
            "chamados_suporte_6m": chamados_suporte_6m,
            "cliente_promocao": cliente_promocao,
            "engajamento_digital_0_100": engajamento_digital_0_100,
            "regiao": regiao,
        }])
        prob = float(model.predict_proba(manual_df)[0, 1])
        st.subheader("Resultado do cliente")
        st.metric("Probabilidade de cancelamento", f"{prob*100:.1f}%")
        st.markdown(f"**Classificação:** {classify_risk(prob)}")
        st.markdown(f"**Ação sugerida:** {action_suggestion(prob)}")

st.divider()
st.subheader("Perguntas para discussão em sala")
st.markdown(
    """
- Você confiaria nessa IA para decidir onde investir dinheiro de retenção?
- Que métricas seriam indispensáveis antes da implantação em produção?
- Qual erro é mais crítico: marcar como risco quem não sairia ou deixar de identificar quem realmente vai cancelar?
- Que riscos de negócio, reputação e governança existem quando o decisor não conhece a qualidade do modelo?
"""
)

with st.expander("Formato esperado do CSV"):
    example = pd.DataFrame([{
        "cliente_id": "C001",
        "tempo_contrato_meses": 8,
        "valor_mensal": 149.0,
        "reclamacoes_6m": 3,
        "atrasos_pagamento_6m": 2,
        "uso_dados_gb_mes": 38.0,
        "tipo_plano": "Basico",
        "chamados_suporte_6m": 4,
        "cliente_promocao": "Sim",
        "engajamento_digital_0_100": 35,
        "regiao": "Metropolitana",
    }])
    st.dataframe(example, use_container_width=True)
