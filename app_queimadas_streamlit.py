import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="IA para Risco de Queimadas", page_icon="🔥", layout="centered")

st.title("🔥 Sistema de IA para Risco de Queimadas")
st.write("""
Este sistema usa dados históricos de focos de queimadas para estimar o risco de queimada
a partir de fatores ambientais e geográficos.
""")

@st.cache_data
def carregar_dados():
    return pd.read_csv("base_queimadas_amostra_balanceada_feira.csv")

@st.cache_resource
def treinar_modelo(df):
    entradas = ["estado", "bioma", "numero_dias_sem_chuva", "precipitacao", "lat", "lon"]
    alvo = "classe_risco"

    X = df[entradas]
    y = df[alvo]

    pre_processamento = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["estado", "bioma"]),
            ("numericas", "passthrough", ["numero_dias_sem_chuva", "precipitacao", "lat", "lon"]),
        ]
    )

    modelo = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
        max_depth=14
    )

    pipeline = Pipeline(steps=[
        ("pre_processamento", pre_processamento),
        ("modelo", modelo)
    ])

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline.fit(X_treino, y_treino)
    pred = pipeline.predict(X_teste)
    acc = accuracy_score(y_teste, pred)

    return pipeline, acc, X_teste, y_teste, pred

df = carregar_dados()
modelo, acc, X_teste, y_teste, pred = treinar_modelo(df)

st.subheader("1. Base de treinamento")
st.write(f"Registros carregados: **{len(df):,}**")
st.write(f"Acurácia aproximada no teste: **{acc:.2%}**")

with st.expander("Ver amostra da base"):
    st.dataframe(df.head(20))

st.subheader("2. Informe os dados da área")

col1, col2 = st.columns(2)

with col1:
    estado = st.selectbox("Estado", sorted(df["estado"].dropna().unique()))
    bioma = st.selectbox("Bioma", sorted(df["bioma"].dropna().unique()))
    dias_sem_chuva = st.slider("Número de dias sem chuva", 0, 120, 10)

with col2:
    precipitacao = st.number_input("Precipitação recente (mm)", min_value=0.0, value=0.0, step=0.1)
    lat = st.number_input("Latitude", value=-8.05, format="%.6f")
    lon = st.number_input("Longitude", value=-34.90, format="%.6f")

novo = pd.DataFrame([{
    "estado": estado,
    "bioma": bioma,
    "numero_dias_sem_chuva": dias_sem_chuva,
    "precipitacao": precipitacao,
    "lat": lat,
    "lon": lon,
}])

if st.button("Calcular risco"):
    classe = modelo.predict(novo)[0]
    probabilidades = modelo.predict_proba(novo)[0]
    classes = modelo.classes_
    prob_df = pd.DataFrame({
        "Classe": classes,
        "Probabilidade": probabilidades
    }).sort_values("Probabilidade", ascending=False)

    st.subheader("3. Resultado da IA")

    if classe == "Alto":
        st.error(f"Risco estimado: {classe}")
    elif classe == "Médio":
        st.warning(f"Risco estimado: {classe}")
    else:
        st.success(f"Risco estimado: {classe}")

    st.write("Probabilidades calculadas pelo modelo:")
    st.dataframe(prob_df, hide_index=True)

    st.write("""
    **Interpretação didática:** o modelo compara os dados informados com padrões históricos
    de queimadas. Em geral, muitos dias sem chuva e pouca precipitação aumentam o risco.
    """)

st.caption("Projeto educacional para Feira de Ciências. O resultado é uma estimativa didática e não substitui sistemas oficiais de monitoramento.")