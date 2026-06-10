import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="IA para Risco de Queimadas",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Sistema de IA para Estimar Risco de Queimadas")
st.markdown(
    """
    Este aplicativo permite que os alunos carreguem uma **base de treinamento** com dados históricos
    de queimadas e, depois, informem novos dados ambientais para estimar o risco de incêndio/queimada
    em uma determinada área.

    **Fluxo da atividade:** carregar a base → treinar a IA → informar novos dados → interpretar o risco.
    """
)

COLUNAS_ENTRADA = [
    "estado",
    "bioma",
    "numero_dias_sem_chuva",
    "precipitacao",
    "lat",
    "lon",
]
ALVO = "classe_risco"


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para reduzir erros de leitura."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def criar_classe_risco(valor: float) -> str:
    """Transforma risco_fogo numérico em classe didática."""
    if pd.isna(valor):
        return np.nan
    if valor < 0.40:
        return "Baixo"
    if valor < 0.70:
        return "Médio"
    return "Alto"


@st.cache_data(show_spinner=False)
def carregar_csv(arquivo) -> pd.DataFrame:
    """Lê CSV com separador automático para aceitar vírgula ou ponto e vírgula."""
    return pd.read_csv(arquivo, sep=None, engine="python")


def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e valida a base para treinamento."""
    df = normalizar_colunas(df)

    # Se a base não tiver classe_risco, mas tiver risco_fogo, cria a classe automaticamente.
    if ALVO not in df.columns and "risco_fogo" in df.columns:
        df["risco_fogo"] = pd.to_numeric(df["risco_fogo"], errors="coerce")
        df.loc[df["risco_fogo"] == -999, "risco_fogo"] = np.nan
        df[ALVO] = df["risco_fogo"].apply(criar_classe_risco)

    colunas_necessarias = COLUNAS_ENTRADA + [ALVO]
    faltantes = [c for c in colunas_necessarias if c not in df.columns]
    if faltantes:
        st.error(
            "A base enviada não possui todas as colunas necessárias. "
            f"Colunas ausentes: {', '.join(faltantes)}."
        )
        st.stop()

    df = df[colunas_necessarias].copy()

    # Trata -999 como ausência de dado.
    for col in ["numero_dias_sem_chuva", "precipitacao", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -999, col] = np.nan

    # Padroniza texto.
    df["estado"] = df["estado"].astype(str).str.strip().str.upper()
    df["bioma"] = df["bioma"].astype(str).str.strip()
    df[ALVO] = df[ALVO].astype(str).str.strip().str.capitalize()

    # Mantém apenas classes esperadas.
    df = df[df[ALVO].isin(["Baixo", "Médio", "Alto"])]

    # Remove linhas incompletas.
    df = df.dropna(subset=colunas_necessarias)

    return df


@st.cache_resource(show_spinner=False)
def treinar_modelo(df: pd.DataFrame):
    X = df[COLUNAS_ENTRADA]
    y = df[ALVO]

    pre_processamento = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["estado", "bioma"]),
            ("numericas", "passthrough", ["numero_dias_sem_chuva", "precipitacao", "lat", "lon"]),
        ]
    )

    modelo = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("pre_processamento", pre_processamento),
            ("modelo", modelo),
        ]
    )

    # Se alguma classe tiver poucos registros, não usa stratify para evitar erro.
    contagem_classes = y.value_counts()
    usar_stratify = y if contagem_classes.min() >= 2 else None

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=usar_stratify,
    )

    pipeline.fit(X_treino, y_treino)
    pred = pipeline.predict(X_teste)
    acc = accuracy_score(y_teste, pred)

    relatorio = classification_report(y_teste, pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_teste, pred, labels=["Baixo", "Médio", "Alto"])

    return pipeline, acc, relatorio, matriz


st.sidebar.header("📁 Base de treinamento")
st.sidebar.markdown(
    """
    Envie um arquivo CSV contendo, no mínimo, as colunas:

    - `estado`
    - `bioma`
    - `numero_dias_sem_chuva`
    - `precipitacao`
    - `lat`
    - `lon`
    - `classe_risco` **ou** `risco_fogo`
    """
)

arquivo = st.sidebar.file_uploader(
    "Carregue a base de treinamento em CSV",
    type=["csv"],
)

if arquivo is None:
    st.info("👈 Para começar, carregue a base de treinamento no menu lateral.")
    st.stop()

try:
    df_original = carregar_csv(arquivo)
    df = preparar_base(df_original)
except Exception as erro:
    st.error("Não foi possível ler ou preparar a base enviada.")
    st.exception(erro)
    st.stop()

if len(df) < 50:
    st.error("A base ficou com poucos registros válidos após a limpeza. Envie uma base maior ou revise os dados.")
    st.stop()

st.subheader("1. Base carregada e preparada")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Registros válidos", f"{len(df):,}".replace(",", "."))
col_b.metric("Estados", df["estado"].nunique())
col_c.metric("Biomas", df["bioma"].nunique())
col_d.metric("Classes", df[ALVO].nunique())

with st.expander("Ver amostra da base preparada"):
    st.dataframe(df.head(30), use_container_width=True)

st.write("Distribuição das classes na base de treinamento:")
st.dataframe(
    df[ALVO].value_counts().rename_axis("classe_risco").reset_index(name="quantidade"),
    use_container_width=True,
    hide_index=True,
)

st.subheader("2. Treinamento da IA")

with st.spinner("Treinando o modelo de IA..."):
    modelo, acc, relatorio, matriz = treinar_modelo(df)

st.success("Modelo treinado com sucesso.")
st.metric("Acurácia aproximada no conjunto de teste", f"{acc:.2%}")

with st.expander("Ver métricas técnicas do modelo"):
    st.markdown("**Matriz de confusão** — linhas = classe real; colunas = classe prevista.")
    matriz_df = pd.DataFrame(
        matriz,
        index=["Real Baixo", "Real Médio", "Real Alto"],
        columns=["Previsto Baixo", "Previsto Médio", "Previsto Alto"],
    )
    st.dataframe(matriz_df, use_container_width=True)

    relatorio_df = pd.DataFrame(relatorio).transpose()
    st.dataframe(relatorio_df, use_container_width=True)

st.subheader("3. Inserir novos dados para estimar o risco")

estados_disponiveis = sorted(df["estado"].dropna().unique())
biomas_disponiveis = sorted(df["bioma"].dropna().unique())

col1, col2, col3 = st.columns(3)

with col1:
    estado = st.selectbox("Estado", estados_disponiveis)
    bioma = st.selectbox("Bioma", biomas_disponiveis)

with col2:
    dias_sem_chuva = st.slider(
        "Número de dias sem chuva",
        min_value=0,
        max_value=120,
        value=10,
        step=1,
    )
    precipitacao = st.number_input(
        "Precipitação recente (mm)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        format="%.2f",
    )

with col3:
    lat = st.number_input("Latitude", value=-8.050000, format="%.6f")
    lon = st.number_input("Longitude", value=-34.900000, format="%.6f")

novo = pd.DataFrame(
    [
        {
            "estado": estado,
            "bioma": bioma,
            "numero_dias_sem_chuva": dias_sem_chuva,
            "precipitacao": precipitacao,
            "lat": lat,
            "lon": lon,
        }
    ]
)

if st.button("🔥 Calcular risco de queimada", type="primary"):
    classe = modelo.predict(novo)[0]

    resultado = st.container(border=True)
    with resultado:
        st.subheader("Resultado da IA")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(novo)[0]
            classes = modelo.classes_
            prob_df = pd.DataFrame(
                {
                    "Classe": classes,
                    "Probabilidade": probabilidades,
                }
            ).sort_values("Probabilidade", ascending=False)
            prob_principal = float(prob_df.iloc[0]["Probabilidade"])
        else:
            prob_df = None
            prob_principal = np.nan

        if classe == "Alto":
            st.error(f"🚨 Risco estimado: **{classe}**")
        elif classe == "Médio":
            st.warning(f"⚠️ Risco estimado: **{classe}**")
        else:
            st.success(f"✅ Risco estimado: **{classe}**")

        if not np.isnan(prob_principal):
            st.write(f"Confiança aproximada do modelo: **{prob_principal:.2%}**")

        if prob_df is not None:
            st.write("Probabilidades por classe:")
            prob_df["Probabilidade"] = prob_df["Probabilidade"].map(lambda x: f"{x:.2%}")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Interpretação didática:** a IA compara os dados informados com padrões históricos.
            Em geral, maior número de dias sem chuva e menor precipitação tendem a elevar o risco,
            mas o modelo também considera localização e bioma.
            """
        )

st.divider()
st.caption(
    "Projeto educacional para Feira de Ciências. A estimativa é didática e não substitui alertas oficiais, Defesa Civil, INPE ou órgãos ambientais."
)
