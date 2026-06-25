import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="IA para Risco de Queimadas - INPE",
    page_icon="🔥",
    layout="wide",
)

DATA_FILE = "base_queimadas_INPE_2020_2025_didatica_limpa.csv"
RISK_ORDER = ["Baixo", "Médio", "Alto"]
RISK_COLOR_MAP = {"Baixo": "green", "Médio": "orange", "Alto": "red"}
MONTH_NAMES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tipos e remove eventuais registros incompletos remanescentes."""
    df = df.copy()

    required = [
        "ano", "mes", "estado", "municipio", "bioma", "latitude", "longitude",
        "numero_dias_sem_chuva", "precipitacao", "area_industrial",
        "risco_fogo", "classe_risco_fogo",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"A base não possui as colunas obrigatórias: {missing}")

    numeric_cols = [
        "ano", "mes", "latitude", "longitude",
        "numero_dias_sem_chuva", "precipitacao", "risco_fogo",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["estado", "municipio", "bioma", "area_industrial", "classe_risco_fogo"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=required)
    df = df[
        (df["risco_fogo"].between(0, 1))
        & (df["numero_dias_sem_chuva"] >= 0)
        & (df["precipitacao"] >= 0)
        & (df["latitude"].between(-90, 90))
        & (df["longitude"].between(-180, 180))
    ]
    df["ano"] = df["ano"].astype(int)
    df["mes"] = df["mes"].astype(int)
    df["mes_nome"] = df["mes"].map(MONTH_NAMES)
    df["classe_risco_fogo"] = pd.Categorical(
        df["classe_risco_fogo"], categories=RISK_ORDER, ordered=True
    )
    return df


@st.cache_data(show_spinner="Carregando base de dados...")
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_columns(df)


@st.cache_data(show_spinner="Carregando base enviada...")
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return _normalize_columns(df)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros da análise")

    years = sorted(df["ano"].dropna().unique().tolist())
    selected_years = st.sidebar.multiselect("Ano", years, default=years)

    states = sorted(df["estado"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("Estado", states, default=states)

    biomes = sorted(df["bioma"].dropna().unique().tolist())
    selected_biomes = st.sidebar.multiselect("Bioma", biomes, default=biomes)

    classes = [c for c in RISK_ORDER if c in df["classe_risco_fogo"].dropna().unique().tolist()]
    selected_classes = st.sidebar.multiselect("Classe de risco", classes, default=classes)

    municipality_text = st.sidebar.text_input(
        "Filtrar município por texto", value="", placeholder="Ex.: Jussara, Balsas, Porto Velho"
    )

    filtered = df[
        df["ano"].isin(selected_years)
        & df["estado"].isin(selected_states)
        & df["bioma"].isin(selected_biomes)
        & df["classe_risco_fogo"].isin(selected_classes)
    ].copy()

    if municipality_text.strip():
        filtered = filtered[
            filtered["municipio"].str.contains(municipality_text.strip(), case=False, na=False)
        ]

    return filtered


def train_model(
    df_model: pd.DataFrame,
    model_name: str,
    sample_size: int,
    include_municipio: bool,
    random_state: int,
    max_depth_tree: int,
    max_depth_forest: int,
    n_estimators: int,
):
    target = "classe_risco_fogo"
    numeric_features = [
        "latitude", "longitude", "mes", "numero_dias_sem_chuva", "precipitacao"
    ]
    categorical_features = ["estado", "bioma", "area_industrial"]
    if include_municipio:
        categorical_features.append("municipio")

    features = categorical_features + numeric_features
    df_train = df_model.dropna(subset=features + [target]).copy()
    df_train = df_train[df_train[target].isin(RISK_ORDER)]

    if sample_size < len(df_train):
        # Amostragem estratificada simples para preservar a proporção das classes.
        parts = []
        for _, group in df_train.groupby(target, observed=False):
            n = max(1, int(round(sample_size * len(group) / len(df_train))))
            n = min(n, len(group))
            parts.append(group.sample(n=n, random_state=random_state))
        df_train = pd.concat(parts).sample(frac=1, random_state=random_state)

    X = df_train[features]
    y = df_train[target].astype(str)
    labels = [label for label in RISK_ORDER if label in sorted(y.unique().tolist())]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("numericas", "passthrough", numeric_features),
        ]
    )

    if model_name == "Árvore de Decisão":
        clf = DecisionTreeClassifier(
            max_depth=max_depth_tree,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth_forest,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    pipeline = Pipeline(steps=[("preprocessamento", preprocessor), ("modelo", clf)])
    pipeline.fit(X, y)

    metadata = {
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target,
        "classes": labels,
        "model_name": model_name,
        "sample_size_used": len(df_train),
    }

    return pipeline, metadata


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("🔥 Sistema de IA para Classificação do Risco de Queimadas")
st.caption("Projeto didático com dados do INPE — focos de queimadas e risco de fogo no Brasil")

with st.sidebar:
    st.markdown("### Base de dados")
    uploaded_file = st.file_uploader(
        "Carregar outro CSV, se desejar", type=["csv"], help="O sistema já procura a base limpa no mesmo diretório do app."
    )

try:
    if uploaded_file is not None:
        df = load_data_from_upload(uploaded_file)
    elif os.path.exists(DATA_FILE):
        df = load_data_from_path(DATA_FILE)
    else:
        st.error(
            "A base não foi encontrada. Coloque o arquivo 'base_queimadas_INPE_2020_2025_didatica_limpa.csv' "
            "na mesma pasta do app.py ou carregue um CSV pela barra lateral."
        )
        st.stop()
except Exception as exc:
    st.error(f"Não foi possível carregar a base: {exc}")
    st.stop()

filtered_df = filter_dataframe(df)

st.sidebar.divider()
st.sidebar.metric("Registros filtrados", f"{len(filtered_df):,}".replace(",", "."))
st.sidebar.download_button(
    "Baixar dados filtrados",
    data=dataframe_to_csv_bytes(filtered_df),
    file_name="dados_filtrados_queimadas.csv",
    mime="text/csv",
    disabled=filtered_df.empty,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Sobre o projeto",
    "2. Explorar os dados",
    "3. Treinar a IA",
    "4. Simular risco",
])

with tab1:
    st.header("Objetivo científico")
    st.write(
        "Este sistema usa dados de focos de queimadas e incêndios florestais do INPE para "
        "classificar o risco de fogo em **baixo**, **médio** ou **alto**. A IA aprende padrões "
        "históricos a partir de variáveis naturais, geográficas e antrópicas."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros na base", f"{len(df):,}".replace(",", "."))
    c2.metric("Anos", f"{df['ano'].min()}–{df['ano'].max()}")
    c3.metric("Estados", df["estado"].nunique())
    c4.metric("Biomas", df["bioma"].nunique())

    st.subheader("Variáveis usadas no projeto")
    variables = pd.DataFrame([
        ["numero_dias_sem_chuva", "Natural", "Número de dias sem chuva até a detecção do foco."],
        ["precipitacao", "Natural", "Precipitação acumulada no dia até o momento da detecção."],
        ["bioma", "Natural/geográfico", "Contexto ambiental do foco."],
        ["mes", "Temporal", "Sazonalidade: meses secos e chuvosos."],
        ["latitude / longitude", "Geográfico", "Localização do foco detectado."],
        ["estado / municipio", "Geográfico/antrópico indireto", "Localização político-administrativa."],
        ["area_industrial", "Antrópico", "Indica se o foco está associado a uma área industrial cadastrada."],
        ["classe_risco_fogo", "Rótulo supervisionado", "Classe criada a partir do valor numérico de risco_fogo."],
    ], columns=["Atributo", "Tipo", "Explicação"])
    st.dataframe(variables, use_container_width=True, hide_index=True)

    st.info(
        "Formulação correta para a Feira: o sistema estima a **classe de risco de fogo** a partir "
        "das condições informadas. Ele não afirma, sozinho, que uma queimada futura ocorrerá ou não."
    )

    with st.expander("Como o rótulo foi criado"):
        st.write(
            "O atributo `risco_fogo`, que varia de 0 a 1, foi transformado em três classes para "
            "facilitar a explicação didática."
        )
        st.dataframe(
            pd.DataFrame({
                "Valor de risco_fogo": ["0,00 a 0,33", "0,34 a 0,66", "0,67 a 1,00"],
                "classe_risco_fogo": ["Baixo", "Médio", "Alto"],
            }),
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    st.header("Exploração visual dos dados")
    if filtered_df.empty:
        st.warning("Nenhum registro encontrado com os filtros atuais.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registros", f"{len(filtered_df):,}".replace(",", "."))
        m2.metric("Média de dias sem chuva", f"{filtered_df['numero_dias_sem_chuva'].mean():.1f}")
        m3.metric("Precipitação média", f"{filtered_df['precipitacao'].mean():.2f} mm")
        m4.metric("Risco médio", f"{filtered_df['risco_fogo'].mean():.2f}")

        st.subheader("Distribuição das classes de risco")
        risk_counts = (
            filtered_df["classe_risco_fogo"]
            .value_counts()
            .reindex(RISK_ORDER)
            .dropna()
            .reset_index()
        )
        risk_counts.columns = ["classe_risco_fogo", "registros"]
        fig_risk = px.bar(
            risk_counts,
            x="classe_risco_fogo",
            y="registros",
            text="registros",
            labels={"classe_risco_fogo": "Classe de risco", "registros": "Registros"},
            title="Quantidade de registros por classe de risco",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            by_biome = (
                filtered_df.groupby(["bioma", "classe_risco_fogo"], observed=False)
                .size()
                .reset_index(name="registros")
            )
            fig_biome = px.bar(
                by_biome,
                x="bioma",
                y="registros",
                color="classe_risco_fogo",
                barmode="group",
                labels={"bioma": "Bioma", "registros": "Registros", "classe_risco_fogo": "Classe"},
                title="Classes de risco por bioma",
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={"classe_risco_fogo": RISK_ORDER},
            )
            st.plotly_chart(fig_biome, use_container_width=True)

        with c2:
            monthly = (
                filtered_df.groupby(["mes", "mes_nome", "classe_risco_fogo"], observed=False)
                .size()
                .reset_index(name="registros")
                .sort_values("mes")
            )
            fig_month = px.line(
                monthly,
                x="mes_nome",
                y="registros",
                color="classe_risco_fogo",
                markers=True,
                labels={"mes_nome": "Mês", "registros": "Registros", "classe_risco_fogo": "Classe"},
                title="Sazonalidade do risco por mês",
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={"classe_risco_fogo": RISK_ORDER},
            )
            st.plotly_chart(fig_month, use_container_width=True)

        st.subheader("Relação entre fatores climáticos e risco")
        scatter_sample_size = min(5000, len(filtered_df))
        scatter_df = filtered_df.sample(scatter_sample_size, random_state=42) if len(filtered_df) > scatter_sample_size else filtered_df

        c3, c4 = st.columns(2)
        with c3:
            fig_dry = px.scatter(
                scatter_df,
                x="numero_dias_sem_chuva",
                y="risco_fogo",
                color="classe_risco_fogo",
                hover_data=["estado", "municipio", "bioma", "ano", "mes"],
                labels={
                    "numero_dias_sem_chuva": "Dias sem chuva",
                    "risco_fogo": "Risco de fogo",
                    "classe_risco_fogo": "Classe",
                },
                title="Dias sem chuva versus risco de fogo",
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={"classe_risco_fogo": RISK_ORDER},
            )
            st.plotly_chart(fig_dry, use_container_width=True)

        with c4:
            fig_rain = px.scatter(
                scatter_df,
                x="precipitacao",
                y="risco_fogo",
                color="classe_risco_fogo",
                hover_data=["estado", "municipio", "bioma", "ano", "mes"],
                labels={
                    "precipitacao": "Precipitação no dia até a detecção (mm)",
                    "risco_fogo": "Risco de fogo",
                    "classe_risco_fogo": "Classe",
                },
                title="Precipitação versus risco de fogo",
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={"classe_risco_fogo": RISK_ORDER},
            )
            st.plotly_chart(fig_rain, use_container_width=True)

        st.subheader("Mapa dos focos filtrados")
        map_size = st.slider("Quantidade máxima de pontos no mapa", 500, 10000, 3000, step=500)
        map_df = filtered_df.sample(min(map_size, len(filtered_df)), random_state=42)
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="classe_risco_fogo",
            hover_name="municipio",
            hover_data=["estado", "bioma", "ano", "mes", "numero_dias_sem_chuva", "precipitacao", "risco_fogo"],
            zoom=3,
            height=550,
            title="Distribuição espacial dos focos",
            color_discrete_map=RISK_COLOR_MAP,
            category_orders={"classe_risco_fogo": RISK_ORDER},
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.header("Treinar a IA")
    st.write(
        "Nesta etapa, o sistema treina um modelo supervisionado. A entrada são os fatores naturais, "
        "geográficos e antrópicos. A saída esperada é `classe_risco_fogo`."
    )

    use_filtered = st.checkbox(
        "Treinar usando apenas os dados filtrados na barra lateral",
        value=False,
        help="Use essa opção para treinar um modelo focado em um estado, bioma ou período específico.",
    )
    df_for_training = filtered_df if use_filtered else df

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Modelo", ["Árvore de Decisão", "Random Forest"], index=1)
    with c2:
        max_records = max(1000, min(100000, len(df_for_training)))
        default_records = min(60000, max_records)
        sample_size = st.slider("Registros para treino", 1000, max_records, default_records, step=1000)
    with c3:
        include_municipio = st.checkbox(
            "Incluir município no modelo",
            value=False,
            help="Pode melhorar o ajuste local, mas deixa o treinamento mais pesado e menos simples de explicar.",
        )

    with st.expander("Parâmetros avançados"):
        random_state = st.number_input("Semente aleatória", min_value=0, max_value=9999, value=42, step=1)
        max_depth_tree = st.slider("Profundidade máxima da Árvore de Decisão", 2, 15, 6)
        max_depth_forest = st.slider("Profundidade máxima da Random Forest", 3, 25, 12)
        n_estimators = st.slider("Número de árvores da Random Forest", 50, 300, 120, step=10)

    st.warning(
        "Para evitar vazamento de informação, o modelo **não usa** `risco_fogo` como entrada. "
        "Ele usa `classe_risco_fogo` apenas como rótulo a ser aprendido. O campo `frp` também fica fora "
        "do modelo inicial, porque mede a potência radiativa do fogo já detectado."
    )

    if len(df_for_training) < 1000:
        st.error("Os filtros atuais deixaram poucos dados para treinamento. Amplie os filtros ou use a base completa.")
    else:
        if st.button("Treinar modelo", type="primary"):
            with st.spinner("Treinando modelo de IA..."):
                try:
                    pipeline, metadata = train_model(
                        df_for_training,
                        model_name=model_name,
                        sample_size=sample_size,
                        include_municipio=include_municipio,
                        random_state=int(random_state),
                        max_depth_tree=max_depth_tree,
                        max_depth_forest=max_depth_forest,
                        n_estimators=n_estimators,
                    )
                    st.session_state["modelo_queimadas"] = pipeline
                    st.session_state["metadata_modelo"] = metadata
                    st.success("Modelo treinado com sucesso. Agora use a aba **Simular risco**.")
                except Exception as exc:
                    st.error(f"Falha no treinamento: {exc}")

with tab4:
    st.header("Simulador de risco")
    st.write(
        "Informe uma situação hipotética. O modelo treinado estimará a classe de risco com base nos padrões aprendidos."
    )

    if "modelo_queimadas" not in st.session_state:
        st.info("Treine um modelo na aba **Treinar a IA** antes de usar o simulador.")
    else:
        model = st.session_state["modelo_queimadas"]
        metadata = st.session_state["metadata_modelo"]
        features = metadata["features"]

        with st.form("form_simulacao"):
            c1, c2, c3 = st.columns(3)
            with c1:
                estado = st.selectbox("Estado", sorted(df["estado"].unique().tolist()))
                bioma = st.selectbox("Bioma", sorted(df["bioma"].unique().tolist()))
                mes = st.slider("Mês", 1, 12, 9, format="%d")
            with c2:
                dias_sem_chuva = st.number_input("Número de dias sem chuva", min_value=0.0, max_value=250.0, value=15.0, step=1.0)
                precipitacao = st.number_input("Precipitação no dia até a detecção (mm)", min_value=0.0, max_value=300.0, value=0.0, step=0.1)
                area_industrial = st.selectbox("Área industrial?", ["Não", "Sim"])
            with c3:
                ref = df[(df["estado"] == estado) & (df["bioma"] == bioma)]
                if ref.empty:
                    ref = df[df["estado"] == estado]
                default_lat = float(ref["latitude"].median()) if not ref.empty else -15.0
                default_lon = float(ref["longitude"].median()) if not ref.empty else -55.0
                latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, step=0.01, format="%.5f")
                longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, step=0.01, format="%.5f")

            municipio = None
            if "municipio" in features:
                municipios_estado = sorted(df[df["estado"] == estado]["municipio"].unique().tolist())
                municipio = st.selectbox("Município", municipios_estado)

            submitted = st.form_submit_button("Classificar risco", type="primary")

        if submitted:
            input_row = {
                "estado": estado,
                "bioma": bioma,
                "area_industrial": area_industrial,
                "latitude": latitude,
                "longitude": longitude,
                "mes": mes,
                "numero_dias_sem_chuva": dias_sem_chuva,
                "precipitacao": precipitacao,
            }
            if "municipio" in features:
                input_row["municipio"] = municipio

            input_df = pd.DataFrame([input_row])[features]
            prediction = model.predict(input_df)[0]
            st.success(f"Risco estimado: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                model_classes = model.classes_
                prob_df = pd.DataFrame({"Classe": model_classes, "Probabilidade": probabilities})
                prob_df["Classe"] = pd.Categorical(prob_df["Classe"], categories=RISK_ORDER, ordered=True)
                prob_df = prob_df.sort_values("Classe")

                fig_prob = px.bar(
                    prob_df,
                    x="Classe",
                    y="Probabilidade",
                    text=prob_df["Probabilidade"].map(lambda x: f"{x:.1%}"),
                    labels={"Probabilidade": "Probabilidade estimada"},
                    title="Confiança do modelo por classe",
                )
                fig_prob.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_prob, use_container_width=True)

            explanation_parts = []
            if dias_sem_chuva >= 15:
                explanation_parts.append("muitos dias sem chuva")
            elif dias_sem_chuva <= 3:
                explanation_parts.append("poucos dias sem chuva")
            if precipitacao <= 0.5:
                explanation_parts.append("baixa precipitação no dia")
            elif precipitacao >= 10:
                explanation_parts.append("precipitação relevante no dia")
            if mes in [7, 8, 9, 10]:
                explanation_parts.append("mês associado ao período seco em muitas regiões do Brasil")
            explanation_parts.append(f"bioma {bioma}")

            st.write("**Leitura didática:**")
            st.write(
                "A classificação foi influenciada pela combinação de " + ", ".join(explanation_parts) + "."
            )

        with st.expander("Sugestão de perguntas para a apresentação na Feira"):
            st.markdown(
                """
- O que acontece com o risco quando aumentamos os dias sem chuva?
- A precipitação no dia sempre reduz o risco? Por quê?
- O risco muda entre biomas diferentes?
- Por que a IA pode errar mesmo usando muitos dados?
- Por que é importante avaliar especialmente a classe **Alto**?
                """
            )
