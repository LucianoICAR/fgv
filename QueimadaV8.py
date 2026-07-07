import os
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Feira de Ciências CMR - Mapa Inteligente do Fogo: IA na análise dos riscos de queimadas no Brasil",
    page_icon="🔥",
    layout="wide",
)
LOCAL_DATA_CANDIDATES = [
    "base_queimadas_INPE_2020_2025_didatica_limpa_ate_200MB.csv",
    "base_queimadas_INPE_2020_2025_didatica_limpa_ate_200MB.zip",
    "base_queimadas_INPE_2020_2025_didatica_limpa.csv",
    "base_queimadas_INPE_2020_2025_didatica_limpa.zip",
]
RISK_ORDER = ["Baixo", "Médio", "Alto"]
RISK_COLOR_MAP = {"Baixo": "green", "Médio": "orange", "Alto": "red"}
MONTH_NAMES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}
HIGH_RISK_THRESHOLD = 0.70
CSV_DTYPE_HINTS = {
    "estado": "string",
    "municipio": "string",
    "bioma": "string",
    "area_industrial": "string",
    "classe_risco_fogo": "string",
}
TRAINING_NUMERIC_FEATURES = [
    "latitude", "longitude", "mes", "numero_dias_sem_chuva", "precipitacao"
]
TRAINING_CATEGORICAL_FEATURES = ["estado", "bioma", "area_industrial"]
TRAINING_TARGET = "classe_risco_fogo"
TEST_SIZE = 0.20
TRAIN_SIZE = 0.80


def _is_zip_source(file_or_path) -> bool:
    """Identifica se a origem é um arquivo ZIP, seja por caminho, seja por upload."""
    name = str(file_or_path).lower()
    if hasattr(file_or_path, "name"):
        name = str(file_or_path.name).lower()
    return name.endswith(".zip")


def _detect_csv_separator(file_or_path) -> str:
    """Detecta separador provável do CSV sem carregar o arquivo inteiro."""
    sample_bytes = b""

    if isinstance(file_or_path, (str, os.PathLike)):
        with open(file_or_path, "rb") as sample_file:
            sample_bytes = sample_file.read(8192)
    else:
        try:
            current_position = file_or_path.tell()
        except Exception:
            current_position = None
        sample_bytes = file_or_path.read(8192)
        try:
            file_or_path.seek(0 if current_position is None else current_position)
        except Exception:
            pass

    sample = sample_bytes.decode("utf-8", errors="ignore")
    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    candidates = {",": first_line.count(","), ";": first_line.count(";"), "\t": first_line.count("\t")}
    return max(candidates, key=candidates.get) if max(candidates.values(), default=0) > 0 else ","


def read_csv_optimized(file_or_path) -> pd.DataFrame:
    """Lê CSVs grandes com dicas de tipo para reduzir inferências caras de memória."""
    sep = _detect_csv_separator(file_or_path)
    return pd.read_csv(file_or_path, sep=sep, dtype=CSV_DTYPE_HINTS, low_memory=False)


def read_csv_from_zip(file_or_path) -> pd.DataFrame:
    """Lê um ZIP contendo um ou mais arquivos CSV e consolida em uma única tabela."""
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    frames = []
    with zipfile.ZipFile(file_or_path) as zip_ref:
        csv_names = [
            name for name in zip_ref.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not os.path.basename(name).startswith(".")
        ]
        if not csv_names:
            raise ValueError("O arquivo ZIP não contém nenhum CSV.")

        for csv_name in sorted(csv_names):
            with zip_ref.open(csv_name) as csv_file:
                frames.append(read_csv_optimized(csv_file))

    if not frames:
        raise ValueError("Carregue uma base para fins de aprendizado e exploração dos dados.")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def read_table(file_or_path) -> pd.DataFrame:
    """Lê a base em CSV direto ou em ZIP contendo CSV."""
    if _is_zip_source(file_or_path):
        return read_csv_from_zip(file_or_path)
    return read_csv_optimized(file_or_path)


def find_local_data_file():
    """Procura a base local em ordem de preferência, priorizando a base didática de até 200 MB."""
    for candidate in LOCAL_DATA_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def join_pt_br(items):
    items = [str(item) for item in items if str(item).strip()]
    if not items:
        return "-"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " e " + items[-1]


def format_month_ranges(months):
    months = sorted({int(m) for m in months if pd.notna(m)})
    if not months:
        return "-"

    ranges = []
    start = previous = months[0]
    for month in months[1:]:
        if month == previous + 1:
            previous = month
        else:
            ranges.append((start, previous))
            start = previous = month
    ranges.append((start, previous))

    labels = []
    for start, end in ranges:
        if start == end:
            labels.append(MONTH_NAMES[start])
        else:
            labels.append(f"{MONTH_NAMES[start]} a {MONTH_NAMES[end]}")
    return join_pt_br(labels)


def build_monthly_biome_distribution(df_analysis: pd.DataFrame):
    """Gera a tabela-resumo e a tabela detalhada mensal de risco por bioma."""
    monthly = (
        df_analysis.groupby(["bioma", "mes"], observed=True)
        .agg(
            registros=("risco_fogo", "size"),
            risco_medio=("risco_fogo", "mean"),
            dias_sem_chuva_medio=("numero_dias_sem_chuva", "mean"),
            precipitacao_media=("precipitacao", "mean"),
            registros_alto_risco=("risco_fogo", lambda s: int((s >= HIGH_RISK_THRESHOLD).sum())),
        )
        .reset_index()
        .sort_values(["bioma", "mes"])
    )
    monthly["mes_nome"] = monthly["mes"].map(MONTH_NAMES)
    monthly["percentual_alto_risco"] = monthly["registros_alto_risco"] / monthly["registros"]

    summary_rows = []
    for biome, biome_monthly in monthly.groupby("bioma", observed=True):
        biome_monthly = biome_monthly.sort_values("mes")
        peak = biome_monthly.loc[biome_monthly["risco_medio"].idxmax()]

        critical = biome_monthly[biome_monthly["risco_medio"] >= HIGH_RISK_THRESHOLD]
        if critical.empty:
            top_months = biome_monthly.sort_values("risco_medio", ascending=False).head(min(3, len(biome_monthly)))
            critical_months = sorted(top_months["mes"].tolist())
            period_label = "Sem mês ≥ 0,70; maior atenção em " + format_month_ranges(critical_months)
        else:
            critical_months = sorted(critical["mes"].tolist())
            period_label = format_month_ranges(critical_months)

        state_base = df_analysis[
            (df_analysis["bioma"] == biome)
            & (df_analysis["mes"].isin(critical_months))
            & (df_analysis["risco_fogo"] >= HIGH_RISK_THRESHOLD)
        ]
        if state_base.empty:
            state_base = df_analysis[
                (df_analysis["bioma"] == biome)
                & (df_analysis["risco_fogo"] >= HIGH_RISK_THRESHOLD)
            ]

        if state_base.empty:
            top_state = "Sem registros de alto risco"
            top_state_records = 0
        else:
            state_counts = state_base.groupby("estado", observed=True).size().sort_values(ascending=False)
            top_state = str(state_counts.index[0])
            top_state_records = int(state_counts.iloc[0])

        summary_rows.append({
            "Bioma": biome,
            "Período crítico observado": period_label,
            "Pico de risco": f"{MONTH_NAMES[int(peak['mes'])]} ({peak['risco_medio']:.3f})",
            "Registros no pico": int(peak["registros"]),
            "% alto risco no pico": f"{peak['percentual_alto_risco']:.1%}",
            "Média de dias sem chuva no pico": round(float(peak["dias_sem_chuva_medio"]), 1),
            "Precipitação média no pico (mm)": round(float(peak["precipitacao_media"]), 2),
            "Região mais afetada nos dados": top_state,
            "Registros de alto risco na região": top_state_records,
        })

    summary = pd.DataFrame(summary_rows).sort_values("Bioma")

    detailed = monthly.copy()
    detailed = detailed[[
        "bioma", "mes", "mes_nome", "registros", "risco_medio", "registros_alto_risco",
        "percentual_alto_risco", "dias_sem_chuva_medio", "precipitacao_media",
    ]]
    detailed = detailed.rename(columns={
        "bioma": "Bioma",
        "mes": "Mês",
        "mes_nome": "Mês nome",
        "registros": "Registros",
        "risco_medio": "Risco médio",
        "registros_alto_risco": "Registros de alto risco",
        "percentual_alto_risco": "% alto risco",
        "dias_sem_chuva_medio": "Média de dias sem chuva",
        "precipitacao_media": "Precipitação média (mm)",
    })
    detailed["Risco médio"] = detailed["Risco médio"].round(3)
    detailed["% alto risco"] = detailed["% alto risco"].map(lambda x: f"{x:.1%}")
    detailed["Média de dias sem chuva"] = detailed["Média de dias sem chuva"].round(1)
    detailed["Precipitação média (mm)"] = detailed["Precipitação média (mm)"].round(2)

    return summary, detailed


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tipos, cria campos didáticos quando necessário e remove registros inválidos."""
    df = df.copy()

    # Compatibilidade com os arquivos ZIP originais do INPE:
    # eles trazem data_pas e id_area_industrial, mas não necessariamente ano, mês,
    # area_industrial e classe_risco_fogo já prontos para a interface didática.
    if "ano" not in df.columns or "mes" not in df.columns:
        date_col = next((c for c in ["data_pas", "data_hora_gmt", "data_hora", "data"] if c in df.columns), None)
        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if "ano" not in df.columns:
                df["ano"] = dates.dt.year
            if "mes" not in df.columns:
                df["mes"] = dates.dt.month

    if "area_industrial" not in df.columns and "id_area_industrial" in df.columns:
        industrial = pd.to_numeric(df["id_area_industrial"], errors="coerce").fillna(0)
        df["area_industrial"] = industrial.apply(lambda value: "Sim" if value > 0 else "Não")

    if "classe_risco_fogo" not in df.columns and "risco_fogo" in df.columns:
        risk_values = pd.to_numeric(df["risco_fogo"], errors="coerce")
        df["classe_risco_fogo"] = pd.cut(
            risk_values,
            bins=[-0.001, 0.33, 0.66, 1.0],
            labels=RISK_ORDER,
            include_lowest=True,
        ).astype("string")

    required = [
        "ano", "mes", "estado", "municipio", "bioma", "latitude", "longitude",
        "numero_dias_sem_chuva", "precipitacao", "area_industrial",
        "risco_fogo", "classe_risco_fogo",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "A base não possui as colunas obrigatórias: "
            f"{missing}. Use a base didática limpa ou um ZIP do INPE contendo CSV com os campos necessários."
        )

    numeric_cols = [
        "ano", "mes", "latitude", "longitude",
        "numero_dias_sem_chuva", "precipitacao", "risco_fogo",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = ["estado", "municipio", "bioma", "area_industrial", "classe_risco_fogo"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})

    df = df.dropna(subset=required)
    df = df[
        (df["risco_fogo"].between(0, 1))
        & (df["numero_dias_sem_chuva"] >= 0)
        & (df["precipitacao"] >= 0)
        & (df["latitude"].between(-90, 90))
        & (df["longitude"].between(-180, 180))
    ]
    df["ano"] = df["ano"].astype("int16")
    df["mes"] = df["mes"].astype("int8")
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    df["numero_dias_sem_chuva"] = df["numero_dias_sem_chuva"].astype("float32")
    df["precipitacao"] = df["precipitacao"].astype("float32")
    df["risco_fogo"] = df["risco_fogo"].astype("float32")
    df["mes_nome"] = df["mes"].map(MONTH_NAMES)
    for col in ["estado", "bioma", "area_industrial"]:
        df[col] = df[col].astype("category")
    df["classe_risco_fogo"] = pd.Categorical(
        df["classe_risco_fogo"], categories=RISK_ORDER, ordered=True
    )
    return df


@st.cache_data(show_spinner="Carregando base de dados...")
def load_data_from_path(path: str) -> pd.DataFrame:
    df = read_table(path)
    return _normalize_columns(df)


@st.cache_data(show_spinner="Carregando base enviada...")
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = read_table(uploaded_file)
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
            filtered["municipio"].astype("string").str.contains(municipality_text.strip(), case=False, na=False)
        ]

    return filtered


def predict_with_high_risk_threshold(model, X, threshold_alto: float = 0.40):
    """
    Aplica uma regra de alerta para a classe Alto.

    Regra padrão do modelo: escolhe a classe com maior probabilidade.
    Regra ajustada: se a probabilidade de Alto atingir o limiar definido,
    a previsão final passa a ser Alto. Isso tende a aumentar o recall de Alto,
    ainda que possa reduzir a precisão.
    """
    default_predictions = model.predict(X)

    if not hasattr(model, "predict_proba"):
        return default_predictions

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if "Alto" not in classes:
        return default_predictions

    alto_index = classes.index("Alto")
    alto_probabilities = probabilities[:, alto_index]

    adjusted_predictions = default_predictions.copy()
    adjusted_predictions[alto_probabilities >= threshold_alto] = "Alto"

    return adjusted_predictions


def train_model(
    df_model: pd.DataFrame,
    model_name: str,
    sample_size: int,
    random_state: int,
    max_depth_tree: int,
    max_depth_forest: int,
    n_estimators: int,
    threshold_alto: float,
):
    target = TRAINING_TARGET
    numeric_features = TRAINING_NUMERIC_FEATURES
    # O município foi retirado do treinamento para evitar cardinalidade alta
    # no OneHotEncoder, que pode deixar o app pesado ou causar erro no Streamlit.
    categorical_features = TRAINING_CATEGORICAL_FEATURES

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

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "A base selecionada possui apenas uma classe de risco. "
            "Para treinar e calcular a acurácia, mantenha pelo menos duas classes nos filtros."
        )
    if class_counts.min() < 2:
        raise ValueError(
            "Há classe de risco com menos de 2 registros na base selecionada. "
            "Amplie os filtros ou use a base completa para permitir a divisão entre treino e teste."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    labels = [label for label in RISK_ORDER if label in sorted(y.unique().tolist())]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features),
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
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    pipeline = Pipeline(steps=[("preprocessamento", preprocessor), ("modelo", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = predict_with_high_risk_threshold(
        pipeline,
        X_test,
        threshold_alto=threshold_alto,
    )
    accuracy = accuracy_score(y_test, y_pred)

    # Métricas focadas na classe Alto, que é a classe mais crítica para o projeto.
    recall_alto = recall_score(
        y_test,
        y_pred,
        labels=["Alto"],
        average="macro",
        zero_division=0,
    )
    precision_alto = precision_score(
        y_test,
        y_pred,
        labels=["Alto"],
        average="macro",
        zero_division=0,
    )
    f1_alto = f1_score(
        y_test,
        y_pred,
        labels=["Alto"],
        average="macro",
        zero_division=0,
    )

    cm_labels = [label for label in RISK_ORDER if label in sorted(set(y_test) | set(y_pred))]
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
    confusion_matrix_df = pd.DataFrame(
        cm,
        index=[f"Real {label}" for label in cm_labels],
        columns=[f"Previsto {label}" for label in cm_labels],
    )

    metadata = {
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target,
        "classes": labels,
        "model_name": model_name,
        "sample_size_used": len(df_train),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": accuracy,
        "recall_alto": recall_alto,
        "precision_alto": precision_alto,
        "f1_alto": f1_alto,
        "confusion_matrix": confusion_matrix_df,
        "threshold_alto": threshold_alto,
        "train_percent": TRAIN_SIZE,
        "test_percent": TEST_SIZE,
    }

    return pipeline, metadata, accuracy


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.title("🔥 Mapa Inteligente do Fogo: IA na análise dos riscos de queimadas no Brasil 🔥")
st.caption("Focos de queimadas e risco de fogo no Brasil")

with st.sidebar:
    st.markdown("Base de dados")
    uploaded_file = st.file_uploader(
        "Carregar CSV",
        type=["csv"],
        help=(
            "O sistema aceita CSV direto"
            "Para a feira, recomenda-se usar a base didática de até 200 MB."
        ),
    )
    st.caption("Carregue uma base de dados .CSV para fins de treinamento")

try:
    loaded_source = None
    if uploaded_file is not None:
        loaded_source = uploaded_file.name
        df = load_data_from_upload(uploaded_file)
    else:
        local_data_file = find_local_data_file()
        if local_data_file is not None:
            loaded_source = local_data_file
            df = load_data_from_path(local_data_file)
        else:
            st.error(
                "A base não foi encontrada. Coloque uma destas bases na mesma pasta do app.py: "
                + ", ".join(LOCAL_DATA_CANDIDATES)
                + ". Você deve carregar um CSV pela barra lateral."
            )
            st.stop()
except Exception as exc:
    st.error(f"Carregue uma base de dados: {exc}")
    st.stop()

st.sidebar.caption(f"Base carregada: {loaded_source}")

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
    "4. Simular risco de queimada",
])

with tab1:
    st.header("Objetivo científico")
    st.write(
        "Este sistema usa dados de focos de queimadas e incêndios florestais do Programa Quimadas do INPE para "
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
        ["area_industrial", "Antrópico", "Indica se o foco está associado a uma área industrial cadastrada; nos arquivos originais do INPE, pode ser derivado de id_area_industrial."],
        ["classe_risco_fogo", "Rótulo supervisionado", "Classe criada a partir do valor numérico de risco_fogo."],
    ], columns=["Atributo", "Tipo", "Explicação"])
    st.dataframe(variables, use_container_width=True, hide_index=True)

    st.info(
        "Formulação correta para a Feira: o sistema estima a **classe de risco de fogo** a partir "
        "das condições informadas. Ele não afirma, sozinho, que uma queimada futura ocorrerá ou não."
    )

    st.subheader("100 primeiros registros extraídos do CSV")
    st.caption(
        "A tabela abaixo mostra uma amostra inicial da base já carregada e padronizada pelo sistema."
    )
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

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

        st.info(
            "Esta aba apresenta análise exploratória e estatística descritiva dos dados. "
            "Ela não executa aprendizado. O treinamento de IA supervisionada ocorre na aba 'Treinar a IA.'"
        )

        st.subheader("Risco de queimadas por mês")
        monthly_risk_chart = (
            filtered_df.groupby(["mes", "mes_nome", "bioma"], observed=True)
            .agg(
                risco_medio=("risco_fogo", "mean"),
                registros=("risco_fogo", "size"),
            )
            .reset_index()
            .sort_values("mes")
        )
        fig_monthly_risk = px.line(
            monthly_risk_chart,
            x="mes_nome",
            y="risco_medio",
            color="bioma",
            markers=True,
            hover_data={"registros": ":,", "risco_medio": ":.3f", "mes": False},
            labels={
                "mes_nome": "Mês",
                "risco_medio": "Risco médio de fogo",
                "bioma": "Bioma",
                "registros": "Registros",
            },
            title="Risco médio de fogo por mês e bioma",
            category_orders={"mes_nome": [MONTH_NAMES[m] for m in range(1, 13)]},
        )
        fig_monthly_risk.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_monthly_risk, use_container_width=True)

        st.subheader("Distribuição Mensal do Risco de Fogo por Bioma")
        st.caption(
            "Tabela calculada com os filtros atuais. O período crítico considera meses com risco médio ≥ 0,70. "
            "A região mais afetada é o estado com maior quantidade de registros de alto risco no período crítico do bioma."
        )
        summary_distribution, detailed_distribution = build_monthly_biome_distribution(filtered_df)
        st.dataframe(summary_distribution, use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Baixar tabela-resumo por bioma",
                data=dataframe_to_csv_bytes(summary_distribution),
                file_name="distribuicao_mensal_risco_fogo_por_bioma.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "Baixar tabela mensal detalhada",
                data=dataframe_to_csv_bytes(detailed_distribution),
                file_name="risco_mensal_detalhado_por_bioma.csv",
                mime="text/csv",
            )

        with st.expander("Ver tabela mensal detalhada por bioma"):
            st.dataframe(detailed_distribution, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Mapa dos focos filtrados")
        map_size = st.slider("Quantidade máxima de pontos no mapa", 500, 1000000, 10000, step=1000)
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
        "geográficos e antrópicos. A saída esperada é `classe_risco_fogo`. "
        "A divisão do experimento é fixa: 80% dos registros selecionados para treino e 20% para teste."
    )

    use_filtered = st.checkbox(
        "Treinar usando apenas os dados filtrados na barra lateral",
        value=False,
        help="Use essa opção para treinar um modelo focado em um estado, bioma ou período específico.",
    )
    df_for_training = filtered_df if use_filtered else df

    training_features = TRAINING_CATEGORICAL_FEATURES + TRAINING_NUMERIC_FEATURES
    training_preview = df_for_training.dropna(subset=training_features + [TRAINING_TARGET])
    training_preview = training_preview[training_preview[TRAINING_TARGET].isin(RISK_ORDER)]
    available_training_records = len(training_preview)

    class_distribution = (
        training_preview[TRAINING_TARGET]
        .value_counts()
        .reindex(RISK_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    class_distribution.columns = ["Classe", "Registros disponíveis"]

    estimated_train_records = int(available_training_records * TRAIN_SIZE)
    estimated_test_records = available_training_records - estimated_train_records

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Registros válidos", f"{available_training_records:,}".replace(",", "."))
    t2.metric("Treino estimado", f"{estimated_train_records:,}".replace(",", "."))
    t3.metric("Teste estimado", f"{estimated_test_records:,}".replace(",", "."))
    t4.metric("Classes disponíveis", int((class_distribution["Registros disponíveis"] > 0).sum()))

    with st.expander("Ver distribuição das classes disponíveis para treinamento"):
        st.dataframe(class_distribution, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("Modelo", ["Árvore de Decisão", "Random Forest"], index=1)
    with c2:
        if available_training_records >= 1000:
            max_records = available_training_records
            if available_training_records <= 300_000:
                default_records = available_training_records
            else:
                default_records = min(available_training_records, max(300_000, int(available_training_records * 0.55)))
            step_records = 50_000 if max_records >= 500_000 else 10_000 if max_records >= 100_000 else 1_000
            min_records = min(1_000, max_records)
            sample_size = st.slider(
                "Registros para o experimento de IA",
                min_value=min_records,
                max_value=max_records,
                value=default_records,
                step=step_records,
                help=(
                    "O valor é calculado a partir dos registros válidos da base carregada. "
                    "Deste total, o sistema separa automaticamente 80% para treino e 20% para teste. "
                    "Usar todos os registros aumenta a representatividade, mas também aumenta o tempo de processamento."
                ),
            )
            st.caption(
                f"Com {sample_size:,} registros no experimento, aproximadamente "
                f"{int(sample_size * TRAIN_SIZE):,} serão usados para treino e "
                f"{sample_size - int(sample_size * TRAIN_SIZE):,} para teste.".replace(",", ".")
            )
        else:
            max_records = available_training_records
            sample_size = available_training_records
            st.info("A base filtrada tem menos de 1.000 registros válidos para treinamento.")

    st.caption(
        "O atributo `municipio` não é usado no treinamento para evitar erro por excesso de categorias "
        "e manter o sistema mais leve para a apresentação. A divisão treino/teste é feita de forma "
        "estratificada, preservando a proporção das classes Baixo, Médio e Alto."
    )

    with st.expander("Parâmetros avançados"):
        random_state = st.number_input("Semente aleatória", min_value=0, max_value=9999, value=42, step=1)
        max_depth_tree = st.slider("Profundidade máxima da Árvore de Decisão", 2, 15, 6)
        max_depth_forest = st.slider("Profundidade máxima da Random Forest", 3, 40, 18)
        n_estimators = st.slider("Número de árvores da Random Forest", 50, 300, 120, step=10)
        threshold_alto = st.slider(
            "Limiar de alerta para classe Alto",
            min_value=0.20,
            max_value=0.70,
            value=0.40,
            step=0.05,
            help=(
                "Quanto menor o limiar, mais sensível o sistema fica para classificar Alto. "
                "Isso tende a aumentar o recall da classe Alto, mas pode reduzir a precisão."
            ),
        )

    st.warning(
        "Para evitar vazamento de informação, o modelo **não usa** `risco_fogo` como entrada. "
        "Ele usa `classe_risco_fogo` apenas como rótulo a ser aprendido. O campo `frp` também fica fora "
        "do modelo inicial, porque mede a potência radiativa do fogo já detectado."
    )

    if available_training_records < 1000:
        st.error("Os filtros atuais deixaram poucos dados válidos para treinamento. Amplie os filtros ou use a base completa.")
    else:
        if st.button("Treinar modelo", type="primary"):
            with st.spinner("Treinando modelo de IA..."):
                try:
                    pipeline, metadata, accuracy = train_model(
                        df_for_training,
                        model_name=model_name,
                        sample_size=sample_size,
                        random_state=int(random_state),
                        max_depth_tree=max_depth_tree,
                        max_depth_forest=max_depth_forest,
                        n_estimators=n_estimators,
                        threshold_alto=threshold_alto,
                    )
                    st.session_state["modelo_queimadas"] = pipeline
                    st.session_state["metadata_modelo"] = metadata
                    st.success("Modelo treinado com sucesso. Agora use a aba **Simular risco**.")
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Acurácia geral", f"{accuracy:.1%}")
                    a2.metric("Recall da classe Alto", f"{metadata['recall_alto']:.1%}")
                    a3.metric("Precisão da classe Alto", f"{metadata['precision_alto']:.1%}")
                    a4.metric("F1-score da classe Alto", f"{metadata['f1_alto']:.1%}")
                    st.caption(
                        f"Regra de alerta usada: classificar como **Alto** sempre que "
                        f"a probabilidade da classe Alto for maior ou igual a "
                        f"{metadata['threshold_alto']:.0%}."
                    )

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Registros do experimento", f"{metadata['sample_size_used']:,}".replace(",", "."))
                    r2.metric("Registros de treino", f"{metadata['train_size']:,}".replace(",", "."))
                    r3.metric("Registros de teste", f"{metadata['test_size']:,}".replace(",", "."))

                    st.caption(
                        "A acurácia mede o acerto geral. O recall da classe Alto mede quantos casos "
                        "realmente classificados como Alto foram encontrados pelo modelo. Para risco de "
                        "queimadas, esse indicador é especialmente importante porque reduz a chance de "
                        "deixar passar situações críticas. A divisão treino/teste é estratificada para "
                        "preservar a proporção das classes Baixo, Médio e Alto."
                    )

                    with st.expander("Ver matriz de confusão do teste"):
                        st.write(
                            "Na matriz de confusão, as linhas representam a classe real e as colunas "
                            "representam a classe prevista pelo modelo."
                        )
                        st.dataframe(metadata["confusion_matrix"], use_container_width=True)
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
            input_df = pd.DataFrame([input_row])[features]
            threshold_alto = metadata.get("threshold_alto", 0.40)
            prediction = predict_with_high_risk_threshold(
                model,
                input_df,
                threshold_alto=threshold_alto,
            )[0]
            st.success(f"Risco estimado: **{prediction}**")
            st.caption(
                f"Regra de alerta aplicada: se a probabilidade de Alto for maior ou igual a "
                f"{threshold_alto:.0%}, o sistema classifica como Alto."
            )

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
                st.caption(
                    "No modo de alerta, a decisão final pode ser Alto mesmo quando Alto não é a maior "
                    "probabilidade, desde que ultrapasse o limiar definido no treinamento."
                )

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
                """
            )
