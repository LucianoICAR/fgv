
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Atividade em Equipe: Métricas & Predição", layout="wide")

st.title("🛍️ Atividade em Equipe — Campanha de E-commerce (Foco em **Recall**)")
st.markdown(
    """
    **Cenário:** Marketing de um e-commerce de moda. Uma campanha de e-mails com ofertas personalizadas foi lançada. 
    Um modelo de ML classifica clientes com maior chance de **comprar** após a campanha.  
    O objetivo é **não perder compradores reais** → priorizamos **Recall** (sensibilidade).
    """
)

with st.expander("ℹ️ O que fazer neste app?", expanded=True):
    st.markdown(
        """
        1. Envie a **base de treino/validação** (com a coluna alvo).  
        2. Escolha o **algoritmo** (Logistic, Random Forest ou KNN) e ajuste as opções.  
        3. Veja as **métricas** (Acurácia, Precisão e Recall) e a **Matriz de Confusão** (com validação).  
        4. Envie a **base de predição** (sem alvo) para gerar os **resultados de predição** e **baixar o CSV**.
        """
    )

# --------------------
# Sidebar: parameters
# --------------------
with st.sidebar:
    st.header("Configurações do Modelo")
    model_name = st.selectbox("Algoritmo", ["Logistic Regression", "Random Forest", "KNN"])
    test_size = st.slider("Proporção de teste (validação)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    standardize_numeric = st.checkbox("Padronizar variáveis numéricas (StandardScaler)", value=True)
    n_neighbors = st.slider("K (KNN)", 1, 25, 7, 1)
    rf_n_estimators = st.slider("Árvores (Random Forest)", 50, 600, 300, 50)
    rf_max_depth = st.select_slider("Profundidade máxima (RF)", options=[None, 5, 10, 20, 30], value=None)

# --------------------
# Uploads
# --------------------
st.subheader("1) Base de **Treino/Validação**")
train_file = st.file_uploader("Envie um CSV para treino/validação (com a coluna alvo)", type=["csv"], key="train")

target_col = None
df_train = None
if train_file is not None:
    try:
        df_train = pd.read_csv(train_file)
    except Exception:
        train_file.seek(0)
        df_train = pd.read_csv(train_file, sep=";")
    st.write("Pré-visualização (treino/validação):", df_train.head())
    cols = list(df_train.columns)
    target_col = st.selectbox("Selecione a coluna alvo (y)", cols, index=len(cols)-1 if cols else 0)

st.subheader("2) Base de **Predição** (sem rótulo)")
pred_file = st.file_uploader("Envie um CSV para predição (sem a coluna alvo)", type=["csv"], key="pred")
df_pred = None
if pred_file is not None:
    try:
        df_pred = pd.read_csv(pred_file)
    except Exception:
        pred_file.seek(0)
        df_pred = pd.read_csv(pred_file, sep=";")
    st.write("Pré-visualização (predição):", df_pred.head())

# --------------------
# Training & eval
# --------------------
if df_train is not None and target_col is not None:
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler() if standardize_numeric else "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state)
    else:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    st.markdown("### ✅ Métricas (validação)")
    average = None
    pos_label = None
    if y_test.dtype == "O" or y_test.nunique() != 2:
        average = "macro"
        st.caption("Problema multiclasse detectado — métricas com **média macro**.")
    else:
        labels_sorted = sorted(y_test.unique().tolist())
        pos_label = st.selectbox("Rótulo positivo (para Precisão/Recall)", labels_sorted, index=1 if len(labels_sorted) > 1 else 0)

    acc = accuracy_score(y_test, y_pred)
    if average is None:
        prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    else:
        prec = precision_score(y_test, y_pred, average=average, zero_division=0)
        rec = recall_score(y_test, y_pred, average=average, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{acc:.3f}")
    c2.metric("Precisão", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")

    st.markdown("#### 🧩 Matriz de Confusão (validação)")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    fig = plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(values_format='d')
    plt.title("Matriz de Confusão (base de validação)")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🔮 Predições (base **sem rótulo**)")
    if df_pred is not None:
        X_deploy = df_pred.copy()
        y_pred_deploy = pipe.predict(X_deploy)

        # Probabilities (if available)
        proba_available = hasattr(pipe.named_steps["clf"], "predict_proba")
        if proba_available:
            y_proba = pipe.predict_proba(X_deploy)
            if len(pipe.named_steps["clf"].classes_) == 2:
                # score for selected positive label if possible
                if pos_label is not None and pos_label in pipe.named_steps["clf"].classes_:
                    pos_index = list(pipe.named_steps["clf"].classes_).index(pos_label)
                else:
                    pos_index = 1 if len(pipe.named_steps["clf"].classes_) > 1 else 0
                score = y_proba[:, pos_index]
                df_out = df_pred.copy()
                df_out["Predicao"] = y_pred_deploy
                df_out["Score_Pos"] = score
            else:
                max_scores = y_proba.max(axis=1)
                df_out = df_pred.copy()
                df_out["Predicao"] = y_pred_deploy
                df_out["Score_Pred"] = max_scores
        else:
            df_out = df_pred.copy()
            df_out["Predicao"] = y_pred_deploy

        st.dataframe(df_out.head(50))
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar predições (.csv)", data=csv_bytes, file_name="predicoes.csv", mime="text/csv")

else:
    st.info("Envie a base de treino/validação (com alvo) e selecione a coluna y para continuar.")

st.markdown("---")
st.caption("Dica: **Recall** = VP / (VP + FN). Em campanhas, perder compradores reais (FN) é caro; por isso focamos em alta sensibilidade.")
