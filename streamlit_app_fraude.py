
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Detecção de Fraude — Precisão vs Recall (com Custos)", layout="wide")
st.title("🛡️ Detecção de Fraude — Precisão × Recall **com Custos**")

st.markdown(
    """
    **Objetivo:** comparar **duas configurações do mesmo modelo** alterando o **limiar** e **medindo custos**:
    - **Config A (Limiar Escolhido 1))**: reduzir fraudes que passam (**FN**).  
    - **Config B (Limiar Escolhido 2))**: reduzir alarmes falsos (**FP**).  
    Exemplo: 
    Limiar Escolhido (Cenário 1)= 20%
    Transações que possuem 20% de chance de serem fraudes serão consideradas fraudes pelo modelo.
    Limiar Escolhido (Cenário 2)= 60%
    Transações que possuem 60% de chance de serem fraudes serão consideradas fraudes pelo modelo.

    Lembre-se: temos apenas um algoritmo, uma base de treino e uma base de teste. Será gerado um único modelo.
    O limiar escolhido muda o comportamento operacional do modelo gerado. 
   
    O app calcula métricas **na validação (train/test)** e mostra **custos** de cada configuração.
    """
)

with st.sidebar:
    st.header("Parâmetros")
    test_size = st.slider("Proporção de teste (validação)", 0.1, 0.5, 0.2, 0.05)
    standardize_numeric = st.checkbox("Padronizar variáveis numéricas (StandardScaler)", value=True)
    class_balanced = st.checkbox("Usar class_weight='balanced' (opcional)", value=False)
    st.markdown("---")
    st.subheader("Limiar de decisão (prob≥limiar ⇒ FRAUDE)")
    th_recall = st.slider("Config A — Limiar 1", 0.0, 1.0, 0.30, 0.01)
    th_precision = st.slider("Config B — Limiar 2", 0.0, 1.0, 0.70, 0.01)
    st.markdown("---")
    st.subheader("Parâmetros de **custo**")
    loss_per_fraud = st.number_input("Perda média se a fraude passar (R$)", min_value=0.0, value=600.0, step=50.0)
    review_cost = st.number_input("Custo por **revisão manual** (R$)", min_value=0.0, value=5.0, step=1.0)
    st.caption("Cálculo: **Custo = (FN × perda) + ((VP + FP) × revisão)**")

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)        # sklearn <1.2

# Upload único (treino/validação)
st.subheader("1) Base de **Treino/Validação** (obrigatória)")
c1, c2 = st.columns([2,1])
with c1:
    train_file = st.file_uploader("Envie CSV com a coluna alvo `fraude` (0/1)", type=["csv"], key="train")
with c2:
    use_example = st.button("Usar dados de exemplo (20% fraude)")

target_col = "fraude"
df_train = None

if use_example and train_file is None:
    st.info("Carregando dados de exemplo embutidos (1000 linhas, 20% fraude)…")
    try:
        df_train = pd.read_csv("fraude_treino_1000.csv")  # arquivo do repositório
    except Exception:
        st.warning("Dados de exemplo não encontrados no repositório. Faça upload manual do CSV com a coluna `fraude`.")
elif train_file is not None:
    try:
        df_train = pd.read_csv(train_file)
    except Exception:
        train_file.seek(0)
        df_train = pd.read_csv(train_file, sep=";")

if df_train is not None:
    if target_col not in df_train.columns:
        st.error(f"A base deve conter a coluna alvo `{target_col}`.")
        df_train = None
    else:
        df_train[target_col] = pd.to_numeric(df_train[target_col], errors="coerce").fillna(0).astype(int)
        st.write("Pré-visualização:", df_train.head())
        st.caption(f"Formato: {df_train.shape[0]} linhas × {df_train.shape[1]} colunas")
        counts_all = df_train[target_col].value_counts().to_dict()
        st.markdown(f"**Distribuição do alvo (dataset completo):** {counts_all}")

if df_train is not None:
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(("cat", make_ohe(), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler() if standardize_numeric else "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(max_iter=1000, class_weight=("balanced" if class_balanced else None))
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])

    # Split robusto garantindo 0 e 1 no teste
    def robust_split(X, y, test_size, max_tries=200):
        for seed in range(1, max_tries+1):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique()>1 else None)
            if y_te.nunique() == 2:
                return X_tr, X_te, y_tr, y_te, seed
        return X_tr, X_te, y_tr, y_te, None

    X_train, X_test, y_train, y_test, used_seed = robust_split(X, y, test_size=test_size)
    if used_seed is None:
        st.warning("Não foi possível garantir positivos e negativos no conjunto de teste. Ajuste o test_size ou verifique o alvo.")
    else:
        st.caption(f"Split estratificado com seed interno = **{used_seed}**")

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    def eval_with_threshold(th):
        y_pred = (y_proba >= th).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        vn, fp, fn, vp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        total_cost = fn * loss_per_fraud + (vp + fp) * review_cost
        return acc, prec, rec, vn, fp, fn, vp, total_cost, cm.sum()

    accA, precA, recA, vnA, fpA, fnA, vpA, costA, totA = eval_with_threshold(th_recall)
    accB, precB, recB, vnB, fpB, fnB, vpB, costB, totB = eval_with_threshold(th_precision)

    st.markdown("### 🔎 Diagnóstico da validação")
    test_counts = y_test.value_counts().to_dict()
    st.markdown(f"**Distribuição do alvo no teste:** {test_counts}")
    if test_counts.get(1, 0) == 0:
        st.error("O conjunto de teste não contém nenhum positivo (fraude). Ajuste o split ou revise a base.")
    if test_counts.get(0, 0) == 0:
        st.error("O conjunto de teste não contém nenhum negativo (legítima). Ajuste o split ou revise a base.")

    st.markdown("### ✅ Resultados por configuração")
    colA, colB = st.columns(2, gap="large")
    def moeda(v): 
        return ("R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
    with colA:
        st.subheader("Teste do Cenário (Limiar Escolhido 1)")
        st.metric("Recall", f"{recA:.3f}")
        st.metric("Precisão", f"{precA:.3f}")
        st.metric("Acurácia", f"{accA:.3f}")
        st.markdown("**Matriz de confusão (valores)**")
        df_cm_A = pd.DataFrame([[vnA, fpA],[fnA, vpA]], index=["Real 0","Real 1"], columns=["Prev 0","Prev 1"])
        st.table(df_cm_A)
        st.caption(f"VP={vpA} • FP={fpA} • FN={fnA} • VN={vnA} • Total={totA}")
        st.markdown(f"**Custo estimado** = FN×perda + (VP+FP)×revisão = {fnA}×{moeda(loss_per_fraud)} + ({vpA}+{fpA})×{moeda(review_cost)} = **{moeda(costA)}**")

    with colB:
        st.subheader("Teste do Cenário (Limiar Escolhido 2)")
        st.metric("Recall", f"{recB:.3f}")
        st.metric("Precisão", f"{precB:.3f}")
        st.metric("Acurácia", f"{accB:.3f}")
        st.markdown("**Matriz de confusão (valores)**")
        df_cm_B = pd.DataFrame([[vnB, fpB],[fnB, vpB]], index=["Real 0","Real 1"], columns=["Prev 0","Prev 1"])
        st.table(df_cm_B)
        st.caption(f"VP={vpB} • FP={fpB} • FN={fnB} • VN={vnB} • Total={totB}")
        st.markdown(f"**Custo estimado** = FN×perda + (VP+FP)×revisão = {fnB}×{moeda(loss_per_fraud)} + ({vpB}+{fpB})×{moeda(review_cost)} = **{moeda(costB)}**")

    st.markdown("---")
    st.markdown("### 📊 Comparativo de custos")
    df_cost = pd.DataFrame([
        {"Configuração":"A (Recall)", "Recall":recA, "Precisão":precA, "Acurácia":accA, "VP":vpA, "FP":fpA, "FN":fnA, "VN":vnA, "Custo (R$)":costA},
        {"Configuração":"B (Precisão)", "Recall":recB, "Precisão":precB, "Acurácia":accB, "VP":vpB, "FP":fpB, "FN":fnB, "VN":vnB, "Custo (R$)":costB},
    ])
    st.dataframe(df_cost.style.format({"Recall":"{:.3f}","Precisão":"{:.3f}","Acurácia":"{:.3f}","Custo (R$)":"{:,.2f}"}), use_container_width=True)

    melhor = "A (Recall)" if costA < costB else "B (Precisão)" if costB < costA else "Empate"
    st.success(f"**Configuração com **menor custo estimado**: {melhor}**")

    st.markdown("---")
    with st.expander("📌 Guia do exercício (resumo)", expanded=True):
        st.markdown(
            """
            1) Ajuste os **limiares** A e B.  
            2) Compare **Precisão, Recall, Acurácia** e **VP/FP/FN/VN**.  
            3) Preencha os **custos** e veja qual configuração **minimiza o custo total**.  
            4) Explique o **trade-off** quando você altera o limiar.  
            """
        )
else:
    st.info("Envie a base de treino/validação (com `fraude`) ou clique em **Usar dados de exemplo** (20% fraude) para começar.")
