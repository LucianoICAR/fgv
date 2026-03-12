# streamlit_app.py (tarefa1_supervisionado_v2)
# ------------------------------------------------------
# Atividade 1: Avaliação Executiva de um Sistema de IA (Supervisionado)
# Requisitos do professor:
# 1) Apresentar um problema de negócio.
# 2) Upload de base histórica e split 70/30 (treino/teste).
# 3) O sistema usa IA SUPERVISIONADA e resolve PARCIALMENTE o problema.
# 4) Exibir APENAS a ACURÁCIA.
# 5) Exibir a base utilizada (mesmo com erros) e DESTACAR erros por cor.
# 6) Permitir inserir UMA NOVA BASE (sem alvo) para obter as PREDIÇÕES do sistema.
# As equipes analisam e propõem ações do EXECUTIVO com base nos 10 passos.
# ------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Atividade 1 — Avaliação Executiva de IA (70/30)", layout="wide")
st.title("Atividade 1 — Avaliação Executiva de um Sistema de IA (70/30)")

# 1) Problema de negócio (exemplo claro para as equipes)
st.markdown(
    """
    ### 🧩 Problema de Negócio (Exemplo)
    A empresa **Entrega Rápida** sofre com **atrasos nas entregas** e quer **priorizar pedidos** com maior risco de atraso.

    **Tipo de IA utilizado:** este sistema é de **aprendizado supervisionado**.

    **Como o sistema pretende resolver o problema:**
    1. A equipe **faz upload** de uma base histórica com uma **coluna alvo** (ex.: `atraso`, `target`, `label`, `classe`) indicando se o pedido atrasou (1) ou não (0).
    2. O sistema faz um **split 70%/30%** (treino/teste), treina um **modelo baseline (Regressão Logística)** e mede **apenas a acurácia** no conjunto de **teste**.
    3. A acurácia indica **o quanto o modelo acerta** ao classificar atrasos vs. não atrasos. É uma **solução parcial**: serve para discutir se **ajuda a priorizar** pedidos com risco, **quais dados faltam** e **quais políticas** o executivo deve definir para o próximo ciclo.

    **O que o sistema NÃO faz:**
    - Não apresenta outras métricas.
    - Não corrige automaticamente os erros da base; apenas **destaca** onde estão, para apoiar **decisões executivas** sobre qualidade de dados.
    """
)

st.markdown("---")

# 2) Upload e seleção de coluna alvo
st.header("1) Upload da Base e Seleção da Classe")
file = st.file_uploader("📥 Envie um CSV (delimitador vírgula)", type=["csv"]) 
random_state = st.sidebar.number_input("Semente (random_state)", min_value=0, value=42, step=1)

if file is None:
    st.info("Envie um arquivo .csv contendo a coluna alvo (ex.: target, label, classe). Mantenha os erros: o objetivo é avaliá-los.")
    st.stop()

# Tenta ler separado por vírgula; se falhar, tenta ponto e vírgula
try:
    df_raw = pd.read_csv(file)
except Exception:
    file.seek(0)
    df_raw = pd.read_csv(file, sep=';')

# 5) Exibir a base (mesmo com erros) + Destacar erros por cor
st.header("2) Base Utilizada e Erros Destacados")

# Heurísticas de problemas: faltantes, duplicados, tipagem inconsistente, outliers (z>4)
dup_mask = df_raw.duplicated(keep=False)
convertible_numeric = []
non_numeric_cells = pd.DataFrame(False, index=df_raw.index, columns=df_raw.columns)
for c in df_raw.columns:
    try:
        coerced = pd.to_numeric(df_raw[c], errors='coerce')
        if coerced.notna().mean() >= 0.5:
            convertible_numeric.append(c)
            non_numeric_cells[c] = coerced.isna() & (~df_raw[c].isna())
    except Exception:
        pass

outlier_cells = pd.DataFrame(False, index=df_raw.index, columns=df_raw.columns)
for c in convertible_numeric:
    coerced = pd.to_numeric(df_raw[c], errors='coerce')
    m = coerced.mean()
    s = coerced.std(ddof=0)
    if s and s > 0:
        z = (coerced - m).abs() / s
        outlier_cells[c] = z > 4

na_cells = df_raw.isna()

def style_errors(val, row_idx, col_name):
    styles = []
    if dup_mask.iloc[row_idx]:
        styles.append("background-color: #ffe0e0")  # duplicados
    if pd.isna(val):
        styles.append("background-color: #fff3cd")  # faltante
    if non_numeric_cells.loc[row_idx, col_name]:
        styles.append("background-color: #e0f7ff")  # tipagem inconsistente
    if outlier_cells.loc[row_idx, col_name]:
        styles.append("background-color: #e6ffe6")  # outlier
    return ";".join(styles) if styles else ""

styled = df_raw.style.format(precision=3)
styled = styled.apply(lambda s: [style_errors(v, s.index[i], s.name) for i, v in enumerate(s)], axis=0)

st.caption("Cores: vermelho=duplicado, amarelo=faltante, azul=tipagem inconsistente, verde=possível outlier.")
st.dataframe(styled, use_container_width=True)

# 3) IA supervisionada simples
st.header("3) Treino 70% / Teste 30% — IA Supervisionada (Protótipo Parcial)")

def infer_target(df: pd.DataFrame):
    for cand in ["target", "label", "classe", "y", "atraso"]:
        if cand in df.columns:
            return cand
    return None

target_guess = infer_target(df_raw)
cols = ["— selecione —"] + df_raw.columns.tolist()
sel_index = cols.index(target_guess) if target_guess in df_raw.columns else 0

label_col = st.selectbox("Coluna alvo (classe)", options=cols, index=sel_index)
if label_col == "— selecione —":
    st.warning("Selecione a coluna alvo para continuar.")
    st.stop()

X = df_raw.drop(columns=[label_col])
y = df_raw[label_col]

# Remover linhas com alvo ausente
st.warning("Linhas sem alvo (classe NaN) não podem treinar/testar. Serão removidas apenas para o modelo.")
missing_target_idx = y[y.isna()].index.tolist()
if missing_target_idx:
    st.write("Linhas removidas por alvo ausente:", missing_target_idx)
    X = X.drop(index=missing_target_idx)
    y = y.drop(index=missing_target_idx)

# Pré-processamento
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.columns.difference(num_cols).tolist()

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ]
)

model = LogisticRegression(max_iter=300)
pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
acc = accuracy_score(y_test, preds)

st.metric("Acurácia (30% teste)", f"{acc:.3f}")

# 5) Testar com nova base
st.header("5) Testar com uma **nova base**")
new_file = st.file_uploader("📥 Envie um CSV para predição (sem coluna alvo)", type=["csv"], key="novo")
if new_file is not None:
    try:
        df_new = pd.read_csv(new_file)
    except Exception:
        new_file.seek(0)
        df_new = pd.read_csv(new_file, sep=';')
    st.write("Prévia da nova base:")
    st.dataframe(df_new, use_container_width=True)
    try:
        preds_new = pipe.predict(df_new)
        out = df_new.copy()
        out["predicao_atraso"] = preds_new
        st.success("Predições geradas. Baixe o resultado para análise em equipe.")
        st.dataframe(out, use_container_width=True)
        st.download_button("⬇️ Baixar predições (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="predicoes_nova_base.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"Não foi possível prever com a nova base: {e}")

# 6) Discussão em Equipe
st.header("6) Discussão em Equipe — Ações do Executivo")
st.markdown(
    """
    1. **Objetivos** — O sistema ajuda a atingir a meta de negócio? 
    2. **Fontes de dados** — Há fontes críticas faltando? 
    3. **Refinamento** — Os erros destacados comprometem decisões? 
    4. **Variáveis** — Quais atributos devem ser exigidos ou criados?
    5. **Restrições** — Há requisitos de explicabilidade, tempo de resposta ou custo?
    6. **Aprendizado** — O tipo (supervisionado) é adequado?
    7. **Algoritmo** — Precisamos de alternativas mais explicáveis/robustas?
    8. **Treinamento** — O 70/30 está ok? Política de versão de modelos/dados?
    9. **Avaliação** — Só acurácia basta? 
    10. **Implantação/Monitoramento** — Que SLAs e auditorias seriam cobrados?
    """
)
st.success("Objetivo pedagógico: evidenciar que **o executivo decide rumos e políticas** em TODAS as etapas, não apenas ao final.")

