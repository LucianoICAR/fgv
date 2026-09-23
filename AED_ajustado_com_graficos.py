
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AED - Análise Exploratória de Dados", page_icon="📊", layout="wide")
st.title("📊 Atividade em Equipe - Análise Exploratória de Dados")
st.caption("Carregue um CSV ou use a base fornecida. Identifique problemas de **Completude**, **Consistência**, **Unicidade** e **Outliers (IQR)**. Gere uma **base limpa**.")

@st.cache_data
def load_sample():
    return pd.read_csv("imoveis.csv")

uploaded = st.file_uploader("Envie um CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ Dataset carregado do upload.")
else:
    st.info("Use a base `imoveis.csv` incluída no app.")
    try:
        df = load_sample()
    except Exception:
        st.error("A base não foi encontrada. Faça upload de um CSV.")
        st.stop()

# Remove colunas técnicas eventualmente presentes em uma base exportada por versões anteriores do app.
# Essas colunas servem apenas à interface e não fazem parte dos dados do negócio.
technical_cols = ["_categorias_problema", "_tem_problema_", "_status"]
df = df.drop(columns=[c for c in technical_cols if c in df.columns], errors="ignore")

# ---------- PASSO 1
st.divider()
st.header("1) Inspeção inicial")
c1,c2,c3 = st.columns(3)
with c1: st.metric("Linhas (n)", f"{df.shape[0]:,}".replace(",","."))
with c2: st.metric("Colunas (p)", df.shape[1])
with c3: st.metric("% Nulos (média)", f"{df.isna().mean().mean()*100:.2f}%")
st.dataframe(df.head(12), use_container_width=True)
with st.expander("Tipos & nulos por coluna"):
    info_df = pd.DataFrame({"dtype": df.dtypes.astype(str), "n_nulos": df.isna().sum(), "%_nulos": (df.isna().mean()*100).round(2)})
    st.dataframe(info_df, use_container_width=True)

# ---------- QUALIDADE: 3Cs
st.divider()
st.header("Qualidade dos Dados (3Cs)")

current_year = pd.Timestamp.now().year
probs = []
cats = []

# Unicidade
dup_id_mask = df["id"].duplicated(keep=False) if "id" in df.columns else pd.Series(False, index=df.index)
# Duplicidade de linha (todas as colunas iguais) - marca as duplicadas (mantém a 1ª)
dup_row_mask = df.duplicated(keep="first")

def categorias_linha(row, idx):
    cat = []
    # Completude
    if row.isna().any(): cat.append("Completude: nulos")
    # Consistência (regras cruzadas e de domínio)
    # domínios básicos
    if "area_m2" in df.columns and pd.notna(row["area_m2"]) and row["area_m2"] <= 0: cat.append("Consistência: area_m2 <= 0")
    if "quartos" in df.columns and pd.notna(row["quartos"]) and row["quartos"] < 1: cat.append("Consistência: quartos < 1")
    if "banheiros" in df.columns and pd.notna(row["banheiros"]) and row["banheiros"] < 1: cat.append("Consistência: banheiros < 1")
    if "vagas" in df.columns and pd.notna(row["vagas"]) and row["vagas"] < 0: cat.append("Consistência: vagas < 0")
    if "preco" in df.columns and pd.notna(row["preco"]) and row["preco"] <= 0: cat.append("Consistência: preco <= 0")
    if "ano_construcao" in df.columns and pd.notna(row["ano_construcao"]) and not (1900 <= int(row["ano_construcao"]) <= current_year):
        cat.append("Consistência: ano_construcao fora do intervalo")
    # regras cruzadas
    if all(c in df.columns for c in ["suites","quartos"]) and pd.notna(row["suites"]) and pd.notna(row["quartos"]) and row["suites"] > row["quartos"]:
        cat.append("Consistência: suites > quartos")
    if all(c in df.columns for c in ["banheiros","quartos"]) and pd.notna(row["banheiros"]) and pd.notna(row["quartos"]) and row["banheiros"] > row["quartos"] + 2:
        cat.append("Consistência: banheiros > quartos + 2")
    if all(c in df.columns for c in ["vagas","quartos"]) and pd.notna(row["vagas"]) and pd.notna(row["quartos"]) and row["vagas"] > row["quartos"] + 3:
        cat.append("Consistência: vagas > quartos + 3")
    if all(c in df.columns for c in ["tipo","area_m2"]) and pd.notna(row["tipo"]) and pd.notna(row["area_m2"]) and (row["tipo"]=="Cobertura") and (row["area_m2"] < 90):
        cat.append("Consistência: cobertura com área < 90m²")
    if all(c in df.columns for c in ["preco","area_m2"]) and pd.notna(row["preco"]) and pd.notna(row["area_m2"]) and row["area_m2"]>0:
        pm2 = row["preco"]/row["area_m2"]
        if pm2 < 1500: cat.append("Consistência: preço/m² irrealmente baixo")
        if pm2 > 50000: cat.append("Consistência: preço/m² irrealmente alto")
    # Unicidade
    if dup_id_mask.iloc[idx]: cat.append("Unicidade: id duplicado")
    if dup_row_mask.iloc[idx]: cat.append("Unicidade: linha duplicada")
    return "; ".join(cat)

cats = [categorias_linha(df.iloc[i], i) for i in range(df.shape[0])]
df["_categorias_problema"] = cats
df["_tem_problema_"] = df["_categorias_problema"].str.len() > 0

st.subheader("Resumo (contagem por tipo de problema)")
if df["_tem_problema_"].any():
    all_cats = []
    for c in df["_categorias_problema"]:
        if c:
            all_cats.extend([x.strip() for x in c.split(";")])
    summary = pd.Series(all_cats).value_counts().rename_axis("categoria").reset_index(name="qtd")
    st.dataframe(summary, use_container_width=True)
else:
    st.success("Nenhum problema de Completude/Consistência/Unicidade foi encontrado.")

st.subheader("Linhas com problemas")
st.dataframe(df[df["_tem_problema_"]], use_container_width=True)

# ---------- OUTLIERS (IQR)
st.divider()
st.header("4) Outliers pelo IQR")

st.info(
    "💡 **Qual é o papel do IQR?** O método do Intervalo Interquartil (IQR) é uma técnica "
    "estatística para **sinalizar valores muito afastados da região central dos dados**. "
    "Ele calcula Q1 (25%), Q3 (75%) e o IQR = Q3 − Q1. Com o fator padrão 1,5, "
    "são sinalizados valores abaixo de Q1 − 1,5×IQR ou acima de Q3 + 1,5×IQR. "
    "**Um outlier não é necessariamente um erro**: pode representar um caso legítimo e raro do negócio. "
    "Por isso, o IQR deve apoiar a investigação e a decisão sobre qualidade dos dados, e não substituir "
    "o julgamento do contexto de negócio."
)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
sel_cols = st.multiselect("Selecione colunas numéricas", options=[c for c in num_cols if c not in ["id"]], default=[c for c in num_cols if c not in ["id"]][:3])
k = st.slider("Fator do IQR (padrão=1.5)", 0.5, 3.0, 1.5, 0.1)

def iqr_bounds(s, kk=1.5):
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - kk*iqr, q3 + kk*iqr, q1, q3, iqr

def get_outlier_mask(data, columns, kk=1.5):
    """Retorna uma máscara de outliers para qualquer uma das colunas selecionadas."""
    mask = pd.Series(False, index=data.index)
    for col in columns:
        s = data[col].dropna()
        if s.empty:
            continue
        lo, hi, *_ = iqr_bounds(s, kk=kk)
        mask |= ((data[col] < lo) | (data[col] > hi)).fillna(False)
    return mask

def remove_outliers_until_stable(data, columns, kk=1.5, max_iter=50):
    """
    Remove outliers de forma iterativa até que o IQR da base remanescente
    não sinalize novos registros. Isso torna a exportação estável quando
    recarregada com as mesmas colunas e o mesmo fator de IQR.
    """
    cleaned = data.copy()
    total_removed = 0
    iterations = 0

    for _ in range(max_iter):
        if cleaned.empty or not columns:
            break
        mask = get_outlier_mask(cleaned, columns, kk=kk)
        n_out = int(mask.sum())
        if n_out == 0:
            break
        cleaned = cleaned.loc[~mask].copy()
        total_removed += n_out
        iterations += 1

    return cleaned, total_removed, iterations

if sel_cols:
    outlier_mask = get_outlier_mask(df, sel_cols, kk=k)
    st.write(f"Total de outliers sinalizados nesta análise: **{int(outlier_mask.sum())}**")

    # Resumo estatístico do IQR para apoiar a interpretação dos boxplots.
    iqr_summary_rows = []
    for col in sel_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        lo, hi, q1, q3, iqr = iqr_bounds(s, kk=k)
        median = s.median()
        col_outliers = ((df[col] < lo) | (df[col] > hi)).fillna(False)
        iqr_summary_rows.append({
            "Variável": col,
            "Q1": q1,
            "Mediana": median,
            "Q3": q3,
            "IQR": iqr,
            "Limite inferior": lo,
            "Limite superior": hi,
            "Outliers": int(col_outliers.sum())
        })

    if iqr_summary_rows:
        st.subheader("Leitura estatística do IQR")
        st.caption(
            "A tabela mostra como os limites são calculados para cada variável selecionada. "
            "Os valores fora dos limites inferior e superior são sinalizados como candidatos a outlier."
        )
        iqr_summary = pd.DataFrame(iqr_summary_rows)
        numeric_summary_cols = [
            "Q1", "Mediana", "Q3", "IQR", "Limite inferior", "Limite superior"
        ]
        iqr_summary[numeric_summary_cols] = iqr_summary[numeric_summary_cols].round(2)
        st.dataframe(iqr_summary, use_container_width=True, hide_index=True)

    st.subheader("Visualização dos outliers — Boxplot")
    st.markdown(
        "O **retângulo** representa os 50% centrais dos dados (de Q1 a Q3), a linha interna é a "
        "**mediana** e os pontos além dos limites calculados pelo IQR são valores que merecem investigação. "
        "As linhas verticais tracejadas indicam exatamente os limites inferior e superior usados pelo sistema."
    )

    for col in sel_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        lo, hi, q1, q3, iqr = iqr_bounds(s, kk=k)
        median = s.median()
        col_outlier_mask = ((df[col] < lo) | (df[col] > hi)).fillna(False)
        n_col_outliers = int(col_outlier_mask.sum())

        with st.expander(f"📦 {col} — {n_col_outliers} outlier(s)", expanded=True):
            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.boxplot(s, vert=False, whis=k, showfliers=True)
            ax.axvline(lo, linestyle="--", linewidth=1.5, label=f"Limite inferior = {lo:,.2f}")
            ax.axvline(hi, linestyle="--", linewidth=1.5, label=f"Limite superior = {hi:,.2f}")
            ax.set_title(f"Boxplot de {col}")
            ax.set_xlabel(col)
            ax.set_yticks([])
            ax.grid(axis="x", alpha=0.25)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Q1", f"{q1:,.2f}")
            m2.metric("Mediana", f"{median:,.2f}")
            m3.metric("Q3", f"{q3:,.2f}")
            m4.metric("IQR", f"{iqr:,.2f}")

            st.caption(
                f"Faixa esperada pelo critério atual: **{lo:,.2f} a {hi:,.2f}** "
                f"(fator IQR = {k:.1f}). Foram sinalizados **{n_col_outliers}** registro(s) nesta variável."
            )

    if outlier_mask.any():
        st.warning(
            "Os registros abaixo são **candidatos a investigação**. Ser outlier pelo IQR não significa, "
            "por si só, que o dado esteja errado."
        )
        st.dataframe(df[outlier_mask], use_container_width=True)
    else:
        st.success("✅ Nenhum outlier foi sinalizado pelo IQR nas colunas selecionadas.")
else:
    outlier_mask = pd.Series(False, index=df.index)
    st.info("Selecione pelo menos 1 coluna.")

# ---------- GERAÇÃO BASE LIMPA
st.divider()
st.header("Gerar base limpa")

st.markdown('''
- 🔴 **Problema **  
- 🟡 **Outlier (IQR)**  
- 🟣 **Ambos**  
- 🟢 **OK**
''')

status = np.where(df["_tem_problema_"] & outlier_mask, "🟣 problema + outlier",
         np.where(df["_tem_problema_"], "🔴 problema",
         np.where(outlier_mask, "🟡 outlier", "🟢 ok")))
df_prev = df.copy()
df_prev["_status"] = status

def highlight_row(row):
    if row["_status"].startswith("🟣"): return ["background-color: #e8ddff"]*len(row)
    if row["_status"].startswith("🔴"): return ["background-color: #ffe5e5"]*len(row)
    if row["_status"].startswith("🟡"): return ["background-color: #fff7d6"]*len(row)
    if row["_status"].startswith("🟢"): return ["background-color: #e8f5e9"]*len(row)
    return [""]*len(row)

st.subheader("Prévia com status (cores)")
st.dataframe(df_prev.head(120).style.apply(highlight_row, axis=1), use_container_width=True)

# 1) Remove problemas determinísticos de Completude, Consistência e Unicidade.
base_sem_3cs = df.loc[~df["_tem_problema_"]].copy()
removed_3cs = df.shape[0] - base_sem_3cs.shape[0]

# 2) Remove outliers pelo IQR até estabilizar. Recalcular o IQR após a primeira
# remoção pode revelar novos valores extremos; por isso a limpeza é iterativa.
if sel_cols:
    clean_df, removed_iqr, iqr_iterations = remove_outliers_until_stable(
        base_sem_3cs, sel_cols, kk=k
    )
else:
    clean_df = base_sem_3cs.copy()
    removed_iqr = 0
    iqr_iterations = 0

# 3) Nunca exportar colunas auxiliares criadas pela interface.
clean_df = clean_df.drop(
    columns=["_categorias_problema", "_tem_problema_3Cs", "_status"],
    errors="ignore"
).copy()

removed_total = df.shape[0] - clean_df.shape[0]
st.write(
    f"Registros originais: **{df.shape[0]}** | "
    f"Removidos - problemas: **{removed_3cs}** | "
    f"Removidos pelo IQR: **{removed_iqr}** | "
    f"Restantes (limpos): **{clean_df.shape[0]}**"
)

if removed_total == 0:
    st.success("✅ A base já está limpa segundo as regras atuais e o IQR selecionado.")
elif sel_cols:
    st.caption(
        f"O IQR foi reaplicado até estabilizar ({iqr_iterations} ciclo(s) de remoção). "
        "Assim, ao recarregar esta base com as mesmas colunas e o mesmo fator de IQR, "
        "o sistema não deverá voltar a sinalizar outliers."
    )

def to_csv_bytes(d):
    return d.to_csv(index=False).encode("utf-8")

st.download_button("⬇️ Baixar BASE LIMPA (.csv)", data=to_csv_bytes(clean_df), file_name="base_limpa.csv", mime="text/csv", use_container_width=True)
