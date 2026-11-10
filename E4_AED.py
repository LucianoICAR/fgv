"""Atividade 4 – Análise Exploratória de Dados (AED).

Aplicativo independente: a base fictícia é gerada dentro deste arquivo.
O participante baixa o CSV, envia-o para análise e exporta uma versão limpa.
Todos os registros são sintéticos e os erros foram inseridos para fins pedagógicos.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="FGV | AED e qualidade de dados", page_icon="📊", layout="wide")

NUM = ["area_m2", "quartos", "banheiros", "suites", "vagas", "ano_construcao",
       "preco", "distancia_centro_km", "visitas_7d", "dias_anuncio"]
CAT = ["tipo", "bairro"]
ALVO = "vendido_90d"
COLUNAS = ["id"] + NUM + CAT + [ALVO]
TIPOS = ["Apartamento", "Casa", "Cobertura"]
BAIRROS = ["Centro", "Zona Norte", "Zona Sul", "Zona Oeste"]

def make_clean(n=2100, seed=2026):
 r=np.random.default_rng(seed)
 tipo=r.choice(['Apartamento','Casa','Cobertura'],n,p=[.68,.28,.04]); bairro=r.choice(['Centro','Zona Norte','Zona Sul','Zona Oeste'],n,p=[.24,.28,.30,.18])
 area=np.clip(np.rint(r.lognormal(np.log(110),.36,n)),40,300).astype(int)
 area=np.where(tipo=='Cobertura', np.maximum(area,112),area)
 area=np.where(tipo=='Casa',np.maximum(area,79),area)
 quartos=np.clip(np.rint(area/63+r.normal(0,.58,n)),1,5).astype(int)
 banheiros=np.maximum(1,np.minimum(quartos+2,quartos+r.integers(-1,3,n)))
 suites=np.array([r.integers(0, min(q,b)+1) for q,b in zip(quartos,banheiros)])
 vagas=np.clip(np.rint(area/85+r.normal(0,.6,n)),0,4).astype(int)
 vagas=np.minimum(vagas,quartos+3)
 ano=r.integers(1985,2027,n)
 distancia_base={'Centro':4,'Zona Norte':12,'Zona Sul':8,'Zona Oeste':17}
 dist=np.clip(np.array([distancia_base[b] for b in bairro])+r.normal(0,3,n),.5,32).round(1)
 base_p={'Centro':6500,'Zona Norte':4900,'Zona Sul':8500,'Zona Oeste':4350}
 preco_m2=np.array([base_p[b] for b in bairro])*r.lognormal(0,.12,n)
 preco_m2*=np.where(tipo=='Cobertura',1.3,np.where(tipo=='Casa',1.05,1.0))
 preco=np.round(area*preco_m2/1000)*1000
 dias=np.clip(r.poisson(28,n),0,150)
 visitas=np.clip(r.poisson(np.maximum(1,8-dias/18),n),0,22)
 # target depends on accessible before sale predictors
 relativo=preco/(area*np.array([base_p[b] for b in bairro]))
 z= -1.65+ .41*visitas -.024*dias - 1.4*(relativo-1) - .022*dist + .28*(tipo=='Apartamento') + .1*(ano>2010) + r.normal(0,.18,n)
 prob=1/(1+np.exp(-z))
 y=r.binomial(1,prob,n)
 return pd.DataFrame({'id':np.arange(seed*10,seed*10+n,dtype=int), 'tipo':tipo,'bairro':bairro,'area_m2':area,'quartos':quartos,'banheiros':banheiros,'suites':suites,'vagas':vagas,'ano_construcao':ano,'preco':preco.astype(float),'distancia_centro_km':dist,'visitas_7d':visitas,'dias_anuncio':dias,'vendido_90d':y})

def make_dirty():
 clean=make_clean()
 d=clean.copy(); r=np.random.default_rng(2121)
 ix=r.permutation(len(d)); t=0
 def choose(k):
  nonlocal t
  out=ix[t:t+k];t+=k;return out
 z=choose(105);d.loc[z,'area_m2']=np.where(np.arange(len(z))%2==0,0,-30)
 z=choose(105);d.loc[z,'preco']=np.where(np.arange(len(z))%2==0,1,1e8)
 z=choose(90);d.loc[z,'visitas_7d']=np.where(np.arange(len(z))%2==0,400,-10)
 z=choose(80);d.loc[z,'area_m2']=np.nan
 z=choose(65);d.loc[z,'quartos']=0
 z=choose(50);d.loc[z,'suites']=d.loc[z,'quartos']+4
 z=choose(45);d.loc[z,'ano_construcao']=2080
 z=choose(50);d.loc[z,'distancia_centro_km']=-10
 z=choose(40);d.loc[z,'preco']=np.nan
 # shuffle some labels on bad records? not yet
 z=r.choice(d.index, size=45, replace=False)
 d=pd.concat([d,d.loc[z].copy()],ignore_index=True)
 return d



@st.cache_data(show_spinner=False)
def dados_da_atividade():
    """2.100 exemplos únicos + 45 duplicados intencionais = 2.145 linhas."""
    return make_dirty()


def converter_csv(dados):
    return dados.to_csv(index=False).encode("utf-8-sig")


def verificar_inconsistencias(dados):
    """Máscara de registros que violam regras de consistência previamente definidas."""
    d = dados
    invalidos = (
        ~d["area_m2"].between(35, 400)
        | ~d["quartos"].between(1, 7)
        | ~d["banheiros"].between(1, 9)
        | ~d["suites"].between(0, 7)
        | (d["suites"] > d["quartos"])
        | (d["banheiros"] > d["quartos"] + 2)
        | ~d["vagas"].between(0, 8)
        | (d["vagas"] > d["quartos"] + 3)
        | ~d["ano_construcao"].between(1900, 2026)
        | ~d["preco"].between(20_000, 10_000_000)
        | ~d["distancia_centro_km"].between(0, 60)
        | ~d["visitas_7d"].between(0, 60)
        | ~d["dias_anuncio"].between(0, 365)
        | ~d["tipo"].isin(TIPOS)
        | ~d["bairro"].isin(BAIRROS)
        | ~d[ALVO].isin([0, 1])
    )
    preco_m2 = d["preco"] / d["area_m2"].where(d["area_m2"] > 0)
    invalidos |= ~preco_m2.between(1_500, 50_000)
    # Nulos são classificados separadamente como completude, não consistência.
    return invalidos & ~d[COLUNAS].isna().any(axis=1)


def limites_iqr(s, fator=1.5):
    numeros = pd.to_numeric(s, errors="coerce").dropna()
    if len(numeros) < 4:
        return None
    q1, q3 = numeros.quantile([0.25, 0.75])
    amplitude = q3 - q1
    return float(q1 - fator * amplitude), float(q3 + fator * amplitude), float(q1), float(q3)


def mascara_iqr(dados, colunas, fator):
    resultado = pd.Series(False, index=dados.index)
    for coluna in colunas:
        limites = limites_iqr(dados[coluna], fator)
        if limites is not None:
            baixo, alto, _, _ = limites
            resultado |= dados[coluna].lt(baixo) | dados[coluna].gt(alto)
    return resultado


st.title("📊 Atividade 4 — Análise Exploratória de Dados")
st.subheader("Caso fictício: Imobiliária Horizonte")
st.write(
    "A empresa quer prever **quais imóveis serão vendidos em até 90 dias**. "
    "Antes de treinar o sistema, a equipe recebeu uma base histórica com problemas "
    "de qualidade. Sua missão é investigar, limpar e depois comparar a acurácia."
)
st.info(
    "**Aprendizado supervisionado:** o resultado histórico `vendido_90d` vale "
    "**1 = vendido em até 90 dias** e **0 = não vendido nesse período**. "
    "A base possui erros intencionais para estudo; não contém dados reais."
)

st.header("1. Baixe a base original e envie-a ao sistema")
base_original = dados_da_atividade()
col_a, col_b = st.columns([1, 1])
with col_a:
    st.metric("Amostras disponíveis", f"{len(base_original):,}".replace(",", "."))
    st.download_button(
        "⬇️ Baixar base ORIGINAL com erros (CSV)",
        data=converter_csv(base_original),
        file_name="imoveis_base_com_erros.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_b:
    st.write("**Passo obrigatório:** baixe o arquivo acima e carregue-o aqui.")
    arquivo = st.file_uploader("Carregar CSV original", type=["csv"], key="upload_base_aed")

if arquivo is None:
    st.warning("A análise será liberada quando você enviar a base baixada.")
    st.stop()

try:
    df = pd.read_csv(arquivo, encoding="utf-8-sig")
except Exception as exc:
    st.error(f"Não foi possível ler o arquivo CSV: {exc}")
    st.stop()

faltantes = [c for c in COLUNAS if c not in df.columns]
if faltantes:
    st.error("Colunas ausentes: " + ", ".join(faltantes))
    st.stop()

# Retira apenas campos auxiliares do próprio aplicativo; preserva as variáveis de negócio.
df = df[COLUNAS].copy()
for c in NUM + [ALVO]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

st.success("Base carregada. Agora você pode investigar sua qualidade.")

with st.expander("📖 Dicionário de dados e interpretação do alvo"):
    st.dataframe(pd.DataFrame([
        ("id", "Identificador do registro — NÃO entra no treinamento"),
        ("area_m2", "Área construída em m²"),
        ("quartos, banheiros, suites, vagas", "Características físicas do imóvel"),
        ("ano_construcao", "Ano de construção"),
        ("preco", "Preço anunciado em reais"),
        ("tipo, bairro", "Tipo e localização agrupada do imóvel"),
        ("distancia_centro_km", "Distância ao centro em quilômetros"),
        ("visitas_7d", "Visitas recebidas nos sete dias anteriores à previsão"),
        ("dias_anuncio", "Dias em anúncio no instante da previsão"),
        ("vendido_90d", "Rótulo histórico: 1 = vendido em 90 dias; 0 = não vendido"),
    ], columns=["Campo", "Significado"]), hide_index=True, use_container_width=True)

st.divider()
st.header("2. Inspeção inicial")
col1, col2, col3 = st.columns(3)
col1.metric("Registros carregados", len(df))
col2.metric("Variáveis + identificador + alvo", len(df.columns))
col3.metric("Células vazias", int(df.isna().sum().sum()))
st.dataframe(df.head(18), use_container_width=True, hide_index=True)
with st.expander("Nulos e tipos por coluna"):
    resumo_colunas = pd.DataFrame({
        "Tipo": df.dtypes.astype(str),
        "Nulos": df.isna().sum(),
        "% de nulos": (100 * df.isna().mean()).round(1),
    })
    st.dataframe(resumo_colunas, use_container_width=True)

st.divider()
st.header("3. Completude, consistência e unicidade")
completude = df[COLUNAS].isna().any(axis=1)
inconsistencia = verificar_inconsistencias(df)
duplicidade = df["id"].duplicated(keep=False)
resumo = pd.DataFrame({
    "Dimensão": ["Completude", "Consistência", "Unicidade"],
    "Registros sinalizados": [int(completude.sum()), int(inconsistencia.sum()), int(duplicidade.sum())],
    "O que verificar": [
        "Campos necessários sem informação",
        "Regras de negócio e valores impossíveis",
        "Identificador repetido (inclui o registro original)",
    ],
})
st.dataframe(resumo, hide_index=True, use_container_width=True)
st.caption("Uma mesma linha pode apresentar mais de um problema; não some as contagens.")
status = np.select(
    [completude, inconsistencia, duplicidade],
    ["Completude", "Consistência", "Unicidade"],
    default="OK",
)
visao = df.copy()
visao.insert(1, "diagnostico", status)
st.subheader("Exemplos de registros com problema")
st.dataframe(visao.loc[status != "OK"].head(35), hide_index=True, use_container_width=True)

st.divider()
st.header("4. Outliers — intervalo interquartil (IQR)")
st.write(
    "O IQR sinaliza observações afastadas da região central. **Outlier não é automaticamente erro**: "
    "o imóvel pode ser realmente diferente dos demais. Analise o contexto antes de removê-lo."
)
selecao = st.multiselect(
    "Escolha as variáveis para investigar",
    options=["area_m2", "preco", "distancia_centro_km", "visitas_7d", "dias_anuncio"],
    default=["area_m2", "preco"],
)
fator = st.slider("Fator do IQR", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
marcados_iqr = mascara_iqr(df, selecao, fator)
st.metric("Registros sinalizados pelo IQR", int(marcados_iqr.sum()))
for coluna in selecao:
    limites = limites_iqr(df[coluna], fator)
    if limites is None:
        continue
    inferior, superior, q1, q3 = limites
    with st.expander(f"📦 {coluna} — boxplot e limites", expanded=(coluna == selecao[0])):
        fig, ax = plt.subplots(figsize=(10, 2.4))
        valores = df[coluna].dropna()
        ax.boxplot(valores, vert=False, whis=fator, showfliers=True)
        ax.axvline(inferior, linestyle="--", label=f"Limite inf. {inferior:,.1f}")
        ax.axvline(superior, linestyle="--", label=f"Limite sup. {superior:,.1f}")
        ax.set_xlabel(coluna)
        ax.set_yticks([])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(f"Q1 = {q1:,.1f}; Q3 = {q3:,.1f}; IQR = {q3 - q1:,.1f}")

st.divider()
st.header("5. Defina a limpeza e baixe a nova base")
st.write(
    "Para esta simulação, a estratégia inicial é **retirar registros com problemas evidentes**. "
    "Em projetos reais, também é possível corrigir na origem ou imputar valores; "
    "excluir linhas pode eliminar observações importantes."
)
col_l, col_r = st.columns(2)
with col_l:
    remover_duplicados = st.checkbox("Remover identificadores repetidos (manter a primeira linha)", value=True)
    remover_incompletos = st.checkbox("Excluir registros com dados essenciais ausentes", value=True)
with col_r:
    remover_inconsistentes = st.checkbox("Excluir registros que violam regras de consistência", value=True)
    remover_iqr = st.checkbox("Excluir também candidatos a outlier pelo IQR", value=False)

limpa = df.copy()
passos = []
if remover_duplicados:
    antes = len(limpa)
    limpa = limpa.drop_duplicates(subset="id", keep="first")
    passos.append(("Duplicados removidos", antes - len(limpa)))
if remover_incompletos:
    antes = len(limpa)
    limpa = limpa.dropna(subset=COLUNAS)
    passos.append(("Incompletos removidos", antes - len(limpa)))
if remover_inconsistentes:
    antes = len(limpa)
    limpa = limpa.loc[~verificar_inconsistencias(limpa)].copy()
    passos.append(("Inconsistentes removidos", antes - len(limpa)))
if remover_iqr and selecao:
    antes = len(limpa)
    limpa = limpa.loc[~mascara_iqr(limpa, selecao, fator)].copy()
    passos.append(("Outliers removidos", antes - len(limpa)))

c1, c2, c3 = st.columns(3)
c1.metric("Original", len(df))
c2.metric("Após limpeza", len(limpa))
c3.metric("Registros descartados", len(df) - len(limpa))
if passos:
    st.dataframe(pd.DataFrame(passos, columns=["Etapa", "Registros descartados"]),
                 hide_index=True, use_container_width=True)
if len(limpa) < 100:
    st.error("A limpeza deixou poucos registros para o experimento. Revise as opções.")
else:
    st.dataframe(limpa.head(12), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Baixar BASE LIMPA para o segundo sistema (.csv)",
        converter_csv(limpa),
        file_name="imoveis_base_limpa.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )
    st.success("Agora abra o sistema E4_Teste.py e compare a acurácia das duas bases.")

st.divider()
st.markdown("**Discussão executiva:** A base ficou mais confiável? Quantas observações foram perdidas? "
            "Excluir todos os outliers seria uma decisão segura? A acurácia deve melhorar necessariamente?")
st.caption("FGV | Dados e resultados inteiramente fictícios para fins didáticos.")
