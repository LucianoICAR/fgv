"""Atividade 4 – teste comparativo: treino com dados brutos versus dados limpos.

Aplicativo independente. O conjunto de avaliação é gerado internamente e
é idêntico para as duas medições, evitando comparação injusta entre testes.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="FGV | Compare a acurácia", page_icon="🎯", layout="wide")

NUM = ["area_m2", "quartos", "banheiros", "suites", "vagas", "ano_construcao",
       "preco", "distancia_centro_km", "visitas_7d", "dias_anuncio"]
CAT = ["tipo", "bairro"]
ALVO = "vendido_90d"
XCOLS = NUM + CAT
COLUNAS = ["id"] + XCOLS + [ALVO]

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



@st.cache_data(show_spinner=False)
def conjunto_teste_fixo():
    """480 imóveis independentes e sem erros artificiais, não usados no treinamento."""
    return make_clean(n=480, seed=9087)


def avaliar_base(arquivo_csv):
    dados = pd.read_csv(arquivo_csv, encoding="utf-8-sig")
    faltantes = [col for col in COLUNAS if col not in dados.columns]
    if faltantes:
        raise ValueError("Colunas ausentes: " + ", ".join(faltantes))
    if len(dados) < 100:
        raise ValueError("A base deve conter pelo menos 100 registros para esta simulação.")
    if not dados[ALVO].isin([0, 1]).all():
        raise ValueError("A coluna vendido_90d deve conter apenas 0 e 1, sem valores ausentes.")
    if dados[ALVO].nunique() < 2:
        raise ValueError("O treinamento requer exemplos das duas classes (0 e 1).")
    dados = dados[COLUNAS].copy()
    for coluna in NUM:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    # Mesmo algoritmo e mesmas configurações nas duas execuções.
    num_pipe = Pipeline([
        ("preencher_ausentes", SimpleImputer(strategy="median")),
        ("padronizar", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("preencher_ausentes", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    preparacao = ColumnTransformer([
        ("num", num_pipe, NUM),
        ("cat", cat_pipe, CAT),
    ])
    modelo = Pipeline([
        ("preparacao", preparacao),
        ("classificador", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    modelo.fit(dados[XCOLS], dados[ALVO].astype(int))

    teste = conjunto_teste_fixo()
    y_real = teste[ALVO].astype(int)
    y_previsto = modelo.predict(teste[XCOLS])
    tn, fp, fn, vp = confusion_matrix(y_real, y_previsto, labels=[0, 1]).ravel()
    return {
        "treino": len(dados),
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "recall": float(recall_score(y_real, y_previsto, zero_division=0)),
        "corretos": int((y_real == y_previsto).sum()),
        "testados": len(teste),
        "vp": int(vp), "fp": int(fp), "fn": int(fn), "vn": int(tn),
    }


st.title("🎯 Atividade 4 — O efeito da qualidade dos dados na acurácia")
st.write(
    "**Problema de negócio:** prever quais imóveis serão vendidos em até **90 dias**, "
    "com base em características conhecidas antes da venda."
)
st.info(
    "Você treinará **o mesmo modelo de classificação duas vezes**: primeiro com a base original "
    "e depois com a base limpa. Nos dois casos, a avaliação utiliza os **mesmos 480 imóveis "
    "de teste**, que não participaram do treinamento."
)
st.caption(
    "O sistema usa regressão logística (classificação binária), com tratamento básico "
    "de valores ausentes para que a base original possa ser treinada. "
    "Acurácia = proporção de previsões corretas no conjunto de teste."
)

st.header("1. Carregue a BASE ORIGINAL com erros")
original = st.file_uploader("imoveis_base_com_erros.csv", type=["csv"], key="base_bruta")
resultado_original = None
if original is not None:
    try:
        resultado_original = avaliar_base(original)
        c1, c2, c3 = st.columns(3)
        c1.metric("Acurácia — original", f"{100 * resultado_original['acuracia']:.1f}%")
        c2.metric("Acertos no teste", f"{resultado_original['corretos']} / {resultado_original['testados']}")
        c3.metric("Registros para treinar", resultado_original["treino"])
    except Exception as erro:
        st.error(f"Não foi possível avaliar a base original: {erro}")
else:
    st.warning("Envie primeiro o arquivo original baixado no sistema de AED.")

st.divider()
st.header("2. Carregue a BASE LIMPA produzida na AED")
limpa = st.file_uploader("imoveis_base_limpa.csv", type=["csv"], key="base_limpa")
resultado_limpo = None
if limpa is not None:
    try:
        resultado_limpo = avaliar_base(limpa)
        c1, c2, c3 = st.columns(3)
        c1.metric("Acurácia — limpa", f"{100 * resultado_limpo['acuracia']:.1f}%")
        c2.metric("Acertos no mesmo teste", f"{resultado_limpo['corretos']} / {resultado_limpo['testados']}")
        c3.metric("Registros para treinar", resultado_limpo["treino"])
    except Exception as erro:
        st.error(f"Não foi possível avaliar a base limpa: {erro}")
else:
    st.info("Depois de limpar e baixar a base no sistema AED, carregue-a aqui.")

if resultado_original and resultado_limpo:
    st.divider()
    st.header("3. Compare os resultados e discuta")
    variacao = 100 * (resultado_limpo["acuracia"] - resultado_original["acuracia"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia antes", f"{100 * resultado_original['acuracia']:.1f}%")
    c2.metric("Acurácia depois", f"{100 * resultado_limpo['acuracia']:.1f}%")
    c3.metric("Diferença", f"{variacao:+.1f} p.p.")

    comparacao = pd.DataFrame({
        "Versão": ["Original (com erros)", "Após AED"],
        "Registros de treinamento": [resultado_original["treino"], resultado_limpo["treino"]],
        "Acurácia (%)": [round(100 * resultado_original["acuracia"], 1),
                          round(100 * resultado_limpo["acuracia"], 1)],
        "Recall de vendas em 90 dias (%)": [round(100 * resultado_original["recall"], 1),
                                             round(100 * resultado_limpo["recall"], 1)],
    })
    st.dataframe(comparacao, hide_index=True, use_container_width=True)
    st.bar_chart(comparacao.set_index("Versão")["Acurácia (%)"], y_label="Acurácia (%)")

    if variacao > 0:
        st.success(
            "Neste experimento sintético, o modelo treinado com a base tratada acertou "
            "mais casos no mesmo conjunto de avaliação."
        )
    elif variacao == 0:
        st.info("A acurácia não se alterou nesta execução. Qualidade dos dados não se resume a uma métrica.")
    else:
        st.warning(
            "Neste experimento a acurácia diminuiu após a limpeza. "
            "Investigue se foram excluídos registros relevantes."
        )
    with st.expander("📌 Quais erros o modelo cometeu?"):
        st.dataframe(pd.DataFrame({
            "Resultado no conjunto fixo": ["Verdadeiro positivo", "Falso positivo", "Falso negativo", "Verdadeiro negativo"],
            "Original": [resultado_original[k] for k in ("vp", "fp", "fn", "vn")],
            "Limpa": [resultado_limpo[k] for k in ("vp", "fp", "fn", "vn")],
        }), hide_index=True, use_container_width=True)
    st.markdown(
        "**Perguntas à equipe**\n\n"
        "1. A acurácia melhorou? De quanto?\n"
        "2. A limpeza eliminou dados que poderiam ser legítimos?\n"
        "3. Só a acurácia basta para aprovar uma IA usada em decisões comerciais?"
    )
    st.caption(
        "Experimento de dados sintéticos e erros inseridos deliberadamente. "
        "A limpeza não melhora necessariamente a acurácia em qualquer problema; "
        "a conclusão exige validação em dados independentes."
    )
