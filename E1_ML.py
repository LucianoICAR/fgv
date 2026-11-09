"""Exercício executivo 1 — Machine Learning supervisionado (caso fictício).

Aplicação de uso educacional. O modelo já está treinado no arquivo
modelo_churn.joblib. Não é feito nenhum treinamento durante a execução.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PASTA = Path(__file__).resolve().parent
CAMPOS = [
    "cliente_id", "meses_contrato", "valor_mensal", "reclamacoes_90d",
    "atrasos_pagamento_6m", "uso_dados_gb", "ticket_suporte_90d",
    "tipo_plano", "regiao", "forma_pagamento", "fidelidade",
]
CAMPOS_MODELO = [c for c in CAMPOS if c != "cliente_id"]
NUMERICOS_INTEIROS = [
    "meses_contrato", "reclamacoes_90d", "atrasos_pagamento_6m",
    "ticket_suporte_90d",
]
NUMERICOS_DECIMAIS = ["valor_mensal", "uso_dados_gb"]
CATEGORIAS = {
    "tipo_plano": ["Básico", "Padrão", "Premium"],
    "regiao": ["Norte/Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
    "forma_pagamento": ["Boleto", "Cartão", "Débito automático"],
    "fidelidade": ["Sem fidelidade", "12 meses", "24 meses"],
}

st.set_page_config(
    page_title="FGV | Exercício executivo 1 — Churn",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def carregar_modelo():
    # Carregue somente arquivos de modelo provenientes de fonte confiável.
    return joblib.load(PASTA / "modelo_churn.joblib")


def validar_base(dados: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    mensagens: list[str] = []
    if dados.empty:
        return None, ["A base está vazia."]
    faltantes = [c for c in CAMPOS if c not in dados.columns]
    if faltantes:
        return None, ["Colunas obrigatórias ausentes: " + ", ".join(faltantes)]
    if len(dados) > 5000:
        mensagens.append("A base supera 5.000 clientes, limite didático deste exercício.")
    df = dados[CAMPOS].copy()
    ids = df["cliente_id"].astype("string").str.strip()
    if ids.isna().any() or (ids == "").any():
        mensagens.append("Há identificadores de clientes vazios.")
    if ids.duplicated().any():
        mensagens.append("Há cliente_id duplicados. Cada cliente deve aparecer uma única vez.")
    df["cliente_id"] = ids.astype(str)

    for coluna in NUMERICOS_INTEIROS + NUMERICOS_DECIMAIS:
        numeros = pd.to_numeric(df[coluna], errors="coerce")
        if numeros.isna().any():
            mensagens.append(f"A coluna '{coluna}' contém valores ausentes ou não numéricos.")
            continue
        if (numeros < 0).any():
            mensagens.append(f"A coluna '{coluna}' contém valores negativos.")
            continue
        if coluna in NUMERICOS_INTEIROS and not np.all(np.mod(numeros, 1) == 0):
            mensagens.append(f"A coluna '{coluna}' deve conter números inteiros.")
            continue
        df[coluna] = numeros.astype(int if coluna in NUMERICOS_INTEIROS else float)

    for coluna, valores_permitidos in CATEGORIAS.items():
        serie = df[coluna].astype("string").str.strip()
        invalidos = serie.isna() | ~serie.isin(valores_permitidos)
        if invalidos.any():
            mensagens.append(
                f"Valores inválidos em '{coluna}'. Aceitos: {', '.join(valores_permitidos)}."
            )
        df[coluna] = serie.astype(str)
    if mensagens:
        return None, mensagens
    return df, []


def faixa_risco(escore: float) -> str:
    # Faixas ilustrativas; não são thresholds validados para uso real.
    if escore >= 0.70:
        return "Alto"
    if escore >= 0.40:
        return "Médio"
    return "Baixo"


st.title("Exercício 1 | Você confiaria nesta IA?")
st.caption("FGV Educação Executiva • Primeira aula • Simulação de decisão gerencial")

st.markdown("### O desafio da operadora Conecta+")
st.write(
    "A Conecta+ enfrenta cancelamentos de contratos e precisa decidir quais clientes "
    "merecem atenção prioritária. Há orçamento para uma campanha de retenção, "
    "mas **não para atender toda a carteira**. A diretoria está avaliando um sistema de IA."
)

c1, c2, c3 = st.columns(3)
c1.metric("Orçamento da campanha", "R$ 4.000")
c2.metric("Custo estimado por ação", "R$ 200")
c3.metric("Capacidade da equipe", "20 clientes")
st.caption("Cenário e valores financeiros fictícios para fins de aprendizagem.")

st.info(
    "**Qual é a tecnologia?** Um sistema de **Machine Learning supervisionado**, "
    "treinado anteriormente com exemplos históricos de clientes que cancelaram "
    "e que permaneceram. **Você não vai treinar o modelo.** Seu papel é "
    "enviar a base atual e julgar se o resultado merece confiança."
)
st.markdown(
    "**Objetivo de negócio:** reduzir perdas de clientes respeitando o orçamento disponível.  \n"
    "**Objetivo da IA:** identificar e ordenar clientes por um escore estimado de risco de cancelamento."
)

st.markdown("---")
st.subheader("Etapa 1 — Envie a base de clientes atuais")
st.write(
    "Baixe a base fictícia de clientes, observe os campos e faça o upload do CSV. "
    "Ela contém **clientes atuais**, sem indicar quais deles cancelarão futuramente."
)
base_exemplo = (PASTA / "base_clientes_atuais_ficticia.csv").read_bytes()
modelo_csv = (PASTA / "modelo_base_para_preencher.csv").read_bytes()
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "1. Baixar base de 80 clientes fictícios (CSV)",
        data=base_exemplo,
        file_name="base_clientes_atuais_ficticia.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_b:
    st.download_button(
        "Baixar modelo de arquivo em branco (CSV)",
        data=modelo_csv,
        file_name="modelo_base_para_preencher.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.warning(
    "**Privacidade:** não envie nomes, contatos nem informações pessoais reais. "
    "Use dados fictícios ou devidamente anonimizados neste exercício."
)
with st.expander("Ver a descrição das colunas esperadas"):
    st.markdown(
        "**Identificação:** cliente_id.  \n"
        "**Relacionamento:** meses_contrato, tipo_plano, fidelidade.  \n"
        "**Consumo e valor:** valor_mensal, uso_dados_gb.  \n"
        "**Experiência:** reclamacoes_90d, ticket_suporte_90d.  \n"
        "**Pagamentos:** atrasos_pagamento_6m, forma_pagamento.  \n"
        "**Contexto:** regiao."
    )
    st.dataframe(
        pd.DataFrame([
            {"Campo": campo, "Valores aceitos": " | ".join(valores)}
            for campo, valores in CATEGORIAS.items()
        ]),
        hide_index=True,
        use_container_width=True,
    )

arquivo = st.file_uploader("2. Envie o CSV dos clientes atuais", type=["csv"])
if arquivo is None:
    st.info("Para começar, baixe a base fictícia e envie o arquivo acima.")
    st.stop()

identificador_arquivo = hashlib.sha256(arquivo.getvalue()).hexdigest()
try:
    entrada = pd.read_csv(arquivo, encoding="utf-8-sig", sep=None, engine="python")
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f"Não foi possível ler o CSV: {exc}")
    st.stop()

entrada, erros = validar_base(entrada)
if erros:
    for erro in erros:
        st.error(erro)
    st.stop()

st.success(f"Base recebida: {len(entrada)} clientes atuais.")
with st.expander("Conferir uma amostra da base enviada"):
    st.dataframe(entrada.head(10), use_container_width=True, hide_index=True)

if st.button("3. Executar previsão da IA", type="primary", use_container_width=True):
    try:
        modelo = carregar_modelo()
        escores = modelo.predict_proba(entrada[CAMPOS_MODELO])[:, 1]
        resultado = entrada.copy()
        resultado["escore_risco"] = escores
        resultado["faixa_ilustrativa"] = [faixa_risco(v) for v in escores]
        resultado = resultado.sort_values(
            ["escore_risco", "valor_mensal"], ascending=[False, False]
        ).reset_index(drop=True)
        st.session_state["analise"] = resultado
        st.session_state["analise_id"] = identificador_arquivo
    except Exception as exc:
        st.error(f"Não foi possível executar a previsão: {exc}")
        st.stop()

if st.session_state.get("analise_id") != identificador_arquivo:
    st.stop()

resultado = st.session_state["analise"]
st.markdown("---")
st.subheader("Etapa 2 — O que a IA está sugerindo?")
st.caption(
    "O modelo fornece um **escore de risco**, não uma certeza sobre o futuro. "
    "As faixas 'alto', 'médio' e 'baixo' são convenções didáticas e não foram validadas "
    "para decidir gastos reais."
)

m1, m2, m3 = st.columns(3)
m1.metric("Clientes analisados", len(resultado))
m2.metric("Faixa de risco alto", int((resultado["faixa_ilustrativa"] == "Alto").sum()))
m3.metric("Faixas médio ou alto", int((resultado["escore_risco"] >= 0.40).sum()))

st.markdown("#### Ranking dos clientes")
visao = resultado[[
    "cliente_id", "escore_risco", "faixa_ilustrativa", "valor_mensal",
    "reclamacoes_90d", "atrasos_pagamento_6m", "meses_contrato", "tipo_plano",
]].copy()
visao["escore_risco"] = (visao["escore_risco"] * 100).round(1)
visao.rename(columns={
    "cliente_id": "Cliente", "escore_risco": "Escore de risco (%)",
    "faixa_ilustrativa": "Faixa ilustrativa", "valor_mensal": "Mensalidade (R$)",
    "reclamacoes_90d": "Reclamações (90 dias)",
    "atrasos_pagamento_6m": "Atrasos (6 meses)",
    "meses_contrato": "Meses de contrato", "tipo_plano": "Plano",
}, inplace=True)
st.dataframe(visao, hide_index=True, use_container_width=True)

st.markdown("---")
st.subheader("Etapa 3 — O orçamento permite seguir todos os alertas?")
col_orc, col_custo, col_cap = st.columns(3)
with col_orc:
    orcamento = st.number_input(
        "Orçamento disponível (R$)", min_value=0, max_value=100000,
        value=4000, step=500,
    )
with col_custo:
    custo = st.number_input(
        "Custo estimado por ação (R$)", min_value=1, max_value=10000,
        value=200, step=50,
    )
with col_cap:
    capacidade = st.number_input(
        "Capacidade máxima da equipe (clientes)", min_value=1,
        max_value=5000, value=20, step=5,
    )

max_acoes = min(len(resultado), int(orcamento // custo), capacidade)
selecionados = resultado.head(max_acoes).copy()
politica_ampla = resultado[resultado["escore_risco"] >= 0.40].copy()
custo_amplo = len(politica_ampla) * custo

p1, p2 = st.columns(2)
with p1:
    st.markdown("**Alternativa A — priorização pelo ranking**")
    st.metric("Clientes que cabem no orçamento", max_acoes)
    st.write(f"Custo estimado: **R$ {max_acoes * custo:,.0f}**")
    st.caption("Seleciona os maiores escores até o limite de recursos e capacidade.")
with p2:
    st.markdown("**Alternativa B — atender alertas médios e altos**")
    st.metric("Clientes sinalizados", len(politica_ampla))
    st.write(f"Custo estimado: **R$ {custo_amplo:,.0f}**")
    if custo_amplo > orcamento or len(politica_ampla) > capacidade:
        st.warning("Essa alternativa excede o orçamento e/ou a capacidade da equipe.")
    else:
        st.success("A alternativa cabe no orçamento e na capacidade estimada.")

st.caption(
    "As duas alternativas são hipóteses de priorização, não recomendações de investimento. "
    "O modelo não informa quem aceitará uma oferta nem o benefício financeiro da ação."
)
st.markdown("**Clientes selecionados na alternativa A**")
st.dataframe(
    visao.head(max_acoes), hide_index=True, use_container_width=True
)

exportacao = resultado.copy()
exportacao["selecionado_alternativa_A"] = [
    "Sim" if indice < max_acoes else "Não" for indice in range(len(exportacao))
]
exportacao["escore_risco_percentual"] = (exportacao["escore_risco"] * 100).round(1)
exportacao.drop(columns=["escore_risco"], inplace=True)
st.download_button(
    "Baixar ranking e proposta de priorização (CSV)",
    data=exportacao.to_csv(index=False).encode("utf-8-sig"),
    file_name="ranking_churn_exercicio1.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("---")
st.subheader("Etapa 4 — Decisão do comitê executivo")
st.warning(
    "**Atenção:** você não recebeu o conjunto de treinamento, resultados de validação "
    "independente nem histórico de sucesso das ações de retenção. "
    "Também não conhece quais clientes da base atual realmente cancelarão."
)
perguntas = [
    "1. Você confiaria nessa IA para direcionar orçamento de retenção? Por quê?",
    "2. Qual o risco de agir sobre previsões erradas?",
    "3. Quem deveria aprovar o uso dessa IA no negócio?",
]
respostas = []
for i, pergunta in enumerate(perguntas, start=1):
    respostas.append(st.text_area(pergunta, key=f"resposta_{i}", height=78))

parecer = st.radio(
    "Qual é a decisão provisória da equipe?",
    [
        "Autorizar uso imediato",
        "Autorizar somente um piloto controlado",
        "Solicitar novas evidências antes de autorizar",
    ],
    index=2,
)
relatorio = (
    "EXERCÍCIO EXECUTIVO 1 — CONECTA+\n"
    "ML supervisionado — simulação didática\n\n"
    f"Clientes avaliados: {len(resultado)}\n"
    f"Orçamento: R$ {orcamento}\n"
    f"Custo por ação: R$ {custo}\n"
    f"Capacidade: {capacidade}\n"
    f"Selecionados pelo ranking: {max_acoes}\n\n"
    + "\n\n".join(f"{p}\n{r}" for p, r in zip(perguntas, respostas))
    + f"\n\nDECISÃO DO GRUPO: {parecer}\n"
)
st.download_button(
    "Baixar parecer executivo da equipe (TXT)",
    data=relatorio.encode("utf-8"),
    file_name="parecer_executivo_conecta_mais.txt",
    mime="text/plain",
    use_container_width=True,
)
st.caption(
    "Experimento educacional com dados inteiramente fictícios. "
    "Não usar o modelo para decisões sobre clientes reais."
)
