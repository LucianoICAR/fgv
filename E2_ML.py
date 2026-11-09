"""FGV Educação Executiva — Atividade 2: Comitê de Investimentos em IA.

Aplicativo didático de um só arquivo. Os CSVs são gerados neste próprio código.
O aluno precisa baixar e enviar as bases de treino e de teste, nessa ordem.
NÃO utilizar os escores gerados para decisões sobre pessoas ou empresas reais.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="FGV | Comitê de IA — Entregas", page_icon="📦", layout="wide")

# -----------------------------------------------------------------------------
# CONTEXTO DIDÁTICO E DEFINIÇÕES
# -----------------------------------------------------------------------------
NUMERICAS = [
    "distancia_km", "prazo_dias", "horas_atraso_coleta",
    "pedidos_pendentes_transportadora", "historico_atrasos_transportadora_pct",
]
CATEGORICAS = ["regiao", "modal", "tipo_carga", "importancia_cliente"]
COLUNAS_MODELO = NUMERICAS + CATEGORICAS
ID = "pedido_id"
ALVO = "atrasou"
CUSTO_ACAO = 200   # R$ por pedido priorizado (hipótese para a aula)
ORCAMENTO_BASE = 3000

DICIONARIO = {
    "pedido_id": "Identificador único do pedido.",
    "distancia_km": "Distância aproximada entre origem e destino (km).",
    "prazo_dias": "Prazo de entrega contratado (dias).",
    "horas_atraso_coleta": "Horas de atraso já registradas na coleta, antes da previsão.",
    "pedidos_pendentes_transportadora": "Volume de pedidos pendentes na transportadora.",
    "historico_atrasos_transportadora_pct": "Percentual histórico de atrasos da transportadora.",
    "regiao": "Região de destino.",
    "modal": "Modal de transporte.",
    "tipo_carga": "Tipo de carga.",
    "importancia_cliente": "Criticidade comercial definida pelo negócio.",
    "atrasou": "Rótulo histórico: 1 = entrega atrasada; 0 = entrega no prazo.",
}


def criar_base(n: int, seed: int, prefixo: str, com_rotulo: bool) -> pd.DataFrame:
    """Gera exemplos fictícios reproduzíveis sem arquivo externo."""
    rng = np.random.default_rng(seed)
    distancia = rng.integers(30, 2201, size=n)
    prazo = rng.integers(2, 11, size=n)
    atraso_coleta = rng.integers(0, 27, size=n)
    pendentes = rng.integers(1, 51, size=n)
    atraso_historico = np.round(rng.uniform(3, 48, n), 1)
    regiao = rng.choice(["Norte/Nordeste", "Centro-Oeste", "Sudeste", "Sul"], n,
                        p=[0.28, 0.13, 0.42, 0.17])
    modal = rng.choice(["Rodoviário", "Aéreo"], n, p=[0.84, 0.16])
    carga = rng.choice(["Geral", "Frágil", "Refrigerada"], n, p=[0.69, 0.21, 0.10])
    importancia = rng.choice(["Alta", "Média", "Baixa"], n, p=[0.20, 0.43, 0.37])

    df = pd.DataFrame({
        ID: [f"{prefixo}-{i:04d}" for i in range(1, n + 1)],
        "distancia_km": distancia,
        "prazo_dias": prazo,
        "horas_atraso_coleta": atraso_coleta,
        "pedidos_pendentes_transportadora": pendentes,
        "historico_atrasos_transportadora_pct": atraso_historico,
        "regiao": regiao,
        "modal": modal,
        "tipo_carga": carga,
        "importancia_cliente": importancia,
    })

    # Mecanismo didático desconhecido pelo aluno. Inclui fatores não observados
    # e aleatoriedade: previsões nunca devem ser tratadas como certezas.
    escore_latente = (
        -2.10 + 0.045 * atraso_coleta + 0.019 * pendentes
        + 0.027 * atraso_historico + 0.00055 * distancia
        - 0.25 * prazo + 0.28 * (modal == "Rodoviário")
        + 0.38 * (carga == "Refrigerada")
        + 0.20 * (regiao == "Norte/Nordeste")
        + rng.normal(0, 0.9, n)
    )
    prob_sintetica = 1.0 / (1.0 + np.exp(-escore_latente))
    if com_rotulo:
        df[ALVO] = rng.binomial(1, prob_sintetica).astype(int)
    return df


# Bases são construídas internamente. Não há chamadas a read_csv de caminhos locais.
BASE_TREINO = criar_base(1200, 20261109, "HIST", True)
BASE_ATUAL = criar_base(70, 20261110, "ATUAL", False)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def preparar_modelo() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    clf = RandomForestClassifier(
        n_estimators=180, max_depth=8, min_samples_leaf=7,
        class_weight="balanced", random_state=17,
    )
    return Pipeline([("preparacao", pre), ("classificador", clf)])


def validar(df: pd.DataFrame, treino: bool) -> pd.DataFrame:
    obrigatorias = [ID] + COLUNAS_MODELO + ([ALVO] if treino else [])
    faltantes = [c for c in obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError("Colunas ausentes: " + ", ".join(faltantes))
    df = df[obrigatorias].copy()
    if df.empty:
        raise ValueError("A base está vazia.")
    if df[ID].isna().any() or df[ID].astype(str).str.strip().eq("").any():
        raise ValueError("Há pedidos sem identificador.")
    if df[ID].duplicated().any():
        raise ValueError("Existem pedidos duplicados. Verifique o campo pedido_id.")
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any() or (df[c] < 0).any():
            raise ValueError(f"A coluna {c} contém valores vazios, inválidos ou negativos.")
    for c in CATEGORICAS:
        if df[c].isna().any() or df[c].astype(str).str.strip().eq("").any():
            raise ValueError(f"A coluna {c} contém valores vazios.")
        df[c] = df[c].astype(str).str.strip()
    if treino:
        df[ALVO] = pd.to_numeric(df[ALVO], errors="coerce")
        if not df[ALVO].isin([0, 1]).all():
            raise ValueError("A coluna atrasou deve conter apenas 0 e 1.")
        df[ALVO] = df[ALVO].astype(int)
        if df[ALVO].nunique() < 2:
            raise ValueError("A base de treino deve conter exemplos com e sem atraso.")
        if len(df) < 60 or df[ALVO].value_counts().min() < 10:
            raise ValueError("A base de treino precisa de ao menos 60 registros e 10 casos por classe.")
    return df


def marcar_download(chave: str) -> None:
    st.session_state[chave] = True


def arquivo_para_df(upload) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(upload.getvalue()), encoding="utf-8-sig")


def moeda(v: float) -> str:
    return "R$ " + f"{v:,.0f}".replace(",", ".")


def quadrante(impacto: int, viabilidade: int) -> str:
    if impacto >= 4 and viabilidade >= 4:
        return "PRIORIZAR"
    if impacto >= 4:
        return "INVESTIGAR"
    if viabilidade >= 4:
        return "IMPLEMENTAR COM AGILIDADE"
    return "POSTERGAR"


# -----------------------------------------------------------------------------
# EXPERIÊNCIA EM SALA — 4 ETAPAS
# -----------------------------------------------------------------------------
st.title("📦 Comitê de Investimentos em IA")
st.subheader("Atividade executiva — IA para antecipar atrasos em entregas")
st.caption("FGV | MBA em Inteligência Artificial • Caso fictício e dados sintéticos")

with st.container(border=True):
    st.markdown("### O problema de negócio")
    st.write(
        "A distribuidora **RotaCerta** sofre com atrasos que geram reclamações, "
        "multas contratuais e perda de clientes. A diretoria avalia um sistema de "
        "**Machine Learning supervisionado** para antecipar entregas de maior risco "
        "e direcionar uma equipe de intervenção."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Orçamento para intervenções", moeda(ORCAMENTO_BASE))
    c2.metric("Custo estimado por intervenção", moeda(CUSTO_ACAO))
    c3.metric("Capacidade inicial", f"{ORCAMENTO_BASE // CUSTO_ACAO} pedidos")
    st.markdown(
        "**Objetivo de negócio:** reduzir atrasos e impactos sobre os clientes.  "
        "**Objetivo do modelo:** identificar quais pedidos têm maior risco de atraso.  "
        "**Decisão executiva:** investir ou não na iniciativa, considerando dados, erros, "
        "governança, custo e indicadores de sucesso."
    )

st.info(
    "**Regra da atividade:** é obrigatório baixar as duas bases disponibilizadas no próprio sistema "
    "e depois enviá-las nos campos indicados. A base histórica tem o resultado real das entregas "
    "para treinar; a base atual não tem rótulos e será usada para prever."
)

# Etapa 1 — Treinamento
st.markdown("---")
st.header("1. Baixe a base histórica e treine o modelo")
c1, c2 = st.columns([1, 1])
with c1:
    st.download_button(
        "⬇️ Baixar base de TREINAMENTO (1.200 pedidos)",
        data=csv_bytes(BASE_TREINO), file_name="base_historica_treinamento.csv",
        mime="text/csv", on_click=marcar_download, args=("baixou_treino",),
        use_container_width=True,
    )
with c2:
    st.write("**O que contém:** histórico dos pedidos e a coluna **atrasou** (0 ou 1).")
    st.caption("A base é sintética. As variáveis representam informações conhecidas antes da entrega.")

baixou_treino = bool(st.session_state.get("baixou_treino", False))
if baixou_treino:
    arq_treino = st.file_uploader("Envie o CSV de treinamento que acabou de baixar", type="csv", key="up_treino")
else:
    arq_treino = None
    st.warning("Primeiro clique no botão de download da base histórica.")

if arq_treino is not None:
    if st.button("🧠 Treinar e avaliar o modelo", type="primary", use_container_width=True):
        try:
            df = validar(arquivo_para_df(arq_treino), treino=True)
            X = df[COLUNAS_MODELO]
            y = df[ALVO]
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=19,
            )
            p_val = preparar_modelo()
            p_val.fit(Xtr, ytr)
            pred_val = p_val.predict(Xva)
            tn, fp, fn, tp = confusion_matrix(yva, pred_val, labels=[0, 1]).ravel()
            # O modelo usado nos pedidos atuais é treinado com toda a base histórica.
            modelo_final = preparar_modelo()
            modelo_final.fit(X, y)
            st.session_state["modelo"] = modelo_final
            st.session_state["resultado_validacao"] = {
                "acuracia": float(accuracy_score(yva, pred_val)),
                "recall": float(recall_score(yva, pred_val, zero_division=0)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "n_validacao": len(yva), "n_treinamento": len(df),
            }
            st.session_state["treino_hash"] = hashlib.sha256(arq_treino.getvalue()).hexdigest()
            st.session_state.pop("resultado_atual", None)
            st.session_state.pop("parecer_txt", None)
            st.success("Modelo treinado! Os resultados abaixo foram obtidos em dados de validação separados.")
        except (ValueError, KeyError, pd.errors.ParserError, UnicodeError) as e:
            st.error(f"Não foi possível treinar: {e}")

if "resultado_validacao" in st.session_state:
    validar_treino = (
        arq_treino is not None and
        hashlib.sha256(arq_treino.getvalue()).hexdigest() == st.session_state.get("treino_hash")
    )
    if not validar_treino:
        st.warning("A base de treino foi alterada ou removida. Treine novamente antes de continuar.")
    else:
        m = st.session_state["resultado_validacao"]
        st.markdown("**Evidências de validação — o modelo acerta, mas também erra**")
        a, b, c, d = st.columns(4)
        a.metric("Previsões corretas", f"{m['acuracia']:.0%}")
        b.metric("Atrasos identificados", str(m["tp"]))
        c.metric("Atrasos não identificados", str(m["fn"]))
        d.metric("Alertas desnecessários", str(m["fp"]))
        st.caption(
            "Esses resultados vêm de 25% da base histórica separados antes do treinamento "
            "de validação. Não demonstram, por si só, que o modelo funcionará da mesma forma em produção."
        )
        st.info(
            "**Questão para o executivo:** atrasos não identificados e alertas desnecessários "
            "têm o mesmo custo? Qual fonte de dados importante está faltando? "
            "Ex.: clima, trânsito ou bloqueios de vias."
        )
else:
    validar_treino = False

# Etapa 2 — Base atual
st.markdown("---")
st.header("2. Baixe a base atual, envie o CSV e gere previsões")
arq_teste = None
if not validar_treino:
    st.warning("Conclua o treinamento acima para liberar a base de pedidos atuais.")
else:
    st.download_button(
        "⬇️ Baixar base de TESTE (70 pedidos atuais)",
        data=csv_bytes(BASE_ATUAL), file_name="base_pedidos_atuais_teste.csv",
        mime="text/csv", on_click=marcar_download, args=("baixou_teste",),
        use_container_width=True,
    )
    st.caption("A base atual não possui a coluna 'atrasou': os pedidos ainda não tiveram seu desfecho.")

    if st.session_state.get("baixou_teste", False):
        arq_teste = st.file_uploader("Envie o CSV de teste que acabou de baixar", type="csv", key="up_teste")
    else:
        arq_teste = None
        st.warning("Baixe a base de teste antes de continuar.")

    if arq_teste is not None:
        if st.button("🔎 Analisar pedidos atuais", type="primary", use_container_width=True):
            try:
                df_atual = validar(arquivo_para_df(arq_teste), treino=False)
                # predict_proba de Random Forest é um ESCORE, não garantia de calibração.
                riscos = st.session_state["modelo"].predict_proba(df_atual[COLUNAS_MODELO])[:, 1]
                resultado = df_atual.copy()
                resultado["escore_risco_atraso"] = np.round(riscos * 100, 1)
                resultado["faixa_risco"] = pd.cut(
                    resultado["escore_risco_atraso"],
                    bins=[-0.1, 39.999, 64.999, 100.001],
                    labels=["Baixo", "Médio", "Alto"],
                ).astype(str)
                resultado = resultado.sort_values("escore_risco_atraso", ascending=False)
                st.session_state["resultado_atual"] = resultado
                st.session_state.pop("parecer_txt", None)
                st.session_state["teste_hash"] = hashlib.sha256(arq_teste.getvalue()).hexdigest()
                st.success("Previsões concluídas. Analise a lista de prioridades.")
            except (ValueError, KeyError, pd.errors.ParserError, UnicodeError) as e:
                st.error(f"Não foi possível analisar os pedidos: {e}")

resultado_pronto = (
    validar_treino and
    arq_teste is not None and
    "resultado_atual" in st.session_state and
    hashlib.sha256(arq_teste.getvalue()).hexdigest() == st.session_state.get("teste_hash")
)
if resultado_pronto:
    df_res = st.session_state["resultado_atual"]
    st.markdown("### Ranking de pedidos para avaliação gerencial")
    b1, b2, b3 = st.columns(3)
    b1.metric("Pedidos avaliados", len(df_res))
    b2.metric("Alto risco", int((df_res["faixa_risco"] == "Alto").sum()))
    b3.metric("Médio ou alto", int((df_res["faixa_risco"] != "Baixo").sum()))
    st.dataframe(
        df_res[[ID, "regiao", "importancia_cliente", "horas_atraso_coleta",
                "escore_risco_atraso", "faixa_risco"]],
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Faixas e escores são ilustrativos. Um escore de 70 não deve ser interpretado como "
        "probabilidade calibrada de 70% sem verificação adicional."
    )

    st.markdown("### Simulação de capacidade e orçamento")
    orcamento = st.slider("Orçamento disponível para intervenções (R$)",
                          min_value=1000, max_value=6000, value=ORCAMENTO_BASE, step=200)
    limite = orcamento // CUSTO_ACAO
    df_prioritarios = df_res.head(limite).copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Capacidade de intervenção", f"{limite} pedidos")
    c2.metric("Custo planejado", moeda(len(df_prioritarios) * CUSTO_ACAO))
    c3.metric("Pedidos sem intervenção", int(len(df_res) - len(df_prioritarios)))
    st.write("**Pedidos que receberiam atenção prioritária (por escore de risco):**")
    st.dataframe(df_prioritarios[[ID, "escore_risco_atraso", "importancia_cliente"]],
                 hide_index=True, use_container_width=True)
    st.download_button("⬇️ Baixar ranking completo", csv_bytes(df_res),
                       file_name="resultado_pedidos_atuais.csv", mime="text/csv")
    st.warning(
        "**Atenção:** priorizar maiores escores não comprova que essas intervenções "
        "evitarão atrasos. O custo das ações foi informado, mas sua efetividade e o "
        "benefício financeiro ainda precisam ser demonstrados."
    )

# Etapa 3 — avaliação de oportunidade
st.markdown("---")
st.header("3. Avalie a oportunidade de IA e decida como executivo")
if not resultado_pronto:
    st.info("Esta etapa é liberada após o treinamento e a análise dos pedidos atuais.")
else:
    st.write(
        "Utilize as seis dimensões do processo gerencial: **problema de negócio, "
        "dados disponíveis, solução de IA, riscos e governança, decisão de investimento "
        "e indicadores de sucesso**."
    )
    with st.form("form_decisao"):
        grupo = st.text_input("Equipe / grupo", placeholder="Ex.: Equipe 3")
        q1 = st.text_area("1. Problema de negócio — Qual resultado concreto buscamos melhorar?", height=75)
        q2 = st.text_area("2. Dados — Que fontes críticas faltam e quais atributos precisam ser acrescentados?", height=75)
        q3 = st.text_area("3. Solução — Que erros observados na validação preocupam o gestor?", height=75)
        q4 = st.text_area("4. Riscos e governança — Que controles e aprovações são indispensáveis?", height=75)
        q5 = st.text_area("5. Investimento — O resultado potencial justifica custos e esforço? O que falta comprovar?", height=75)
        q6 = st.text_area("6. Indicadores — Como acompanharíamos sucesso, custos e impacto após um piloto?", height=75)
        st.markdown("**Posicionamento na matriz impacto × viabilidade**")
        co1, co2 = st.columns(2)
        with co1:
            impacto = st.slider("Impacto potencial para o negócio", 1, 5, 3)
        with co2:
            viabilidade = st.slider("Viabilidade de implantação", 1, 5, 3)
        aprovacao = st.radio(
            "Qual decisão a equipe recomenda?",
            ["Não aprovar ainda; solicitar evidências", "Autorizar um piloto controlado", "Implantar imediatamente"],
        )
        justificativa = st.text_area("Justifique a decisão em até cinco linhas", height=100)
        enviar = st.form_submit_button("📋 Consolidar parecer executivo", type="primary", use_container_width=True)
    if enviar:
        recomendacao = quadrante(impacto, viabilidade)
        st.markdown(f"### Quadrante da iniciativa: **{recomendacao}**")
        st.write(f"Impacto: **{impacto}/5** • Viabilidade: **{viabilidade}/5**")
        if aprovacao == "Implantar imediatamente":
            st.warning(
                "O grupo escolheu implantação imediata. Quais evidências faltam para "
                "garantir que essa decisão não expõe a empresa a riscos inaceitáveis?"
            )
        elif aprovacao == "Autorizar um piloto controlado":
            st.info("O piloto deve ter responsável, custo máximo, critérios de sucesso e critérios de suspensão.")
        else:
            st.info("Registre as evidências mínimas exigidas e quem será responsável por produzi-las.")
        texto = "\n".join([
            "FGV — PARECER EXECUTIVO (SIMULAÇÃO DIDÁTICA)",
            f"Equipe: {grupo or 'Não informada'}",
            f"Quadrante: {recomendacao} | Impacto: {impacto}/5 | Viabilidade: {viabilidade}/5",
            f"Decisão: {aprovacao}", "",
            "1. Problema de negócio: " + q1,
            "2. Dados disponíveis: " + q2,
            "3. Solução e erros: " + q3,
            "4. Riscos e governança: " + q4,
            "5. Investimento: " + q5,
            "6. Indicadores: " + q6,
            "Justificativa: " + justificativa,
        ])
        st.session_state["parecer_txt"] = texto
    if st.session_state.get("parecer_txt"):
        st.download_button("⬇️ Baixar parecer da equipe (.txt)",
                           data=st.session_state["parecer_txt"].encode("utf-8-sig"),
                           file_name="parecer_executivo_ia.txt", mime="text/plain")

st.markdown("---")
with st.expander("📘 Dicionário dos dados / informações para a equipe"):
    st.dataframe(pd.DataFrame([{"Campo": c, "Significado": d} for c, d in DICIONARIO.items()]),
                 hide_index=True, use_container_width=True)
    st.markdown(
        "**Limitação intencional:** trânsito, clima, bloqueios de vias e eventos excepcionais "
        "não estão incluídos na base. O exercício simula a avaliação gerencial de uma IA "
        "ainda não aprovada para uso real."
    )
st.caption("FGV | Caso, bases, valores e resultados sintéticos, apenas para fins educacionais.")
