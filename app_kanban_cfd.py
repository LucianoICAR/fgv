
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Exercício Kanban com CFD",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exercício Kanban: Construção e Análise do CFD")
st.caption("Cumulative Flow Diagram para diagnóstico da saúde do fluxo em um sistema Kanban.")

# ---------------------------------------------------------
# Dataset padrão
# ---------------------------------------------------------

def dataset_padrao():
    return pd.DataFrame({
        "Dia": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Backlog": [20, 18, 16, 15, 14, 13, 12, 11, 10, 9],
        "Análise": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Desenvolvimento": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "Testes": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
        "Concluído": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    })


# ---------------------------------------------------------
# Funções de análise
# ---------------------------------------------------------

def validar_colunas(df):
    colunas_necessarias = ["Dia", "Backlog", "Análise", "Desenvolvimento", "Testes", "Concluído"]
    faltantes = [c for c in colunas_necessarias if c not in df.columns]
    return faltantes


def calcular_metricas(df):
    etapas_wip = ["Análise", "Desenvolvimento", "Testes"]
    df = df.copy()

    df["WIP Total"] = df[etapas_wip].sum(axis=1)
    df["Throughput Diário"] = df["Concluído"].diff().fillna(df["Concluído"])

    wip_inicial = df["WIP Total"].iloc[0]
    wip_final = df["WIP Total"].iloc[-1]
    variacao_wip = wip_final - wip_inicial

    throughput_medio = df["Throughput Diário"].mean()
    throughput_final = df["Throughput Diário"].tail(3).mean()

    crescimento_por_etapa = {
        etapa: df[etapa].iloc[-1] - df[etapa].iloc[0]
        for etapa in etapas_wip
    }

    gargalo = max(crescimento_por_etapa, key=crescimento_por_etapa.get)

    if variacao_wip <= 2 and throughput_medio >= 1:
        saude = "🟢 Saudável"
        diagnostico = "O WIP está relativamente controlado e o fluxo apresenta sinais de estabilidade."
    elif variacao_wip <= 6:
        saude = "🟡 Atenção"
        diagnostico = "O WIP está crescendo. O sistema ainda pode ser recuperado com ajustes de fluxo e limites de WIP."
    else:
        saude = "🔴 Fora de controle"
        diagnostico = "O WIP cresce de forma acentuada. O sistema está acumulando trabalho e perdendo previsibilidade."

    return {
        "df": df,
        "wip_inicial": wip_inicial,
        "wip_final": wip_final,
        "variacao_wip": variacao_wip,
        "throughput_medio": throughput_medio,
        "throughput_final": throughput_final,
        "crescimento_por_etapa": crescimento_por_etapa,
        "gargalo": gargalo,
        "saude": saude,
        "diagnostico": diagnostico
    }


def plotar_cfd(df):
    etapas = ["Backlog", "Análise", "Desenvolvimento", "Testes", "Concluído"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(df["Dia"], [df[e] for e in etapas], labels=etapas)

    ax.set_title("CFD — Cumulative Flow Diagram")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Quantidade de itens")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return fig


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("⚙️ Configurações")

usar_upload = st.sidebar.checkbox("Usar arquivo CSV próprio", value=False)

st.sidebar.markdown("""
### Formato esperado do CSV

O arquivo deve conter as colunas:

`Dia`, `Backlog`, `Análise`, `Desenvolvimento`, `Testes`, `Concluído`
""")

arquivo = None
if usar_upload:
    arquivo = st.sidebar.file_uploader("Enviar CSV", type=["csv"])

modo_professor = st.sidebar.checkbox("Mostrar gabarito / modo professor", value=False)

# ---------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------

if usar_upload and arquivo is not None:
    df = pd.read_csv(arquivo)
else:
    df = dataset_padrao()

faltantes = validar_colunas(df)

if faltantes:
    st.error(f"O arquivo não contém as colunas obrigatórias: {faltantes}")
    st.stop()

df = df.sort_values("Dia").reset_index(drop=True)
metricas = calcular_metricas(df)
df_metricas = metricas["df"]

# ---------------------------------------------------------
# Abas
# ---------------------------------------------------------

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📘 Case",
    "📊 CFD",
    "🔍 Diagnóstico",
    "📝 Exercício",
    "✅ Gabarito"
])

# ---------------------------------------------------------
# Aba 1
# ---------------------------------------------------------

with aba1:
    st.header("📘 Case: FlowTech Digital")

    st.markdown("""
A empresa **FlowTech Digital** implantou um quadro Kanban para controlar demandas de desenvolvimento de software.

Após alguns dias de operação, a diretoria solicitou uma análise da saúde do fluxo.

O objetivo é responder:

> **O sistema Kanban está saudável, previsível e estável?**

Para isso, a equipe deverá construir e interpretar um **CFD — Cumulative Flow Diagram**.
""")

    st.subheader("Dados coletados do Kanban")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Baixar dataset padrão em CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dados_cfd_kanban.csv",
        mime="text/csv"
    )

    st.info("""
Missão dos alunos: construir o CFD, analisar o comportamento das faixas, identificar gargalos e avaliar a saúde do projeto.
""")

# ---------------------------------------------------------
# Aba 2
# ---------------------------------------------------------

with aba2:
    st.header("📊 Cumulative Flow Diagram — CFD")

    st.markdown("""
O **CFD** mostra a evolução acumulada dos itens em cada etapa do fluxo.

Em um sistema saudável, as faixas tendem a permanecer relativamente estáveis e paralelas.
Quando uma faixa começa a abrir demais, isso indica **acúmulo de trabalho** naquela etapa.
""")

    fig = plotar_cfd(df)
    st.pyplot(fig)

    st.subheader("Dados com métricas calculadas")
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Aba 3
# ---------------------------------------------------------

with aba3:
    st.header("🔍 Diagnóstico Automático do Fluxo")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("WIP inicial", int(metricas["wip_inicial"]))
    c2.metric("WIP final", int(metricas["wip_final"]))
    c3.metric("Variação do WIP", int(metricas["variacao_wip"]))
    c4.metric("Throughput médio", f"{metricas['throughput_medio']:.2f}/dia")

    st.subheader("Saúde do sistema")
    st.markdown(f"## {metricas['saude']}")
    st.write(metricas["diagnostico"])

    st.subheader("Possível gargalo")
    st.warning(f"A etapa com maior crescimento de acúmulo é: **{metricas['gargalo']}**.")

    st.subheader("Crescimento do WIP por etapa")
    df_gargalos = pd.DataFrame({
        "Etapa": list(metricas["crescimento_por_etapa"].keys()),
        "Crescimento no período": list(metricas["crescimento_por_etapa"].values())
    })
    st.dataframe(df_gargalos, use_container_width=True, hide_index=True)
    st.bar_chart(df_gargalos.set_index("Etapa"))

    st.subheader("Interpretação executiva")
    if metricas["saude"].startswith("🟢"):
        st.success("""
O fluxo apresenta boa estabilidade. A recomendação é manter os limites de WIP, acompanhar throughput e preservar a cadência de entrega.
""")
    elif metricas["saude"].startswith("🟡"):
        st.warning("""
O fluxo exige atenção. A recomendação é reduzir o início de novas demandas, revisar limites de WIP e atuar na etapa que está acumulando trabalho.
""")
    else:
        st.error("""
O fluxo está fora de controle. A recomendação é parar de iniciar novas demandas, focar em finalizar itens em andamento, atuar no gargalo e redefinir limites de WIP.
""")

# ---------------------------------------------------------
# Aba 4
# ---------------------------------------------------------

with aba4:
    st.header("📝 Exercício para os Alunos")

    st.markdown("""
## Objetivo

Construir e interpretar o **CFD** para avaliar a saúde do sistema Kanban.

---

## Entregáveis

### 1. Construção do CFD

A equipe deve gerar o gráfico CFD com:
- eixo X: dias;
- eixo Y: quantidade de itens;
- faixas: Backlog, Análise, Desenvolvimento, Testes e Concluído.

---

### 2. Diagnóstico visual

Responder:

1. As faixas do CFD estão paralelas ou abrindo?
2. O WIP está aumentando, diminuindo ou estável?
3. Existe alguma etapa acumulando trabalho?
4. O throughput parece constante?
5. O sistema é previsível?

---

### 3. Classificação da saúde do projeto

Classificar o sistema como:

- 🟢 Saudável;
- 🟡 Atenção;
- 🔴 Fora de controle.

A equipe deve justificar a classificação.

---

### 4. Decisão executiva

Responder:

> Se você fosse o gerente do projeto, qual decisão tomaria imediatamente?

A resposta deve considerar:
- limites de WIP;
- gargalos;
- fluxo puxado;
- foco em finalizar antes de iniciar;
- previsibilidade de entrega.

---

### 5. Proposta de melhoria

A equipe deve propor pelo menos **3 ações práticas** para melhorar o fluxo.
""")

    st.download_button(
        label="⬇️ Baixar enunciado da atividade",
        data="""ATIVIDADE — KANBAN COM CFD

Case:
A FlowTech Digital implantou um sistema Kanban e coletou dados diários do fluxo. A diretoria deseja saber se o sistema está saudável, previsível e estável.

Objetivo:
Construir e interpretar o CFD — Cumulative Flow Diagram.

Entregáveis:
1. Construção do CFD
2. Diagnóstico visual
3. Classificação da saúde do projeto
4. Decisão executiva
5. Proposta de melhoria

Perguntas:
1. As faixas do CFD estão paralelas ou abrindo?
2. O WIP está aumentando, diminuindo ou estável?
3. Existe alguma etapa acumulando trabalho?
4. O throughput parece constante?
5. O sistema é previsível?
6. Qual decisão executiva deve ser tomada imediatamente?
""",
        file_name="atividade_kanban_cfd.txt",
        mime="text/plain"
    )

# ---------------------------------------------------------
# Aba 5
# ---------------------------------------------------------

with aba5:
    st.header("✅ Gabarito / Modo Professor")

    if not modo_professor:
        st.info("Ative o modo professor na barra lateral para visualizar o gabarito.")
    else:
        st.subheader("Diagnóstico esperado")

        st.markdown(f"""
### 1. Saúde do sistema

Classificação esperada: **{metricas['saude']}**

Justificativa:
- WIP inicial: **{int(metricas['wip_inicial'])}**
- WIP final: **{int(metricas['wip_final'])}**
- Variação do WIP: **{int(metricas['variacao_wip'])}**
- Throughput médio: **{metricas['throughput_medio']:.2f} item(ns)/dia**

---

### 2. Interpretação do CFD

As faixas do CFD não permanecem paralelas.  
O aumento da largura das faixas intermediárias indica acúmulo de trabalho em progresso.

---

### 3. Gargalo provável

A etapa com maior crescimento é: **{metricas['gargalo']}**.

Isso sugere que essa etapa está recebendo mais trabalho do que consegue processar.

---

### 4. Decisão executiva recomendada

A decisão mais adequada é:

1. Reduzir ou congelar temporariamente a entrada de novas demandas;
2. Atuar na etapa gargalo;
3. Redefinir limites de WIP;
4. Rebalancear a equipe;
5. Priorizar a conclusão dos itens já iniciados.

---

### 5. Mensagem executiva

O CFD mostra que o problema não é apenas volume de trabalho, mas **instabilidade do fluxo**.  
O projeto perde previsibilidade quando o WIP cresce continuamente e as entregas não acompanham o ritmo de entrada ou movimentação das demandas.
""")

st.divider()
st.caption("Aplicação didática para ensino de Kanban, CFD, WIP, throughput, gargalos e saúde do fluxo.")
