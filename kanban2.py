
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Simulador Kanban — Gargalos, WIP e Tempo de Ciclo")
st.caption("Exercicio pratico para identificar gargalos, controlar WIP e reduzir tempo de ciclo")

# =========================
# PARAMETROS
# =========================
st.sidebar.header("Configuracao do Sistema")

total_itens = st.sidebar.slider("Quantidade de demandas", 5, 60, 24)
meta_ciclo = st.sidebar.slider("Meta de tempo de ciclo (dias)", 1, 20, 5)

analise = st.sidebar.number_input("Analise (historias/dia)", 0.5, 10.0, 4.0)
dev = st.sidebar.number_input("Desenvolvimento (historias/dia)", 0.5, 10.0, 1.0)
teste = st.sidebar.number_input("Testes (historias/dia)", 0.5, 10.0, 1.0)

wip_analise = st.sidebar.slider("WIP Analise", 1, 20, 6)
wip_dev = st.sidebar.slider("WIP Desenvolvimento", 1, 20, 6)
wip_teste = st.sidebar.slider("WIP Testes", 1, 20, 6)

# =========================
# CASE
# =========================
st.header("Case")

st.markdown("""
A empresa esta enfrentando problemas no fluxo de desenvolvimento.

As demandas estao levando mais de 15 dias para serem entregues.

A diretoria definiu a meta de reduzir o tempo de ciclo para 5 dias.

O problema e que a equipe de analise esta muito rapida, gerando grande volume de entrada,
mas as etapas seguintes nao conseguem acompanhar.

Isso gera:

- acumulacao de demandas
- filas internas
- gargalo severo
- baixa previsibilidade
""")

df_case = pd.DataFrame({
    "Etapa": ["Analise", "Desenvolvimento", "Testes"],
    "Capacidade": ["4 historias/dia", "1 historia/dia", "1 historia/dia"],
    "Pessoas": [4, 5, 3],
    "Observacao": [
        "Equipe rapida, gera alto volume de entrada",
        "Alta complexidade tecnica",
        "Processo lento e rigoroso"
    ]
})

st.dataframe(df_case, use_container_width=True)

# =========================
# SIMULACAO
# =========================
st.header("Simulacao do Fluxo")

analise_q = 0
dev_q = 0
teste_q = 0
done = 0

historico = []

for dia in range(1, 30):
    entrada = analise

    espaco_analise = wip_analise - analise_q
    entrada_real = min(entrada, espaco_analise)
    analise_q += entrada_real

    move_dev = min(analise_q, dev, wip_dev - dev_q)
    analise_q -= move_dev
    dev_q += move_dev

    move_teste = min(dev_q, teste, wip_teste - teste_q)
    dev_q -= move_teste
    teste_q += move_teste

    concluido = min(teste_q, teste)
    teste_q -= concluido
    done += concluido

    historico.append({
        "Dia": dia,
        "Analise": analise_q,
        "Desenvolvimento": dev_q,
        "Testes": teste_q,
        "Concluido": done
    })

df = pd.DataFrame(historico)

st.line_chart(df.set_index("Dia"))

# =========================
# RESULTADOS
# =========================
st.header("Resultados")

throughput = min(dev, teste)
wip_total = analise_q + dev_q + teste_q

tempo_ciclo = wip_total / throughput if throughput > 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("Throughput", round(throughput,2))
col2.metric("WIP Total", round(wip_total,2))
col3.metric("Tempo de Ciclo", round(tempo_ciclo,2))

if tempo_ciclo > meta_ciclo:
    st.error("Tempo de ciclo acima da meta")
else:
    st.success("Tempo de ciclo dentro da meta")

# =========================
# INSIGHT
# =========================
st.header("Analise")

st.markdown(f"""
- Gargalo em Desenvolvimento/Testes  
- Entrada maior que saida  
- WIP cresce continuamente  
- Tempo de ciclo estimado: {round(tempo_ciclo,2)} dias  

O sistema esta desbalanceado
""")

st.header("Perguntas")

st.markdown("""
1. Qual e o gargalo do sistema?  
2. O WIP esta adequado?  
3. O que acontece se aumentar ainda mais a analise?  
4. Como reduzir o tempo de ciclo para 5 dias?  
""")
