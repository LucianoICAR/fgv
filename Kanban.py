
import streamlit as st
import pandas as pd
import math

st.set_page_config(
    page_title="Simulador Kanban | Gargalos e WIP",
    page_icon="🟦",
    layout="wide"
)

st.title("🟦 Simulador Kanban: Gargalos, WIP e Tempo de Ciclo")
st.caption("Atividade prática para alunos construírem um Quadro Kanban, identificarem gargalos e ajustarem limites de WIP.")

# ---------------------------------------------------------
# Funções
# ---------------------------------------------------------

def calc_system(analise, dev, teste, wip_analise, wip_dev, wip_teste, demandas, dias_simulacao):
    throughput = min(analise, dev, teste)
    gargalo = min(
        [("Análise", analise), ("Desenvolvimento", dev), ("Testes", teste)],
        key=lambda x: x[1]
    )[0]

    wip_total = wip_analise + wip_dev + wip_teste

    # Lei de Little: Cycle Time = WIP / Throughput
    tempo_ciclo_estimado = wip_total / throughput if throughput > 0 else 0
    dias_para_entregar = demandas / throughput if throughput > 0 else 0
    entregas_por_periodo = min(demandas, math.floor(throughput * dias_simulacao))

    return {
        "throughput": throughput,
        "gargalo": gargalo,
        "wip_total": wip_total,
        "tempo_ciclo_estimado": tempo_ciclo_estimado,
        "dias_para_entregar": dias_para_entregar,
        "entregas_por_periodo": entregas_por_periodo
    }


def status_tempo_ciclo(ct, meta):
    if ct <= meta:
        return "🟢 Dentro da meta"
    elif ct <= meta * 2:
        return "🟡 Atenção"
    else:
        return "🔴 Fora da meta"


def gerar_historias(qtd):
    return [f"História #{i}" for i in range(1, qtd + 1)]


def distribuir_historias(qtd_demandas, wip_analise, wip_dev, wip_teste, concluidas):
    """
    Distribui as histórias mantendo numeração única de 1 até N.

    A numeração nunca reinicia por coluna.
    Exemplo:
    - Concluído: História #1, #2
    - Testes: História #3, #4
    - Desenvolvimento: História #5
    - Análise: História #6, #7
    - Backlog: História #8 em diante
    """
    historias = gerar_historias(qtd_demandas)

    concluidas = min(concluidas, qtd_demandas)
    restantes = historias[concluidas:]

    col_concluido = historias[:concluidas]

    col_teste = restantes[:min(wip_teste, len(restantes))]
    restantes = restantes[len(col_teste):]

    col_dev = restantes[:min(wip_dev, len(restantes))]
    restantes = restantes[len(col_dev):]

    col_analise = restantes[:min(wip_analise, len(restantes))]
    restantes = restantes[len(col_analise):]

    col_backlog = restantes

    return {
        "Backlog": col_backlog,
        "Análise": col_analise,
        "Desenvolvimento": col_dev,
        "Testes": col_teste,
        "Concluído": col_concluido
    }


def mostrar_coluna(titulo, historias, wip=None):
    if wip is not None:
        st.subheader(titulo)
        st.caption(f"WIP: {wip} | Itens na coluna: {len(historias)}")
    else:
        st.subheader(titulo)
        st.caption(f"Itens na coluna: {len(historias)}")

    if len(historias) == 0:
        st.write("—")

    for h in historias[:10]:
        st.info(h)

    if len(historias) > 10:
        st.caption(f"+ {len(historias) - 10} histórias")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("⚙️ Configurações do Case")

demandas = st.sidebar.slider("Quantidade total de Histórias", 5, 60, 12)
dias_simulacao = st.sidebar.slider("Dias de simulação", 3, 30, 12)
meta_ciclo = st.sidebar.number_input("Meta de tempo de ciclo médio (dias)", min_value=1, max_value=30, value=3)

st.sidebar.divider()

st.sidebar.subheader("Capacidade por etapa")
analise = st.sidebar.number_input("Análise — histórias/dia", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
dev = st.sidebar.number_input("Desenvolvimento — histórias/dia", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
teste = st.sidebar.number_input("Testes — histórias/dia", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

st.sidebar.divider()

st.sidebar.subheader("Limites de WIP")
wip_analise = st.sidebar.slider("WIP Análise", 1, 15, 3)
wip_dev = st.sidebar.slider("WIP Desenvolvimento", 1, 15, 2)
wip_teste = st.sidebar.slider("WIP Testes", 1, 15, 2)

st.sidebar.divider()

st.sidebar.subheader("Evolução no Quadro")
concluidas = st.sidebar.slider(
    "Histórias já concluídas",
    0,
    demandas,
    0,
    help="Use este controle para simular a evolução do fluxo. A numeração das Histórias permanece fixa."
)

# ---------------------------------------------------------
# Abas
# ---------------------------------------------------------

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📘 Case",
    "🧱 Quadro Kanban",
    "📊 Diagnóstico",
    "🔁 Cenários",
    "📝 Atividade dos Alunos"
])

# ---------------------------------------------------------
# Aba 1
# ---------------------------------------------------------

with aba1:
    st.header("📘 Case: TechFlow Solutions")

    st.markdown("""
A empresa **TechFlow Solutions** está enfrentando problemas no fluxo de desenvolvimento de software.

Os clientes estão reclamando do **tempo de ciclo das demandas**, pois algumas histórias de usuário permanecem mais de **12 dias** no sistema.

A diretoria definiu uma meta: reduzir o **tempo médio de ciclo para 3 dias**.

A equipe utiliza um quadro visual, mas ainda não aplica corretamente:
- limites de WIP;
- sistema puxado;
- políticas explícitas;
- análise de gargalos.
""")

    st.subheader("Processo atual")
    df_case = pd.DataFrame({
        "Etapa": ["Análise", "Desenvolvimento", "Testes"],
        "Capacidade atual": ["3 histórias/dia", "1 história/dia", "2 histórias/dia"],
        "Pessoas": [3, 4, 3],
        "Observação": [
            "Uma pessoa também consegue desenvolver",
            "Pode dobrar a capacidade com mais duas pessoas",
            "Há ociosidade parcial; uma pessoa também pode ajudar em análise ou desenvolvimento"
        ]
    })
    st.dataframe(df_case, use_container_width=True, hide_index=True)

    st.info("Missão: construir o Quadro Kanban, identificar o gargalo, definir limites de WIP e propor ajustes para reduzir o tempo de ciclo.")

# ---------------------------------------------------------
# Aba 2
# ---------------------------------------------------------

with aba2:
    st.header("🧱 Quadro Kanban com Histórias Numeradas")

    st.markdown("""
Cada História possui uma **numeração única**, de acordo com a quantidade total de demandas.

Exemplo: se existem 12 demandas, as Histórias serão numeradas de **História #1** até **História #12**.

Essa numeração permanece a mesma mesmo quando a História sai do Backlog e avança para Análise, Desenvolvimento, Testes ou Concluído.
""")

    colunas = distribuir_historias(demandas, wip_analise, wip_dev, wip_teste, concluidas)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mostrar_coluna("Backlog", colunas["Backlog"])
    with col2:
        mostrar_coluna("Análise", colunas["Análise"], wip_analise)
    with col3:
        mostrar_coluna("Desenvolvimento", colunas["Desenvolvimento"], wip_dev)
    with col4:
        mostrar_coluna("Testes", colunas["Testes"], wip_teste)
    with col5:
        mostrar_coluna("Concluído", colunas["Concluído"])

    st.warning("Use o controle 'Histórias já concluídas' na barra lateral para simular a evolução do quadro mantendo a numeração fixa das Histórias.")

    with st.expander("Ver lista completa das Histórias por coluna"):
        linhas = []
        for coluna, historias in colunas.items():
            for historia in historias:
                linhas.append({"Coluna": coluna, "História": historia})
        if linhas:
            st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
        else:
            st.write("Não há histórias no quadro.")

# ---------------------------------------------------------
# Aba 3
# ---------------------------------------------------------

with aba3:
    st.header("📊 Diagnóstico do Sistema")

    resultado = calc_system(
        analise, dev, teste,
        wip_analise, wip_dev, wip_teste,
        demandas, dias_simulacao
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Throughput do sistema", f"{resultado['throughput']:.1f}/dia")
    c2.metric("Gargalo", resultado["gargalo"])
    c3.metric("WIP total", resultado["wip_total"])
    c4.metric("Tempo de ciclo estimado", f"{resultado['tempo_ciclo_estimado']:.1f} dias")

    st.subheader("Interpretação")
    st.markdown(f"""
- O sistema consegue entregar, no máximo, **{resultado['throughput']:.1f} história(s) por dia**.
- O gargalo atual é: **{resultado['gargalo']}**.
- Com o WIP total atual de **{resultado['wip_total']}**, o tempo de ciclo estimado é de **{resultado['tempo_ciclo_estimado']:.1f} dias**.
- Status em relação à meta de **{meta_ciclo} dias**: **{status_tempo_ciclo(resultado['tempo_ciclo_estimado'], meta_ciclo)}**.
""")

    if resultado["tempo_ciclo_estimado"] > meta_ciclo:
        st.error("O sistema está acima da meta. Reduza o WIP, aumente a capacidade do gargalo ou faça as duas coisas.")
    else:
        st.success("O sistema está dentro da meta de tempo de ciclo.")

    st.subheader("Capacidade por etapa")
    df_cap = pd.DataFrame({
        "Etapa": ["Análise", "Desenvolvimento", "Testes"],
        "Capacidade": [analise, dev, teste],
        "É gargalo?": [
            "Sim" if resultado["gargalo"] == "Análise" else "Não",
            "Sim" if resultado["gargalo"] == "Desenvolvimento" else "Não",
            "Sim" if resultado["gargalo"] == "Testes" else "Não"
        ]
    })
    st.bar_chart(df_cap.set_index("Etapa")["Capacidade"])
    st.dataframe(df_cap, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Aba 4
# ---------------------------------------------------------

with aba4:
    st.header("🔁 Comparação de Cenários")

    cenarios = {
        "A — Apenas aplicar WIP": {
            "analise": 3.0,
            "dev": 1.0,
            "teste": 2.0,
            "wip_analise": 2,
            "wip_dev": 1,
            "wip_teste": 2
        },
        "B — Realocar 1 pessoa para Desenvolvimento": {
            "analise": 3.0,
            "dev": 1.5,
            "teste": 1.5,
            "wip_analise": 2,
            "wip_dev": 2,
            "wip_teste": 2
        },
        "C — Aumentar capacidade do Desenvolvimento": {
            "analise": 3.0,
            "dev": 2.0,
            "teste": 2.0,
            "wip_analise": 2,
            "wip_dev": 2,
            "wip_teste": 2
        },
        "D — Sistema desbalanceado, sem limite adequado": {
            "analise": 3.0,
            "dev": 1.0,
            "teste": 2.0,
            "wip_analise": 6,
            "wip_dev": 6,
            "wip_teste": 6
        }
    }

    linhas = []
    for nome, cfg in cenarios.items():
        r = calc_system(
            cfg["analise"], cfg["dev"], cfg["teste"],
            cfg["wip_analise"], cfg["wip_dev"], cfg["wip_teste"],
            demandas, dias_simulacao
        )
        linhas.append({
            "Cenário": nome,
            "Throughput": r["throughput"],
            "Gargalo": r["gargalo"],
            "WIP Total": r["wip_total"],
            "Tempo de Ciclo Estimado": round(r["tempo_ciclo_estimado"], 1),
            "Status": status_tempo_ciclo(r["tempo_ciclo_estimado"], meta_ciclo)
        })

    df_cenarios = pd.DataFrame(linhas)
    st.dataframe(df_cenarios, use_container_width=True, hide_index=True)

    st.subheader("Gráfico comparativo")
    st.bar_chart(df_cenarios.set_index("Cenário")["Tempo de Ciclo Estimado"])

    melhor = df_cenarios.sort_values("Tempo de Ciclo Estimado").iloc[0]
    st.success(f"Melhor cenário pelo menor tempo de ciclo estimado: **{melhor['Cenário']}**.")

# ---------------------------------------------------------
# Aba 5
# ---------------------------------------------------------

with aba5:
    st.header("📝 Atividade dos Alunos")

    st.markdown("""
## Entregáveis da equipe

Cada equipe deve entregar:

### 1. Quadro Kanban proposto
Desenhar o quadro com as colunas:
- Backlog
- Análise
- Desenvolvimento
- Testes
- Concluído

As Histórias devem ter **numeração única**.  
Exemplo: História #1, História #2, História #3, até a última História do case.

### 2. Diagnóstico do fluxo
Responder:
1. Qual é o gargalo do sistema?
2. Qual etapa gera fila?
3. Onde existe ociosidade?
4. Qual é o throughput máximo do sistema?

### 3. Limites de WIP
Definir os limites de WIP para:
- Análise
- Desenvolvimento
- Testes

A equipe deve justificar cada limite.

### 4. Políticas explícitas
Definir:
- regra para puxar uma nova demanda;
- critério de pronto da análise;
- critério de pronto do desenvolvimento;
- critério de pronto dos testes;
- regra para tratar demandas bloqueadas.

### 5. Decisão executiva
Escolher um cenário de melhoria e justificar:

- Cenário A: apenas aplicar WIP;
- Cenário B: realocar pessoas;
- Cenário C: aumentar capacidade do desenvolvimento;
- Cenário D: manter o sistema atual.

### 6. Conclusão
Explicar como a proposta reduz o tempo de ciclo e aumenta a previsibilidade.
""")

    st.download_button(
        label="⬇️ Baixar enunciado da atividade",
        data="""ATIVIDADE — KANBAN, GARGALOS, WIP E HISTÓRIAS NUMERADAS

Contexto:
A empresa TechFlow Solutions está enfrentando atrasos no fluxo de desenvolvimento de software. O tempo de ciclo médio atual é de 12 dias e a meta é reduzi-lo para 3 dias.

Processo:
- Análise: 3 histórias/dia
- Desenvolvimento: 1 história/dia
- Testes: 2 histórias/dia

Regra importante:
Cada História de Usuário deve ter uma numeração única. Se o sistema possui 12 demandas, as Histórias devem ser numeradas de História #1 até História #12. Essa numeração deve permanecer a mesma quando a História avançar no quadro Kanban.

Missão:
Construir o Quadro Kanban, identificar gargalos, definir limites de WIP e propor ajustes de fluxo.

Entregáveis:
1. Quadro Kanban proposto com Histórias numeradas
2. Diagnóstico do fluxo
3. Limites de WIP
4. Políticas explícitas
5. Decisão executiva
6. Conclusão
""",
        file_name="atividade_kanban_wip_historias_numeradas.txt",
        mime="text/plain"
    )

st.divider()
st.caption("Modelo didático para ensino de Kanban, gargalos, WIP, throughput, tempo de ciclo e rastreabilidade visual das Histórias.")
