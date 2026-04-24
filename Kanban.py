
import streamlit as st
import pandas as pd
import math

st.set_page_config(
    page_title="Simulador Kanban | Gargalos, WIP e Capacidade",
    page_icon="🟦",
    layout="wide"
)

st.title("🟦 Simulador Kanban: Gargalos, WIP, Capacidade e Tempo de Ciclo")
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
    return [{"id": i, "nome": f"História #{i}", "idade": 0} for i in range(1, qtd + 1)]


def capacidade_do_dia(capacidade, dia):
    """
    Permite simular capacidades fracionadas.
    Exemplo: capacidade 1.5 histórias/dia.
    Em 1 dia move 1 história; em 2 dias move 3 histórias no acumulado.
    """
    return math.floor(capacidade * dia) - math.floor(capacidade * (dia - 1))


def simular_fluxo(qtd_demandas, dias, cap_analise, cap_dev, cap_teste, wip_analise, wip_dev, wip_teste):
    """
    Simula o quadro Kanban considerando:
    1. Numeração única das Histórias.
    2. Limite de WIP por etapa.
    3. Capacidade diária de cada etapa.
    4. Sistema puxado: o fluxo anda da direita para a esquerda:
       Testes -> Concluído; Desenvolvimento -> Testes; Análise -> Desenvolvimento; Backlog -> Análise.
    """

    backlog = gerar_historias(qtd_demandas)
    analise = []
    dev = []
    teste = []
    concluido = []

    historico = []

    for dia in range(1, dias + 1):
        mov_teste = capacidade_do_dia(cap_teste, dia)
        mov_dev = capacidade_do_dia(cap_dev, dia)
        mov_analise = capacidade_do_dia(cap_analise, dia)

        eventos = []

        # 1. Testes -> Concluído
        qtd = min(mov_teste, len(teste))
        for _ in range(qtd):
            h = teste.pop(0)
            h["idade"] = dia
            concluido.append(h)
            eventos.append(f'{h["nome"]}: Testes → Concluído')

        # 2. Desenvolvimento -> Testes, respeitando WIP de Testes
        espaco_teste = max(0, wip_teste - len(teste))
        qtd = min(mov_dev, len(dev), espaco_teste)
        for _ in range(qtd):
            h = dev.pop(0)
            teste.append(h)
            eventos.append(f'{h["nome"]}: Desenvolvimento → Testes')

        # 3. Análise -> Desenvolvimento, respeitando WIP de Desenvolvimento
        espaco_dev = max(0, wip_dev - len(dev))
        qtd = min(mov_analise, len(analise), espaco_dev)
        for _ in range(qtd):
            h = analise.pop(0)
            dev.append(h)
            eventos.append(f'{h["nome"]}: Análise → Desenvolvimento')

        # 4. Backlog -> Análise, respeitando WIP de Análise
        espaco_analise = max(0, wip_analise - len(analise))
        qtd = min(mov_analise, len(backlog), espaco_analise)
        for _ in range(qtd):
            h = backlog.pop(0)
            analise.append(h)
            eventos.append(f'{h["nome"]}: Backlog → Análise')

        historico.append({
            "Dia": dia,
            "Backlog": len(backlog),
            "Análise": len(analise),
            "Desenvolvimento": len(dev),
            "Testes": len(teste),
            "Concluído": len(concluido),
            "Movimentos do dia": "; ".join(eventos) if eventos else "Sem movimentação"
        })

    return {
        "Backlog": backlog,
        "Análise": analise,
        "Desenvolvimento": dev,
        "Testes": teste,
        "Concluído": concluido,
        "Histórico": pd.DataFrame(historico)
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
        st.info(h["nome"])

    if len(historias) > 10:
        st.caption(f"+ {len(historias) - 10} histórias")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("⚙️ Configurações do Case")

demandas = st.sidebar.slider("Quantidade total de Histórias", 5, 60, 12)
dias_simulacao = st.sidebar.slider("Dias de simulação", 1, 30, 1)
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

# ---------------------------------------------------------
# Simulação
# ---------------------------------------------------------

simulacao = simular_fluxo(
    demandas,
    dias_simulacao,
    analise,
    dev,
    teste,
    wip_analise,
    wip_dev,
    wip_teste
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
O quadro agora considera **duas restrições simultâneas**:

1. **Limite de WIP**: quantidade máxima de Histórias que podem ficar em cada etapa.
2. **Capacidade diária**: quantidade máxima de Histórias que cada etapa consegue processar por dia.

As Histórias mantêm numeração única durante toda a simulação.
""")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mostrar_coluna("Backlog", simulacao["Backlog"])
    with col2:
        mostrar_coluna("Análise", simulacao["Análise"], wip_analise)
    with col3:
        mostrar_coluna("Desenvolvimento", simulacao["Desenvolvimento"], wip_dev)
    with col4:
        mostrar_coluna("Testes", simulacao["Testes"], wip_teste)
    with col5:
        mostrar_coluna("Concluído", simulacao["Concluído"])

    st.warning("Altere os dias de simulação, os limites de WIP e as capacidades para observar a evolução das Histórias no quadro.")

    with st.expander("Ver histórico diário da simulação"):
        st.dataframe(simulacao["Histórico"], use_container_width=True, hide_index=True)

    with st.expander("Ver lista completa das Histórias por coluna"):
        linhas = []
        for coluna in ["Backlog", "Análise", "Desenvolvimento", "Testes", "Concluído"]:
            for historia in simulacao[coluna]:
                linhas.append({"Coluna": coluna, "História": historia["nome"]})
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

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
    c3.metric("WIP total permitido", resultado["wip_total"])
    c4.metric("Tempo de ciclo estimado", f"{resultado['tempo_ciclo_estimado']:.1f} dias")

    st.subheader("Interpretação")
    st.markdown(f"""
- O sistema consegue entregar, no máximo, **{resultado['throughput']:.1f} história(s) por dia**.
- O gargalo atual é: **{resultado['gargalo']}**.
- Com WIP total permitido de **{resultado['wip_total']}**, o tempo de ciclo estimado pela Lei de Little é de **{resultado['tempo_ciclo_estimado']:.1f} dias**.
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

    st.subheader("Evolução acumulada")
    df_hist = simulacao["Histórico"].copy()
    st.line_chart(df_hist.set_index("Dia")[["Backlog", "Análise", "Desenvolvimento", "Testes", "Concluído"]])

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

        sim_cenario = simular_fluxo(
            demandas,
            dias_simulacao,
            cfg["analise"],
            cfg["dev"],
            cfg["teste"],
            cfg["wip_analise"],
            cfg["wip_dev"],
            cfg["wip_teste"]
        )

        linhas.append({
            "Cenário": nome,
            "Throughput teórico": r["throughput"],
            "Gargalo": r["gargalo"],
            "WIP Total": r["wip_total"],
            "Tempo de Ciclo Estimado": round(r["tempo_ciclo_estimado"], 1),
            "Histórias Concluídas na Simulação": len(sim_cenario["Concluído"]),
            "Status": status_tempo_ciclo(r["tempo_ciclo_estimado"], meta_ciclo)
        })

    df_cenarios = pd.DataFrame(linhas)
    st.dataframe(df_cenarios, use_container_width=True, hide_index=True)

    st.subheader("Gráfico comparativo — Tempo de Ciclo Estimado")
    st.bar_chart(df_cenarios.set_index("Cenário")["Tempo de Ciclo Estimado"])

    st.subheader("Gráfico comparativo — Histórias Concluídas na Simulação")
    st.bar_chart(df_cenarios.set_index("Cenário")["Histórias Concluídas na Simulação"])

    melhor = df_cenarios.sort_values(["Histórias Concluídas na Simulação", "Tempo de Ciclo Estimado"], ascending=[False, True]).iloc[0]
    st.success(f"Melhor cenário pela simulação: **{melhor['Cenário']}**.")

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

### 4. Simulação de capacidade
Executar a simulação alterando:
- dias de simulação;
- capacidade das etapas;
- limites de WIP.

Explicar como essas variáveis mudam a evolução das Histórias no quadro.

### 5. Políticas explícitas
Definir:
- regra para puxar uma nova demanda;
- critério de pronto da análise;
- critério de pronto do desenvolvimento;
- critério de pronto dos testes;
- regra para tratar demandas bloqueadas.

### 6. Decisão executiva
Escolher um cenário de melhoria e justificar:

- Cenário A: apenas aplicar WIP;
- Cenário B: realocar pessoas;
- Cenário C: aumentar capacidade do desenvolvimento;
- Cenário D: manter o sistema atual.

### 7. Conclusão
Explicar como a proposta reduz o tempo de ciclo e aumenta a previsibilidade.
""")

    st.download_button(
        label="⬇️ Baixar enunciado da atividade",
        data="""ATIVIDADE — KANBAN, GARGALOS, WIP, CAPACIDADE E HISTÓRIAS NUMERADAS

Contexto:
A empresa TechFlow Solutions está enfrentando atrasos no fluxo de desenvolvimento de software. O tempo de ciclo médio atual é de 12 dias e a meta é reduzi-lo para 3 dias.

Processo:
- Análise: 3 histórias/dia
- Desenvolvimento: 1 história/dia
- Testes: 2 histórias/dia

Regra importante:
Cada História de Usuário deve ter uma numeração única. Se o sistema possui 12 demandas, as Histórias devem ser numeradas de História #1 até História #12. Essa numeração deve permanecer a mesma quando a História avançar no quadro Kanban.

O quadro deve considerar simultaneamente:
1. Limites de WIP.
2. Capacidade diária das etapas.

Missão:
Construir o Quadro Kanban, identificar gargalos, definir limites de WIP, simular a capacidade diária e propor ajustes de fluxo.

Entregáveis:
1. Quadro Kanban proposto com Histórias numeradas
2. Diagnóstico do fluxo
3. Limites de WIP
4. Simulação de capacidade
5. Políticas explícitas
6. Decisão executiva
7. Conclusão
""",
        file_name="atividade_kanban_wip_capacidade_historias_numeradas.txt",
        mime="text/plain"
    )

st.divider()
st.caption("Modelo didático para ensino de Kanban, gargalos, WIP, capacidade, throughput, tempo de ciclo e rastreabilidade visual das Histórias.")
