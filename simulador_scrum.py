
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Simulador Scrum | Histórias, Tarefas, Review e Retrospectiva",
    page_icon="🏉",
    layout="wide"
)

# =========================================================
# Dados
# =========================================================

BACKLOG_PADRAO = [
    {"ID": "US1", "História": "Login seguro com autenticação forte", "Valor": 100, "Esforço": 5, "Risco": "Médio"},
    {"ID": "US2", "História": "Cadastro digital de cliente", "Valor": 80, "Esforço": 3, "Risco": "Baixo"},
    {"ID": "US3", "História": "Integração com API bancária externa", "Valor": 120, "Esforço": 8, "Risco": "Alto"},
    {"ID": "US4", "História": "Dashboard executivo de transações", "Valor": 90, "Esforço": 5, "Risco": "Médio"},
    {"ID": "US5", "História": "Relatório regulatório automático", "Valor": 150, "Esforço": 13, "Risco": "Alto"},
    {"ID": "US6", "História": "Trilha de auditoria das operações", "Valor": 130, "Esforço": 8, "Risco": "Alto"},
    {"ID": "US7", "História": "Notificações para clientes", "Valor": 60, "Esforço": 3, "Risco": "Baixo"},
    {"ID": "US8", "História": "Tela de contestação de transações", "Valor": 110, "Esforço": 8, "Risco": "Médio"},
    {"ID": "US9", "História": "Consulta de limites e taxas", "Valor": 70, "Esforço": 3, "Risco": "Baixo"},
    {"ID": "US10", "História": "Mecanismo antifraude simplificado", "Valor": 140, "Esforço": 13, "Risco": "Alto"},
]

PROBLEMAS_REVIEW = [
    "O Relatório Regulatório foi apresentado, mas não exibiu trilha de auditoria suficiente para aprovação da área de Compliance.",
    "O Dashboard Executivo apresentou dados inconsistentes entre transações aprovadas e transações contestadas.",
    "A integração com a API bancária externa funcionou no ambiente de teste, mas apresentou lentidão e falhas intermitentes na demonstração."
]

PROBLEMA_ESCOLHIDO = "A integração com a API bancária externa apresentou lentidão e falhas intermitentes na demonstração do incremento."

ISHIKAWA = pd.DataFrame({
    "6M": ["Método", "Máquina", "Mão de obra", "Material", "Medição", "Meio ambiente"],
    "Causas potenciais": [
        "Ausência de protocolo claro para testes de integração ponta a ponta antes da Sprint Review.",
        "Ambiente de homologação instável e sem monitoramento adequado de latência.",
        "Time com pouca familiaridade com a API externa e suas limitações operacionais.",
        "Documentação da API incompleta, desatualizada ou com exemplos insuficientes.",
        "Não havia métrica objetiva de tempo de resposta aceitável para considerar a integração pronta.",
        "Dependência de fornecedor externo sem acordo claro de disponibilidade para janelas de teste."
    ]
})

CINCO_PORQUES = pd.DataFrame({
    "Nível": ["Problema", "Por quê 1?", "Por quê 2?", "Por quê 3?", "Por quê 4?", "Por quê 5? / Causa-raiz"],
    "Resposta": [
        "A API bancária externa apresentou lentidão e falhas intermitentes durante a apresentação do incremento.",
        "Porque a integração não foi validada sob condições próximas ao uso real.",
        "Porque o time testou apenas cenários funcionais simples e não realizou teste de estabilidade.",
        "Porque a Definition of Done não exigia validação de desempenho e disponibilidade da integração.",
        "Porque o time tratou a integração como uma tarefa técnica interna, e não como um risco crítico de produto.",
        "Causa-raiz: ausência de critério explícito de pronto para integrações externas críticas, incluindo desempenho, estabilidade e validação em ambiente representativo."
    ]
})

PLANO_ACAO = pd.DataFrame({
    "5W2H": ["What/O quê", "Why/Por quê", "Where/Onde", "When/Quando", "Who/Quem", "How/Como", "How much/Quanto"],
    "Plano": [
        "Criar uma política de Definition of Done específica para integrações externas críticas.",
        "Evitar que integrações sejam apresentadas como prontas sem validação mínima de estabilidade, desempenho e disponibilidade.",
        "No fluxo de desenvolvimento, testes integrados, homologação e Sprint Review.",
        "Antes do fechamento da próxima Sprint e aplicado em todas as próximas integrações.",
        "Scrum Master facilita; Time de Desenvolvimento define critérios técnicos; Product Owner valida impacto de negócio.",
        "Adicionar critérios objetivos: teste de latência, teste de falha, evidência de logs, simulação de indisponibilidade e aprovação em ambiente de homologação.",
        "Baixo custo financeiro direto; custo principal é a reserva de capacidade técnica de 1 a 2 dias por Sprint para validação e automação."
    ]
})


# =========================================================
# Funções
# =========================================================

def iniciar():
    st.session_state.setdefault("backlog", pd.DataFrame(BACKLOG_PADRAO))
    st.session_state.setdefault("sprint", 1)
    st.session_state.setdefault("historico", [])
    st.session_state.setdefault("capacidade", 18)


def resetar():
    st.session_state.clear()
    iniciar()


def montar_tarefas(sprint_df):
    linhas = []

    for _, row in sprint_df.iterrows():
        us_id = row["ID"]
        historia = row["História"]

        tarefa_1 = st.session_state.get(f"tarefa_{us_id}_1", "").strip()
        tarefa_2 = st.session_state.get(f"tarefa_{us_id}_2", "").strip()

        linhas.append({
            "ID História": us_id,
            "História": historia,
            "ID Tarefa": f"{us_id}-T1",
            "Tarefa": tarefa_1
        })

        linhas.append({
            "ID História": us_id,
            "História": historia,
            "ID Tarefa": f"{us_id}-T2",
            "Tarefa": tarefa_2
        })

    return pd.DataFrame(linhas)


def executar_sprint(sprint_df, tarefas_df, capacidade):
    temp = sprint_df.copy()
    temp["Valor por Ponto"] = temp["Valor"] / temp["Esforço"]
    temp = temp.sort_values(["Valor por Ponto", "Valor"], ascending=False)

    entregues = []
    esforco_usado = 0

    for _, row in temp.iterrows():
        if esforco_usado + row["Esforço"] <= capacidade:
            entregues.append(row.drop(labels=["Valor por Ponto"]).to_dict())
            esforco_usado += int(row["Esforço"])

    entregues_df = pd.DataFrame(entregues)

    ids_entregues = entregues_df["ID"].tolist() if not entregues_df.empty else []
    nao_entregues_df = sprint_df[~sprint_df["ID"].isin(ids_entregues)].copy()

    tarefas_entregues_df = tarefas_df[tarefas_df["ID História"].isin(ids_entregues)].copy()
    tarefas_nao_entregues_df = tarefas_df[~tarefas_df["ID História"].isin(ids_entregues)].copy()

    valor_entregue = int(entregues_df["Valor"].sum()) if not entregues_df.empty else 0
    esforco_planejado = int(sprint_df["Esforço"].sum()) if not sprint_df.empty else 0
    previsibilidade = round((esforco_usado / esforco_planejado) * 100, 1) if esforco_planejado > 0 else 0

    return {
        "entregues": entregues_df,
        "nao_entregues": nao_entregues_df,
        "tarefas_entregues": tarefas_entregues_df,
        "tarefas_nao_entregues": tarefas_nao_entregues_df,
        "valor": valor_entregue,
        "esforco": esforco_usado,
        "previsibilidade": previsibilidade
    }


# =========================================================
# Aplicação
# =========================================================

iniciar()

st.title("🏉 Simulador Scrum — Histórias, Tarefas, Review e Retrospectiva")
st.caption("Sprint Planning com decomposição de Histórias em tarefas, Execução, Sprint Review e Retrospectiva das 4 tias.")

with st.sidebar:
    st.header("⚙️ Configuração")
    st.session_state.capacidade = st.slider("Capacidade da Sprint", 5, 40, st.session_state.capacidade)
    st.write(f"**Sprint atual:** {st.session_state.sprint}")

    if st.button("🔄 Reiniciar simulação"):
        resetar()
        st.rerun()

aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
    "📘 Case",
    "📦 Backlog",
    "🧭 Planejamento",
    "🚀 Execução",
    "🧪 Revisão do Produto",
    "🔁 Retrospectiva"
])

# =========================================================
# Aba 1
# =========================================================

with aba1:
    st.header("📘 Case da Simulação")

    st.markdown("""
Você faz parte de um time Scrum responsável por desenvolver uma plataforma digital para uma **FinTech regulada**.

## Missão do time

1. Selecionar Histórias de Usuário para a Sprint.
2. Transformar cada História selecionada em **duas tarefas**.
3. Executar a Sprint dentro da capacidade disponível.
4. Apresentar o incremento na Sprint Review.
5. Realizar a Retrospectiva da Sprint usando a técnica das **4 tias**.
6. Escolher um problema e transformá-lo em plano de ação.
""")

    st.info("Nesta versão, o Planejamento exige obrigatoriamente 2 tarefas para cada História selecionada.")

# =========================================================
# Aba 2
# =========================================================

with aba2:
    st.header("📦 Product Backlog")
    st.dataframe(st.session_state.backlog, use_container_width=True, hide_index=True)

# =========================================================
# Aba 3
# =========================================================

with aba3:
    st.header("🧭 Sprint Planning")

    backlog = st.session_state.backlog.copy()
    st.write(f"**Capacidade disponível da Sprint:** {st.session_state.capacidade} pontos")

    opcoes = (
        backlog["ID"] + " — " + backlog["História"] +
        " | Valor: " + backlog["Valor"].astype(str) +
        " | Esforço: " + backlog["Esforço"].astype(str) +
        " | Risco: " + backlog["Risco"]
    ).tolist()

    selecionadas = st.multiselect("Selecione as Histórias para a Sprint", opcoes)

    ids = [item.split(" — ")[0] for item in selecionadas]
    sprint_df = backlog[backlog["ID"].isin(ids)].copy()

    if sprint_df.empty:
        st.warning("Selecione pelo menos uma História para montar o Sprint Backlog.")
    else:
        esforco_total = int(sprint_df["Esforço"].sum())
        valor_total = int(sprint_df["Valor"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Esforço planejado", esforco_total)
        c2.metric("Valor planejado", valor_total)
        c3.metric("Capacidade", st.session_state.capacidade)

        if esforco_total > st.session_state.capacidade:
            st.error("O esforço planejado excede a capacidade. Nem todas as Histórias poderão ser entregues.")
        else:
            st.success("O planejamento está dentro da capacidade.")

        st.subheader("Histórias selecionadas")
        st.dataframe(sprint_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Decomposição obrigatória: 2 tarefas por História")

        for _, row in sprint_df.iterrows():
            us_id = row["ID"]
            historia = row["História"]

            st.markdown(f"### {us_id} — {historia}")

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.text_input(
                    f"Tarefa 1 de {us_id}",
                    key=f"tarefa_{us_id}_1",
                    placeholder="Ex.: Implementar componente, regra ou integração"
                )

            with col_t2:
                st.text_input(
                    f"Tarefa 2 de {us_id}",
                    key=f"tarefa_{us_id}_2",
                    placeholder="Ex.: Criar teste, validação ou evidência de aceite"
                )

        tarefas_previa = montar_tarefas(sprint_df)

        with st.expander("Pré-visualizar tarefas criadas"):
            st.dataframe(tarefas_previa, use_container_width=True, hide_index=True)

    objetivo = st.text_area(
        "Objetivo da Sprint",
        placeholder="Ex.: Entregar funcionalidades críticas de segurança, auditoria e integração bancária."
    )

    if st.button("✅ Confirmar Planejamento"):
        if sprint_df.empty:
            st.warning("Selecione pelo menos uma História.")
        elif not objetivo.strip():
            st.warning("Defina o Objetivo da Sprint.")
        else:
            tarefas_df = montar_tarefas(sprint_df)

            if tarefas_df["Tarefa"].eq("").any():
                st.warning("Preencha exatamente 2 tarefas para cada História selecionada.")
            else:
                st.session_state.sprint_df = sprint_df
                st.session_state.tarefas_df = tarefas_df
                st.session_state.objetivo = objetivo

                for k in ["resultado", "review_feita"]:
                    st.session_state.pop(k, None)

                st.success("Planejamento confirmado com Histórias e tarefas. Vá para a aba Execução.")

# =========================================================
# Aba 4
# =========================================================

with aba4:
    st.header("🚀 Execução da Sprint")

    if "sprint_df" not in st.session_state or "tarefas_df" not in st.session_state:
        st.warning("Realize e confirme o Planejamento da Sprint antes da Execução.")
    else:
        st.write(f"**Objetivo da Sprint:** {st.session_state.objetivo}")

        st.subheader("Sprint Backlog — Histórias")
        st.dataframe(st.session_state.sprint_df, use_container_width=True, hide_index=True)

        st.subheader("Sprint Backlog — Tarefas associadas às Histórias")
        st.dataframe(st.session_state.tarefas_df, use_container_width=True, hide_index=True)

        with st.expander("Ver tarefas agrupadas por História"):
            for us_id in st.session_state.sprint_df["ID"].tolist():
                historia = st.session_state.sprint_df.loc[
                    st.session_state.sprint_df["ID"] == us_id, "História"
                ].iloc[0]

                st.markdown(f"### {us_id} — {historia}")

                tarefas_us = st.session_state.tarefas_df[
                    st.session_state.tarefas_df["ID História"] == us_id
                ]

                for _, tarefa in tarefas_us.iterrows():
                    st.write(f"- **{tarefa['ID Tarefa']}**: {tarefa['Tarefa']}")

        if st.button("🚀 Executar Sprint"):
            st.session_state.resultado = executar_sprint(
                st.session_state.sprint_df,
                st.session_state.tarefas_df,
                st.session_state.capacidade
            )

            st.success("Sprint executada. Confira o resultado abaixo e depois vá para a Revisão do Produto.")

        if "resultado" in st.session_state:
            r = st.session_state.resultado

            c1, c2, c3 = st.columns(3)
            c1.metric("Valor entregue", r["valor"])
            c2.metric("Esforço entregue", r["esforco"])
            c3.metric("Previsibilidade", f'{r["previsibilidade"]}%')

            st.subheader("Histórias entregues")
            if r["entregues"].empty:
                st.write("Nenhuma História entregue.")
            else:
                st.dataframe(r["entregues"], use_container_width=True, hide_index=True)

            st.subheader("Tarefas entregues")
            if r["tarefas_entregues"].empty:
                st.write("Nenhuma tarefa entregue.")
            else:
                st.dataframe(r["tarefas_entregues"], use_container_width=True, hide_index=True)

            st.subheader("Histórias não entregues")
            if r["nao_entregues"].empty:
                st.write("Todas as Histórias planejadas foram entregues.")
            else:
                st.dataframe(r["nao_entregues"], use_container_width=True, hide_index=True)

            st.subheader("Tarefas não entregues")
            if r["tarefas_nao_entregues"].empty:
                st.write("Todas as tarefas planejadas foram entregues.")
            else:
                st.dataframe(r["tarefas_nao_entregues"], use_container_width=True, hide_index=True)

# =========================================================
# Aba 5
# =========================================================

with aba5:
    st.header("🧪 Revisão do Produto — Sprint Review")

    if "resultado" not in st.session_state:
        st.warning("Execute a Sprint antes da Revisão do Produto.")
    else:
        st.markdown("""
A Sprint Review serve para inspecionar o **incremento do produto** com stakeholders.

O foco é o produto: valor entregue, aderência às expectativas e adaptações necessárias no Product Backlog.
""")

        st.subheader("Incremento apresentado — Histórias entregues")
        if st.session_state.resultado["entregues"].empty:
            st.write("Nenhuma História entregue.")
        else:
            st.dataframe(st.session_state.resultado["entregues"], use_container_width=True, hide_index=True)

        st.subheader("Tarefas concluídas associadas ao incremento")
        if st.session_state.resultado["tarefas_entregues"].empty:
            st.write("Nenhuma tarefa concluída.")
        else:
            st.dataframe(st.session_state.resultado["tarefas_entregues"], use_container_width=True, hide_index=True)

        st.subheader("Três problemas apontados na apresentação do incremento")
        for i, problema in enumerate(PROBLEMAS_REVIEW, start=1):
            st.error(f"Problema {i}: {problema}")

        decisao_produto = st.text_area(
            "Decisão de produto após a Review",
            placeholder="Ex.: reordenar o backlog, criar histórias de correção, ajustar critérios de aceite."
        )

        adaptacao_backlog = st.text_area(
            "Adaptações necessárias no Product Backlog",
            placeholder="Ex.: criar item para estabilizar API externa e revisar requisitos do relatório regulatório."
        )

        if st.button("✅ Registrar Sprint Review"):
            st.session_state.review_feita = {
                "decisao_produto": decisao_produto,
                "adaptacao_backlog": adaptacao_backlog
            }
            st.success("Sprint Review registrada. Vá para a Retrospectiva.")

# =========================================================
# Aba 6
# =========================================================

with aba6:
    st.header("🔁 Retrospectiva da Sprint")

    if "resultado" not in st.session_state:
        st.warning("Execute a Sprint antes da Retrospectiva.")
    else:
        st.markdown("""
A Retrospectiva olha para o **processo de trabalho do time**.

Nesta simulação, será usada a **Retrospectiva das 4 tias**:

- **Alegria:** o que funcionou bem?
- **Tristeza:** o que causou frustração?
- **Medo:** o que preocupa o time?
- **Raiva:** o que incomodou e precisa ser enfrentado?
""")

        col1, col2 = st.columns(2)

        with col1:
            alegria = st.text_area("😊 Alegria — O que deixou o time satisfeito nesta Sprint?")
            tristeza = st.text_area("😢 Tristeza — O que gerou frustração, atraso ou perda de qualidade?")

        with col2:
            medo = st.text_area("😨 Medo — Quais riscos ou preocupações permanecem para a próxima Sprint?")
            raiva = st.text_area("😠 Raiva — O que incomodou o time e precisa ser enfrentado?")

        st.markdown("---")
        st.header("🛠️ Transformando problema em plano de ação")

        st.subheader("Problema escolhido")
        st.error(PROBLEMA_ESCOLHIDO)

        st.subheader("1. Diagrama de Causa-e-Efeito — 6M")
        st.dataframe(ISHIKAWA, use_container_width=True, hide_index=True)

        st.subheader("2. Cinco Porquês")
        st.dataframe(CINCO_PORQUES, use_container_width=True, hide_index=True)

        st.subheader("3. Plano de Ação — 5W2H")
        st.dataframe(PLANO_ACAO, use_container_width=True, hide_index=True)

        plano_aluno = st.text_area(
            "Plano de ação proposto pela equipe",
            placeholder="Descreva as ações que a equipe adotaria para evitar recorrência do problema."
        )

        if st.button("📌 Encerrar Sprint"):
            r = st.session_state.resultado

            registro = {
                "Sprint": st.session_state.sprint,
                "Objetivo": st.session_state.objetivo,
                "Valor Entregue": r["valor"],
                "Esforço Entregue": r["esforco"],
                "Previsibilidade": r["previsibilidade"],
                "Alegria": alegria,
                "Tristeza": tristeza,
                "Medo": medo,
                "Raiva": raiva,
                "Plano de Ação": plano_aluno
            }

            st.session_state.historico.append(registro)

            ids_entregues = r["entregues"]["ID"].tolist() if not r["entregues"].empty else []
            st.session_state.backlog = st.session_state.backlog[
                ~st.session_state.backlog["ID"].isin(ids_entregues)
            ].copy()

            for k in ["sprint_df", "tarefas_df", "resultado", "objetivo", "review_feita"]:
                st.session_state.pop(k, None)

            st.session_state.sprint += 1
            st.success("Sprint encerrada. Você pode planejar a próxima Sprint.")

        if st.session_state.historico:
            st.markdown("---")
            st.subheader("Histórico das Sprints")
            hist = pd.DataFrame(st.session_state.historico)
            st.dataframe(hist, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Baixar histórico da simulação",
                hist.to_csv(index=False, encoding="utf-8-sig"),
                "historico_simulacao_scrum.csv",
                "text/csv"
            )

st.divider()
st.caption("Simulador didático de Scrum com Histórias, tarefas, Sprint Review, Retrospectiva das 4 tias, Ishikawa, 5 Porquês e 5W2H.")
