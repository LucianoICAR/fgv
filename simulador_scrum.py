
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Simulador Scrum | Review e Retrospectiva",
    page_icon="🏉",
    layout="wide"
)

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

RETRO_4TIAS = {
    "Alegria": "O que deixou o time satisfeito nesta Sprint?",
    "Tristeza": "O que gerou frustração, atraso ou perda de qualidade?",
    "Medo": "Quais riscos ou preocupações permanecem para a próxima Sprint?",
    "Raiva": "O que incomodou o time e precisa ser enfrentado com transparência?"
}

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


def iniciar():
    st.session_state.setdefault("backlog", pd.DataFrame(BACKLOG_PADRAO))
    st.session_state.setdefault("sprint", 1)
    st.session_state.setdefault("historico", [])
    st.session_state.setdefault("capacidade", 18)


def resetar():
    st.session_state.clear()
    iniciar()


def executar_sprint(df, capacidade):
    temp = df.copy()
    temp["Valor por Ponto"] = temp["Valor"] / temp["Esforço"]
    temp = temp.sort_values(["Valor por Ponto", "Valor"], ascending=False)

    entregues = []
    esforco_usado = 0

    for _, row in temp.iterrows():
        if esforco_usado + row["Esforço"] <= capacidade:
            item = row.drop(labels=["Valor por Ponto"]).to_dict()
            entregues.append(item)
            esforco_usado += int(row["Esforço"])

    entregues_df = pd.DataFrame(entregues)
    ids_entregues = set(entregues_df["ID"].tolist()) if not entregues_df.empty else set()
    nao_entregues_df = df[~df["ID"].isin(ids_entregues)].copy()

    valor_entregue = int(entregues_df["Valor"].sum()) if not entregues_df.empty else 0
    esforco_planejado = int(df["Esforço"].sum()) if not df.empty else 0
    previsibilidade = round((esforco_usado / esforco_planejado) * 100, 1) if esforco_planejado > 0 else 0

    return entregues_df, nao_entregues_df, valor_entregue, esforco_usado, previsibilidade


iniciar()

st.title("🏉 Simulador Scrum — Review do Produto e Retrospectiva da Sprint")
st.caption("Exercício prático com Sprint Planning, execução, Sprint Review, Retrospectiva das 4 tias e plano de ação.")

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

with aba1:
    st.header("📘 Case da Simulação")

    st.markdown("""
Você faz parte de um time Scrum responsável por desenvolver uma plataforma digital para uma **FinTech regulada**.

A organização deseja entregar valor rapidamente, mas precisa manter atenção especial a:

- segurança;
- integração com APIs externas;
- rastreabilidade;
- auditoria;
- experiência do usuário;
- confiabilidade do incremento.

## Missão do time

1. Selecionar Histórias de Usuário para a Sprint.
2. Executar a Sprint dentro da capacidade disponível.
3. Apresentar o incremento na Revisão do Produto.
4. Analisar os problemas encontrados.
5. Realizar a Retrospectiva da Sprint usando a técnica das **4 tias**.
6. Escolher um problema e transformá-lo em plano de ação.
""")

    st.info("A simulação diferencia claramente Sprint Review, que olha o produto, e Retrospectiva, que olha o processo de trabalho do time.")

with aba2:
    st.header("📦 Product Backlog")
    st.dataframe(st.session_state.backlog, use_container_width=True, hide_index=True)

    st.markdown("""
### Critérios de leitura

- **Valor:** impacto de negócio.
- **Esforço:** tamanho estimado em pontos.
- **Risco:** incerteza técnica, regulatória ou operacional.
""")

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
    ids = [x.split(" — ")[0] for x in selecionadas]
    sprint_df = backlog[backlog["ID"].isin(ids)].copy()

    if not sprint_df.empty:
        esforco = int(sprint_df["Esforço"].sum())
        valor = int(sprint_df["Valor"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Esforço planejado", esforco)
        c2.metric("Valor planejado", valor)
        c3.metric("Capacidade", st.session_state.capacidade)

        st.dataframe(sprint_df, use_container_width=True, hide_index=True)

        if esforco > st.session_state.capacidade:
            st.error("O esforço planejado excede a capacidade da Sprint. O time poderá não entregar tudo.")
        else:
            st.success("O planejamento está dentro da capacidade.")

    objetivo = st.text_area(
        "Objetivo da Sprint",
        placeholder="Exemplo: entregar funcionalidades críticas de segurança, auditoria e integração bancária."
    )

    if st.button("✅ Confirmar Planejamento"):
        if sprint_df.empty:
            st.warning("Selecione pelo menos uma História.")
        elif not objetivo.strip():
            st.warning("Defina o Objetivo da Sprint.")
        else:
            st.session_state.sprint_df = sprint_df
            st.session_state.objetivo = objetivo
            for k in ["resultado", "review_feita", "retro_feita"]:
                st.session_state.pop(k, None)
            st.success("Planejamento confirmado. Vá para Execução.")

with aba4:
    st.header("🚀 Execução da Sprint")

    if "sprint_df" not in st.session_state:
        st.warning("Realize o Planejamento da Sprint antes.")
    else:
        st.write(f"**Objetivo da Sprint:** {st.session_state.objetivo}")
        st.dataframe(st.session_state.sprint_df, use_container_width=True, hide_index=True)

        if st.button("🚀 Executar Sprint"):
            ent, nao, valor, esforco, previs = executar_sprint(
                st.session_state.sprint_df,
                st.session_state.capacidade
            )

            st.session_state.resultado = {
                "entregues": ent,
                "nao_entregues": nao,
                "valor": valor,
                "esforco": esforco,
                "previsibilidade": previs
            }
            st.success("Sprint executada. Vá para Revisão do Produto.")

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

            st.subheader("Histórias não entregues")
            if r["nao_entregues"].empty:
                st.write("Todas as Histórias planejadas foram entregues.")
            else:
                st.dataframe(r["nao_entregues"], use_container_width=True, hide_index=True)

with aba5:
    st.header("🧪 Revisão do Produto — Sprint Review")

    if "resultado" not in st.session_state:
        st.warning("Execute a Sprint antes da Revisão do Produto.")
    else:
        st.markdown("""
A Sprint Review é o momento de **inspecionar o incremento do produto** com stakeholders.

O foco não é avaliar o comportamento do time, mas sim responder:

- O incremento gera valor?
- O produto atende às expectativas?
- O Product Backlog precisa ser adaptado?
- Quais problemas foram identificados na apresentação?
""")

        st.subheader("Incremento apresentado")
        if st.session_state.resultado["entregues"].empty:
            st.write("Nenhuma História foi entregue para apresentação.")
        else:
            st.dataframe(st.session_state.resultado["entregues"], use_container_width=True, hide_index=True)

        st.subheader("Problemas apontados pelos stakeholders")

        for i, problema in enumerate(PROBLEMAS_REVIEW, start=1):
            st.error(f"Problema {i}: {problema}")

        st.markdown("""
### Atividade do aluno

A equipe deve registrar a decisão de produto após a Review.
""")

        decisao_produto = st.text_area(
            "Qual decisão de produto o Product Owner deveria tomar após a Review?",
            placeholder="Exemplo: reordenar o backlog, criar histórias de correção, revisar critérios de aceite e priorizar estabilidade da integração."
        )

        adaptacao_backlog = st.text_area(
            "Quais adaptações devem ser feitas no Product Backlog?",
            placeholder="Exemplo: criar item de correção para a API externa, adicionar critérios de auditoria e revisar história do dashboard."
        )

        if st.button("✅ Registrar Sprint Review"):
            st.session_state.review_feita = {
                "decisao_produto": decisao_produto,
                "adaptacao_backlog": adaptacao_backlog
            }
            st.success("Sprint Review registrada. Vá para Retrospectiva.")

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

        st.subheader("Retrospectiva das 4 tias")

        col1, col2 = st.columns(2)
        with col1:
            alegria = st.text_area(f"😊 Alegria — {RETRO_4TIAS['Alegria']}")
            tristeza = st.text_area(f"😢 Tristeza — {RETRO_4TIAS['Tristeza']}")
        with col2:
            medo = st.text_area(f"😨 Medo — {RETRO_4TIAS['Medo']}")
            raiva = st.text_area(f"😠 Raiva — {RETRO_4TIAS['Raiva']}")

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

        st.markdown("""
### Atividade do aluno

A equipe deve propor um plano de ação próprio ou ajustar o plano sugerido.
""")

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

            for k in ["sprint_df", "resultado", "objetivo", "review_feita"]:
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
                "historico_simulacao_scrum_review_retro.csv",
                "text/csv"
            )

st.divider()
st.caption("Simulador didático de Scrum com Sprint Review, Retrospectiva das 4 tias, Ishikawa, 5 Porquês e 5W2H.")
