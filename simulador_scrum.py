
import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Simulador Scrum", page_icon="🏉", layout="wide")

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

EVENTOS = [
    {"nome": "Mudança regulatória", "descricao": "O regulador exigiu evidências adicionais.", "efeito": "US5 e US6 recebem +3 pontos.", "tipo": "regulatorio"},
    {"nome": "Ausência de desenvolvedor", "descricao": "Um membro-chave ficou indisponível.", "efeito": "Capacidade -3 pontos.", "tipo": "capacidade3"},
    {"nome": "Bug crítico em produção", "descricao": "Correção urgente antes de continuar.", "efeito": "Capacidade -4 pontos.", "tipo": "capacidade4"},
    {"nome": "Cliente mudou prioridade", "descricao": "Contestação de transações virou prioridade.", "efeito": "US8 recebe +50 valor.", "tipo": "valor"},
    {"nome": "Dependência técnica bloqueada", "descricao": "API externa atrasou.", "efeito": "US3 recebe +4 pontos.", "tipo": "dependencia"},
    {"nome": "Automação de testes disponível", "descricao": "Baixo risco teve menos retrabalho.", "efeito": "Histórias de baixo risco recebem -1 ponto.", "tipo": "melhoria"},
]

def iniciar():
    st.session_state.setdefault("backlog", pd.DataFrame(BACKLOG_PADRAO))
    st.session_state.setdefault("sprint", 1)
    st.session_state.setdefault("historico", [])
    st.session_state.setdefault("capacidade_base", 18)

def resetar():
    st.session_state.clear()
    iniciar()

def aplicar_evento(df, capacidade, evento):
    df = df.copy()
    cap = capacidade
    if evento["tipo"] == "regulatorio":
        df.loc[df["ID"].isin(["US5", "US6"]), "Esforço"] += 3
    elif evento["tipo"] == "capacidade3":
        cap = max(1, capacidade - 3)
    elif evento["tipo"] == "capacidade4":
        cap = max(1, capacidade - 4)
    elif evento["tipo"] == "valor":
        df.loc[df["ID"] == "US8", "Valor"] += 50
    elif evento["tipo"] == "dependencia":
        df.loc[df["ID"] == "US3", "Esforço"] += 4
    elif evento["tipo"] == "melhoria":
        mask = df["Risco"] == "Baixo"
        df.loc[mask, "Esforço"] = df.loc[mask, "Esforço"].apply(lambda x: max(1, x - 1))
    return df, cap

def avaliar(df, capacidade):
    if df.empty:
        return df, df, 0, 0, 0
    temp = df.copy()
    temp["Valor por Ponto"] = temp["Valor"] / temp["Esforço"]
    temp = temp.sort_values(["Valor por Ponto", "Valor"], ascending=False)
    entregues = []
    esforco = 0
    for _, row in temp.iterrows():
        if esforco + row["Esforço"] <= capacidade:
            d = row.drop(labels=["Valor por Ponto"]).to_dict()
            entregues.append(d)
            esforco += int(row["Esforço"])
    entregues_df = pd.DataFrame(entregues)
    ids = set(entregues_df["ID"].tolist()) if not entregues_df.empty else set()
    nao = df[~df["ID"].isin(ids)].copy()
    valor = int(entregues_df["Valor"].sum()) if not entregues_df.empty else 0
    planejado = int(df["Esforço"].sum())
    previs = round((esforco / planejado) * 100, 1) if planejado else 0
    return entregues_df, nao, valor, esforco, previs

iniciar()

st.title("🏉 Simulador Scrum — Entregando Valor sob Pressão")
st.caption("Sprint Planning, evento surpresa, execução, Review e Retrospectiva.")

with st.sidebar:
    st.header("⚙️ Configuração")
    st.session_state.capacidade_base = st.slider("Capacidade base da Sprint", 5, 40, st.session_state.capacidade_base)
    st.write(f"**Sprint atual:** {st.session_state.sprint}")
    if st.button("🔄 Reiniciar simulação"):
        resetar()
        st.rerun()

aba1, aba2, aba3, aba4, aba5 = st.tabs(["📘 Case", "📦 Backlog", "🧭 Planejamento", "⚡ Execução", "📊 Review"])

with aba1:
    st.header("📘 Case")
    st.markdown("""
Você faz parte de um time Scrum em uma **FinTech regulada**.

A diretoria quer entregar valor rapidamente, mas o time precisa lidar com mudanças, dependências técnicas,
bugs inesperados, requisitos regulatórios e capacidade limitada.

**Missão:** planejar uma Sprint, lidar com eventos inesperados e entregar o maior valor possível sem ultrapassar a capacidade.
""")

with aba2:
    st.header("📦 Product Backlog")
    st.dataframe(st.session_state.backlog, use_container_width=True, hide_index=True)

with aba3:
    st.header("🧭 Sprint Planning")
    backlog = st.session_state.backlog.copy()
    st.write(f"**Capacidade base:** {st.session_state.capacidade_base} pontos")
    opcoes = (backlog["ID"] + " — " + backlog["História"] + " | Valor: " +
              backlog["Valor"].astype(str) + " | Esforço: " + backlog["Esforço"].astype(str)).tolist()
    selecionadas = st.multiselect("Selecione as histórias para a Sprint", opcoes)
    ids = [x.split(" — ")[0] for x in selecionadas]
    sprint_df = backlog[backlog["ID"].isin(ids)].copy()
    if not sprint_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Esforço planejado", int(sprint_df["Esforço"].sum()))
        c2.metric("Valor planejado", int(sprint_df["Valor"].sum()))
        c3.metric("Capacidade", st.session_state.capacidade_base)
        st.dataframe(sprint_df, use_container_width=True, hide_index=True)
        if int(sprint_df["Esforço"].sum()) > st.session_state.capacidade_base:
            st.error("O plano excede a capacidade.")
        else:
            st.success("O plano está dentro da capacidade.")
    objetivo = st.text_area("Objetivo da Sprint")
    if st.button("✅ Confirmar Sprint Planning"):
        if sprint_df.empty or not objetivo.strip():
            st.warning("Selecione histórias e defina o objetivo da Sprint.")
        else:
            st.session_state.sprint_df = sprint_df
            st.session_state.objetivo = objetivo
            st.session_state.evento = None
            st.success("Sprint planejada. Vá para Execução.")

with aba4:
    st.header("⚡ Execução")
    if "sprint_df" not in st.session_state:
        st.warning("Planeje a Sprint antes.")
    else:
        st.write(f"**Objetivo:** {st.session_state.objetivo}")
        st.dataframe(st.session_state.sprint_df, use_container_width=True, hide_index=True)
        if st.session_state.get("evento") is None:
            if st.button("🎲 Gerar evento surpresa"):
                evento = random.choice(EVENTOS)
                st.session_state.evento = evento
                novo, cap = aplicar_evento(st.session_state.sprint_df, st.session_state.capacidade_base, evento)
                st.session_state.sprint_df_ajustado = novo
                st.session_state.capacidade_ajustada = cap
                st.rerun()
        else:
            evento = st.session_state.evento
            st.error(f"Evento: {evento['nome']}")
            st.write(evento["descricao"])
            st.write(f"**Efeito:** {evento['efeito']}")
            c1, c2 = st.columns(2)
            c1.metric("Capacidade original", st.session_state.capacidade_base)
            c2.metric("Capacidade ajustada", st.session_state.capacidade_ajustada)
            st.dataframe(st.session_state.sprint_df_ajustado, use_container_width=True, hide_index=True)
            if st.button("🚀 Executar Sprint"):
                ent, nao, valor, esforco, previs = avaliar(st.session_state.sprint_df_ajustado, st.session_state.capacidade_ajustada)
                st.session_state.resultado = {"ent": ent, "nao": nao, "valor": valor, "esforco": esforco, "previs": previs}
                st.success("Sprint executada. Vá para Review.")

with aba5:
    st.header("📊 Review e Retrospectiva")
    if "resultado" not in st.session_state:
        st.warning("Execute a Sprint antes.")
    else:
        r = st.session_state.resultado
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor entregue", r["valor"])
        c2.metric("Esforço entregue", r["esforco"])
        c3.metric("Previsibilidade", f'{r["previs"]}%')
        st.subheader("Entregues")
        st.dataframe(r["ent"], use_container_width=True, hide_index=True) if not r["ent"].empty else st.write("Nenhuma.")
        st.subheader("Não entregues")
        st.dataframe(r["nao"], use_container_width=True, hide_index=True) if not r["nao"].empty else st.write("Todas foram entregues.")

        st.subheader("Perguntas de Retrospectiva")
        r1 = st.text_area("1. O Objetivo da Sprint foi preservado?")
        r2 = st.text_area("2. O planejamento estava adequado à capacidade real?")
        r3 = st.text_area("3. O PO priorizou valor, risco ou quantidade?")
        r4 = st.text_area("4. Que decisão aumentaria a previsibilidade?")
        r5 = st.text_area("5. Qual melhoria para a próxima Sprint?")

        if st.button("📌 Encerrar Sprint e avançar"):
            st.session_state.historico.append({
                "Sprint": st.session_state.sprint,
                "Objetivo": st.session_state.objetivo,
                "Evento": st.session_state.evento["nome"],
                "Valor Entregue": r["valor"],
                "Esforço Entregue": r["esforco"],
                "Previsibilidade": r["previs"],
                "Melhoria": r5
            })
            ids_ent = r["ent"]["ID"].tolist() if not r["ent"].empty else []
            st.session_state.backlog = st.session_state.backlog[~st.session_state.backlog["ID"].isin(ids_ent)].copy()
            for k in ["sprint_df", "sprint_df_ajustado", "resultado", "evento", "objetivo", "capacidade_ajustada"]:
                st.session_state.pop(k, None)
            st.session_state.sprint += 1
            st.success("Sprint encerrada.")

        if st.session_state.historico:
            st.subheader("Histórico")
            hist = pd.DataFrame(st.session_state.historico)
            st.dataframe(hist, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Baixar histórico", hist.to_csv(index=False, encoding="utf-8-sig"), "historico_simulacao_scrum.csv", "text/csv")
