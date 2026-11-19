"""AuditLab FGV — checklist de auditoria de requisitos.

Atividade didática de 25 minutos, baseada em um caso fictício.
PMBOK Guide — 8ª edição: Elicit and Analyze Requirements (2.2.2.2).
ISO 19011:2026: diretrizes de auditoria, sem reprodução da norma.

Execução: streamlit run app.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
from io import StringIO
import re
import unicodedata

import pandas as pd
import streamlit as st


PROJECT = {
    "name": "Hospital Inteligente Horizonte",
    "budget": "R$ 28 milhões",
    "duration": "18 meses",
    "requirements": 80,
    "sample": "10 registros (2 RN, 6 RF, 1 RT e 1 RP)",
    "audit_scope": "Processo de Elicitar e Analisar os Requisitos — registros da amostra",
}

PAGES = ["1 · Contexto e critérios", "2 · Checklist e evidências", "3 · Resultado"]
EMPTY_TOPIC = "— Selecione um foco —"
EMPTY_RESULT = "Não avaliado"
RESULTS = [EMPTY_RESULT, "Conforme", "Não conforme", "Evidência insuficiente"]

RULES = [
    {
        "id": "R01", "short": "Identificação única",
        "topic": "Identificação única",
        "rule": (
            "100% dos requisitos devem possuir identificador único no padrão "
            "RN-###, RF-###, RT-### ou RP-###, sem duplicidade."
        ),
        "doc": "D01", "allowed_docs": {"D01"}, "expected": "Conforme",
        "criterion_groups": [
            ("id", "identific"), ("unico", "duplic"), ("rn", "rf", "rt", "rp")
        ],
        "anchors": ("10", "amostra", "unico", "duplic", "padrao", "identificador"),
        "reason": (
            "Os 10 registros da amostra têm códigos distintos e seguem o padrão "
            "RN/RF/RT/RP. A conclusão de conformidade vale apenas para os registros examinados."
        ),
    },
    {
        "id": "R02", "short": "Rastreabilidade RN–RF",
        "topic": "Rastreabilidade RN–RF",
        "rule": "100% dos requisitos funcionais (RF) devem possuir vínculo documentado com um requisito de negócio (RN) na MRR.",
        "doc": "D02", "allowed_docs": {"D02"}, "expected": "Não conforme",
        "criterion_groups": [("100", "todos"), ("rf", "funcionais"), ("rn", "negocio", "rastre")],
        "anchors": ("rf-024", "rf024", "ausente", "sem rn", "vazio", "nao possui", "lacuna"),
        "reason": (
            "A MRR registra o RF-024 sem RN de origem. Isso contraria a regra de "
            "rastreabilidade de 100% dos RF da amostra."
        ),
    },
    {
        "id": "R03", "short": "Métricas dos RF",
        "topic": "Métricas dos RF",
        "rule": "Cada RF deve registrar Critério de Aceitação, Teste e Tolerância, nos termos do plano.",
        "doc": "D03", "allowed_docs": {"D03"}, "expected": "Não conforme",
        "criterion_groups": [("aceitacao",), ("teste",), ("tolerancia",)],
        "anchors": ("rf-014", "rf014", "ausente", "nao definido", "sem tolerancia", "vago"),
        "reason": (
            "O RF-014 usa o termo 'rapidamente', sem critério de aceitação mensurável, "
            "teste ou tolerância definidos."
        ),
    },
    {
        "id": "R04", "short": "Participação das áreas",
        "topic": "Participação das áreas",
        "rule": (
            "A elicitação deve ter entrevistas documentadas com representantes "
            "das cinco áreas críticas: Corpo Clínico, Enfermagem, TI, Gestão Hospitalar e Engenharia Clínica."
        ),
        "doc": "D04", "allowed_docs": {"D04"}, "expected": "Conforme",
        "criterion_groups": [("5", "cinco"), ("area", "grupo"), ("entrevista", "consulta", "particip")],
        "anchors": ("5", "cinco", "todas", "areas", "entrevista", "ata"),
        "reason": (
            "Existem registros de entrevista das cinco áreas críticas estipuladas "
            "pelo plano de requisitos."
        ),
    },
    {
        "id": "R05", "short": "Aprovação formal",
        "topic": "Aprovação formal",
        "rule": (
            "Todos os requisitos liberados para a linha de base devem possuir "
            "aprovação formal registrada pela Diretora de Projetos, até 05/11/2026. "
            "Manifestações verbais não substituem o registro formal."
        ),
        "doc": "D05", "allowed_docs": {"D05", "D06"}, "expected": "Não conforme",
        "criterion_groups": [("aprova",), ("formal", "registro", "assinatura"), ("diretora", "responsavel", "05/11")],
        "anchors": ("rf-024", "rf024", "pendente", "assinatura", "verbal", "sem aprovacao"),
        "reason": (
            "O RF-024 consta como liberado para desenvolvimento, mas não tem "
            "aprovação formal da Diretora. A autorização verbal relatada na D06 "
            "não substitui a assinatura exigida pelo plano."
        ),
    },
]

RULE_BY_ID = {r["id"]: r for r in RULES}
TOPIC_OPTIONS = [EMPTY_TOPIC] + [f'{r["id"]} · {r["short"]}' for r in RULES]

DOCS = [
    {
        "id": "D01", "name": "Cadastro de requisitos",
        "intro": (
            "Extrato **integral da amostra sorteada**, com 10 dos 80 requisitos "
            "do projeto. Os códigos listados representam registros distintos."
        ),
        "columns": ["ID", "Tipo", "Descrição resumida"],
        "rows": [
            ["RN-001", "RN", "Aumentar a segurança assistencial"],
            ["RN-002", "RN", "Diminuir o tempo de acesso a dados clínicos"],
            ["RF-011", "RF", "Consultar prontuário do paciente"],
            ["RF-012", "RF", "Emitir alertas clínicos"],
            ["RF-013", "RF", "Registrar medições de sinais vitais"],
            ["RF-014", "RF", "Apresentar informações clínicas em painel"],
            ["RF-015", "RF", "Disponibilizar painel ao corpo clínico"],
            ["RF-024", "RF", "Integrar resultados do laboratório"],
            ["RT-001", "RT", "Protocolo seguro de integração"],
            ["RP-001", "RP", "Validar requisitos em reunião formal"],
        ],
        "foot": "Não é possível extrapolar a conformidade dessa amostra para todos os 80 requisitos.",
    },
    {
        "id": "D02", "name": "Matriz de rastreabilidade (MRR)",
        "intro": "Trecho da MRR correspondente a **todos os seis RF da amostra auditada**.",
        "columns": ["RF", "RN de origem", "Situação da ligação"],
        "rows": [
            ["RF-011", "RN-002", "Registrada"],
            ["RF-012", "RN-001", "Registrada"],
            ["RF-013", "RN-001", "Registrada"],
            ["RF-014", "RN-002", "Registrada"],
            ["RF-015", "RN-002", "Registrada"],
            ["RF-024", "—", "Sem RN de origem informado"],
        ],
        "foot": "A coluna 'RN de origem' precisa ser preenchida conforme a regra R02.",
    },
    {
        "id": "D03", "name": "Fichas de qualidade dos RF",
        "intro": "Campos de métricas dos **seis RF examinados**.",
        "columns": ["RF", "Critério de Aceitação", "Teste", "Tolerância"],
        "rows": [
            ["RF-011", "95% das consultas em até 2 s", "100 consultas cronometradas", "Até 5% acima de 2 s"],
            ["RF-012", "Alerta em até 5 s", "Simular 20 alertas", "Máximo de 1 alerta acima de 5 s"],
            ["RF-013", "100% das medições registradas", "Registrar 30 medições", "Nenhuma omissão"],
            ["RF-014", "Mostrar informações rapidamente", "Não definido", "Não definida"],
            ["RF-015", "Disponibilidade mensal ≥ 99,9%", "Monitoramento mensal", "Até 0,1% de indisponibilidade"],
            ["RF-024", "Resultado do laboratório em até 60 s", "30 integrações simuladas", "Até 1 falha em 30"],
        ],
        "foot": "O critério exige simultaneamente os três campos: aceitação, teste e tolerância.",
    },
    {
        "id": "D04", "name": "Registros de elicitação",
        "intro": "Atas de entrevistas realizadas em outubro de 2026.",
        "columns": ["Área crítica", "Data", "Registro de entrevista"],
        "rows": [
            ["Corpo Clínico", "26/10/2026", "ATA-01 — registrada"],
            ["Enfermagem", "26/10/2026", "ATA-02 — registrada"],
            ["TI", "27/10/2026", "ATA-03 — registrada"],
            ["Gestão Hospitalar", "27/10/2026", "ATA-04 — registrada"],
            ["Engenharia Clínica", "28/10/2026", "ATA-05 — registrada"],
        ],
        "foot": "As cinco áreas previstas em R04 têm atas documentadas.",
    },
    {
        "id": "D05", "name": "Registro de aprovações",
        "intro": "Situação da aprovação formal dos **dez requisitos da amostra**, em 06/11/2026.",
        "columns": ["Requisitos", "Situação", "Registro de aprovação"],
        "rows": [
            ["RN-001, RN-002", "Aprovados", "Assinatura da Diretora em 04/11/2026"],
            ["RF-011 a RF-015", "Aprovados", "Assinatura da Diretora em 04/11/2026"],
            ["RT-001, RP-001", "Aprovados", "Assinatura da Diretora em 05/11/2026"],
            ["RF-024", "Liberado para desenvolvimento", "Assinatura pendente"],
        ],
        "foot": "Todos os requisitos da amostra foram considerados liberados pela equipe do projeto.",
    },
    {
        "id": "D06", "name": "Ata de reunião de status",
        "intro": (
            "Trecho da reunião de 05/11/2026: 'A equipe entendeu que o RF-024 "
            "estava aprovado verbalmente para não atrasar o projeto. O registro "
            "formal ficará para a próxima reunião.'"
        ),
        "columns": ["Ponto", "Informação encontrada"],
        "rows": [
            ["Autorização verbal", "Gerente relatou concordância informal com RF-024"],
            ["Assinatura formal", "Não consta assinatura neste documento"],
            ["Providência", "Formalizar em reunião posterior"],
        ],
        "foot": "A ata contextualiza o processo, mas não substitui a aprovação formal exigida por R05.",
    },
]
DOC_BY_ID = {d["id"]: d for d in DOCS}
DOC_OPTIONS = [f'{d["id"]} · {d["name"]}' for d in DOCS]


def normalize(value: str) -> str:
    """Remove acentos e normaliza espaços e maiúsculas para verificação indicativa."""
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.casefold()).strip()


def rule_id_from_option(value: str) -> str | None:
    match = re.match(r"^(R\d{2})\s*·", str(value))
    return match.group(1) if match and match.group(1) in RULE_BY_ID else None


def doc_id_from_option(value: str) -> str | None:
    match = re.match(r"^(D\d{2})\s*·", str(value))
    return match.group(1) if match and match.group(1) in DOC_BY_ID else None


def empty_answer(index: int) -> dict:
    return {
        "number": index + 1, "rule_id": None, "question": "",
        "criterion": "", "docs": [], "result": EMPTY_RESULT, "note": "",
    }


def read_responses(state: dict) -> list[dict]:
    """Usa armazenamento permanente; widgets somem ao trocar de página."""
    return [dict(item, docs=list(item["docs"])) for item in state.get("audit_answers", [empty_answer(i) for i in range(5)])]


def hydrate_inputs() -> None:
    """Restaura campos quando o usuário retorna à tela do checklist."""
    answers = st.session_state["audit_answers"]
    for i, answer in enumerate(answers):
        values = {
            f"topic_{i}": next((v for v in TOPIC_OPTIONS if rule_id_from_option(v) == answer["rule_id"]), EMPTY_TOPIC),
            f"question_{i}": answer["question"],
            f"criterion_{i}": answer["criterion"],
            f"docs_{i}": [v for v in DOC_OPTIONS if doc_id_from_option(v) in answer["docs"]],
            f"result_{i}": answer["result"],
            f"note_{i}": answer["note"],
        }
        for key, value in values.items():
            if key not in st.session_state:
                st.session_state[key] = value


def sync_answer(index: int) -> None:
    """Copia valores de widget para chaves persistentes antes da limpeza do Streamlit."""
    i = index
    st.session_state["audit_answers"][i] = {
        "number": i + 1,
        "rule_id": rule_id_from_option(st.session_state.get(f"topic_{i}", EMPTY_TOPIC)),
        "question": str(st.session_state.get(f"question_{i}", "")).strip(),
        "criterion": str(st.session_state.get(f"criterion_{i}", "")).strip(),
        "docs": [doc_id_from_option(v) for v in st.session_state.get(f"docs_{i}", []) if doc_id_from_option(v)],
        "result": str(st.session_state.get(f"result_{i}", EMPTY_RESULT)),
        "note": str(st.session_state.get(f"note_{i}", "")).strip(),
    }
    st.session_state["submitted"] = False


def navigate(page: str) -> None:
    st.session_state["nav"] = page


def finish_audit() -> None:
    for i in range(5):
        sync_answer(i)
    st.session_state["submitted"] = True
    navigate(PAGES[2])


def score_row(row: dict, rule: dict) -> dict:
    """Pontuação didática transparente: 6 + 5 + 4 + 5 = 20 pontos por tema.

    Há validações indicativas de texto. Não substituem avaliação humana do conteúdo.
    """
    parts = {"Abrangência": 0, "Critério": 0, "Evidência": 0, "Resultado": 0}
    question = normalize(row["question"])
    criterion = normalize(row["criterion"])
    note = normalize(row["note"])
    docs = set(row["docs"])

    # Pergunta de auditoria: formulação verificável, não um campo em branco.
    if len(question) >= 10:
        parts["Abrangência"] = 3
        if "?" in row["question"] and len(question) >= 18:
            parts["Abrangência"] = 6

    # Critério: escrita mínima + três elementos-chave previstos no plano fictício.
    if len(criterion) >= 12:
        parts["Critério"] = 2
        for variants in rule["criterion_groups"]:
            if any(normalize(v) in criterion for v in variants):
                parts["Critério"] += 1

    # Evidência: documento principal e seleção enxuta de registros pertinentes.
    primary_found = rule["doc"] in docs
    if primary_found:
        parts["Evidência"] = 3
        if docs.issubset(rule["allowed_docs"]):
            parts["Evidência"] += 1

    # Resultado: a classificação correta só recebe nota máxima se estiver apoiada
    # na evidência indispensável. Nota é um comentário curto, não um relatório.
    if row["result"] == rule["expected"]:
        parts["Resultado"] = 3 if primary_found else 1
        if primary_found and len(note) >= 12 and any(normalize(v) in note for v in rule["anchors"]):
            parts["Resultado"] += 2
    elif row["result"] == "Evidência insuficiente" and not primary_found:
        parts["Resultado"] = 1

    return {"parts": parts, "score": sum(parts.values()), "primary_found": primary_found}


def evaluate_responses(rows: list[dict]) -> dict:
    """Só o melhor registro de cada foco pontua (evita duplicidade artificial)."""
    records = []
    scores = {"Abrangência": 0, "Critério": 0, "Evidência": 0, "Resultado": 0}
    for rule in RULES:
        candidates = [r for r in rows if r["rule_id"] == rule["id"]]
        chosen = None
        if candidates:
            chosen = max(candidates, key=lambda r: score_row(r, rule)["score"])
            scored = score_row(chosen, rule)
        else:
            scored = {"parts": {k: 0 for k in scores}, "score": 0, "primary_found": False}
        for key, value in scored["parts"].items():
            scores[key] += value
        records.append({
            "rule": rule, "item": chosen, "score": scored["score"],
            "parts": scored["parts"], "has_primary": scored["primary_found"],
        })
    return {
        "total": sum(scores.values()), "by_dimension": scores, "records": records,
        "covered_topics": sum(bool(r["item"]) for r in records),
    }


def printable_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Item de auditoria": row["question"],
            "Critério": row["criterion"],
            "Evidência examinada": ", ".join(row["docs"]) if row["docs"] else "—",
            "Resultado": row["result"],
        }
        for row in rows
    ]


def csv_report(team: str, rows: list[dict], assessment: dict) -> bytes:
    stream = StringIO()
    writer = csv.writer(stream, delimiter=";")
    writer.writerow(["AuditLab FGV — Checklist de Auditoria de Requisitos"])
    writer.writerow(["Equipe", team or "Não identificada"])
    writer.writerow(["Projeto", PROJECT["name"]])
    writer.writerow(["Escopo amostral", PROJECT["sample"]])
    writer.writerow(["Pontuação indicativa", f'{assessment["total"]}/100'])
    writer.writerow([])
    writer.writerow(["Foco", "Item de auditoria", "Critério", "Evidência examinada", "Resultado", "Fundamentação"])
    for row in rows:
        writer.writerow([
            row["rule_id"] or "—", row["question"], row["criterion"],
            ", ".join(row["docs"]), row["result"], row["note"],
        ])
    writer.writerow([])
    writer.writerow(["Nota", "As conclusões se restringem às evidências examinadas na simulação."])
    writer.writerow(["Nota", "A pontuação textual é heurística; professor deve revisar clareza e fundamentação."])
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def html_report(team: str, rows: list[dict], assessment: dict) -> bytes:
    """Relatório autocontido que pode ser aberto e impresso pelo navegador."""
    def h(value) -> str:
        return escape(str(value or "—"))

    trs = "".join(
        "<tr>" + "".join(f"<td>{h(v)}</td>" for v in [
            row["rule_id"], row["question"], row["criterion"],
            ", ".join(row["docs"]), row["result"], row["note"]
        ]) + "</tr>"
        for row in rows
    )
    body = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"/>
    <title>AuditLab FGV — Relatório</title>
    <style>body{{font-family:Arial,sans-serif;margin:35px;color:#18283d;}}
    h1{{color:#113765;font-size:25px}}table{{width:100%;border-collapse:collapse;font-size:12px;}}
    th,td{{border:1px solid #b8c5d4;padding:9px;text-align:left;vertical-align:top;}}
    th{{background:#eef3f8}}.small{{color:#5a6470;font-size:12px}}
    @media print{{body{{margin:12mm}}}}</style></head><body>
    <h1>AuditLab FGV — Checklist de auditoria</h1>
    <p><strong>Equipe:</strong> {h(team)} &nbsp; | &nbsp; <strong>Projeto:</strong> {h(PROJECT['name'])}</p>
    <p><strong>Escopo:</strong> {h(PROJECT['sample'])} &nbsp; | &nbsp; <strong>Nota indicativa:</strong> {assessment['total']}/100</p>
    <table><thead><tr><th>Foco</th><th>Item de auditoria</th><th>Critério</th>
    <th>Evidência examinada</th><th>Resultado</th><th>Fundamentação</th></tr></thead>
    <tbody>{trs}</tbody></table>
    <p class="small">Exercício fictício. Conclusões limitadas à amostra examinada.
    A pontuação automática não substitui a avaliação do professor.</p></body></html>"""
    return body.encode("utf-8")


def reset_activity() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("topic_", "question_", "criterion_", "docs_", "result_", "note_")):
            del st.session_state[key]
    st.session_state["audit_answers"] = [empty_answer(i) for i in range(5)]
    if "team" in st.session_state:
        del st.session_state["team"]
    st.session_state["submitted"] = False
    st.session_state["started_at"] = datetime.now(timezone.utc)
    st.session_state["nav"] = PAGES[0]


def show_doc(doc: dict) -> None:
    st.markdown(doc["intro"])
    df = pd.DataFrame(doc["rows"], columns=doc["columns"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(doc["foot"])


def show_context() -> None:
    st.subheader("Missão da equipe auditora")
    st.markdown(
        "Você foi designado para auditar **Elicitar e Analisar os Requisitos** "
        "de um projeto de hospital inteligente. O objetivo é verificar a aplicação "
        "das **regras do Plano de Gerenciamento dos Requisitos** a uma amostra de registros."
    )
    a, b, c, d = st.columns(4)
    a.metric("Orçamento do projeto", PROJECT["budget"])
    b.metric("Prazo", PROJECT["duration"])
    c.metric("Requisitos totais", str(PROJECT["requirements"]))
    d.metric("Amostra examinada", "10 registros")

    st.info(
        "**Critérios de auditoria:** neste exercício, as regras abaixo são exigências "
        "do projeto fictício. O PMBOK 8 orienta o processo; a ISO 19011:2026 orienta "
        "a condução da auditoria. Nenhuma das cinco regras deve ser interpretada "
        "como exigência universal do PMBOK ou da ISO."
    )
    st.markdown("#### Extrato do Plano de Gerenciamento dos Requisitos")
    for rule in RULES:
        st.markdown(f'**{rule["id"]} — {rule["short"]}:** {rule["rule"]}')

    st.markdown("#### Enunciado da atividade")
    st.markdown(
        "1. Construa **cinco perguntas** de auditoria, uma para cada foco do plano.  \n"
        "2. Para cada pergunta, redija o **critério** e indique a **evidência efetivamente examinada**.  \n"
        "3. Examine os **seis documentos** simulados disponíveis na próxima tela.  \n"
        "4. Classifique o resultado: *Conforme*, *Não conforme* ou *Evidência insuficiente*, "
        "com uma fundamentação curta.  \n"
        "5. Envie as respostas e discuta o feedback com o professor."
    )
    st.caption(
        "Duração sugerida: 5 min de leitura · 10 min de construção · "
        "7 min de exame das evidências · 3 min de feedback."
    )
    st.warning(
        "**Atenção à amostragem:** um registro conforme na amostra não comprova, "
        "por si só, que todos os 80 requisitos do projeto estejam conformes."
    )
    st.button("Ir para a elaboração do checklist", type="primary", use_container_width=True,
              on_click=navigate, args=(PAGES[1],))


def show_checklist() -> None:
    hydrate_inputs()
    st.subheader("Construir e aplicar o checklist")
    st.caption(
        "Selecione um foco para cada item. Redija sua pergunta e critério, examine "
        "os documentos e registre a conclusão. Os campos são livres para edição."
    )

    with st.expander("📁 Repositório de evidências — abrir os 6 documentos", expanded=False):
        tabs = st.tabs([f'{doc["id"]} · {doc["name"]}' for doc in DOCS])
        for tab, doc in zip(tabs, DOCS):
            with tab:
                show_doc(doc)
    st.caption("Dica: D06 é uma ata de status; verifique se ela atende ao critério de aprovação formal.")

    for i in range(5):
        title = f"Item {i+1}"
        with st.expander(title, expanded=(i == 0)):
            st.selectbox(
                "Foco de auditoria", TOPIC_OPTIONS, key=f"topic_{i}",
                on_change=sync_answer, args=(i,),
                help="A associação com o foco do plano permite avaliar cobertura e evitar itens duplicados.",
            )
            st.text_input(
                "Item de auditoria — formule uma pergunta verificável",
                key=f"question_{i}", placeholder="Ex.: Os requisitos examinados possuem ...?",
                on_change=sync_answer, args=(i,),
            )
            st.text_input(
                "Critério — qual regra objetiva deve ser atendida?",
                key=f"criterion_{i}", placeholder="Registre o padrão, a meta e/ou o responsável previsto no plano.",
                on_change=sync_answer, args=(i,),
            )
            left, right = st.columns([3, 2])
            with left:
                st.multiselect(
                    "Evidência efetivamente examinada", DOC_OPTIONS,
                    key=f"docs_{i}", on_change=sync_answer, args=(i,),
                    help="Marque apenas os documentos que a equipe de fato examinou.",
                )
            with right:
                st.selectbox(
                    "Resultado", RESULTS, key=f"result_{i}",
                    on_change=sync_answer, args=(i,),
                )
            st.text_input(
                "Fundamentação — uma frase curta indicando o que a evidência demonstrou",
                key=f"note_{i}", placeholder="Ex.: No documento D__, o RF-___ apresenta ...",
                on_change=sync_answer, args=(i,),
            )

    responses = read_responses(st.session_state)
    st.markdown("#### Prévia do checklist — 4 colunas")
    st.dataframe(
        pd.DataFrame(printable_rows(responses)), use_container_width=True,
        hide_index=True, column_config={
            "Item de auditoria": st.column_config.TextColumn(width="large"),
            "Critério": st.column_config.TextColumn(width="large"),
            "Evidência examinada": st.column_config.TextColumn(width="medium"),
            "Resultado": st.column_config.TextColumn(width="medium"),
        }
    )
    answered = sum(bool(r["rule_id"] and r["question"] and r["criterion"] and r["docs"] and r["result"] != EMPTY_RESULT) for r in responses)
    st.progress(answered / 5, text=f"{answered}/5 itens com os quatro campos essenciais preenchidos")
    st.caption("A fundamentação melhora a avaliação; registros incompletos também podem ser enviados para receber feedback.")
    st.button("Concluir auditoria e receber feedback", type="primary", use_container_width=True,
              on_click=finish_audit)


def show_results() -> None:
    st.subheader("Feedback da auditoria")
    if not st.session_state.get("submitted", False):
        st.info("O resultado permanece fechado até a equipe concluir e enviar o checklist na Etapa 2.")
        st.button("Voltar ao checklist", type="primary", on_click=navigate, args=(PAGES[1],))
        return

    responses = read_responses(st.session_state)
    assessment = evaluate_responses(responses)
    team = st.session_state.get("team", "").strip()
    a, b, c = st.columns(3)
    a.metric("Pontuação indicativa", f'{assessment["total"]}/100')
    b.metric("Focos contemplados", f'{assessment["covered_topics"]}/5')
    c.metric("Itens elaborados", f'{sum(bool(r["question"]) for r in responses)}/5')
    st.progress(assessment["total"] / 100)
    st.warning(
        "**Limite da correção automática:** a ferramenta verifica cobertura, "
        "termos essenciais, documentos selecionados e classificação dos achados. "
        "Ela não compreende plenamente textos livres; a clareza dos itens e a "
        "adequação das justificativas devem ser confirmadas pelo professor."
    )

    st.markdown("#### Pontuação por dimensão")
    max_scores = {"Abrangência": 30, "Critério": 25, "Evidência": 20, "Resultado": 25}
    dim_cols = st.columns(4)
    for col, (name, maximum) in zip(dim_cols, max_scores.items()):
        col.metric(name, f'{assessment["by_dimension"][name]}/{maximum}')

    st.markdown("#### Achados esperados no caso fictício")
    for rec in assessment["records"]:
        rule = rec["rule"]
        row = rec["item"]
        received = row["result"] if row else "Não abordado"
        with st.expander(f'{rule["id"]} · {rule["short"]} — {rec["score"]}/20 pts'):
            st.markdown(f'**Critério do case:** {rule["rule"]}')
            st.markdown(f'**Evidência principal:** {rule["doc"]} · {DOC_BY_ID[rule["doc"]]["name"]}')
            st.markdown(f'**Resultado esperado:** {rule["expected"]}')
            st.markdown(f'**Resultado da equipe:** {received}')
            st.markdown(f'**Justificativa técnica:** {rule["reason"]}')
            if row:
                if not rec["has_primary"]:
                    st.info("A equipe não indicou o documento principal que sustenta este achado.")
                if rec["parts"]["Critério"] < 5:
                    st.info("Revise a redação do critério: explicite os elementos verificáveis da regra indicada.")
                if rec["parts"]["Abrangência"] < 6:
                    st.info("Reformule o item como pergunta clara, específica e verificável.")
            else:
                st.info("O checklist não cobriu este foco do Plano de Gerenciamento dos Requisitos.")

    st.markdown("#### Checklist entregue")
    st.dataframe(pd.DataFrame(printable_rows(responses)), use_container_width=True, hide_index=True)
    x, y = st.columns(2)
    with x:
        st.download_button(
            "Baixar checklist (CSV para Excel)", data=csv_report(team, responses, assessment),
            file_name="AuditLab_FGV_Checklist.csv", mime="text/csv", use_container_width=True,
        )
    with y:
        st.download_button(
            "Baixar relatório para impressão (HTML)", data=html_report(team, responses, assessment),
            file_name="AuditLab_FGV_Relatorio.html", mime="text/html", use_container_width=True,
        )
    st.caption("Para imprimir ou salvar em PDF: abra o HTML baixado e use Imprimir no navegador.")


def main() -> None:
    st.set_page_config(page_title="AuditLab FGV — Auditoria de Requisitos", page_icon="📋", layout="wide")
    st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {color: #123A67;}
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    div[data-testid="stExpander"] {border: 1px solid #d7dfeb; border-radius: 9px;}
    </style>
    """, unsafe_allow_html=True)

    if "started_at" not in st.session_state:
        st.session_state["started_at"] = datetime.now(timezone.utc)
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False
    if "nav" not in st.session_state:
        st.session_state["nav"] = PAGES[0]
    if "audit_answers" not in st.session_state:
        st.session_state["audit_answers"] = [empty_answer(i) for i in range(5)]

    st.title("AuditLab FGV | Checklist de Auditoria")
    st.caption("MBA em Gerenciamento de Projetos · Gerenciamento do Escopo e da Qualidade · 25 minutos")

    with st.sidebar:
        st.header("Equipe auditora")
        st.text_input("Nome da equipe", key="team", placeholder="Ex.: Equipe Alfa")
        st.markdown("**Roteiro sugerido · 25 min**")
        st.markdown("📘 5 min — leitura do plano  \n✏️ 10 min — checklist  \n🔎 7 min — evidências  \n✅ 3 min — feedback")
        elapsed = max(0, int((datetime.now(timezone.utc) - st.session_state["started_at"]).total_seconds() // 60))
        st.caption(f"Tempo decorrido aproximado: {elapsed} min (atualiza ao interagir)")
        st.divider()
        st.radio("Navegação", PAGES, key="nav")
        st.divider()
        st.caption("Exercício fictício. Dados não são armazenados em banco de dados.")
        st.button("Reiniciar a atividade", use_container_width=True, on_click=reset_activity)

    current = st.session_state["nav"]
    if current == PAGES[0]:
        show_context()
    elif current == PAGES[1]:
        show_checklist()
    else:
        show_results()

    st.divider()
    st.caption(
        "Referências metodológicas: PMI, PMBOK® Guide — 8ª edição, "
        "processo 2.2.2.2; ISO 19011:2026. "
        "Critérios R01–R05 foram elaborados para este exercício."
    )


if __name__ == "__main__":
    main()
