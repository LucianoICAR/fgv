"""Editor de Estrutura Analítica do Projeto (EAP) para Streamlit Cloud.

Execução local: streamlit run app.py
Publicação: envie app.py, eap_core.py, requirements.txt e packages.txt ao GitHub.
"""

from __future__ import annotations

import re

import graphviz
import pandas as pd
import streamlit as st

from eap_core import (
    adicionar,
    agregar,
    carregar_json,
    criar_exemplo,
    criar_projeto,
    descendentes,
    dinheiro_br,
    editar,
    excluir,
    exportar_csv,
    exportar_json,
    gerar_grafo,
    indice,
    linhas,
    reordenar,
    texto_hierarquico,
)

st.set_page_config(
    page_title="Editor de EAP | Estrutura Analítica do Projeto",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
      .eap-eyebrow {font-size: .78rem; font-weight: 750; letter-spacing: .12em;
                    color: #3d6e9f; text-transform: uppercase;}
      .eap-sub {color: #56667a; margin-top: -.7rem;}
      [data-testid="stMetric"] {border: 1px solid #dde6ef; border-radius: 10px;
                                  padding: 13px 17px; background: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)


def atribuir_projeto(documento: dict, aviso: str, selecionar: str = "root") -> None:
    st.session_state["projeto"] = documento
    st.session_state["revisao"] = st.session_state.get("revisao", 0) + 1
    st.session_state["selecionado_editar"] = selecionar
    st.session_state["aviso"] = aviso
    st.rerun()


def mostrar_erro(exc: Exception) -> None:
    st.error(str(exc))


def nome_seguro(nome: str) -> str:
    base = re.sub(r"[^a-z0-9_-]+", "_", nome.lower().strip())
    return (base.strip("_") or "projeto")[:50]


@st.cache_data(show_spinner=False, max_entries=30)
def renderizar_codigo_dot(codigo_dot: str, formato: str) -> bytes:
    return graphviz.Source(codigo_dot, engine="dot").pipe(format=formato)


if "projeto" not in st.session_state:
    st.session_state["projeto"] = criar_projeto()
if "revisao" not in st.session_state:
    st.session_state["revisao"] = 0

projeto = st.session_state["projeto"]
registros = linhas(projeto)
nos = indice(projeto)
raiz = nos["root"]
totais = agregar(projeto)
contador_entregas = sum(n["tipo"] == "entrega" for n in projeto["nodes"])
contador_pacotes = sum(n["tipo"] == "pacote" for n in projeto["nodes"])
nomes_exibicao = {
    n["id"]: f"{r['codigo']} · {n['nome']} ({'projeto' if n['tipo'] == 'projeto' else 'entrega' if n['tipo'] == 'entrega' else 'pacote'})"
    for r in registros
    for n in [nos[r["id"]]]
}

with st.sidebar:
    st.header("Meu trabalho")
    st.caption("O projeto fica na sessão atual. Salve um JSON para continuar depois.")

    st.download_button(
        "💾 Salvar projeto (.json)",
        data=exportar_json(projeto),
        file_name=f"eap_{nome_seguro(raiz['nome'])}.json",
        mime="application/json",
        use_container_width=True,
        on_click="ignore",
    )

    st.divider()
    st.subheader("Abrir trabalho salvo")
    arquivo = st.file_uploader("Selecione um arquivo JSON", type=["json"], help="JSON gerado por este aplicativo")
    if st.button("Importar EAP", use_container_width=True, disabled=arquivo is None):
        try:
            projeto_lido = carregar_json(arquivo.getvalue())
            atribuir_projeto(projeto_lido, "EAP importada com sucesso.")
        except ValueError as exc:
            mostrar_erro(exc)

    st.divider()
    st.subheader("Demonstração e reinício")
    st.caption("Carregar um exemplo substitui o projeto que está aberto.")
    confirma_exemplo = st.checkbox("Autorizo substituir pelo exemplo", key="confirma_exemplo")
    if st.button("Carregar exemplo didático", use_container_width=True, disabled=not confirma_exemplo):
        atribuir_projeto(criar_exemplo(), "Exemplo didático carregado.")

    confirma_novo = st.checkbox("Autorizo criar uma nova EAP", key="confirma_novo")
    if st.button("Nova EAP em branco", use_container_width=True, disabled=not confirma_novo):
        atribuir_projeto(criar_projeto(), "Nova EAP criada.")

    st.divider()
    st.caption("**Regras do modelo:** o projeto é o nível 0; uma entrega pode ter subentregas ou pacotes; um pacote de trabalho não possui filhos.")

st.markdown('<div class="eap-eyebrow">Ferramenta didática · Gestão de Projetos</div>', unsafe_allow_html=True)
st.title("Estrutura Analítica do Projeto — EAP")
st.markdown(
    '<p class="eap-sub">Construa uma EAP orientada a entregas, com níveis hierárquicos, '
    'pacotes de trabalho, duração estimada e custo.</p>',
    unsafe_allow_html=True,
)

if "aviso" in st.session_state:
    st.success(st.session_state.pop("aviso"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entregas", contador_entregas)
c2.metric("Pacotes de trabalho", contador_pacotes)
c3.metric("Custo total estimado", dinheiro_br(totais["root"]["custo"]))
c4.metric("Níveis da EAP", max(r["nivel"] for r in registros) + 1)

aba_editor, aba_diagrama, aba_lista = st.tabs([
    "✏️ Construir e editar", "🌳 Diagrama hierárquico", "📋 Lista da EAP"
])

with aba_editor:
    st.subheader("Projeto — nível 0")
    with st.form("form_projeto"):
        nome_projeto = st.text_input(
            "Nome do projeto", value=raiz["nome"], max_chars=140,
            key=f"raiz_nome_{st.session_state['revisao']}",
            placeholder="Ex.: Implantação de unidade de atendimento",
        )
        descricao_projeto = st.text_area(
            "Descrição (opcional)", value=raiz["descricao"], max_chars=1000, height=76,
            key=f"raiz_desc_{st.session_state['revisao']}",
        )
        salvar_projeto = st.form_submit_button("Salvar dados do projeto")
    if salvar_projeto:
        try:
            novo = editar(projeto, "root", nome_projeto, descricao_projeto)
            atribuir_projeto(novo, "Dados do projeto atualizados.")
        except ValueError as exc:
            mostrar_erro(exc)

    st.divider()
    aba_adicionar, aba_editar = st.tabs(["➕ Adicionar componente", "🛠️ Editar, mover ou excluir"])

    with aba_adicionar:
        st.markdown("**Escolha o tipo e vincule o componente ao projeto ou a uma entrega existente.**")
        novo_tipo = st.radio(
            "Tipo de componente", ["entrega", "pacote"],
            format_func=lambda t: "Entrega / subentrega" if t == "entrega" else "Pacote de trabalho",
            horizontal=True, key="tipo_adicionado",
        )
        pais = [r["id"] for r in registros if r["tipo"] != "pacote"]
        with st.form("form_adicionar"):
            pai = st.selectbox(
                "Componente pai", pais, format_func=lambda i: nomes_exibicao[i],
                key=f"novo_pai_{st.session_state['revisao']}",
            )
            nome_novo = st.text_input(
                "Nome da entrega ou do pacote", max_chars=140,
                placeholder="Descreva um resultado verificável, não uma atividade",
                key=f"novo_nome_{st.session_state['revisao']}",
            )
            descricao_nova = st.text_area(
                "Descrição / critério resumido (opcional)", height=76, max_chars=1000,
                key=f"novo_desc_{st.session_state['revisao']}",
            )
            duracao_nova, custo_novo = 0.0, 0.0
            if novo_tipo == "pacote":
                ac, bc = st.columns(2)
                duracao_nova = ac.number_input(
                    "Duração estimada (dias úteis)", min_value=0.0, max_value=1_000_000.0,
                    value=1.0, step=0.5, format="%.2f",
                    help="Estimativa do pacote. Não é a duração total do projeto.",
                    key=f"novo_dias_{st.session_state['revisao']}",
                )
                custo_novo = bc.number_input(
                    "Custo estimado do pacote (R$)", min_value=0.0, max_value=1_000_000_000.0,
                    value=0.0, step=100.0, format="%.2f",
                    key=f"novo_custo_{st.session_state['revisao']}",
                )
            submeteu = st.form_submit_button("Adicionar à EAP", type="primary")
        if submeteu:
            try:
                novo, id_adicionado = adicionar(
                    projeto, pai, novo_tipo, nome_novo, descricao_nova,
                    duracao_nova, custo_novo,
                )
                atribuir_projeto(novo, "Componente adicionado à EAP.", id_adicionado)
            except ValueError as exc:
                mostrar_erro(exc)

    with aba_editar:
        selecionaveis = [r["id"] for r in registros]
        if st.session_state.get("selecionado_editar") not in selecionaveis:
            st.session_state["selecionado_editar"] = "root"
        id_atual = st.selectbox(
            "Selecione o componente da EAP", selecionaveis,
            format_func=lambda i: nomes_exibicao[i], key="selecionado_editar",
        )
        no_atual = nos[id_atual]
        rev = st.session_state["revisao"]
        eh_raiz = id_atual == "root"
        if eh_raiz:
            tipo_alvo = "projeto"
            novo_pai = None
            st.caption("Este é o projeto (nível 0). É possível alterar seu nome e descrição; não pode ser movido nem excluído.")
        else:
            tipo_alvo = st.radio(
                "Tipo após a edição", ["entrega", "pacote"],
                index=0 if no_atual["tipo"] == "entrega" else 1,
                format_func=lambda t: "Entrega / subentrega" if t == "entrega" else "Pacote de trabalho",
                horizontal=True, key=f"tipo_editar_{id_atual}_{rev}",
            )
            bloqueados = descendentes(projeto, id_atual)
            pais_permitidos = [
                r["id"] for r in registros
                if r["tipo"] != "pacote" and r["id"] not in bloqueados
            ]
            novo_pai = st.selectbox(
                "Mover para dentro de", pais_permitidos,
                index=pais_permitidos.index(no_atual["parent_id"]),
                format_func=lambda i: nomes_exibicao[i],
                key=f"editar_pai_{id_atual}_{rev}",
            )
            if tipo_alvo == "pacote" and len(bloqueados) > 1:
                st.warning("Esta entrega possui componentes filhos. Mova ou exclua os filhos antes de convertê-la em pacote.")

        with st.form("form_editar"):
            nome_editado = st.text_input(
                "Nome do componente", value=no_atual["nome"], max_chars=140,
                key=f"editar_nome_{id_atual}_{rev}",
            )
            descricao_editada = st.text_area(
                "Descrição", value=no_atual["descricao"], max_chars=1000, height=76,
                key=f"editar_desc_{id_atual}_{rev}",
            )
            tempo_editado = 0.0
            custo_editado = 0.0
            if tipo_alvo == "pacote":
                col_dias, col_custo = st.columns(2)
                tempo_editado = col_dias.number_input(
                    "Duração do pacote (dias úteis)", min_value=0.0, max_value=1_000_000.0,
                    value=float(no_atual["duracao_dias"]), step=0.5, format="%.2f",
                    key=f"editar_dias_{id_atual}_{rev}",
                )
                custo_editado = col_custo.number_input(
                    "Custo do pacote (R$)", min_value=0.0, max_value=1_000_000_000.0,
                    value=float(no_atual["custo_reais"]), step=100.0, format="%.2f",
                    key=f"editar_custo_{id_atual}_{rev}",
                )
            salvar_editado = st.form_submit_button("Salvar alterações", type="primary")
        if salvar_editado:
            try:
                modificado = editar(
                    projeto, id_atual, nome_editado, descricao_editada,
                    tipo_alvo, novo_pai, tempo_editado, custo_editado,
                )
                atribuir_projeto(modificado, "Alterações salvas.", id_atual)
            except ValueError as exc:
                mostrar_erro(exc)

        if not eh_raiz:
            st.divider()
            st.markdown("**Ordem e exclusão**")
            irmaos = [n["id"] for n in projeto["nodes"] if n["parent_id"] == no_atual["parent_id"]]
            posicao = irmaos.index(id_atual)
            x, y = st.columns(2)
            if x.button("↑ Subir na EAP", disabled=posicao == 0, use_container_width=True):
                try:
                    atribuir_projeto(reordenar(projeto, id_atual, -1), "Ordem alterada.", id_atual)
                except ValueError as exc:
                    mostrar_erro(exc)
            if y.button("↓ Descer na EAP", disabled=posicao == len(irmaos) - 1, use_container_width=True):
                try:
                    atribuir_projeto(reordenar(projeto, id_atual, +1), "Ordem alterada.", id_atual)
                except ValueError as exc:
                    mostrar_erro(exc)

            apagar = descendentes(projeto, id_atual)
            st.caption(f"A exclusão deste componente removerá {len(apagar)} componente(s), incluindo os descendentes.")
            confirmado = st.checkbox(
                "Confirmo a exclusão deste componente e de seus descendentes",
                key=f"confirmar_excluir_{id_atual}_{rev}",
            )
            if st.button("Excluir componente", type="secondary", disabled=not confirmado):
                try:
                    novo, quantidade = excluir(projeto, id_atual)
                    atribuir_projeto(novo, f"{quantidade} componente(s) excluído(s).")
                except ValueError as exc:
                    mostrar_erro(exc)

with aba_diagrama:
    st.subheader("Visualização hierárquica")
    st.caption("Caixas azuis representam projeto/entregas; caixas amarelas representam pacotes de trabalho. Os custos das entregas são a soma dos pacotes subordinados.")
    orientacao = st.radio(
        "Orientação do desenho", ["vertical", "horizontal"],
        format_func=lambda o: "De cima para baixo" if o == "vertical" else "Da esquerda para a direita",
        horizontal=True, key="orientacao_grafo",
    )
    modo_exibicao = st.radio(
        "Modo de visualização", ["navegavel", "imagem"],
        format_func=lambda m: "Navegável (com rolagem)" if m == "navegavel" else "Imagem ajustada à tela",
        horizontal=True, key="modo_grafo",
    )
    fonte = gerar_grafo(projeto, orientacao).source
    try:
        with st.spinner("Gerando diagrama..."):
            imagem_png = renderizar_codigo_dot(fonte, "png")
            imagem_svg = renderizar_codigo_dot(fonte, "svg")
        if modo_exibicao == "navegavel":
            st.graphviz_chart(
                fonte, width="stretch", height=660,
                alt="Diagrama da EAP organizado em níveis hierárquicos",
            )
            st.caption("A área do diagrama pode ser rolada. Para ampliar sem perda de qualidade, baixe a versão SVG.")
        else:
            st.image(imagem_png, width="stretch", alt="Diagrama hierárquico da EAP")
        a, b = st.columns(2)
        a.download_button(
            "⬇️ Baixar imagem PNG", imagem_png,
            file_name=f"eap_{nome_seguro(raiz['nome'])}.png", mime="image/png",
            use_container_width=True, on_click="ignore",
        )
        b.download_button(
            "⬇️ Baixar imagem SVG (vetorial)", imagem_svg,
            file_name=f"eap_{nome_seguro(raiz['nome'])}.svg", mime="image/svg+xml",
            use_container_width=True, on_click="ignore",
        )
    except graphviz.backend.ExecutableNotFound:
        st.error("O programa Graphviz não foi encontrado. Inclua 'graphviz' no arquivo packages.txt do repositório.")
    except graphviz.backend.CalledProcessError as exc:
        st.error("O Graphviz não conseguiu renderizar esta EAP. Tente simplificar os nomes ou reduzir o número de componentes.")
        st.code(str(exc))

    st.info("**Atenção:** os dias úteis informados são estimativas de cada pacote. A duração total do projeto depende das dependências e do sequenciamento e não é obtida somando os prazos de todos os pacotes.")

with aba_lista:
    st.subheader("Lista numerada da EAP")
    st.caption("A raiz está no nível 0, com código 1. As entregas recebem códigos como 1.1; as subentregas e pacotes recebem os níveis subsequentes.")

    tabela = pd.DataFrame([
        {
            "Código": r["codigo"],
            "Nível": r["nivel"],
            "Tipo": "Projeto" if r["tipo"] == "projeto" else "Entrega" if r["tipo"] == "entrega" else "Pacote",
            "Componente / entrega": ("    " * r["nivel"]) + r["nome"],
            "Duração (dias úteis)": r["duracao_dias"],
            "Custo do pacote (R$)": r["custo_pacote"],
            "Custo consolidado (R$)": r["custo_consolidado"],
            "Descrição": r["descricao"],
        }
        for r in registros
    ])
    st.dataframe(
        tabela, hide_index=True, use_container_width=True,
        column_config={
            "Código": st.column_config.TextColumn("Código EAP", width="small"),
            "Nível": st.column_config.NumberColumn("Nível", format="%d", width="small"),
            "Componente / entrega": st.column_config.TextColumn("Componente / entrega", width="large"),
            "Duração (dias úteis)": st.column_config.NumberColumn("Duração (dias úteis)", format="%.2f"),
            "Custo do pacote (R$)": st.column_config.NumberColumn("Custo do pacote (R$)", format="R$ %.2f"),
            "Custo consolidado (R$)": st.column_config.NumberColumn("Custo consolidado (R$)", format="R$ %.2f"),
            "Descrição": st.column_config.TextColumn("Descrição", width="medium"),
        },
        height=min(680, 95 + (len(tabela) + 1) * 36),
    )

    st.markdown("**Versão textual hierarquizada**")
    st.code(texto_hierarquico(projeto), language=None)
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Baixar lista CSV (Excel)", exportar_csv(projeto),
        file_name=f"lista_eap_{nome_seguro(raiz['nome'])}.csv",
        mime="text/csv", use_container_width=True, on_click="ignore",
    )
    d2.download_button(
        "⬇️ Baixar lista TXT", texto_hierarquico(projeto).encode("utf-8-sig"),
        file_name=f"lista_eap_{nome_seguro(raiz['nome'])}.txt",
        mime="text/plain", use_container_width=True, on_click="ignore",
    )

    entregas_sem_filhos = [
        r for r in registros
        if r["tipo"] == "entrega" and not any(n["parent_id"] == r["id"] for n in projeto["nodes"])
    ]
    if entregas_sem_filhos:
        st.warning(
            f"**EAP ainda em elaboração:** {len(entregas_sem_filhos)} entrega(s) não foram "
            "desdobradas em subentregas ou pacotes de trabalho."
        )

st.divider()
st.caption(
    "Editor didático de EAP · inspirado na visualização hierárquica do WBS Schedule Pro "
    "(sem associação ou afiliação com seu fabricante). "
    "As durações são estimativas dos pacotes; este aplicativo não constrói cronograma."
)
