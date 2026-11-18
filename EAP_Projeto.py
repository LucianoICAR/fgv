import streamlit as st
import pandas as pd
import graphviz
import base64

st.set_page_config(page_title="Construtor de EAP", layout="wide")

st.title("📊 Construtor Interativo de EAP")
st.markdown("Decomponha o **Projeto** em **Entregas** e **Pacotes de Trabalho**, edite valores e ajuste o zoom.")

# Inicialização dos dados na sessão
if "eap_dados" not in st.session_state:
    st.session_state.eap_dados = [
        {"Código": "1", "Pai": None, "Tipo": "Projeto", "Nome": "CONSTRUÇÃO DA CASA", "Custo": 0.0, "Horas": 0},
        {"Código": "1.1", "Pai": "1", "Tipo": "Entrega", "Nome": "ESCRITÓRIO DE ARQUITETURA", "Custo": 0.0, "Horas": 0},
        {"Código": "1.1.1", "Pai": "1.1", "Tipo": "Pacote de Trabalho", "Nome": "Ante-Projeto Arquitetônico", "Custo": 2500.0, "Horas": 40},
        {"Código": "1.1.2", "Pai": "1.1", "Tipo": "Pacote de Trabalho", "Nome": "Aprovação na Prefeitura", "Custo": 1200.0, "Horas": 16},
        {"Código": "1.2", "Pai": "1", "Tipo": "Entrega", "Nome": "INFRA-ESTRUTURA", "Custo": 0.0, "Horas": 0},
        {"Código": "1.2.1", "Pai": "1.2", "Tipo": "Pacote de Trabalho", "Nome": "Limpeza do Terreno", "Custo": 800.0, "Horas": 24},
        {"Código": "1.2.2", "Pai": "1.2", "Tipo": "Pacote de Trabalho", "Nome": "Terraplanagem", "Custo": 3500.0, "Horas": 32},
    ]

df = pd.DataFrame(st.session_state.eap_dados)

# Função para gerar código sequencial (1 -> 1.1 -> 1.1.1)
def gerar_proximo_codigo(codigo_pai, dataframe):
    filhos = dataframe[dataframe["Pai"] == codigo_pai]
    return f"{codigo_pai}.{len(filhos) + 1}"

# Função de Roll-up (Soma custos e horas dos pacotes para as entregas e projeto)
def calcular_rollup(dataframe):
    df_calc = dataframe.copy()
    df_calc["Profundidade"] = df_calc["Código"].apply(lambda x: len(str(x).split(".")))
    df_calc = df_calc.sort_values(by="Profundidade", ascending=False)
    
    for idx, row in df_calc.iterrows():
        cod = row["Código"]
        filhos = df_calc[df_calc["Pai"] == cod]
        if not filhos.empty:
            df_calc.at[idx, "Custo"] = filhos["Custo"].sum()
            df_calc.at[idx, "Horas"] = filhos["Horas"].sum()
            
    return df_calc.sort_values(by="Código")

# Formatação de moeda
def fmt_moeda(val):
    return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1. Adicionar Novo Item")
    possiveis_pais = df[df["Tipo"] != "Pacote de Trabalho"]
    opcoes_pai = {f"{row['Código']} - {row['Nome']}": row['Código'] for _, row in possiveis_pais.iterrows()}
    
    pai_selecionado = st.selectbox("Adicionar abaixo de:", list(opcoes_pai.keys()))
    codigo_pai = opcoes_pai[pai_selecionado]
    
    tipo_item = st.radio(
        "Tipo de Elemento:",
        ["Entrega", "Pacote de Trabalho"],
        key="seletor_tipo_elemento"
    )
    
    nome_novo_item = st.text_input("Nome do Novo Item:", key="input_nome_item")
    
    if tipo_item == "Pacote de Trabalho":
        col1, col2 = st.columns(2)
        with col1:
            custo_item = st.number_input("Custo (R$):", min_value=0.0, value=0.0, step=100.0, key="input_custo")
        with col2:
            horas_item = st.number_input("Prazo (Horas):", min_value=0, value=0, step=4, key="input_horas")
    else:
        custo_item = 0.0
        horas_item = 0
            
    if st.button("➕ Adicionar Item", type="primary", use_container_width=True):
        if nome_novo_item.strip() == "":
            st.error("Digite um nome para o item.")
        else:
            novo_cod = gerar_proximo_codigo(codigo_pai, df)
            st.session_state.eap_dados.append({
                "Código": novo_cod,
                "Pai": codigo_pai,
                "Tipo": tipo_item,
                "Nome": nome_novo_item.upper() if tipo_item == "Entrega" else nome_novo_item,
                "Custo": float(custo_item),
                "Horas": int(horas_item)
            })
            st.rerun()

    st.divider()
    st.header("2. Editar ou Excluir Item")
    opcoes_editar = {f"{r['Código']} - {r['Nome']} ({r['Tipo']})": r['Código'] for _, r in df.iterrows()}
    item_edit_label = st.selectbox("Selecione o item para editar:", list(opcoes_editar.keys()))
    cod_edit = opcoes_editar[item_edit_label]
    
    item_atual = next(item for item in st.session_state.eap_dados if item["Código"] == cod_edit)
    
    novo_nome_edit = st.text_input("Editar Nome:", value=item_atual["Nome"], key=f"edit_nome_{cod_edit}")
    
    if item_atual["Tipo"] == "Pacote de Trabalho":
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            novo_custo_edit = st.number_input("Editar Custo (R$):", min_value=0.0, value=float(item_atual["Custo"]), step=100.0, key=f"edit_custo_{cod_edit}")
        with col_e2:
            novo_horas_edit = st.number_input("Editar Prazo (h):", min_value=0, value=int(item_atual["Horas"]), step=4, key=f"edit_horas_{cod_edit}")
    else:
        novo_custo_edit = 0.0
        novo_horas_edit = 0

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("💾 Salvar Alteração", use_container_width=True):
            for item in st.session_state.eap_dados:
                if item["Código"] == cod_edit:
                    item["Nome"] = novo_nome_edit.upper() if item["Tipo"] in ["Projeto", "Entrega"] else novo_nome_edit
                    if item["Tipo"] == "Pacote de Trabalho":
                        item["Custo"] = float(novo_custo_edit)
                        item["Horas"] = int(novo_horas_edit)
            st.rerun()
            
    with col_btn2:
        if cod_edit != "1":
            if st.button("🗑️ Excluir Item", use_container_width=True):
                st.session_state.eap_dados = [
                    item for item in st.session_state.eap_dados
                    if item["Código"] != cod_edit and not item["Código"].startswith(f"{cod_edit}.")
                ]
                st.rerun()

    st.divider()
    st.header("3. Recolher Entregas")
    entregas_disponiveis = df[df["Tipo"] == "Entrega"]
    opcoes_recolher = {f"{r['Código']} - {r['Nome']}": r['Código'] for _, r in entregas_disponiveis.iterrows()}
    entregas_fechadas_labels = st.multiselect(
        "Ocultar pacotes das entregas:",
        options=list(opcoes_recolher.keys())
    )
    codigos_fechados = [opcoes_recolher[lbl] for lbl in entregas_fechadas_labels]

    st.divider()
    if st.button("🔄 Resetar Projeto", use_container_width=True):
        st.session_state.eap_dados = [
            {"Código": "1", "Pai": None, "Tipo": "Projeto", "Nome": "NOVO PROJETO", "Custo": 0.0, "Horas": 0}
        ]
        st.rerun()

# Processa cálculos atualizados
df_final = calcular_rollup(pd.DataFrame(st.session_state.eap_dados))

# --- ÁREA PRINCIPAL ---
tab_diagrama, tab_tabela = st.tabs(["🌳 Diagrama EAP", "📋 Tabela Editável"])

with tab_diagrama:
    zoom_pct = st.slider("🔍 Zoom do Diagrama (%):", min_value=40, max_value=200, value=100, step=10)
    escala = zoom_pct / 100.0
    
    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.25", ranksep="0.35")
    dot.attr("node", shape="plain", fontname="Arial", fontsize="10")
    dot.attr("edge", arrowhead="none", color="#5D6D7E", penwidth="1.2")

    # 1. Desenha o Projeto (Nó raiz "1")
    proj = df_final[df_final["Código"] == "1"].iloc[0]
    label_proj = f'''<
    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#85C1E9">
        <TR><TD COLSPAN="2"><B>{proj['Nome']}</B></TD></TR>
        <TR><TD>{fmt_moeda(proj['Custo'])}</TD><TD>{int(proj['Horas'])}h</TD></TR>
    </TABLE>>'''
    dot.node("1", label=label_proj)

    # 2. Desenha todas as ENTREGAS em formato hierárquico
    entregas = df_final[df_final["Tipo"] == "Entrega"]
    for _, ent in entregas.iterrows():
        cod_ent = ent["Código"]
        pai_ent = ent["Pai"]
        
        if any(cod_ent.startswith(f"{f}.") for f in codigos_fechados):
            continue
            
        indicador_fechado = " [+]" if cod_ent in codigos_fechados else ""
        label_ent = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" BGCOLOR="#D6EAF8">
            <TR><TD COLSPAN="2" ALIGN="LEFT"><B>{cod_ent}{indicador_fechado}</B></TD></TR>
            <TR><TD COLSPAN="2"><B>{ent['Nome']}</B></TD></TR>
            <TR><TD>{fmt_moeda(ent['Custo'])}</TD><TD>{int(ent['Horas'])}h</TD></TR>
        </TABLE>>'''
        dot.node(cod_ent, label=label_ent)
        dot.edge(pai_ent, cod_ent)

    # 3. Desenha os PACOTES DE TRABALHO empilhados verticalmente
    for _, ent in entregas.iterrows():
        cod_ent = ent["Código"]
        
        if cod_ent in codigos_fechados or any(cod_ent.startswith(f"{f}.") for f in codigos_fechados):
            continue
            
        pacotes = df_final[(df_final["Pai"] == cod_ent) & (df_final["Tipo"] == "Pacote de Trabalho")]
        
        ultimo_no = cod_ent
        for _, pct in pacotes.iterrows():
            cod_pct = pct["Código"]
            label_pct = f'''<
            <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" BGCOLOR="#FFFFFF">
                <TR><TD COLSPAN="2" ALIGN="LEFT"><B>{cod_pct}</B></TD></TR>
                <TR><TD COLSPAN="2">{pct['Nome']}</TD></TR>
                <TR><TD>{fmt_moeda(pct['Custo'])}</TD><TD>{int(pct['Horas'])}h</TD></TR>
            </TABLE>>'''
            dot.node(cod_pct, label=label_pct)
            dot.edge(ultimo_no, cod_pct)
            ultimo_no = cod_pct

    # Renderiza em SVG vetorial com Zoom real via CSS e barra de rolagem
    svg_bytes = dot.pipe(format="svg")
    svg_base64 = base64.b64encode(svg_bytes).decode("utf-8")
    
    html_zoom = f"""
    <div style="width: 100%; height: 600px; overflow: auto; border: 1px solid #E0E0E0; border-radius: 8px; padding: 16px; background-color: #FAFAFA;">
Use o código com cuidado.
"""
st.components.v1.html(html_zoom, height=630)
with tab_tabela:
st.subheader("Tabela de Escopo, Prazo e Custo")
st.caption("Você também pode editar nomes, custos e prazos diretamente nas células abaixo (para Pacotes de Trabalho).")
df_editado = st.data_editor(
df_final[["Código", "Tipo", "Nome", "Custo", "Horas"]],
disabled=["Código", "Tipo"],
use_container_width=True,
hide_index=True
)
if st.button("Atualizar pela Tabela"):
for _, row in df_editado.iterrows():
for item in st.session_state.eap_dados:
if item["Código"] == row["Código"]:
item["Nome"] = row["Nome"].upper() if item["Tipo"] in ["Projeto", "Entrega"] else row["Nome"]
if item["Tipo"] == "Pacote de Trabalho":
item["Custo"] = float(row["Custo"])
item["Horas"] = int(row["Horas"])
st.rerun()
