import streamlit as st
import pandas as pd
import graphviz
import base64

st.set_page_config(page_title="WBS / EAP Builder - Escopo e Qualidade", layout="wide")

st.title("📊 Construtor Interativo de EAP")

# Inicialização dos dados
if "eap_dados" not in st.session_state:
    st.session_state.eap_dados = [
        {"Código": "0", "Pai": None, "Tipo": "Projeto", "Nome": "CONSTRUÇÃO DA CASA", "Custo": 0.0, "Horas": 0},
        {"Código": "1", "Pai": "0", "Tipo": "Entrega", "Nome": "ESCRITÓRIO DE ARQUITETURA", "Custo": 0.0, "Horas": 0},
        {"Código": "1.1", "Pai": "1", "Tipo": "Pacote de Trabalho", "Nome": "Ante-Projeto Arquitetônico", "Custo": 2500.0, "Horas": 40},
        {"Código": "1.2", "Pai": "1", "Tipo": "Pacote de Trabalho", "Nome": "Aprovação na Prefeitura", "Custo": 1200.0, "Horas": 16},
        {"Código": "2", "Pai": "0", "Tipo": "Entrega", "Nome": "INFRA-ESTRUTURA", "Custo": 0.0, "Horas": 0},
        {"Código": "2.1", "Pai": "2", "Tipo": "Pacote de Trabalho", "Nome": "Limpeza do Terreno", "Custo": 800.0, "Horas": 24},
        {"Código": "2.2", "Pai": "2", "Tipo": "Pacote de Trabalho", "Nome": "Terraplanagem", "Custo": 3500.0, "Horas": 32},
        {"Código": "3", "Pai": "0", "Tipo": "Entrega", "Nome": "FUNDAÇÕES", "Custo": 0.0, "Horas": 0},
        {"Código": "3.1", "Pai": "3", "Tipo": "Entrega", "Nome": "Fundações Profundas", "Custo": 0.0, "Horas": 0},
        {"Código": "3.1.1", "Pai": "3.1", "Tipo": "Pacote de Trabalho", "Nome": "Perfuração de Estacas", "Custo": 4200.0, "Horas": 48},
        {"Código": "3.1.2", "Pai": "3.1", "Tipo": "Pacote de Trabalho", "Nome": "Concretagem", "Custo": 3100.0, "Horas": 24},
    ]

df = pd.DataFrame(st.session_state.eap_dados)

def gerar_proximo_codigo(codigo_pai, dataframe):
    filhos = dataframe[dataframe["Pai"] == codigo_pai]
    if codigo_pai == "0":
        return str(len(filhos) + 1)
    return f"{codigo_pai}.{len(filhos) + 1}"

def calcular_rollup(dataframe):
    df_calc = dataframe.copy()
    df_calc["Profundidade"] = df_calc["Código"].apply(lambda x: 0 if x == "0" else len(str(x).split(".")))
    df_calc = df_calc.sort_values(by="Profundidade", ascending=False)
    
    for idx, row in df_calc.iterrows():
        cod = row["Código"]
        filhos = df_calc[df_calc["Pai"] == cod]
        if not filhos.empty:
            df_calc.at[idx, "Custo"] = filhos["Custo"].sum()
            df_calc.at[idx, "Horas"] = filhos["Horas"].sum()
            
    return df_calc.sort_values(by="Código")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1. Projeto")
    nome_projeto = st.text_input("Nome do Projeto:", value=df.loc[df["Código"] == "0", "Nome"].values)
    if st.button("Atualizar Nome"):
        for item in st.session_state.eap_dados:
            if item["Código"] == "0":
                item["Nome"] = nome_projeto.upper()
        st.rerun()

    st.divider()
    st.header("2. Decompor Escopo")
    
    possiveis_pais = df[df["Tipo"] != "Pacote de Trabalho"]
    opcoes_pai = {
        (f"{row['Nome']} (Projeto)" if row['Código'] == "0" else f"{row['Código']} - {row['Nome']}"): row['Código'] 
        for _, row in possiveis_pais.iterrows()
    }
    
    pai_selecionado_label = st.selectbox("Adicionar abaixo de:", list(opcoes_pai.keys()))
    codigo_pai = opcoes_pai[pai_selecionado_label]
    
    tipo_item = st.radio(
        "Tipo de Elemento:",
        ["Entrega", "Pacote de Trabalho"],
        help="Entregas abrem na horizontal. Pacotes de Trabalho ficam empilhados na vertical."
    )
    
    nome_novo_item = st.text_input("Nome da Entrega ou Pacote:")
    
    custo_item = 0.0
    horas_item = 0
    if tipo_item == "Pacote de Trabalho":
        col1, col2 = st.columns(2)
        with col1:
            custo_item = st.number_input("Custo (R$):", min_value=0.0, step=100.0)
        with col2:
            horas_item = st.number_input("Duração (Horas):", min_value=0, step=4)
            
    if st.button("➕ Adicionar à EAP", type="primary"):
        if nome_novo_item.strip() == "":
            st.error("Digite um nome para o item.")
        else:
            novo_cod = gerar_proximo_codigo(codigo_pai, df)
            st.session_state.eap_dados.append({
                "Código": novo_cod,
                "Pai": codigo_pai,
                "Tipo": tipo_item,
                "Nome": nome_novo_item.upper() if tipo_item == "Entrega" else nome_novo_item,
                "Custo": custo_item,
                "Horas": horas_item
            })
            st.rerun()

    st.divider()
    st.header("3. Visualização (Recolher)")
    # Permite fechar o que está abaixo de qualquer Entrega
    entregas_disponiveis = df[df["Tipo"] == "Entrega"]
    opcoes_recolher = {f"{row['Código']} - {row['Nome']}": row['Código'] for _, row in entregas_disponiveis.iterrows()}
    entregas_fechadas_labels = st.multiselect(
        "Fechar itens abaixo de:",
        options=list(opcoes_recolher.keys()),
        help="Oculta sub-entregas e pacotes abaixo da entrega selecionada."
    )
    codigos_fechados = [opcoes_recolher[lbl] for lbl in entregas_fechadas_labels]

    st.divider()
    if st.button("🗑️ Limpar Tudo"):
        st.session_state.eap_dados = [
            {"Código": "0", "Pai": None, "Tipo": "Projeto", "Nome": "NOVO PROJETO", "Custo": 0.0, "Horas": 0}
        ]
        st.rerun()

# Calcula os totais (Roll-up) ANTES de filtrar os nós recolhidos para manter os valores corretos
df_final = calcular_rollup(pd.DataFrame(st.session_state.eap_dados))

# --- ÁREA PRINCIPAL ---
tab_diagrama, tab_tabela = st.tabs(["🌳 Diagrama EAP", "📋 Tabela Detalhada"])

with tab_diagrama:
    # Controle de Zoom
    col_z1, col_z2 = st.columns([1, 3])
    with col_z1:
        zoom = st.slider("🔍 Zoom do Diagrama (%)", min_value=40, max_value=180, value=90, step=10)

    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.25", ranksep="0.35")
    dot.attr("node", shape="plain", fontname="Arial", fontsize="9")
    # dir="none" remove todas as setas das linhas
    dot.attr("edge", dir="none", color="#333333", penwidth="1.0")

    def fmt_moeda(val):
        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # Verifica se um nó deve ser ocultado porque algum ancestral seu está recolhido
    def esta_oculto(codigo):
        if codigo in ["0", None]:
            return False
        for fechado in codigos_fechados:
            if codigo.startswith(f"{fechado}."):
                return True
        return False

    # 1. Nó Raiz (Projeto) - Sem texto "Nível 0"
    proj = df_final[df_final["Código"] == "0"].iloc
    label_proj = f'''<
    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#85C1E9">
        <TR><TD COLSPAN="2"><B>{proj['Nome']}</B></TD></TR>
        <TR><TD>{fmt_moeda(proj['Custo'])}</TD><TD>{int(proj['Horas'])} horas</TD></TR>
    </TABLE>>'''
    dot.node("0", label=label_proj)

    # 2. Entregas (Hierarquia em árvore)
    entregas = df_final[df_final["Tipo"] == "Entrega"]
    for _, ent in entregas.iterrows():
        cod_ent = ent["Código"]
        if esta_oculto(cod_ent):
            continue
            
        # Sinaliza visualmente com [+] se a entrega estiver recolhida
        indicador_fechado = "  [+]" if cod_ent in codigos_fechados else ""
        
        label_ent = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" BGCOLOR="#D6EAF8">
            <TR><TD COLSPAN="2" ALIGN="LEFT"><B>{cod_ent}{indicador_fechado}</B></TD></TR>
            <TR><TD COLSPAN="2"><B>{ent['Nome']}</B></TD></TR>
            <TR><TD>{fmt_moeda(ent['Custo'])}</TD><TD>{int(ent['Horas'])} horas</TD></TR>
        </TABLE>>'''
        dot.node(cod_ent, label=label_ent)
        dot.edge(ent["Pai"], cod_ent)

    # 3. Pacotes de Trabalho (Empilhados verticalmente abaixo de sua entrega pai)
    pacotes = df_final[df_final["Tipo"] == "Pacote de Trabalho"]
    # Agrupa pacotes pelo Pai para empilhar um embaixo do outro
    for pai_cod, grupo_pacotes in pacotes.groupby("Pai"):
        if esta_oculto(pai_cod) or pai_cod in codigos_fechados:
            continue
            
        ultimo_no_coluna = pai_cod
        for _, pct in grupo_pacotes.iterrows():
            cod_pct = pct["Código"]
            label_pct = f'''<
            <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3" BGCOLOR="#FFFFFF">
                <TR><TD COLSPAN="2" ALIGN="LEFT"><B>{cod_pct}</B></TD></TR>
                <TR><TD COLSPAN="2">{pct['Nome']}</TD></TR>
                <TR><TD>{fmt_moeda(pct['Custo'])}</TD><TD>{int(pct['Horas'])} horas</TD></TR>
            </TABLE>>'''
            dot.node(cod_pct, label=label_pct)
            # Conecta em cadeia vertical (Pai -> Pacote 1 -> Pacote 2...) sem setas
            dot.edge(ultimo_no_coluna, cod_pct)
            ultimo_no_coluna = cod_pct

    # Renderiza em SVG dentro de um container com Zoom e barra de rolagem
    svg_bytes = dot.pipe(format="svg")
    svg_b64 = base64.b64encode(svg_bytes).decode("utf-8")
    
    html_zoom = f"""
    <div style="width: 100%; height: 600px; overflow: auto; border: 1px solid #ddd; border-radius: 6px; padding: 15px; background-color: #fafafa; text-align: center;">
        <img src="data:image/svg+xml;base64,{svg_b64}" style="width: {zoom}%; height: auto; transition: width 0.2s ease;" />
    </div>
    """
    st.markdown(html_zoom, unsafe_allow_html=True)

with tab_tabela:
    st.subheader("Estrutura Analítica Detalhada")
    st.dataframe(
        df_final[["Código", "Tipo", "Nome", "Custo", "Horas"]],
        use_container_width=True,
        hide_index=True
    )
