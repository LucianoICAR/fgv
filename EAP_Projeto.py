import streamlit as st
import pandas as pd
import graphviz

st.set_page_config(page_title="Construtor de EAP", layout="wide")

st.title("📊 Construtor Interativo de EAP")
st.markdown("Decomponha o **Projeto** em **Entregas** e **Pacotes de Trabalho**.")

# Inicialização dos dados (Projeto começa como Código "1")
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

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1. Projeto Principal")
    nome_proj_atual = df.loc[df["Código"] == "1", "Nome"].iloc[0]
    novo_nome_proj = st.text_input("Nome do Projeto:", value=nome_proj_atual)
    if st.button("Atualizar Projeto"):
        for item in st.session_state.eap_dados:
            if item["Código"] == "1":
                item["Nome"] = novo_nome_proj.upper()
        st.rerun()

    st.divider()
    st.header("2. Decompor Escopo")
    
    possiveis_pais = df[df["Tipo"] != "Pacote de Trabalho"]
    opcoes_pai = {f"{row['Código']} - {row['Nome']}": row['Código'] for _, row in possiveis_pais.iterrows()}
    
    pai_selecionado = st.selectbox("Adicionar abaixo de:", list(opcoes_pai.keys()))
    codigo_pai = opcoes_pai[pai_selecionado]
    
    # Seleção do tipo fora de qualquer formulário para atualizar a tela na hora
    tipo_item = st.radio(
        "Tipo de Elemento:",
        ["Entrega", "Pacote de Trabalho"],
        key="seletor_tipo_elemento"
    )
    
    nome_novo_item = st.text_input("Nome do Item:", key="input_nome_item")
    
    # Campos de Custo e Prazo sempre visíveis quando Pacote de Trabalho está marcado
    if tipo_item == "Pacote de Trabalho":
        st.markdown("**Estimativas do Pacote:**")
        col1, col2 = st.columns(2)
        with col1:
            custo_item = st.number_input("Custo (R$):", min_value=0.0, value=0.0, step=100.0, key="input_custo")
        with col2:
            horas_item = st.number_input("Prazo (Horas):", min_value=0, value=0, step=4, key="input_horas")
    else:
        custo_item = 0.0
        horas_item = 0
        st.caption("ℹ️ Entregas somam automaticamente os custos e prazos dos seus pacotes.")
            
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
    st.header("3. Visualização (Recolher)")
    entregas_disponiveis = df[df["Tipo"] == "Entrega"]
    opcoes_recolher = {f"{r['Código']} - {r['Nome']}": r['Código'] for _, r in entregas_disponiveis.iterrows()}
    entregas_fechadas_labels = st.multiselect(
        "Ocultar pacotes das entregas:",
        options=list(opcoes_recolher.keys()),
        help="Selecione entregas para recolher seus pacotes de trabalho no diagrama."
    )
    codigos_fechados = [opcoes_recolher[lbl] for lbl in entregas_fechadas_labels]

    st.divider()
    if st.button("🗑️ Limpar Tudo", use_container_width=True):
        st.session_state.eap_dados = [
            {"Código": "1", "Pai": None, "Tipo": "Projeto", "Nome": "NOVO PROJETO", "Custo": 0.0, "Horas": 0}
        ]
        st.rerun()

# Processa cálculos
df_final = calcular_rollup(pd.DataFrame(st.session_state.eap_dados))

def fmt_moeda(val):
    return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- ÁREA PRINCIPAL ---
tab_diagrama, tab_tabela = st.tabs(["🌳 Diagrama EAP", "📋 Tabela Detalhada"])

with tab_diagrama:
    st.caption("💡 Dica: Passe o mouse sobre o diagrama e clique no ícone de expandir (canto superior direito) ou use o controle abaixo para dar zoom.")
    
    zoom_escala = st.slider("Ajustar tamanho / zoom do diagrama:", min_value=0.5, max_value=1.8, value=0.8, step=0.1)
    
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.25", ranksep="0.35", size=f"{12*zoom_escala},{8*zoom_escala}")
    dot.attr("node", shape="plain", fontname="Arial", fontsize="9")
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

    # 3. Desenha os PACOTES DE TRABALHO empilhados verticalmente abaixo de sua Entrega
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

    st.graphviz_chart(dot, use_container_width=False)

with tab_tabela:
    st.subheader("Resumo de Escopo, Prazo e Custo")
    st.dataframe(
        df_final[["Código", "Tipo", "Nome", "Custo", "Horas"]],
        use_container_width=True,
        hide_index=True
    )
