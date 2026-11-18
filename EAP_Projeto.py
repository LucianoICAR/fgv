import streamlit as st
import pandas as pd
import graphviz

# Configuração da página
st.set_page_config(page_title="WBS / EAP Builder - Escopo e Qualidade", layout="wide")

st.title("📊 Construtor Interativo de EAP (Estilo WBS Chart Pro)")
st.markdown("Decomponha o projeto a partir do **Nível 0** até os **Pacotes de Trabalho**.")

# Inicialização do banco de dados na sessão do navegador do aluno
if "eap_dados" not in st.session_state:
    st.session_state.eap_dados = [
        {"Código": "0", "Pai": None, "Tipo": "Projeto (Nível 0)", "Nome": "CONSTRUÇÃO DA CASA", "Custo": 0.0, "Horas": 0},
        {"Código": "1", "Pai": "0", "Tipo": "Entrega", "Nome": "ESCRITÓRIO DE ARQUITETURA", "Custo": 0.0, "Horas": 0},
        {"Código": "1.1", "Pai": "1", "Tipo": "Pacote de Trabalho", "Nome": "Ante-Projeto Arquitetônico", "Custo": 2500.0, "Horas": 40},
        {"Código": "1.2", "Pai": "1", "Tipo": "Pacote de Trabalho", "Nome": "Aprovação na Prefeitura", "Custo": 1200.0, "Horas": 16},
        {"Código": "2", "Pai": "0", "Tipo": "Entrega", "Nome": "INFRA-ESTRUTURA", "Custo": 0.0, "Horas": 0},
        {"Código": "2.1", "Pai": "2", "Tipo": "Pacote de Trabalho", "Nome": "Limpeza do Terreno", "Custo": 800.0, "Horas": 24},
        {"Código": "2.2", "Pai": "2", "Tipo": "Pacote de Trabalho", "Nome": "Terraplanagem", "Custo": 3500.0, "Horas": 32},
    ]

df = pd.DataFrame(st.session_state.eap_dados)

# Função para gerar o próximo código EAP automaticamente (ex: 1 -> 1.1 -> 1.2)
def gerar_proximo_codigo(codigo_pai, dataframe):
    filhos = dataframe[dataframe["Pai"] == codigo_pai]
    if codigo_pai == "0":
        return str(len(filhos) + 1)
    return f"{codigo_pai}.{len(filhos) + 1}"

# Função de Roll-up (Soma custos e horas dos pacotes para as entregas e projeto)
def calcular_rollup(dataframe):
    df_calc = dataframe.copy()
    # Calcula a profundidade do nó pelo número de pontos no código
    df_calc["Profundidade"] = df_calc["Código"].apply(lambda x: 0 if x == "0" else len(str(x).split(".")))
    # Ordena do nível mais baixo para o mais alto
    df_calc = df_calc.sort_values(by="Profundidade", ascending=False)
    
    for idx, row in df_calc.iterrows():
        cod = row["Código"]
        filhos = df_calc[df_calc["Pai"] == cod]
        if not filhos.empty:
            df_calc.at[idx, "Custo"] = filhos["Custo"].sum()
            df_calc.at[idx, "Horas"] = filhos["Horas"].sum()
            
    return df_calc.sort_values(by="Código")

# --- BARRA LATERAL: CONTROLES DE DECOMPOSIÇÃO ---
with st.sidebar:
    st.header("1. Definir Projeto (Nível 0)")
    nome_projeto = st.text_input("Nome do Projeto:", value=df.loc[df["Código"] == "0", "Nome"].values[0])
    if st.button("Atualizar Nome do Projeto"):
        for item in st.session_state.eap_dados:
            if item["Código"] == "0":
                item["Nome"] = nome_projeto.upper()
        st.rerun()

    st.divider()
    st.header("2. Decompor Escopo")
    
    # Filtra apenas itens que NÃO são pacotes de trabalho para serem "Pais"
    possiveis_pais = df[df["Tipo"] != "Pacote de Trabalho"]
    opcoes_pai = {f"{row['Código']} - {row['Nome']}": row['Código'] for _, row in possiveis_pais.iterrows()}
    
    pai_selecionado_label = st.selectbox("Selecione onde adicionar (Item Pai):", list(opcoes_pai.keys()))
    codigo_pai = opcoes_pai[pai_selecionado_label]
    
    tipo_item = st.radio(
        "Tipo de Elemento:",
        ["Entrega / Sub-entrega", "Pacote de Trabalho"],
        help="Pacotes de Trabalho ficam na ponta da EAP e recebem Custo e Tempo."
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
                "Nome": nome_novo_item.upper() if tipo_item != "Pacote de Trabalho" else nome_novo_item,
                "Custo": custo_item,
                "Horas": horas_item
            })
            st.rerun()

    st.divider()
    if st.button("🗑️ Limpar Tudo (Novo Projeto)"):
        st.session_state.eap_dados = [
            {"Código": "0", "Pai": None, "Tipo": "Projeto (Nível 0)", "Nome": "NOVO PROJETO", "Custo": 0.0, "Horas": 0}
        ]
        st.rerun()

# Processa os cálculos atualizados
df_final = calcular_rollup(pd.DataFrame(st.session_state.eap_dados))

# --- ÁREA PRINCIPAL: DIAGRAMA VISUAL E TABELA ---
tab_diagrama, tab_tabela = st.tabs(["🌳 Diagrama EAP (Visual)", "📋 Tabela e Dicionário"])

with tab_diagrama:
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.3", ranksep="0.4")
    dot.attr("node", shape="plain", fontname="Arial", fontsize="10")

    # Função auxiliar para formatar moeda
    def fmt_moeda(val):
        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # 1. Desenha o Nível 0 (Projeto)
    proj = df_final[df_final["Código"] == "0"].iloc[0]
    label_proj = f'''<
    <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5" BGCOLOR="#85C1E9">
        <TR><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="8">Nível 0</FONT></TD></TR>
        <TR><TD COLSPAN="2"><B>{proj['Nome']}</B></TD></TR>
        <TR><TD>{fmt_moeda(proj['Custo'])}</TD><TD>{int(proj['Horas'])} horas</TD></TR>
    </TABLE>>'''
    dot.node("0", label=label_proj)

    # 2. Desenha as Entregas de Nível 1 e empilha seus filhos verticalmente (Estilo WBS Chart Pro)
    entregas_n1 = df_final[df_final["Pai"] == "0"]
    
    for _, ent in entregas_n1.iterrows():
        cod_n1 = ent["Código"]
        label_n1 = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#D6EAF8">
            <TR><TD COLSPAN="2" ALIGN="LEFT"><B>{cod_n1}</B></TD></TR>
            <TR><TD COLSPAN="2"><B>{ent['Nome']}</B></TD></TR>
            <TR><TD>{fmt_moeda(ent['Custo'])}</TD><TD>{int(ent['Horas'])} horas</TD></TR>
        </TABLE>>'''
        dot.node(cod_n1, label=label_n1)
        dot.edge("0", cod_n1)
        
        # Busca todos os descendentes desta entrega (Nível 2, 3...)
        descendentes = df_final[df_final["Código"].str.startswith(f"{cod_n1}.")]
        
        # Cria uma linha vertical invisível para empilhar os itens exatamente como na imagem anexada
        ultimo_no_coluna = cod_n1
        for _, desc in descendentes.iterrows():
            cod_desc = desc["Código"]
            is_pacote = desc["Tipo"] == "Pacote de Trabalho"
            cor_bg = "#FFFFFF" if is_pacote else "#EBF5FB"
            borda = "2" if is_pacote else "1"
            
            label_desc = f'''<
            <TABLE BORDER="{borda}" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="{cor_bg}">
                <TR><TD COLSPAN="2" ALIGN="LEFT"><B>{cod_desc}</B> <FONT POINT-SIZE="8">({desc['Tipo']})</FONT></TD></TR>
                <TR><TD COLSPAN="2">{desc['Nome']}</TD></TR>
                <TR><TD>{fmt_moeda(desc['Custo'])}</TD><TD>{int(desc['Horas'])} horas</TD></TR>
            </TABLE>>'''
            
            dot.node(cod_desc, label=label_desc)
            # Conecta verticalmente para manter o alinhamento em coluna única abaixo da entrega
            dot.edge(ultimo_no_coluna, cod_desc)
            ultimo_no_coluna = cod_desc

    st.graphviz_chart(dot, use_container_width=True)

with tab_tabela:
    st.subheader("Estrutura Analítica Detalhada")
    st.dataframe(
        df_final[["Código", "Tipo", "Nome", "Custo", "Horas"]],
        use_container_width=True,
        hide_index=True
    )
