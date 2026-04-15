
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Agente de Reposição de Estoque de TI", layout="wide")

st.title("🤖 Agente de Reposição de Estoque de TI")
st.caption("Demonstração didática de agente: perceber → interpretar → decidir → agir")

st.markdown("""
Este agente recebe dados de estoque, calcula risco de ruptura e recomenda ações automáticas.
Ele foi feito para demonstrar, em sala de aula, como um agente de IA pode:
- observar dados operacionais;
- interpretar indicadores;
- tomar decisões com base em regras;
- propor ações automáticas com justificativa;
- registrar a lógica da decisão.
""")

with st.expander("📘 Estrutura esperada do arquivo CSV"):
    st.markdown("""
**Colunas obrigatórias**
- `item`
- `categoria`
- `estoque_atual`
- `consumo_medio_mensal`
- `prazo_reposicao_dias`
- `criticidade`

**Valores esperados em `criticidade`**
- baixa
- media
- alta
    """)

# -----------------------------
# Funções do agente
# -----------------------------
def normalizar_criticidade(valor: str) -> str:
    if pd.isna(valor):
        return "media"
    v = str(valor).strip().lower()
    mapa = {
        "baixa": "baixa",
        "media": "media",
        "média": "media",
        "alta": "alta"
    }
    return mapa.get(v, "media")


def calcular_cobertura_dias(estoque_atual, consumo_medio_mensal):
    consumo_diario = consumo_medio_mensal / 30 if consumo_medio_mensal > 0 else 0
    if consumo_diario == 0:
        return np.inf
    return estoque_atual / consumo_diario


def calcular_risco(cobertura_dias, prazo_reposicao_dias, criticidade):
    margem = cobertura_dias - prazo_reposicao_dias

    if np.isinf(cobertura_dias):
        return "baixo", "Sem consumo registrado; não há evidência de ruptura no horizonte analisado."

    if criticidade == "alta":
        if margem < 0:
            return "crítico", "Cobertura inferior ao prazo de reposição em item de alta criticidade."
        elif margem <= 15:
            return "alto", "Cobertura muito próxima do prazo de reposição em item de alta criticidade."
        elif margem <= 30:
            return "moderado", "Cobertura aceitável, porém com pouca folga para item de alta criticidade."
        else:
            return "baixo", "Cobertura superior ao prazo de reposição com folga adequada."

    if criticidade == "media":
        if margem < 0:
            return "alto", "Cobertura inferior ao prazo de reposição em item de criticidade média."
        elif margem <= 15:
            return "moderado", "Cobertura próxima do prazo de reposição."
        elif margem <= 30:
            return "baixo", "Cobertura razoável para o horizonte de reposição."
        else:
            return "baixo", "Cobertura confortável para o horizonte de reposição."

    # baixa criticidade
    if margem < 0:
        return "moderado", "Cobertura inferior ao prazo de reposição, mas o item tem baixa criticidade."
    elif margem <= 15:
        return "baixo", "Cobertura próxima ao prazo de reposição, sem impacto elevado."
    else:
        return "baixo", "Cobertura adequada para item de baixa criticidade."


def definir_acao(risco, criticidade):
    if risco == "crítico":
        return "comprar imediatamente"
    if risco == "alto" and criticidade == "alta":
        return "priorizar compra"
    if risco == "alto":
        return "avaliar compra"
    if risco == "moderado":
        return "monitorar semanalmente"
    return "manter monitoramento"


def definir_prioridade(risco, criticidade):
    tabela = {
        ("crítico", "alta"): 1,
        ("crítico", "media"): 2,
        ("crítico", "baixa"): 3,
        ("alto", "alta"): 2,
        ("alto", "media"): 3,
        ("alto", "baixa"): 4,
        ("moderado", "alta"): 4,
        ("moderado", "media"): 5,
        ("moderado", "baixa"): 6,
        ("baixo", "alta"): 7,
        ("baixo", "media"): 8,
        ("baixo", "baixa"): 9,
    }
    return tabela.get((risco, criticidade), 9)


def resumo_executivo(df):
    total = len(df)
    criticos = (df["risco_ruptura"] == "crítico").sum()
    altos = (df["risco_ruptura"] == "alto").sum()
    compra_imediata = (df["acao_agente"] == "comprar imediatamente").sum()
    priorizar = (df["acao_agente"] == "priorizar compra").sum()

    itens_top = df.sort_values(["prioridade_agente", "cobertura_dias"]).head(5)["item"].tolist()
    itens_top_str = ", ".join(itens_top) if itens_top else "nenhum item"

    return f"""
Foram analisados **{total} itens** de TIC. O agente identificou **{criticos} itens em risco crítico**
e **{altos} itens em risco alto** de ruptura de estoque. Como ação automática, recomendou
**{compra_imediata} compras imediatas** e **{priorizar} priorizações de compra**.

Os itens mais sensíveis para decisão gerencial são: **{itens_top_str}**.

**Leitura executiva:** este painel permite antecipar ruptura, reduzir indisponibilidade de ativos
e priorizar alocação orçamentária com base em criticidade operacional e tempo de reposição.
"""


def processar_base(df):
    df = df.copy()

    colunas_obrigatorias = [
        "item", "categoria", "estoque_atual",
        "consumo_medio_mensal", "prazo_reposicao_dias", "criticidade"
    ]

    faltantes = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(faltantes)}")

    df["criticidade"] = df["criticidade"].apply(normalizar_criticidade)
    df["estoque_atual"] = pd.to_numeric(df["estoque_atual"], errors="coerce").fillna(0)
    df["consumo_medio_mensal"] = pd.to_numeric(df["consumo_medio_mensal"], errors="coerce").fillna(0)
    df["prazo_reposicao_dias"] = pd.to_numeric(df["prazo_reposicao_dias"], errors="coerce").fillna(0)

    df["cobertura_dias"] = df.apply(
        lambda row: calcular_cobertura_dias(row["estoque_atual"], row["consumo_medio_mensal"]),
        axis=1
    )

    riscos = df.apply(
        lambda row: calcular_risco(row["cobertura_dias"], row["prazo_reposicao_dias"], row["criticidade"]),
        axis=1
    )

    df["risco_ruptura"] = [r[0] for r in riscos]
    df["justificativa_risco"] = [r[1] for r in riscos]

    df["acao_agente"] = df.apply(
        lambda row: definir_acao(row["risco_ruptura"], row["criticidade"]),
        axis=1
    )

    df["prioridade_agente"] = df.apply(
        lambda row: definir_prioridade(row["risco_ruptura"], row["criticidade"]),
        axis=1
    )

    df["observacao_agente"] = df.apply(
        lambda row: (
            f"Ação recomendada: {row['acao_agente']}. "
            f"Motivo: {row['justificativa_risco']}"
        ),
        axis=1
    )

    df = df.sort_values(["prioridade_agente", "cobertura_dias"], ascending=[True, True]).reset_index(drop=True)
    return df


# -----------------------------
# Dados de exemplo
# -----------------------------
exemplo = pd.DataFrame([
    ["Notebook corporativo", "Computadores", 12, 20, 45, "alta"],
    ["Impressora térmica", "Impressoras", 8, 4, 30, "media"],
    ["Roteador de borda", "Redes", 3, 2, 60, "alta"],
    ["Mouse USB", "Periféricos", 120, 30, 20, "baixa"],
    ["Monitor 24 polegadas", "Monitores", 10, 9, 40, "media"],
    ["Switch 24 portas", "Redes", 2, 1, 90, "alta"],
    ["Teclado ABNT2", "Periféricos", 60, 15, 25, "baixa"],
    ["HD SSD 512GB", "Armazenamento", 6, 7, 35, "alta"],
], columns=[
    "item", "categoria", "estoque_atual",
    "consumo_medio_mensal", "prazo_reposicao_dias", "criticidade"
])

st.sidebar.header("⚙️ Fonte de dados")
fonte = st.sidebar.radio(
    "Escolha a base a ser analisada:",
    ["Usar base de exemplo", "Enviar arquivo CSV"]
)

if fonte == "Enviar arquivo CSV":
    arquivo = st.sidebar.file_uploader("Envie seu CSV", type=["csv"])
    if arquivo is not None:
        df_entrada = pd.read_csv(arquivo)
    else:
        st.info("Envie um arquivo CSV ou selecione a base de exemplo.")
        st.stop()
else:
    df_entrada = exemplo

st.subheader("📥 Base recebida pelo agente")
st.dataframe(df_entrada, use_container_width=True)

if st.button("Executar agente de reposição"):
    try:
        resultado = processar_base(df_entrada)

        st.success("Agente executado com sucesso.")

        # Métricas
        total_itens = len(resultado)
        criticos = int((resultado["risco_ruptura"] == "crítico").sum())
        altos = int((resultado["risco_ruptura"] == "alto").sum())
        compras_imediatas = int((resultado["acao_agente"] == "comprar imediatamente").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Itens analisados", total_itens)
        c2.metric("Risco crítico", criticos)
        c3.metric("Risco alto", altos)
        c4.metric("Compras imediatas", compras_imediatas)

        st.subheader("🧠 Resumo executivo do agente")
        st.markdown(resumo_executivo(resultado))

        st.subheader("📊 Decisões automáticas do agente")
        st.dataframe(
            resultado[
                [
                    "item", "categoria", "criticidade", "estoque_atual",
                    "consumo_medio_mensal", "prazo_reposicao_dias",
                    "cobertura_dias", "risco_ruptura", "acao_agente",
                    "prioridade_agente", "observacao_agente"
                ]
            ],
            use_container_width=True
        )

        st.subheader("🚨 Itens que exigem atenção gerencial")
        atencao = resultado[resultado["prioridade_agente"] <= 4]
        if len(atencao) > 0:
            st.dataframe(
                atencao[
                    [
                        "item", "categoria", "criticidade",
                        "cobertura_dias", "risco_ruptura",
                        "acao_agente", "observacao_agente"
                    ]
                ],
                use_container_width=True
            )
        else:
            st.info("Nenhum item crítico ou altamente prioritário foi identificado.")

        st.subheader("📤 Download da decisão do agente")
        csv_saida = resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultado em CSV",
            data=csv_saida,
            file_name="decisoes_agente_reposicao.csv",
            mime="text/csv"
        )

        with st.expander("🔍 Log conceitual do agente"):
            st.markdown("""
**1. Percepção**  
Leitura da base de estoque enviada.

**2. Interpretação**  
Cálculo da cobertura em dias a partir do estoque atual e do consumo médio mensal.

**3. Decisão**  
Classificação do risco com base em prazo de reposição e criticidade do item.

**4. Ação**  
Geração da recomendação automática: comprar imediatamente, priorizar compra, avaliar compra, monitorar semanalmente ou manter monitoramento.

**5. Rastreabilidade**  
Registro da justificativa da decisão para cada item.
            """)

    except Exception as e:
        st.error(f"Erro ao processar a base: {e}")

st.markdown("---")
st.markdown("""
### Como usar em aula
Este sistema demonstra que um **agente** não apenas responde perguntas.  
Ele:
- lê o ambiente (base de dados),
- calcula indicadores,
- toma uma decisão,
- propõe ações,
- e registra a justificativa.

### Gancho executivo para debate
**“Até que ponto eu deixo um agente automatizar a priorização de compras sem validação humana?”**
""")
