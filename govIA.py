import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Laboratório de Governança de TI com IA",
    page_icon="⚖️",
    layout="wide",
)

CATEGORIAS = {
    "Baixo": 1,
    "Médio": 2,
    "Alto": 3,
    "Baixa": 1,
    "Média": 2,
    "Alta": 3,
}

COLUNAS_MODELO = [
    "fonte_do_pedido",
    "area_solicitante",
    "impacto_negocio",
    "urgencia",
    "risco_operacional",
    "alinhamento_estrategico",
    "custo_estimado",
    "usuarios_afetados",
]

COLUNAS_CATEGORICAS = [
    "fonte_do_pedido",
    "area_solicitante",
    "impacto_negocio",
    "urgencia",
    "risco_operacional",
    "alinhamento_estrategico",
]

COLUNAS_NUMERICAS = [
    "custo_estimado",
    "usuarios_afetados",
]


def n_cat(valor):
    return CATEGORIAS[valor]


def validar_base_treino(df):
    faltantes = [c for c in COLUNAS_MODELO + ["aprovado_historico"] if c not in df.columns]
    if faltantes:
        return False, faltantes
    return True, []


def validar_base_demandas(df):
    faltantes = [c for c in COLUNAS_MODELO if c not in df.columns]
    if faltantes:
        return False, faltantes
    return True, []


def escore_referencia_negocio(row):
    escore = (
        2.0 * n_cat(row["impacto_negocio"])
        + 1.8 * n_cat(row["urgencia"])
        + 1.6 * n_cat(row["risco_operacional"])
        + 1.7 * n_cat(row["alinhamento_estrategico"])
        + min(row["usuarios_afetados"] / 1200, 2.0)
        - min(row["custo_estimado"] / 220000, 1.6)
    )
    return round(escore, 2)


def decisao_referencia(escore):
    return "APROVAR pedido" if escore >= 9.2 else "REJEITAR pedido"


def treinar_modelo(df_treino, algoritmo):
    X = df_treino[COLUNAS_MODELO]
    y = df_treino["aprovado_historico"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
            ("num", "passthrough", COLUNAS_NUMERICAS),
        ]
    )

    if algoritmo == "Árvore de Decisão":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

    model = Pipeline(
        steps=[
            ("preprocessamento", preprocessor),
            ("modelo", clf),
        ]
    )

    model.fit(X, y)
    return model


def obter_importancias(model):
    clf = model.named_steps["modelo"]
    pre = model.named_steps["preprocessamento"]

    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame()

    nomes_cat = list(pre.named_transformers_["cat"].get_feature_names_out(COLUNAS_CATEGORICAS))
    nomes = nomes_cat + COLUNAS_NUMERICAS
    imp = clf.feature_importances_

    return (
        pd.DataFrame({"variavel": nomes, "importancia": imp})
        .sort_values("importancia", ascending=False)
        .head(20)
    )


def aplicar_modelo(model, df_demandas):
    df = df_demandas.copy()

    prob = model.predict_proba(df[COLUNAS_MODELO])[:, 1]
    pred = (prob >= 0.5).astype(int)

    df["probabilidade_aprovacao_ia"] = np.round(prob, 3)
    df["decisao_da_ia"] = np.where(pred == 1, "APROVAR pedido", "REJEITAR pedido")

    df["escore_referencia_negocio"] = df.apply(escore_referencia_negocio, axis=1)
    df["decisao_referencia_negocio"] = df["escore_referencia_negocio"].apply(decisao_referencia)

    df["alerta_governanca"] = np.where(
        df["decisao_da_ia"] != df["decisao_referencia_negocio"],
        "Divergência relevante",
        "Sem divergência evidente",
    )

    df["tipo_de_alerta"] = "Sem alerta"
    df.loc[
        (df["decisao_da_ia"] == "REJEITAR pedido")
        & (df["decisao_referencia_negocio"] == "APROVAR pedido"),
        "tipo_de_alerta",
    ] = "IA rejeitou pedido relevante para o negócio"

    df.loc[
        (df["decisao_da_ia"] == "APROVAR pedido")
        & (df["decisao_referencia_negocio"] == "REJEITAR pedido"),
        "tipo_de_alerta",
    ] = "IA aprovou pedido fraco para o negócio"

    return df


def gerar_relatorio_texto(resultado):
    total = len(resultado)
    divergencias = resultado[resultado["alerta_governanca"] == "Divergência relevante"]

    taxa_fonte = (
        resultado.groupby("fonte_do_pedido")["decisao_da_ia"]
        .apply(lambda s: (s == "APROVAR pedido").mean())
        .sort_values(ascending=False)
    )

    linhas = [
        "# Relatório preliminar de governança",
        "",
        f"Total de pedidos analisados: {total}.",
        f"Divergências relevantes entre a IA e a referência de negócio: {len(divergencias)}.",
        "",
        "## Fontes/cargos aparentemente favorecidos pela IA",
    ]

    for fonte in taxa_fonte.head(5).index.tolist():
        linhas.append(f"- {fonte}: {taxa_fonte[fonte]:.0%} dos pedidos aprovados pela IA.")

    linhas.append("")
    linhas.append("## Fontes/cargos aparentemente prejudicados pela IA")
    for fonte in taxa_fonte.tail(5).index.tolist():
        linhas.append(f"- {fonte}: {taxa_fonte[fonte]:.0%} dos pedidos aprovados pela IA.")

    linhas.append("")
    linhas.append("## Pedidos com divergência relevante")
    if divergencias.empty:
        linhas.append("- Nenhuma divergência detectada pelos critérios didáticos.")
    else:
        for _, row in divergencias.iterrows():
            linhas.append(
                f"- {row['id_pedido']} | {row['pedido']} | Fonte: {row['fonte_do_pedido']} | "
                f"Impacto: {row['impacto_negocio']} | Urgência: {row['urgencia']} | "
                f"IA: {row['decisao_da_ia']} | Negócio: {row['decisao_referencia_negocio']} | "
                f"Alerta: {row['tipo_de_alerta']}."
            )

    linhas.extend(
        [
            "",
            "## Diagnóstico esperado",
            "- A IA aprendeu uma cultura organizacional inadequada.",
            "- A fonte/cargo do pedido passou a influenciar mais a decisão do que urgência e impacto no negócio.",
            "- A organização automatizou um padrão político-hierárquico de atendimento.",
            "- A decisão automatizada pode rejeitar pedidos críticos quando eles vêm de fontes menos prestigiadas.",
            "- A decisão automatizada pode aprovar pedidos pouco relevantes quando eles vêm de cargos superiores.",
            "",
            "## Práticas de governança recomendadas",
            "- Definir critérios formais de priorização por valor, risco, urgência e alinhamento estratégico.",
            "- Remover ou controlar variáveis com potencial discriminatório ou político, como fonte/cargo do pedido.",
            "- Realizar avaliação de viés antes da implantação do modelo.",
            "- Exigir validação independente por comitê de governança de TI.",
            "- Implantar revisão humana para decisões de alto impacto.",
            "- Monitorar taxa de aprovação por fonte, área, impacto e urgência.",
            "- Auditar periodicamente decisões automatizadas e dados de treinamento.",
        ]
    )

    return "\n".join(linhas)


st.title("⚖️ Laboratório de Governança Corporativa de TI com IA")
st.caption("Sistema didático: a IA aprende uma cultura enviesada que prioriza a fonte/cargo do pedido em vez do valor para o negócio.")

with st.sidebar:
    st.header("Configurações")
    algoritmo = st.selectbox("Modelo supervisionado", ["Árvore de Decisão", "Random Forest"])

    st.divider()
    st.subheader("Arquivos esperados")
    st.write("O sistema não gera bases de dados. Carregue os CSVs fornecidos pelo professor.")
    st.code(
        "base_treino_vies_cargo.csv\nbase_demandas_atuais_vies_cargo.csv",
        language="text",
    )

abas = st.tabs(
    [
        "1. Contexto",
        "2. Carregar dados",
        "3. Treinamento",
        "4. Decisão da IA",
        "5. Evidências do viés",
        "6. Governança",
        "7. Apresentação",
    ]
)

if "df_treino" not in st.session_state:
    st.session_state.df_treino = None
if "df_demandas" not in st.session_state:
    st.session_state.df_demandas = None
if "model" not in st.session_state:
    st.session_state.model = None
if "resultado" not in st.session_state:
    st.session_state.resultado = None

with abas[0]:
    st.subheader("Caso de negócio")

    st.markdown(
        """
        Uma organização usa IA para decidir se pedidos de TI devem ser **aprovados** ou **rejeitados**.

        O problema é que a base histórica reflete uma cultura organizacional inadequada:
        pedidos vindos de cargos mais altos foram atendidos com mais frequência, mesmo quando tinham baixa urgência
        e baixo impacto no negócio. Já pedidos vindos de fontes operacionais foram rejeitados com mais frequência,
        mesmo quando eram urgentes e relevantes.

        A IA aprende esse padrão histórico e passa a reproduzi-lo.
        """
    )

    st.warning(
        "Objetivo didático: mostrar que o modelo pode aprender decisões erradas quando a organização usa dados históricos contaminados por cultura, hierarquia e poder político."
    )

    st.markdown(
        """
        **Hipótese que os alunos devem investigar**

        A IA está considerando principalmente a **fonte/cargo do pedido** e não os critérios corretos de priorização:
        urgência, impacto no negócio, risco operacional e alinhamento estratégico.
        """
    )

with abas[1]:
    st.subheader("Carregar bases de dados")

    col1, col2 = st.columns(2)

    with col1:
        arquivo_treino = st.file_uploader(
            "1. Carregue a base histórica de treino",
            type=["csv"],
            key="upload_treino",
        )

    with col2:
        arquivo_demandas = st.file_uploader(
            "2. Carregue a base de pedidos atuais",
            type=["csv"],
            key="upload_demandas",
        )

    if arquivo_treino is not None:
        df_treino = pd.read_csv(arquivo_treino)
        ok, faltantes = validar_base_treino(df_treino)
        if not ok:
            st.error(f"A base histórica está sem as colunas: {', '.join(faltantes)}")
        else:
            st.session_state.df_treino = df_treino
            st.success("Base histórica de treino carregada com sucesso.")
            st.dataframe(df_treino.head(30), use_container_width=True, hide_index=True)

    if arquivo_demandas is not None:
        df_demandas = pd.read_csv(arquivo_demandas)
        ok, faltantes = validar_base_demandas(df_demandas)
        if not ok:
            st.error(f"A base de pedidos atuais está sem as colunas: {', '.join(faltantes)}")
        else:
            st.session_state.df_demandas = df_demandas
            st.success("Base de pedidos atuais carregada com sucesso.")
            st.dataframe(df_demandas, use_container_width=True, hide_index=True)

    st.info("Depois de carregar as duas bases, vá para a aba '3. Treinamento'.")

with abas[2]:
    st.subheader("Treinamento do modelo")

    if st.session_state.df_treino is None:
        st.warning("Carregue primeiro a base histórica de treino.")
    else:
        df_treino = st.session_state.df_treino

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Registros históricos", len(df_treino))
        col_b.metric("Pedidos aprovados no histórico", int(df_treino["aprovado_historico"].sum()))
        col_c.metric("Taxa histórica de aprovação", f"{df_treino['aprovado_historico'].mean():.1%}")

        st.write("Aprovação histórica por fonte/cargo")
        resumo_fonte = (
            df_treino.groupby("fonte_do_pedido")["aprovado_historico"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(columns={"count": "pedidos_historicos", "sum": "aprovados_historicos", "mean": "taxa_aprovacao_historica"})
            .sort_values("taxa_aprovacao_historica", ascending=False)
        )
        st.dataframe(resumo_fonte, use_container_width=True, hide_index=True)

        if st.button("Treinar IA com a base histórica enviesada", type="primary"):
            model = treinar_modelo(df_treino, algoritmo)
            st.session_state.model = model

            X_train, X_test, y_train, y_test = train_test_split(
                df_treino[COLUNAS_MODELO],
                df_treino["aprovado_historico"],
                test_size=0.25,
                random_state=42,
                stratify=df_treino["aprovado_historico"],
            )

            model_eval = treinar_modelo(df_treino.loc[X_train.index].copy(), algoritmo)
            y_pred = model_eval.predict(X_test)

            st.success("Modelo treinado com sucesso.")

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            m1, m2 = st.columns(2)
            m1.metric("Acurácia em teste", f"{acc:.1%}")
            m2.metric("Algoritmo", algoritmo)

            st.warning(
                "Acurácia alta pode apenas indicar que a IA aprendeu bem o padrão enviesado do histórico."
            )

            st.write("Matriz de confusão")
            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Real: Rejeitar", "Real: Aprovar"],
                    columns=["Pred: Rejeitar", "Pred: Aprovar"],
                ),
                use_container_width=True,
            )

            imp = obter_importancias(model)
            if not imp.empty:
                st.write("Variáveis mais influentes no modelo")
                st.dataframe(imp, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots()
                ax.barh(imp["variavel"], imp["importancia"])
                ax.set_xlabel("Importância")
                ax.set_ylabel("Variável")
                ax.invert_yaxis()
                st.pyplot(fig)

with abas[3]:
    st.subheader("Decisão explícita da IA: APROVAR ou REJEITAR pedidos")

    if st.session_state.model is None:
        st.warning("Treine primeiro o modelo na aba '3. Treinamento'.")
    elif st.session_state.df_demandas is None:
        st.warning("Carregue primeiro a base de pedidos atuais.")
    else:
        resultado = aplicar_modelo(st.session_state.model, st.session_state.df_demandas)
        st.session_state.resultado = resultado

        total_aprovados = int((resultado["decisao_da_ia"] == "APROVAR pedido").sum())
        total_rejeitados = int((resultado["decisao_da_ia"] == "REJEITAR pedido").sum())
        divergencias = int((resultado["alerta_governanca"] == "Divergência relevante").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Pedidos aprovados pela IA", total_aprovados)
        c2.metric("Pedidos rejeitados pela IA", total_rejeitados)
        c3.metric("Divergências relevantes", divergencias)

        st.markdown(
            """
            Os alunos devem observar se a decisão da IA faz sentido. A referência de negócio considera principalmente:
            **impacto, urgência, risco, alinhamento estratégico, usuários afetados e custo**.

            O viés esperado é a IA favorecer o **cargo/fonte do pedido**.
            """
        )

        colunas = [
            "id_pedido",
            "pedido",
            "fonte_do_pedido",
            "area_solicitante",
            "impacto_negocio",
            "urgencia",
            "risco_operacional",
            "alinhamento_estrategico",
            "custo_estimado",
            "usuarios_afetados",
            "probabilidade_aprovacao_ia",
            "decisao_da_ia",
            "escore_referencia_negocio",
            "decisao_referencia_negocio",
            "alerta_governanca",
            "tipo_de_alerta",
        ]

        st.dataframe(resultado[colunas], use_container_width=True, hide_index=True)

        st.subheader("Leitura executiva das decisões")
        for _, row in resultado.iterrows():
            detalhes = (
                f"**{row['id_pedido']} — {row['pedido']}**  \n"
                f"Fonte/cargo: **{row['fonte_do_pedido']}** | Área: {row['area_solicitante']}  \n"
                f"Impacto: {row['impacto_negocio']} | Urgência: {row['urgencia']} | "
                f"Risco: {row['risco_operacional']} | Alinhamento: {row['alinhamento_estrategico']}  \n"
                f"Probabilidade de aprovação pela IA: {row['probabilidade_aprovacao_ia']:.0%}  \n"
                f"Referência de negócio: **{row['decisao_referencia_negocio']}**"
            )

            if row["decisao_da_ia"] == "APROVAR pedido":
                st.success(f"✅ IA recomenda **APROVAR pedido**\n\n{detalhes}")
            else:
                st.error(f"⛔ IA recomenda **REJEITAR pedido**\n\n{detalhes}")

        st.download_button(
            "Baixar decisões da IA em CSV",
            data=resultado.to_csv(index=False).encode("utf-8"),
            file_name="decisoes_ia_vies_cargo.csv",
            mime="text/csv",
        )

with abas[4]:
    st.subheader("Evidências do viés")

    if st.session_state.resultado is None:
        st.warning("Gere primeiro as decisões da IA.")
    else:
        resultado = st.session_state.resultado

        st.markdown("### Aprovação pela IA por fonte/cargo do pedido")
        resumo_fonte = (
            resultado.assign(aprovado_ia=(resultado["decisao_da_ia"] == "APROVAR pedido").astype(int))
            .groupby("fonte_do_pedido")
            .agg(
                pedidos=("id_pedido", "count"),
                aprovados_pela_ia=("aprovado_ia", "sum"),
                taxa_aprovacao_ia=("aprovado_ia", "mean"),
                media_probabilidade_aprovacao=("probabilidade_aprovacao_ia", "mean"),
                media_escore_negocio=("escore_referencia_negocio", "mean"),
            )
            .reset_index()
            .sort_values("taxa_aprovacao_ia", ascending=False)
        )
        st.dataframe(resumo_fonte, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots()
        ax.bar(resumo_fonte["fonte_do_pedido"], resumo_fonte["taxa_aprovacao_ia"])
        ax.set_ylabel("Taxa de aprovação pela IA")
        ax.set_xlabel("Fonte/cargo do pedido")
        ax.tick_params(axis="x", rotation=70)
        st.pyplot(fig)

        st.markdown("### Pedidos críticos rejeitados pela IA")
        criticos_rejeitados = resultado[
            (resultado["decisao_da_ia"] == "REJEITAR pedido")
            & (resultado["decisao_referencia_negocio"] == "APROVAR pedido")
        ]

        if criticos_rejeitados.empty:
            st.info("Nenhum pedido crítico foi rejeitado pela IA neste conjunto.")
        else:
            st.dataframe(criticos_rejeitados, use_container_width=True, hide_index=True)

        st.markdown("### Pedidos fracos aprovados pela IA")
        fracos_aprovados = resultado[
            (resultado["decisao_da_ia"] == "APROVAR pedido")
            & (resultado["decisao_referencia_negocio"] == "REJEITAR pedido")
        ]

        if fracos_aprovados.empty:
            st.info("Nenhum pedido fraco foi aprovado pela IA neste conjunto.")
        else:
            st.dataframe(fracos_aprovados, use_container_width=True, hide_index=True)

with abas[5]:
    st.subheader("Diagnóstico de governança")

    st.markdown(
        """
        ### Perguntas orientadoras

        1. A IA está considerando mais a **fonte/cargo do pedido** ou o **valor para o negócio**?
        2. Quais pedidos críticos foram rejeitados por virem de fontes menos prestigiadas?
        3. Quais pedidos fracos foram aprovados por virem de cargos superiores?
        4. Que risco essa prática gera para a organização?
        5. Que falha de governança permitiu que esse modelo fosse treinado com dados históricos enviesados?
        6. Que critérios deveriam ser obrigatórios para aprovar ou rejeitar pedidos de TI?
        7. Quem deveria validar o modelo antes da entrada em produção?
        """
    )

    st.markdown("### Práticas de governança sugeridas")
    st.dataframe(
        pd.DataFrame(
            [
                ["Fonte/cargo domina a decisão", "Definir critérios formais de priorização por valor, risco, urgência e alinhamento estratégico"],
                ["Dados históricos refletem cultura política", "Executar curadoria e avaliação de viés antes do treinamento"],
                ["Pedidos críticos são rejeitados", "Implantar regra de exceção para alto impacto, alta urgência e alto risco"],
                ["Pedidos fracos são aprovados por hierarquia", "Exigir justificativa de valor para demandas de cargos superiores"],
                ["Modelo opaco para os usuários", "Exigir explicabilidade mínima e comunicação dos critérios decisórios"],
                ["Ausência de responsabilização", "Definir dono do processo, dono dos dados, dono do modelo e comitê aprovador"],
                ["Uso sem controle contínuo", "Monitorar taxa de aprovação por fonte, área, impacto, urgência e risco"],
                ["Automação de decisão sensível", "Manter revisão humana para decisões de alto impacto"],
            ],
            columns=["Problema observado", "Prática de governança recomendada"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Relação com COBIT 2019")
    st.dataframe(
        pd.DataFrame(
            [
                ["EDM01", "Framework de governança", "Definir como a organização governa IA aplicada a decisões de TI"],
                ["EDM02", "Entrega de benefícios", "Garantir que a priorização produza valor real para o negócio"],
                ["EDM03", "Otimização de riscos", "Tratar viés cultural e erro decisório como riscos corporativos"],
                ["EDM05", "Transparência", "Comunicar critérios e limitações da IA às partes interessadas"],
                ["APO12", "Gestão de riscos", "Avaliar e tratar riscos de uso do modelo"],
                ["APO14", "Gestão de dados", "Assegurar qualidade, representatividade e rastreabilidade dos dados"],
                ["BAI03", "Construção de soluções", "Validar requisitos, testes e controles antes da implantação"],
                ["MEA01", "Monitoramento", "Acompanhar indicadores do modelo em operação"],
                ["MEA03", "Conformidade", "Verificar aderência às políticas internas"],
            ],
            columns=["Objetivo", "Foco", "Aplicação no caso"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state.resultado is not None:
        relatorio = gerar_relatorio_texto(st.session_state.resultado)
        st.download_button(
            "Baixar relatório preliminar em Markdown",
            data=relatorio.encode("utf-8"),
            file_name="relatorio_preliminar_vies_cargo.md",
            mime="text/markdown",
        )

with abas[6]:
    st.subheader("Roteiro da apresentação da equipe")

    st.dataframe(
        pd.DataFrame(
            [
                ["1", "Decisões da IA", "Quais pedidos foram aprovados e quais foram rejeitados?"],
                ["2", "Evidência do viés", "A fonte/cargo do pedido influenciou mais do que urgência e impacto?"],
                ["3", "Decisões erradas", "Quais pedidos críticos foram rejeitados e quais pedidos fracos foram aprovados?"],
                ["4", "Causa provável", "Que padrão cultural/hierárquico contaminou os dados históricos?"],
                ["5", "Riscos", "Quais riscos estratégicos, operacionais, reputacionais e financeiros existem?"],
                ["6", "Governança", "Que práticas, papéis, políticas e controles deveriam existir?"],
            ],
            columns=["Slide", "Tema", "Pergunta-chave"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "Mensagem central: a IA pode automatizar a cultura organizacional errada quando os dados históricos refletem poder, hierarquia e preferência política em vez de valor para o negócio."
    )
