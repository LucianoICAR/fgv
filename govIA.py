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
    "area_solicitante",
    "impacto_negocio",
    "urgencia",
    "risco_operacional",
    "alinhamento_estrategico",
    "custo_estimado",
    "usuarios_afetados",
    "historico_aprovacoes_area",
]

COLUNAS_CATEGORICAS = [
    "area_solicitante",
    "impacto_negocio",
    "urgencia",
    "risco_operacional",
    "alinhamento_estrategico",
]

COLUNAS_NUMERICAS = [
    "custo_estimado",
    "usuarios_afetados",
    "historico_aprovacoes_area",
]

AREAS = [
    "Diretoria",
    "Financeiro",
    "Operações",
    "Comercial",
    "Recursos Humanos",
    "Jurídico",
    "Atendimento ao Cliente",
    "Segurança da Informação",
    "Logística",
]

DEMANDAS = [
    "Atualização de firewall crítico",
    "Automação de relatório gerencial",
    "Correção de falhas no sistema de atendimento",
    "Implantação de chatbot interno",
    "Adequação à LGPD",
    "Modernização de servidores",
    "Dashboard executivo",
    "Integração com sistema legado",
    "Monitoramento de incidentes de segurança",
    "Portal de autosserviço para colaboradores",
    "Rastreabilidade logística",
    "Redução de indisponibilidade de aplicação crítica",
    "Revisão de controles de acesso",
    "Automação de conciliação financeira",
    "Sistema de priorização de chamados",
    "Melhoria no cadastro de clientes",
    "Migração de banco de dados",
    "Controle de contratos jurídicos",
    "Aplicativo para equipe de campo",
    "Gestão de vulnerabilidades",
]


def categoria_numerica(valor):
    return CATEGORIAS[valor]


@st.cache_data
def gerar_base_treino_exemplo(n=900, seed=42):
    rng = np.random.default_rng(seed)
    linhas = []

    for i in range(n):
        area = rng.choice(
            AREAS,
            p=[0.12, 0.15, 0.14, 0.11, 0.09, 0.08, 0.11, 0.10, 0.10],
        )
        impacto = rng.choice(["Baixo", "Médio", "Alto"], p=[0.25, 0.45, 0.30])
        urgencia = rng.choice(["Baixa", "Média", "Alta"], p=[0.30, 0.45, 0.25])
        risco = rng.choice(["Baixo", "Médio", "Alto"], p=[0.35, 0.40, 0.25])
        alinhamento = rng.choice(["Baixo", "Médio", "Alto"], p=[0.25, 0.45, 0.30])
        custo = int(rng.choice([20000, 35000, 50000, 75000, 100000, 150000, 220000, 300000]))
        usuarios = int(rng.choice([30, 80, 150, 300, 600, 1200, 2500, 5000]))

        historico_aprovacoes_area = {
            "Diretoria": rng.integers(75, 96),
            "Financeiro": rng.integers(70, 93),
            "Operações": rng.integers(45, 75),
            "Comercial": rng.integers(40, 70),
            "Recursos Humanos": rng.integers(20, 50),
            "Jurídico": rng.integers(25, 55),
            "Atendimento ao Cliente": rng.integers(15, 45),
            "Segurança da Informação": rng.integers(20, 55),
            "Logística": rng.integers(25, 60),
        }[area]

        escore_governanca = (
            1.8 * categoria_numerica(impacto)
            + 1.4 * categoria_numerica(urgencia)
            + 1.6 * categoria_numerica(risco)
            + 1.8 * categoria_numerica(alinhamento)
            + min(usuarios / 1200, 2.0)
            - min(custo / 200000, 1.8)
        )

        vies_area = {
            "Diretoria": 3.2,
            "Financeiro": 2.6,
            "Operações": 0.5,
            "Comercial": 0.2,
            "Recursos Humanos": -0.8,
            "Jurídico": -0.4,
            "Atendimento ao Cliente": -1.4,
            "Segurança da Informação": -1.1,
            "Logística": -0.6,
        }[area]

        ruido = rng.normal(0, 0.8)
        escore_enviesado = escore_governanca + vies_area + 0.04 * historico_aprovacoes_area + ruido
        aprovado_historico = 1 if escore_enviesado >= 9.4 else 0

        linhas.append(
            {
                "id_demanda": f"HIST-{i+1:04d}",
                "demanda": rng.choice(DEMANDAS),
                "area_solicitante": area,
                "impacto_negocio": impacto,
                "urgencia": urgencia,
                "risco_operacional": risco,
                "alinhamento_estrategico": alinhamento,
                "custo_estimado": custo,
                "usuarios_afetados": usuarios,
                "historico_aprovacoes_area": historico_aprovacoes_area,
                "aprovado_historico": aprovado_historico,
            }
        )

    return pd.DataFrame(linhas)


@st.cache_data
def gerar_base_demandas_exemplo():
    return pd.DataFrame(
        [
            ["ATUAL-001", "Atualização de firewall crítico", "Segurança da Informação", "Alto", "Alta", "Alto", "Alto", 150000, 5000, 35],
            ["ATUAL-002", "Dashboard executivo", "Diretoria", "Médio", "Média", "Baixo", "Médio", 90000, 60, 91],
            ["ATUAL-003", "Correção de falhas no sistema de atendimento", "Atendimento ao Cliente", "Alto", "Alta", "Alto", "Alto", 110000, 2500, 28],
            ["ATUAL-004", "Automação de relatório gerencial", "Financeiro", "Médio", "Baixa", "Baixo", "Médio", 70000, 80, 88],
            ["ATUAL-005", "Gestão de vulnerabilidades", "Segurança da Informação", "Alto", "Alta", "Alto", "Alto", 130000, 1200, 33],
            ["ATUAL-006", "Portal de autosserviço para colaboradores", "Recursos Humanos", "Médio", "Média", "Médio", "Alto", 85000, 3000, 40],
            ["ATUAL-007", "Automação de conciliação financeira", "Financeiro", "Médio", "Média", "Baixo", "Médio", 100000, 120, 86],
            ["ATUAL-008", "Rastreabilidade logística", "Logística", "Alto", "Alta", "Alto", "Alto", 160000, 900, 44],
            ["ATUAL-009", "Controle de contratos jurídicos", "Jurídico", "Médio", "Média", "Médio", "Médio", 75000, 100, 48],
            ["ATUAL-010", "Redução de indisponibilidade de aplicação crítica", "Operações", "Alto", "Alta", "Alto", "Alto", 180000, 5000, 62],
            ["ATUAL-011", "Melhoria no cadastro de clientes", "Comercial", "Alto", "Média", "Médio", "Alto", 95000, 1400, 58],
            ["ATUAL-012", "Integração com sistema legado", "Operações", "Alto", "Média", "Alto", "Alto", 220000, 1800, 60],
        ],
        columns=[
            "id_demanda",
            "demanda",
            "area_solicitante",
            "impacto_negocio",
            "urgencia",
            "risco_operacional",
            "alinhamento_estrategico",
            "custo_estimado",
            "usuarios_afetados",
            "historico_aprovacoes_area",
        ],
    )


def escore_referencia(row):
    escore = (
        1.8 * CATEGORIAS[row["impacto_negocio"]]
        + 1.4 * CATEGORIAS[row["urgencia"]]
        + 1.6 * CATEGORIAS[row["risco_operacional"]]
        + 1.8 * CATEGORIAS[row["alinhamento_estrategico"]]
        + min(row["usuarios_afetados"] / 1200, 2.0)
        - min(row["custo_estimado"] / 200000, 1.8)
    )
    return round(escore, 2)


def decisao_referencia(escore):
    return "APROVAR pedido" if escore >= 8.7 else "REJEITAR pedido"


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
        clf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)

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
        .head(15)
    )


def aplicar_modelo(model, df_demandas):
    df = df_demandas.copy()
    prob = model.predict_proba(df[COLUNAS_MODELO])[:, 1]
    pred = (prob >= 0.5).astype(int)

    df["probabilidade_aprovacao_ia"] = np.round(prob, 3)
    df["decisao_da_ia"] = np.where(pred == 1, "APROVAR pedido", "REJEITAR pedido")
    df["escore_referencia_governanca"] = df.apply(escore_referencia, axis=1)
    df["decisao_referencia_governanca"] = df["escore_referencia_governanca"].apply(decisao_referencia)
    df["alerta_governanca"] = np.where(
        df["decisao_da_ia"] != df["decisao_referencia_governanca"],
        "Divergência relevante",
        "Sem divergência evidente",
    )
    return df


def gerar_relatorio_texto(resultado):
    total = len(resultado)
    divergencias = resultado[resultado["alerta_governanca"] == "Divergência relevante"]
    taxa_area = (
        resultado.groupby("area_solicitante")["decisao_da_ia"]
        .apply(lambda s: (s == "APROVAR pedido").mean())
        .sort_values(ascending=False)
    )

    linhas = [
        "# Relatório preliminar de governança",
        "",
        f"Total de demandas analisadas: {total}.",
        f"Divergências relevantes entre a IA e a referência didática de governança: {len(divergencias)}.",
        "",
        "## Áreas aparentemente favorecidas pela IA",
    ]

    for area in taxa_area.head(3).index.tolist():
        linhas.append(f"- {area}: {taxa_area[area]:.0%} dos pedidos aprovados pela IA.")

    linhas.append("")
    linhas.append("## Áreas aparentemente prejudicadas pela IA")
    for area in taxa_area.tail(3).index.tolist():
        linhas.append(f"- {area}: {taxa_area[area]:.0%} dos pedidos aprovados pela IA.")

    linhas.append("")
    linhas.append("## Pedidos com divergência relevante")
    if divergencias.empty:
        linhas.append("- Nenhuma divergência detectada pelos critérios didáticos.")
    else:
        for _, row in divergencias.iterrows():
            linhas.append(
                f"- {row['id_demanda']} | {row['demanda']} | Área: {row['area_solicitante']} | "
                f"IA: {row['decisao_da_ia']} | Referência: {row['decisao_referencia_governanca']}."
            )

    linhas.extend(
        [
            "",
            "## Práticas de governança recomendadas",
            "- Política de governança para uso de IA em decisões de TI.",
            "- Curadoria, qualidade, representatividade e linhagem dos dados.",
            "- Avaliação de viés antes da implantação.",
            "- Revisão humana obrigatória para decisões de alto impacto.",
            "- Comitê de aprovação para sistemas de IA usados em decisões corporativas.",
            "- Monitoramento periódico de desempenho, equidade, risco e aderência estratégica.",
            "- Auditoria periódica do modelo e das decisões automatizadas.",
        ]
    )
    return "\n".join(linhas)


st.title("⚖️ Laboratório de Governança Corporativa de TI com IA")
st.caption("Sistema didático: a IA recomenda APROVAR ou REJEITAR pedidos de TI, permitindo discutir viés, risco e governança.")

with st.sidebar:
    st.header("Configurações")
    algoritmo = st.selectbox("Modelo supervisionado", ["Árvore de Decisão", "Random Forest"])
    mostrar_referencia = st.checkbox("Mostrar referência didática de governança", value=True)

    st.divider()
    st.subheader("Bases de exemplo")
    st.write("O sistema não carrega dados automaticamente. Use estes arquivos apenas se desejar uma base didática pronta.")

    base_treino_exemplo = gerar_base_treino_exemplo()
    base_demandas_exemplo = gerar_base_demandas_exemplo()

    st.download_button(
        "Baixar base de treino exemplo",
        data=base_treino_exemplo.to_csv(index=False).encode("utf-8"),
        file_name="base_treino_enviesada.csv",
        mime="text/csv",
    )

    st.download_button(
        "Baixar base de demandas atuais exemplo",
        data=base_demandas_exemplo.to_csv(index=False).encode("utf-8"),
        file_name="base_demandas_atuais.csv",
        mime="text/csv",
    )

abas = st.tabs(
    [
        "1. Contexto",
        "2. Carregar dados",
        "3. Treinamento",
        "4. Decisão da IA",
        "5. Análise do viés",
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
        Uma organização implantou um sistema de IA para apoiar o Comitê de Governança de TI na decisão sobre pedidos de investimento.
        Para cada demanda, a IA deve recomendar uma decisão objetiva:

        **APROVAR pedido** ou **REJEITAR pedido**.

        A proposta didática é usar uma base histórica enviesada. Algumas áreas foram mais aprovadas no passado,
        independentemente do real impacto estratégico, do risco operacional ou do número de usuários afetados.

        Os alunos devem perceber se a IA está rejeitando pedidos críticos ou aprovando pedidos menos relevantes por causa do padrão histórico aprendido.
        """
    )

    st.warning(
        "Nesta versão, nenhum dado é carregado automaticamente. O professor ou os alunos precisam carregar a base de treino e a base de demandas atuais."
    )

with abas[1]:
    st.subheader("Carregar bases de dados")

    col1, col2 = st.columns(2)

    with col1:
        arquivo_treino = st.file_uploader(
            "1. Carregue a base de treino histórica",
            type=["csv"],
            key="upload_treino",
        )

    with col2:
        arquivo_demandas = st.file_uploader(
            "2. Carregue a base de demandas atuais",
            type=["csv"],
            key="upload_demandas",
        )

    if arquivo_treino is not None:
        df_treino = pd.read_csv(arquivo_treino)
        ok, faltantes = validar_base_treino(df_treino)
        if not ok:
            st.error(f"A base de treino está sem as colunas: {', '.join(faltantes)}")
        else:
            st.session_state.df_treino = df_treino
            st.success("Base de treino carregada com sucesso.")
            st.dataframe(df_treino.head(20), use_container_width=True)

    if arquivo_demandas is not None:
        df_demandas = pd.read_csv(arquivo_demandas)
        ok, faltantes = validar_base_demandas(df_demandas)
        if not ok:
            st.error(f"A base de demandas atuais está sem as colunas: {', '.join(faltantes)}")
        else:
            st.session_state.df_demandas = df_demandas
            st.success("Base de demandas atuais carregada com sucesso.")
            st.dataframe(df_demandas, use_container_width=True)

    st.info(
        "Depois de carregar as duas bases, vá para a aba '3. Treinamento' e clique no botão para treinar o modelo."
    )

with abas[2]:
    st.subheader("Treinamento do modelo supervisionado")

    if st.session_state.df_treino is None:
        st.warning("Carregue primeiro a base de treino na aba '2. Carregar dados'.")
    else:
        df_treino = st.session_state.df_treino

        st.write("Resumo da base de treino carregada")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", len(df_treino))
        c2.metric("Pedidos aprovados no histórico", int(df_treino["aprovado_historico"].sum()))
        c3.metric("Taxa histórica de aprovação", f"{df_treino['aprovado_historico'].mean():.1%}")

        if st.button("Treinar modelo com a base histórica carregada", type="primary"):
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
            m2.metric("Algoritmo utilizado", algoritmo)

            st.warning(
                "Acurácia técnica não significa boa governança. O modelo pode estar apenas reproduzindo o viés histórico da organização."
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
                st.write("Variáveis mais influentes")
                st.dataframe(imp, use_container_width=True)

                fig, ax = plt.subplots()
                ax.barh(imp["variavel"], imp["importancia"])
                ax.set_xlabel("Importância")
                ax.set_ylabel("Variável")
                ax.invert_yaxis()
                st.pyplot(fig)

with abas[3]:
    st.subheader("Decisão explícita da IA: APROVAR ou REJEITAR pedidos")

    if st.session_state.model is None:
        st.warning("Treine o modelo na aba '3. Treinamento'.")
    elif st.session_state.df_demandas is None:
        st.warning("Carregue a base de demandas atuais na aba '2. Carregar dados'.")
    else:
        resultado = aplicar_modelo(st.session_state.model, st.session_state.df_demandas)
        st.session_state.resultado = resultado

        total_aprovados = int((resultado["decisao_da_ia"] == "APROVAR pedido").sum())
        total_rejeitados = int((resultado["decisao_da_ia"] == "REJEITAR pedido").sum())
        total_divergencias = int((resultado["alerta_governanca"] == "Divergência relevante").sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Pedidos aprovados pela IA", total_aprovados)
        m2.metric("Pedidos rejeitados pela IA", total_rejeitados)
        m3.metric("Sinais de possível viés", total_divergencias)

        st.markdown(
            """
            Abaixo está o ponto central da atividade. Os alunos devem observar as decisões da IA e questionar:
            **a IA aprovou pedidos menos críticos? A IA rejeitou pedidos críticos? Alguma área foi favorecida ou prejudicada?**
            """
        )

        colunas = [
            "id_demanda",
            "demanda",
            "area_solicitante",
            "impacto_negocio",
            "urgencia",
            "risco_operacional",
            "alinhamento_estrategico",
            "custo_estimado",
            "usuarios_afetados",
            "probabilidade_aprovacao_ia",
            "decisao_da_ia",
        ]

        if mostrar_referencia:
            colunas += [
                "escore_referencia_governanca",
                "decisao_referencia_governanca",
                "alerta_governanca",
            ]

        st.dataframe(resultado[colunas], use_container_width=True, hide_index=True)

        st.subheader("Leitura executiva das decisões")
        for _, row in resultado.iterrows():
            texto = (
                f"**{row['id_demanda']} — {row['demanda']}**  \n"
                f"Área: {row['area_solicitante']} | Impacto: {row['impacto_negocio']} | "
                f"Urgência: {row['urgencia']} | Risco: {row['risco_operacional']} | "
                f"Alinhamento: {row['alinhamento_estrategico']} | "
                f"Probabilidade de aprovação pela IA: {row['probabilidade_aprovacao_ia']:.0%}"
            )

            if row["decisao_da_ia"] == "APROVAR pedido":
                st.success(f"✅ IA recomenda **APROVAR pedido**\n\n{texto}")
            else:
                st.error(f"⛔ IA recomenda **REJEITAR pedido**\n\n{texto}")

        st.download_button(
            "Baixar decisões da IA em CSV",
            data=resultado.to_csv(index=False).encode("utf-8"),
            file_name="decisoes_ia_aprovar_rejeitar.csv",
            mime="text/csv",
        )

with abas[4]:
    st.subheader("Análise do viés")

    if st.session_state.resultado is None:
        st.warning("Gere primeiro as decisões da IA na aba '4. Decisão da IA'.")
    else:
        resultado = st.session_state.resultado

        resumo_area = (
            resultado.assign(aprovada_ia=(resultado["decisao_da_ia"] == "APROVAR pedido").astype(int))
            .groupby("area_solicitante")
            .agg(
                demandas=("id_demanda", "count"),
                pedidos_aprovados_pela_ia=("aprovada_ia", "sum"),
                taxa_aprovacao_ia=("aprovada_ia", "mean"),
                media_probabilidade_aprovacao_ia=("probabilidade_aprovacao_ia", "mean"),
                media_escore_governanca=("escore_referencia_governanca", "mean"),
            )
            .reset_index()
            .sort_values("taxa_aprovacao_ia", ascending=False)
        )

        st.write("Taxa de aprovação por área solicitante")
        st.dataframe(resumo_area, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots()
        ax.bar(resumo_area["area_solicitante"], resumo_area["taxa_aprovacao_ia"])
        ax.set_ylabel("Taxa de aprovação pela IA")
        ax.set_xlabel("Área solicitante")
        ax.tick_params(axis="x", rotation=70)
        st.pyplot(fig)

        st.markdown("### Pedidos críticos rejeitados pela IA")
        criticos_rejeitados = resultado[
            (resultado["decisao_da_ia"] == "REJEITAR pedido")
            & (
                (resultado["impacto_negocio"] == "Alto")
                | (resultado["risco_operacional"] == "Alto")
                | (resultado["alinhamento_estrategico"] == "Alto")
            )
        ]

        if criticos_rejeitados.empty:
            st.info("Nenhum pedido crítico foi rejeitado pela IA neste conjunto.")
        else:
            st.dataframe(criticos_rejeitados, use_container_width=True, hide_index=True)

        st.markdown("### Pedidos menos críticos aprovados pela IA")
        menos_criticos_aprovados = resultado[
            (resultado["decisao_da_ia"] == "APROVAR pedido")
            & (resultado["impacto_negocio"].isin(["Baixo", "Médio"]))
            & (resultado["risco_operacional"].isin(["Baixo", "Médio"]))
            & (resultado["alinhamento_estrategico"].isin(["Baixo", "Médio"]))
        ]

        if menos_criticos_aprovados.empty:
            st.info("Nenhum pedido menos crítico foi aprovado pela IA neste conjunto.")
        else:
            st.dataframe(menos_criticos_aprovados, use_container_width=True, hide_index=True)

with abas[5]:
    st.subheader("Diagnóstico de governança")

    st.markdown(
        """
        ### Perguntas orientadoras

        1. Quais pedidos a IA mandou **rejeitar**, apesar de terem alto impacto, alto risco ou alto alinhamento estratégico?
        2. Quais pedidos a IA mandou **aprovar**, apesar de parecerem menos críticos?
        3. Quais áreas parecem favorecidas pela IA?
        4. Quais áreas parecem prejudicadas pela IA?
        5. O problema é técnico, gerencial, ético, estratégico ou de governança?
        6. Quem deveria aprovar o uso desse sistema antes da implantação?
        7. Que controles deveriam existir antes, durante e depois da implantação?
        """
    )

    st.markdown("### Práticas de governança sugeridas")
    st.dataframe(
        pd.DataFrame(
            [
                ["Dados históricos enviesados", "Política de qualidade, curadoria, representatividade e linhagem de dados"],
                ["IA rejeita pedidos críticos", "Validação independente com critérios mínimos de risco, impacto e alinhamento estratégico"],
                ["IA aprova pedidos menos críticos", "Critérios formais de priorização e revisão por comitê"],
                ["Modelo favorece áreas historicamente privilegiadas", "Avaliação de viés por área solicitante e auditoria periódica"],
                ["Decisão automatizada sem revisão", "Human-in-the-loop para decisões de alto impacto"],
                ["Falta de transparência", "Critérios mínimos de explicabilidade e comunicação às partes interessadas"],
                ["Ausência de accountability", "Definição de dono do processo, dono dos dados, dono do modelo e comitê aprovador"],
                ["Falta de monitoramento", "Indicadores periódicos de desempenho, equidade, risco e aderência à estratégia"],
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
                ["EDM01", "Framework de governança", "Definir como a organização governa o uso de IA em decisões de TI"],
                ["EDM02", "Entrega de benefícios", "Verificar se a IA gera valor real e não apenas reproduz histórico"],
                ["EDM03", "Otimização de riscos", "Tratar viés e erro decisório como riscos corporativos"],
                ["EDM05", "Transparência", "Comunicar critérios, limitações e impactos das recomendações"],
                ["APO12", "Gestão de riscos", "Identificar, avaliar e tratar riscos do modelo"],
                ["APO14", "Gestão de dados", "Assegurar qualidade, representatividade e rastreabilidade dos dados"],
                ["BAI03", "Construção de soluções", "Validar requisitos, testes e controles antes da implantação"],
                ["MEA01", "Monitoramento de desempenho", "Acompanhar indicadores do modelo em operação"],
                ["MEA03", "Conformidade", "Avaliar aderência a políticas, normas e requisitos regulatórios"],
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
            file_name="relatorio_preliminar_governanca.md",
            mime="text/markdown",
        )

with abas[6]:
    st.subheader("Roteiro da apresentação da equipe")

    st.dataframe(
        pd.DataFrame(
            [
                ["1", "Decisões da IA", "Quais pedidos foram aprovados e quais foram rejeitados?"],
                ["2", "Incoerências percebidas", "Quais aprovações ou rejeições não fazem sentido?"],
                ["3", "Causa provável", "Qual viés ou falha de dados pode ter influenciado o modelo?"],
                ["4", "Riscos", "Quais riscos estratégicos, operacionais, reputacionais, legais ou financeiros existem?"],
                ["5", "Governança", "Que práticas, papéis, políticas e controles deveriam existir?"],
                ["6", "Modelo proposto", "Como a organização deveria aprovar, monitorar e auditar sistemas de IA?"],
            ],
            columns=["Slide", "Tema", "Pergunta-chave"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "Mensagem central: a IA não apenas automatiza decisões; ela pode institucionalizar padrões históricos inadequados quando a governança não atua."
    )
