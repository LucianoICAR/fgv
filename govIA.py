import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
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
def gerar_base_treino(n=900, seed=42):
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
def gerar_base_atual():
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


@st.cache_data
def carregar_csv_ou_gerar(caminho, tipo):
    arquivo = Path(caminho)
    if arquivo.exists():
        return pd.read_csv(arquivo)
    if tipo == "treino":
        return gerar_base_treino()
    if tipo == "atual":
        return gerar_base_atual()
    raise ValueError("Tipo de base inválido.")


def escore_referencia(row):
    impacto = CATEGORIAS[row["impacto_negocio"]]
    urgencia = CATEGORIAS[row["urgencia"]]
    risco = CATEGORIAS[row["risco_operacional"]]
    alinhamento = CATEGORIAS[row["alinhamento_estrategico"]]
    custo = row["custo_estimado"]
    usuarios = row["usuarios_afetados"]

    escore = (
        1.8 * impacto
        + 1.4 * urgencia
        + 1.6 * risco
        + 1.8 * alinhamento
        + min(usuarios / 1200, 2.0)
        - min(custo / 200000, 1.8)
    )
    return round(escore, 2)


def classificacao_referencia(escore):
    return "APROVAR pedido" if escore >= 8.7 else "REJEITAR pedido"


def validar_colunas(df, nome_base):
    faltantes = [c for c in COLUNAS_MODELO if c not in df.columns]
    if faltantes:
        st.error(f"A base {nome_base} está sem as colunas: {', '.join(faltantes)}")
        st.stop()


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

    df_imp = pd.DataFrame({"variavel": nomes, "importancia": imp})
    return df_imp.sort_values("importancia", ascending=False).head(15)


def aplicar_modelo(model, df_atual):
    df = df_atual.copy()
    prob = model.predict_proba(df[COLUNAS_MODELO])[:, 1]
    pred = (prob >= 0.5).astype(int)

    df["probabilidade_aprovacao_ia"] = np.round(prob, 3)
    df["decisao_da_ia"] = np.where(pred == 1, "APROVAR pedido", "REJEITAR pedido")
    df["escore_referencia_governanca"] = df.apply(escore_referencia, axis=1)
    df["decisao_referencia_governanca"] = df["escore_referencia_governanca"].apply(classificacao_referencia)
    df["alerta_governanca"] = np.where(
        df["decisao_da_ia"] != df["decisao_referencia_governanca"],
        "Divergência relevante",
        "Sem divergência evidente",
    )
    df["percepcao_didatica"] = np.where(
        df["alerta_governanca"] == "Divergência relevante",
        "Investigar possível viés ou falha de governança",
        "Decisão aparentemente coerente com a referência",
    )
    return df


def estilo_decisao(valor):
    if valor == "APROVAR pedido":
        return "background-color: #d1fae5; color: #065f46; font-weight: 800;"
    if valor == "REJEITAR pedido":
        return "background-color: #fee2e2; color: #991b1b; font-weight: 800;"
    return ""


def estilo_alerta(valor):
    if valor == "Divergência relevante":
        return "background-color: #fef3c7; color: #92400e; font-weight: 800;"
    return ""


def gerar_relatorio_texto(resultado):
    total = len(resultado)
    divergencias = resultado[resultado["alerta_governanca"] == "Divergência relevante"]
    prior_area = (
        resultado.groupby("area_solicitante")["decisao_da_ia"]
        .apply(lambda s: (s == "APROVAR pedido").mean())
        .sort_values(ascending=False)
    )

    linhas = [
        "# Relatório preliminar de governança",
        "",
        f"Total de demandas analisadas: {total}.",
        f"Divergências relevantes entre a IA e a referência de governança: {len(divergencias)}.",
        "",
        "## Áreas aparentemente favorecidas pela IA",
    ]

    for area in prior_area.head(3).index.tolist():
        linhas.append(f"- {area}: {prior_area[area]:.0%} das demandas aprovadas pela IA.")

    linhas.append("")
    linhas.append("## Áreas aparentemente subpriorizadas pela IA")
    for area in prior_area.tail(3).index.tolist():
        linhas.append(f"- {area}: {prior_area[area]:.0%} das demandas aprovadas pela IA.")

    linhas.append("")
    linhas.append("## Demandas com divergência relevante")
    if divergencias.empty:
        linhas.append("- Nenhuma divergência foi detectada pelos critérios didáticos adotados.")
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
            "- Definir política de governança para uso de IA em decisões de TI.",
            "- Estabelecer papéis: dono do processo, dono dos dados, dono do modelo e comitê aprovador.",
            "- Exigir avaliação de qualidade, representatividade e viés dos dados antes do treinamento.",
            "- Definir critérios de aprovação alinhados à estratégia, risco e valor.",
            "- Implantar revisão humana obrigatória para decisões de alto impacto.",
            "- Monitorar indicadores de desempenho, equidade, explicabilidade e aderência à política.",
            "- Auditar periodicamente os modelos e as decisões automatizadas.",
        ]
    )
    return "\n".join(linhas)


st.title("⚖️ Laboratório de Governança Corporativa de TI com IA")
st.caption("Sistema didático para demonstrar como dados de treino enviesados podem levar uma IA a aprovar ou rejeitar pedidos de TI de forma inadequada.")

with st.sidebar:
    st.header("Configurações")
    algoritmo = st.selectbox("Modelo supervisionado", ["Árvore de Decisão", "Random Forest"])
    mostrar_referencia = st.checkbox("Mostrar referência didática de governança", value=True)
    st.divider()
    st.info(
        "O ponto central da atividade é observar pedidos aprovados ou rejeitados pela IA e questionar se a decisão faz sentido sob a ótica da governança."
    )

abas = st.tabs(
    [
        "1. Contexto",
        "2. Dados e treinamento",
        "3. Decisão da IA",
        "4. Análise do viés",
        "5. Diagnóstico de governança",
        "6. Apresentação da equipe",
    ]
)

df_treino = carregar_csv_ou_gerar("base_treino_enviesada.csv", "treino")
df_atual = carregar_csv_ou_gerar("base_demandas_atuais.csv", "atual")
validar_colunas(df_treino, "de treino")
validar_colunas(df_atual, "de demandas atuais")

if "aprovado_historico" not in df_treino.columns:
    st.error("A base de treino precisa ter a coluna aprovado_historico.")
    st.stop()

model = treinar_modelo(df_treino, algoritmo)
resultado = aplicar_modelo(model, df_atual)

with abas[0]:
    st.subheader("Caso de negócio")
    st.markdown(
        """
        Uma organização implantou um sistema de IA para apoiar o Comitê de Governança de TI na decisão sobre pedidos de investimento.
        Para cada demanda, a IA recomenda uma decisão objetiva:

        **APROVAR pedido** ou **REJEITAR pedido**.

        O modelo foi treinado com dados históricos de aprovações. Porém, a base histórica contém um viés: algumas áreas receberam
        mais aprovações no passado, independentemente do real impacto estratégico, risco operacional ou número de usuários afetados.

        A tarefa das equipes é observar as decisões da IA, identificar pedidos que foram aprovados ou rejeitados de maneira incoerente
        e propor práticas de governança para evitar esse tipo de falha.
        """
    )

    st.warning(
        "Atenção: o objetivo do laboratório é fazer os alunos perceberem que uma IA pode rejeitar pedidos críticos e aprovar pedidos menos relevantes quando aprende padrões históricos enviesados."
    )

with abas[1]:
    st.subheader("Carregamento das bases")

    col1, col2 = st.columns(2)
    with col1:
        treino_upload = st.file_uploader("Base de treino enviesada — opcional", type=["csv"], key="treino")
    with col2:
        atual_upload = st.file_uploader("Base de demandas atuais — opcional", type=["csv"], key="atual")

    if treino_upload is not None:
        df_treino = pd.read_csv(treino_upload)
        validar_colunas(df_treino, "de treino enviada")
        model = treinar_modelo(df_treino, algoritmo)
        resultado = aplicar_modelo(model, df_atual)

    if atual_upload is not None:
        df_atual = pd.read_csv(atual_upload)
        validar_colunas(df_atual, "de demandas atuais enviada")
        resultado = aplicar_modelo(model, df_atual)

    st.success(
        "Base de demandas atuais carregada. Acesse a aba '3. Decisão da IA' para visualizar, de forma explícita, quais pedidos foram APROVADOS ou REJEITADOS."
    )

    st.write("Amostra da base de treino")
    st.dataframe(df_treino.head(20), use_container_width=True)

    st.write("Demandas atuais carregadas")
    st.dataframe(df_atual, use_container_width=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df_treino[COLUNAS_MODELO],
        df_treino["aprovado_historico"],
        test_size=0.25,
        random_state=42,
        stratify=df_treino["aprovado_historico"],
    )

    df_treino_eval = df_treino.loc[X_train.index].copy()
    model_eval = treinar_modelo(df_treino_eval, algoritmo)
    y_pred = model_eval.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Métricas técnicas do modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia em teste", f"{acc:.1%}")
    c2.metric("Registros de treino", len(df_treino))
    c3.metric("Taxa histórica de aprovação", f"{df_treino['aprovado_historico'].mean():.1%}")

    st.warning(
        "Uma boa métrica técnica não garante boa decisão de governança. O modelo pode ter aprendido a reproduzir aprovações históricas enviesadas."
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
        st.subheader("Variáveis que mais influenciaram o modelo")
        st.dataframe(imp, use_container_width=True)

        fig, ax = plt.subplots()
        ax.barh(imp["variavel"], imp["importancia"])
        ax.set_xlabel("Importância")
        ax.set_ylabel("Variável")
        ax.invert_yaxis()
        st.pyplot(fig)

with abas[2]:
    st.subheader("Decisão explícita da IA: aprovar ou rejeitar pedidos")

    total_aprovados = int((resultado["decisao_da_ia"] == "APROVAR pedido").sum())
    total_rejeitados = int((resultado["decisao_da_ia"] == "REJEITAR pedido").sum())
    total_divergencias = int((resultado["alerta_governanca"] == "Divergência relevante").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Pedidos aprovados pela IA", total_aprovados)
    m2.metric("Pedidos rejeitados pela IA", total_rejeitados)
    m3.metric("Sinais de possível viés", total_divergencias)

    st.markdown(
        """
        A tabela abaixo é o principal ponto da atividade. Os alunos devem observar a coluna **DECISÃO DA IA** e avaliar:
        a IA está aprovando pedidos de menor criticidade? Está rejeitando pedidos de alto impacto, alto risco ou alto alinhamento estratégico?
        """
    )

    colunas_visao_executiva = [
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
        colunas_visao_executiva += [
            "escore_referencia_governanca",
            "decisao_referencia_governanca",
            "alerta_governanca",
            "percepcao_didatica",
        ]

    st.dataframe(
        resultado[colunas_visao_executiva]
        .style
        .applymap(estilo_decisao, subset=["decisao_da_ia"])
        .applymap(estilo_decisao, subset=["decisao_referencia_governanca"] if "decisao_referencia_governanca" in colunas_visao_executiva else [])
        .applymap(estilo_alerta, subset=["alerta_governanca"] if "alerta_governanca" in colunas_visao_executiva else []),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Leitura executiva das decisões")
    for _, row in resultado.iterrows():
        if row["decisao_da_ia"] == "APROVAR pedido":
            st.success(
                f"{row['id_demanda']} — IA recomenda APROVAR pedido: {row['demanda']} | Área: {row['area_solicitante']} | Probabilidade: {row['probabilidade_aprovacao_ia']:.0%}"
            )
        else:
            st.error(
                f"{row['id_demanda']} — IA recomenda REJEITAR pedido: {row['demanda']} | Área: {row['area_solicitante']} | Probabilidade de aprovação: {row['probabilidade_aprovacao_ia']:.0%}"
            )

    csv_resultado = resultado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar decisões da IA em CSV",
        data=csv_resultado,
        file_name="decisoes_ia_aprovar_rejeitar.csv",
        mime="text/csv",
    )

with abas[3]:
    st.subheader("Análise do viés percebido")

    st.markdown(
        """
        Esta aba ajuda os alunos a perceberem se a IA está favorecendo algumas áreas e rejeitando outras,
        independentemente da criticidade real dos pedidos.
        """
    )

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

    st.dataframe(resumo_area, use_container_width=True, hide_index=True)

    fig2, ax2 = plt.subplots()
    ax2.bar(resumo_area["area_solicitante"], resumo_area["taxa_aprovacao_ia"])
    ax2.set_ylabel("Taxa de aprovação pela IA")
    ax2.set_xlabel("Área solicitante")
    ax2.tick_params(axis="x", rotation=70)
    st.pyplot(fig2)

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
        st.dataframe(
            criticos_rejeitados[
                [
                    "id_demanda",
                    "demanda",
                    "area_solicitante",
                    "impacto_negocio",
                    "urgencia",
                    "risco_operacional",
                    "alinhamento_estrategico",
                    "decisao_da_ia",
                    "decisao_referencia_governanca",
                    "alerta_governanca",
                ]
            ]
            .style
            .applymap(estilo_decisao, subset=["decisao_da_ia", "decisao_referencia_governanca"])
            .applymap(estilo_alerta, subset=["alerta_governanca"]),
            use_container_width=True,
            hide_index=True,
        )

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
        st.dataframe(
            menos_criticos_aprovados[
                [
                    "id_demanda",
                    "demanda",
                    "area_solicitante",
                    "impacto_negocio",
                    "urgencia",
                    "risco_operacional",
                    "alinhamento_estrategico",
                    "decisao_da_ia",
                    "decisao_referencia_governanca",
                    "alerta_governanca",
                ]
            ]
            .style
            .applymap(estilo_decisao, subset=["decisao_da_ia", "decisao_referencia_governanca"])
            .applymap(estilo_alerta, subset=["alerta_governanca"]),
            use_container_width=True,
            hide_index=True,
        )

with abas[4]:
    st.subheader("Diagnóstico de governança")

    divergencias = resultado[resultado["alerta_governanca"] == "Divergência relevante"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Demandas analisadas", len(resultado))
    c2.metric("Divergências relevantes", len(divergencias))
    c3.metric("Percentual com divergência", f"{len(divergencias) / len(resultado):.1%}")

    st.markdown("### Perguntas orientadoras")
    st.markdown(
        """
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

    relatorio = gerar_relatorio_texto(resultado)
    st.download_button(
        "Baixar relatório preliminar em Markdown",
        data=relatorio.encode("utf-8"),
        file_name="relatorio_preliminar_governanca.md",
        mime="text/markdown",
    )

with abas[5]:
    st.subheader("Roteiro da apresentação da equipe")

    st.markdown("Cada equipe deve preparar uma apresentação executiva de 5 a 7 minutos.")

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

    st.markdown("### Critérios de avaliação sugeridos")
    st.markdown(
        """
        - Clareza na identificação dos pedidos aprovados e rejeitados pela IA.
        - Capacidade de identificar rejeições de pedidos críticos e aprovações de pedidos menos relevantes.
        - Capacidade de relacionar o problema a dados enviesados.
        - Qualidade da análise de riscos.
        - Coerência das práticas de governança propostas.
        - Uso adequado de conceitos de COBIT, governança de dados, accountability e monitoramento.
        - Postura executiva na apresentação.
        """
    )

    st.success(
        "Mensagem central: a IA não apenas automatiza decisões; ela pode institucionalizar padrões históricos inadequados quando a governança não atua."
    )
