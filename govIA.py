import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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


@st.cache_data
def carregar_csv_padrao(caminho):
    return pd.read_csv(caminho)


def escore_governanca(row):
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


def classificar_referencia_governanca(escore):
    return "Priorizar" if escore >= 8.7 else "Não priorizar"


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
    df_imp = df_imp.sort_values("importancia", ascending=False).head(15)
    return df_imp


def aplicar_modelo(model, df_atual):
    df = df_atual.copy()
    prob = model.predict_proba(df[COLUNAS_MODELO])[:, 1]
    pred = (prob >= 0.5).astype(int)

    df["probabilidade_priorizacao_ia"] = np.round(prob, 3)
    df["recomendacao_ia"] = np.where(pred == 1, "Priorizar", "Não priorizar")
    df["escore_referencia_governanca"] = df.apply(escore_governanca, axis=1)
    df["recomendacao_referencia_governanca"] = df["escore_referencia_governanca"].apply(classificar_referencia_governanca)
    df["alerta_governanca"] = np.where(
        df["recomendacao_ia"] != df["recomendacao_referencia_governanca"],
        "Divergência relevante",
        "Sem divergência evidente",
    )
    return df


def gerar_relatorio_texto(resultado):
    total = len(resultado)
    divergencias = resultado[resultado["alerta_governanca"] == "Divergência relevante"]
    prior_area = (
        resultado.groupby("area_solicitante")["recomendacao_ia"]
        .apply(lambda s: (s == "Priorizar").mean())
        .sort_values(ascending=False)
    )

    mais_fav = prior_area.head(3).index.tolist()
    mais_sub = prior_area.tail(3).index.tolist()

    linhas = []
    linhas.append("# Relatório preliminar de governança")
    linhas.append("")
    linhas.append(f"Total de demandas analisadas: {total}.")
    linhas.append(f"Divergências relevantes entre a IA e a referência de governança: {len(divergencias)}.")
    linhas.append("")
    linhas.append("## Áreas aparentemente favorecidas pela IA")
    for area in mais_fav:
        linhas.append(f"- {area}: {prior_area[area]:.0%} das demandas priorizadas.")
    linhas.append("")
    linhas.append("## Áreas aparentemente subpriorizadas pela IA")
    for area in mais_sub:
        linhas.append(f"- {area}: {prior_area[area]:.0%} das demandas priorizadas.")
    linhas.append("")
    linhas.append("## Demandas com divergência relevante")
    if divergencias.empty:
        linhas.append("- Nenhuma divergência foi detectada pelos critérios didáticos adotados.")
    else:
        for _, row in divergencias.iterrows():
            linhas.append(
                f"- {row['id_demanda']} | {row['demanda']} | Área: {row['area_solicitante']} | "
                f"IA: {row['recomendacao_ia']} | Referência: {row['recomendacao_referencia_governanca']}."
            )
    linhas.append("")
    linhas.append("## Práticas de governança recomendadas")
    linhas.append("- Definir política de governança para uso de IA em decisões de TI.")
    linhas.append("- Estabelecer papéis: dono do processo, dono dos dados, dono do modelo e comitê aprovador.")
    linhas.append("- Exigir avaliação de qualidade, representatividade e viés dos dados antes do treinamento.")
    linhas.append("- Definir critérios de priorização alinhados à estratégia, risco e valor.")
    linhas.append("- Implantar revisão humana obrigatória para decisões de alto impacto.")
    linhas.append("- Monitorar indicadores de desempenho, equidade, explicabilidade e aderência à política.")
    linhas.append("- Auditar periodicamente os modelos e as decisões automatizadas.")
    return "\n".join(linhas)


st.title("⚖️ Laboratório de Governança Corporativa de TI com IA")
st.caption("Sistema didático em Streamlit para demonstrar como dados de treino enviesados podem induzir decisões inadequadas em TI.")

with st.sidebar:
    st.header("Configurações")
    algoritmo = st.selectbox("Modelo supervisionado", ["Árvore de Decisão", "Random Forest"])
    mostrar_alertas = st.checkbox("Mostrar alertas de governança", value=True)
    st.divider()
    st.write("Arquivos esperados:")
    st.code("base_treino_enviesada.csv\nbase_demandas_atuais.csv")

abas = st.tabs([
    "1. Contexto",
    "2. Dados e treinamento",
    "3. Recomendações da IA",
    "4. Diagnóstico de governança",
    "5. Apresentação da equipe",
])

with abas[0]:
    st.subheader("Caso de negócio")

    st.markdown(
        """
        Uma organização implantou um sistema de IA para apoiar o Comitê de Governança de TI na priorização de demandas.
        O modelo foi treinado com dados históricos de aprovações de projetos.

        O problema didático deste laboratório é que a base histórica contém um viés: algumas áreas receberam mais aprovações
        no passado, independentemente do real impacto estratégico, risco operacional ou número de usuários afetados.

        A tarefa das equipes não é apenas avaliar o modelo técnico. A tarefa principal é diagnosticar falhas de governança
        e propor práticas que evitariam o uso inadequado de IA em decisões corporativas de TI.
        """
    )

    st.info(
        "Premissa didática: a IA pode estar estatisticamente coerente com o passado, mas estrategicamente incoerente com a boa governança."
    )

    st.markdown(
        """
        **Missão da equipe**

        1. Executar o sistema de IA.
        2. Identificar recomendações incoerentes.
        3. Diagnosticar a provável causa.
        4. Relacionar o problema a falhas de governança de TI.
        5. Propor práticas, controles e papéis para evitar esse tipo de decisão automatizada.
        """
    )

with abas[1]:
    st.subheader("Bases de dados")

    col1, col2 = st.columns(2)

    with col1:
        treino_upload = st.file_uploader("Base de treino enviesada", type=["csv"], key="treino")
    with col2:
        atual_upload = st.file_uploader("Base de demandas atuais", type=["csv"], key="atual")

    if treino_upload is not None:
        df_treino = pd.read_csv(treino_upload)
    else:
        df_treino = carregar_csv_padrao("base_treino_enviesada.csv")

    if atual_upload is not None:
        df_atual = pd.read_csv(atual_upload)
    else:
        df_atual = carregar_csv_padrao("base_demandas_atuais.csv")

    validar_colunas(df_treino, "de treino")
    validar_colunas(df_atual, "de demandas atuais")

    if "aprovado_historico" not in df_treino.columns:
        st.error("A base de treino precisa ter a coluna aprovado_historico.")
        st.stop()

    st.write("Amostra da base de treino")
    st.dataframe(df_treino.head(20), use_container_width=True)

    st.write("Amostra da base de demandas atuais")
    st.dataframe(df_atual.head(20), use_container_width=True)

    model = treinar_modelo(df_treino, algoritmo)

    X_train, X_test, y_train, y_test = train_test_split(
        df_treino[COLUNAS_MODELO],
        df_treino["aprovado_historico"],
        test_size=0.25,
        random_state=42,
        stratify=df_treino["aprovado_historico"],
    )
    model_eval = treinar_modelo(df_treino.loc[X_train.index].copy(), algoritmo)
    y_pred = model_eval.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Métricas técnicas do modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia em teste", f"{acc:.1%}")
    c2.metric("Registros de treino", len(df_treino))
    c3.metric("Taxa histórica de aprovação", f"{df_treino['aprovado_historico'].mean():.1%}")

    st.warning(
        "Atenção didática: uma boa métrica técnica não garante boa decisão de governança. "
        "O modelo pode estar aprendendo um padrão histórico inadequado."
    )

    st.write("Matriz de confusão")
    st.dataframe(
        pd.DataFrame(cm, index=["Real: Não priorizar", "Real: Priorizar"], columns=["Pred: Não priorizar", "Pred: Priorizar"]),
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
    st.subheader("Recomendações geradas pela IA")

    try:
        df_treino
    except NameError:
        df_treino = carregar_csv_padrao("base_treino_enviesada.csv")
        df_atual = carregar_csv_padrao("base_demandas_atuais.csv")
        model = treinar_modelo(df_treino, algoritmo)

    resultado = aplicar_modelo(model, df_atual)

    st.markdown(
        """
        A coluna **recomendacao_ia** mostra a decisão do modelo treinado com dados históricos.
        A coluna **recomendacao_referencia_governanca** é uma referência didática baseada em impacto, urgência, risco,
        alinhamento, usuários afetados e custo. Ela não representa uma verdade absoluta; serve para provocar a análise.
        """
    )

    colunas_resultado = [
        "id_demanda",
        "demanda",
        "area_solicitante",
        "impacto_negocio",
        "urgencia",
        "risco_operacional",
        "alinhamento_estrategico",
        "custo_estimado",
        "usuarios_afetados",
        "probabilidade_priorizacao_ia",
        "recomendacao_ia",
        "escore_referencia_governanca",
        "recomendacao_referencia_governanca",
        "alerta_governanca",
    ]

    if mostrar_alertas:
        st.dataframe(resultado[colunas_resultado], use_container_width=True)
    else:
        st.dataframe(resultado[[c for c in colunas_resultado if c != "alerta_governanca"]], use_container_width=True)

    st.subheader("Priorização por área solicitante")
    resumo_area = (
        resultado.assign(priorizada_ia=(resultado["recomendacao_ia"] == "Priorizar").astype(int))
        .groupby("area_solicitante")
        .agg(
            demandas=("id_demanda", "count"),
            priorizadas_ia=("priorizada_ia", "sum"),
            media_probabilidade_ia=("probabilidade_priorizacao_ia", "mean"),
            media_escore_governanca=("escore_referencia_governanca", "mean"),
        )
        .reset_index()
        .sort_values("media_probabilidade_ia", ascending=False)
    )
    st.dataframe(resumo_area, use_container_width=True)

    fig2, ax2 = plt.subplots()
    ax2.bar(resumo_area["area_solicitante"], resumo_area["media_probabilidade_ia"])
    ax2.set_ylabel("Probabilidade média de priorização pela IA")
    ax2.set_xlabel("Área solicitante")
    ax2.tick_params(axis="x", rotation=70)
    st.pyplot(fig2)

    csv_resultado = resultado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar recomendações da IA em CSV",
        data=csv_resultado,
        file_name="recomendacoes_ia_governanca.csv",
        mime="text/csv",
    )

with abas[3]:
    st.subheader("Diagnóstico de governança")

    try:
        resultado
    except NameError:
        df_treino = carregar_csv_padrao("base_treino_enviesada.csv")
        df_atual = carregar_csv_padrao("base_demandas_atuais.csv")
        model = treinar_modelo(df_treino, algoritmo)
        resultado = aplicar_modelo(model, df_atual)

    divergencias = resultado[resultado["alerta_governanca"] == "Divergência relevante"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Demandas analisadas", len(resultado))
    c2.metric("Divergências relevantes", len(divergencias))
    c3.metric("Percentual com divergência", f"{len(divergencias) / len(resultado):.1%}")

    st.markdown("### Perguntas orientadoras")
    st.markdown(
        """
        1. Quais recomendações da IA parecem inadequadas?
        2. Quais variáveis parecem influenciar indevidamente o modelo?
        3. O problema é técnico, gerencial, ético, estratégico ou de governança?
        4. Quem deveria aprovar o uso desse sistema?
        5. Que riscos a organização corre ao usar esse modelo sem supervisão?
        6. Que controles deveriam existir antes, durante e depois da implantação?
        """
    )

    st.markdown("### Práticas de governança sugeridas")
    st.dataframe(
        pd.DataFrame(
            [
                ["Dados históricos enviesados", "Política de qualidade, curadoria, representatividade e linhagem de dados"],
                ["Modelo prioriza áreas favorecidas historicamente", "Avaliação de viés e validação independente antes da implantação"],
                ["Decisão automatizada sem revisão", "Human-in-the-loop para decisões de alto impacto"],
                ["Falta de transparência", "Critérios mínimos de explicabilidade e comunicação às partes interessadas"],
                ["Ausência de accountability", "Definição de dono do processo, dono dos dados, dono do modelo e comitê aprovador"],
                ["Falta de monitoramento", "Indicadores periódicos de desempenho, equidade, risco e aderência à estratégia"],
                ["Sistema desalinhado à estratégia", "Critérios de priorização vinculados ao planejamento estratégico e ao portfólio"],
                ["Uso sem controle formal", "Auditoria periódica, gestão de mudanças e registro de decisões"],
            ],
            columns=["Problema observado", "Prática de governança recomendada"],
        ),
        use_container_width=True,
    )

    st.markdown("### Relação com COBIT 2019")
    st.dataframe(
        pd.DataFrame(
            [
                ["EDM01", "Garantir a configuração e manutenção do framework de governança", "Definir como a organização governa o uso de IA em decisões de TI"],
                ["EDM02", "Garantir a entrega de benefícios", "Verificar se a IA gera valor real e não apenas reproduz histórico"],
                ["EDM03", "Garantir a otimização de riscos", "Tratar viés e erro decisório como riscos corporativos"],
                ["EDM05", "Garantir transparência às partes interessadas", "Comunicar critérios, limitações e impactos das recomendações"],
                ["APO12", "Gerenciar riscos", "Identificar, avaliar e tratar riscos do modelo"],
                ["APO14", "Gerenciar dados", "Assegurar qualidade, representatividade e rastreabilidade dos dados"],
                ["BAI03", "Gerenciar identificação e construção de soluções", "Validar requisitos, testes e controles antes da implantação"],
                ["MEA01", "Monitorar desempenho e conformidade", "Acompanhar indicadores do modelo em operação"],
                ["MEA03", "Monitorar conformidade", "Avaliar aderência a políticas, normas e requisitos regulatórios"],
            ],
            columns=["Objetivo", "Descrição", "Aplicação no caso"],
        ),
        use_container_width=True,
    )

    relatorio = gerar_relatorio_texto(resultado)
    st.download_button(
        "Baixar relatório preliminar em Markdown",
        data=relatorio.encode("utf-8"),
        file_name="relatorio_preliminar_governanca.md",
        mime="text/markdown",
    )

with abas[4]:
    st.subheader("Roteiro da apresentação da equipe")

    st.markdown(
        """
        Cada equipe deve preparar uma apresentação executiva de 5 a 7 minutos, com a seguinte estrutura:
        """
    )

    st.dataframe(
        pd.DataFrame(
            [
                ["1", "Diagnóstico", "Quais decisões da IA não fazem sentido?"],
                ["2", "Causa provável", "Qual viés ou falha de dados pode ter influenciado o modelo?"],
                ["3", "Riscos", "Quais riscos estratégicos, operacionais, reputacionais, legais ou financeiros existem?"],
                ["4", "Governança", "Que práticas, papéis, políticas e controles deveriam existir?"],
                ["5", "Modelo proposto", "Como a organização deveria aprovar, monitorar e auditar sistemas de IA?"],
            ],
            columns=["Slide", "Tema", "Pergunta-chave"],
        ),
        use_container_width=True,
    )

    st.markdown("### Critérios de avaliação sugeridos")
    st.markdown(
        """
        - Clareza na identificação das decisões incoerentes.
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
