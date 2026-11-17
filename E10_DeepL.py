"""Oficina final FGV: classificacao multiclasse de pecas por MLP real.
Inicie com: streamlit run app.py
Dados inteiramente sintéticos; nao substitui engenharia de manutencao.
"""
import copy
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
PECAS = [f"Peça {i}" for i in range(1, 6)]
NUMERIC = ["temperatura_media", "nivel_de_vibracao", "pressao_do_sistema",
           "horas_de_uso", "idade_da_maquina"] + [f"historico_trocas_peca_{i}" for i in range(1, 6)]
CATEGORICAL = [f"tipo_peca_{i}" for i in range(1, 6)]
FEATURES = NUMERIC + CATEGORICAL
PIECE_TYPES = ["Original", "Recondicionada", "Compatível"]
CUSTO_PECAS = {"Peça 1": 18000, "Peça 2": 14000, "Peça 3": 23000,
               "Peça 4": 11000, "Peça 5": 26000}


def gerar_dados(n_historico=3000, n_atuais=60, seed=SEED):
    """Gera observacoes independentes, com interacoes nao lineares para a classe alvo."""
    rng = np.random.default_rng(seed)
    n = n_historico + n_atuais
    temp = rng.uniform(45, 108, n)
    vib = rng.uniform(1, 13, n)
    press = rng.uniform(45, 125, n)
    horas = rng.uniform(400, 21000, n)
    idade = np.clip(horas / 2700 + rng.normal(0.5, 1.5, n), 0.3, 18)
    types = rng.choice(PIECE_TYPES, size=(n, 5), p=[0.57, 0.23, 0.20])
    trocas = rng.poisson(lam=np.clip(idade[:, None] / 5 + np.arange(5) / 10, .15, 5), size=(n, 5))
    t = (temp - 73) / 15
    v = (vib - 6.5) / 2.8
    p = (press - 83) / 17
    h = (horas - 10500) / 6500
    a = (idade - 5) / 3
    tr = (trocas - 1.1) / 1.7
    rec = (types == "Recondicionada").astype(float)
    com = (types == "Compatível").astype(float)
    # Padrões distinguiveis por classe que incluem interacoes e transformacoes nao lineares.
    z = np.column_stack([
        1.10 * t + .92 * v + 1.70 * np.sin(1.2 * t * v) + .32 * tr[:, 0] + .30 * rec[:, 0] - .18 * p,
        1.17 * p + .65 * h + 1.65 * np.cos(1.1 * p * h) + .34 * tr[:, 1] + .30 * com[:, 1],
        -1.10 * p + .78 * v + .58 * np.maximum(v, 0) * np.maximum(-p, 0) + .35 * tr[:, 2] + .34 * rec[:, 2],
        .95 * t - .78 * v + .70 * a + .48 * np.maximum(t, 0) * np.maximum(a, 0) + .32 * tr[:, 3] + .28 * com[:, 3],
        -.93 * t + .65 * p + .97 * h + 1.45 * np.sin(1.8 * h * t) + .35 * tr[:, 4] + .36 * rec[:, 4],
    ])
    # Ruido controla a dificuldade; usar maximo determina somente a PECA dentre maquinas ja triadas.
    z += rng.normal(0, 0.60, size=z.shape)
    labels = np.argmax(z, axis=1)
    df = pd.DataFrame({
        "temperatura_media": temp.round(1), "nivel_de_vibracao": vib.round(2),
        "pressao_do_sistema": press.round(1), "horas_de_uso": horas.round(0).astype(int),
        "idade_da_maquina": idade.round(1),
    })
    for i in range(5):
        df[f"tipo_peca_{i+1}"] = types[:, i]
        df[f"historico_trocas_peca_{i+1}"] = trocas[:, i]
    hist = df.iloc[:n_historico].copy().reset_index(drop=True)
    hist.insert(0, "id_maquina", [f"H-{i:04d}" for i in range(1, n_historico + 1)])
    hist["peca_com_problema"] = [PECAS[i] for i in labels[:n_historico]]
    atual = df.iloc[n_historico:].copy().reset_index(drop=True)
    atual.insert(0, "id_maquina", [f"M-{i:03d}" for i in range(1, n_atuais + 1)])
    atual["criticidade_maquina"] = rng.integers(1, 6, n_atuais)
    # Indice de severidade tecnico simulado: nao e probabilidade de falha.
    sev = 42 + .42 * (atual["temperatura_media"].to_numpy() - 65) + 2.3 * (atual["nivel_de_vibracao"].to_numpy() - 5) + rng.normal(0, 9, n_atuais)
    atual["severidade_sensores"] = np.clip(np.rint(sev), 0, 100).astype(int)
    return hist, atual


def preparar_modelagem(hist):
    """Split antes de aprender quaisquer parametros de pre-processamento."""
    X = hist[FEATURES]
    y = hist["peca_com_problema"]
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, random_state=SEED, stratify=y_dev)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop")
    # Ajustrar SOMENTE no treinamento; aplicar aos outros splits e à operação.
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_val)
    Xe = pre.transform(X_test)
    return pre, (Xt, Xv, Xe), (y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy())


def avaliar(y_true, y_pred):
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão macro": precision_score(y_true, y_pred, labels=PECAS, average="macro", zero_division=0),
        "Recall macro": recall_score(y_true, y_pred, labels=PECAS, average="macro", zero_division=0),
    }


def treinar_modelos(hist, epocas=48):
    pre, (Xt, Xv, Xe), (yt, yv, ye) = preparar_modelagem(hist)
    # Referencia linear e explicavel por coeficientes, nao uma segunda rede neural.
    referencia = LogisticRegression(max_iter=700, random_state=SEED)
    referencia.fit(Xt, yt)
    # MLP com duas camadas ocultas, ReLU, saida softmax implicita no sklearn.
    rede = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                         alpha=0.015, learning_rate_init=0.0018,
                         batch_size=64, max_iter=1, warm_start=False, random_state=SEED)
    classes = np.asarray(PECAS)
    trilha = []
    best_loss = float("inf")
    best_model = None
    patience = 9
    espera = 0
    for ep in range(1, epocas + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rede.partial_fit(Xt, yt, classes=classes if ep == 1 else None)
        probs_train = rede.predict_proba(Xt)
        probs_val = rede.predict_proba(Xv)
        lt = log_loss(yt, probs_train, labels=rede.classes_)
        lv = log_loss(yv, probs_val, labels=rede.classes_)
        trilha.append({"Época": ep, "Loss treino": lt, "Loss validação": lv})
        if lv < best_loss - 0.0005:
            best_loss = lv
            best_model = copy.deepcopy(rede)
            espera = 0
        else:
            espera += 1
        if espera >= patience:
            break
    rede = best_model
    preditos = {"Regressão logística": referencia.predict(Xe), "Rede neural MLP": rede.predict(Xe)}
    resultados = pd.DataFrame([{"Modelo": m, **avaliar(ye, p)} for m, p in preditos.items()])
    relatorio = classification_report(ye, preditos["Rede neural MLP"], labels=PECAS,
                                     output_dict=True, zero_division=0)
    por_peca = pd.DataFrame([{"Peça": p, "Suporte": int(relatorio[p]["support"]),
                              "Precisão": relatorio[p]["precision"], "Recall": relatorio[p]["recall"]} for p in PECAS])
    return {"pre": pre, "rede": rede, "baseline": referencia, "resultados": resultados,
            "yt": ye, "yp": preditos["Rede neural MLP"], "por_peca": por_peca,
            "trilha": pd.DataFrame(trilha), "epocas": len(trilha), "test_size": len(ye)}


def prever_atual(modelos, atual):
    X = modelos["pre"].transform(atual[FEATURES])
    probs = modelos["rede"].predict_proba(X)
    out = atual.copy()
    ordered_classes = list(modelos["rede"].classes_)
    for p in PECAS:
        out[f"escore_peca_{PECAS.index(p) + 1}"] = probs[:, ordered_classes.index(p)]
    out["peca_predita"] = np.asarray(ordered_classes)[probs.argmax(axis=1)]
    out["maior_escore"] = probs.max(axis=1)
    out["escore_incerteza"] = 1 - out["maior_escore"]
    # Politica gerencial explicita, distinta da rede neural: nao e probabilidade de falha.
    out["prioridade_operacional"] = (out["criticidade_maquina"] * 20 + out["severidade_sensores"] * .35).round(1)
    out = out.sort_values(["prioridade_operacional", "maior_escore"], ascending=[False, True]).reset_index(drop=True)
    return out


def csv_bytes(df):
    return df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")


st.set_page_config(page_title="FGV | Oficina final de Deep Learning", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.45rem;max-width:1220px}
h1,h2,h3{color:#173E70}
[data-testid="stMetricValue"]{font-size:1.65rem}
</style>""", unsafe_allow_html=True)
st.title("Oficina Final — Deep Learning na Manutenção Industrial")
st.caption("FGV | Caso fictício MetalFort • Rede neural MLP com duas camadas ocultas • Classificação de cinco peças")
st.info("**Missão executiva:** a MetalFort precisa priorizar inspeções de máquinas já selecionadas para manutenção preventiva. "
        "Escolham até **20 máquinas**, interpretem o desempenho da rede neural e justifiquem os riscos da decisão.")
st.warning("**Premissa:** todas as máquinas desta oficina já exigem manutenção. O modelo prevê **qual peça** é candidata a inspeção, "
           "e não se a máquina terá uma falha. Os escores da rede não representam probabilidades validadas de falha.")

with st.sidebar:
    st.header("Configuração da oficina")
    st.write("Base histórica: **3.000** observações")
    st.write("Máquinas atuais: **60**")
    st.write("Peças possíveis: **5**")
    epocas = st.slider("Máximo de épocas do treinamento", min_value=20, max_value=70, value=48, step=2,
                       help="A rede pode parar antes se a perda de validação não melhorar.")
    st.caption("Divisão: 1.800 treino / 600 validação / 600 teste. Semente fixa para comparar os resultados.")
    if st.button("Reiniciar resultados da oficina"):
        for k in ["pacote_modelo", "epocas_treinadas", "selecionadas", "parecer"]:
            st.session_state.pop(k, None)
        st.rerun()

hist, atuais = gerar_dados()
t1, t2, t3, t4 = st.tabs(["1. Dados", "2. Rede neural e métricas", "3. Priorizar inspeções", "4. Decisão executiva"])

with t1:
    st.subheader("Problema e objetivo empresarial")
    st.markdown("""**Problema:** falhas inesperadas em componentes críticos prejudicam a continuidade da produção,
    elevam os custos de manutenção e atrasam encomendas. A equipe não consegue identificar com segurança
    qual das cinco peças exige maior atenção nas máquinas já indicadas para inspeção.

    **Objetivo:** reduzir em **25% os custos mensais de manutenção não planejada em seis meses**, considerando
    cada máquina selecionada para manutenção como unidade de decisão. O alvo do modelo é **a peça com problema nos próximos sete dias**.
    A meta financeira exige um piloto operacional posterior; não é comprovada pela métrica do classificador.
    """)
    a, b, c = st.columns(3)
    a.metric("Registros históricos", len(hist))
    b.metric("Máquinas atuais", len(atuais))
    c.metric("Categorias de peças", len(PECAS))
    st.write("**Distribuição de peças nos registros históricos**")
    st.bar_chart(hist["peca_com_problema"].value_counts().reindex(PECAS))
    st.write("**Exemplo da base rotulada — treinamento**")
    st.dataframe(hist.head(8), use_container_width=True, hide_index=True)
    st.write("**Exemplo das máquinas atuais — sem o rótulo**")
    st.dataframe(atuais.head(8), use_container_width=True, hide_index=True)
    ca, cb = st.columns(2)
    with ca: st.download_button("Baixar base histórica (CSV)", csv_bytes(hist), "metalf_historico.csv", "text/csv")
    with cb: st.download_button("Baixar máquinas atuais (CSV)", csv_bytes(atuais), "metalf_maquinas_atuais.csv", "text/csv")
    st.caption("Dados gerados internamente com padrões sintéticos não lineares. Não representam sensores ou falhas reais.")

with t2:
    st.subheader("Treinamento de uma rede neural real")
    st.markdown("**Arquitetura:** atributos numéricos e categóricos → 64 neurônios (ReLU) → 32 neurônios (ReLU) "
                "→ 5 saídas (*Softmax* para classificação multiclasse). O modelo de comparação é uma regressão logística multiclasse.")
    st.caption("Para tornar a oficina leve no Streamlit Cloud, a rede utiliza MLPClassifier/scikit-learn, "
               "uma rede neural genuína com duas camadas ocultas.")
    if st.button("Treinar e avaliar modelos", type="primary"):
        with st.spinner("Treinando rede neural e comparando com o modelo de referência..."):
            st.session_state["pacote_modelo"] = treinar_modelos(hist, epocas=epocas)
            st.session_state["epocas_treinadas"] = epocas
            st.session_state.pop("selecionadas", None)
    if "pacote_modelo" in st.session_state:
        mod = st.session_state["pacote_modelo"]
        st.success(f"Treinamento concluído em {mod['epocas']} épocas. Conjunto de teste reservado: {mod['test_size']} observações.")
        st.write("**Comparação em dados de teste que não orientaram a escolha das configurações**")
        tabela = mod["resultados"].copy()
        for col in ["Acurácia", "Precisão macro", "Recall macro"]:
            tabela[col] = tabela[col].map(lambda x: f"{x:.1%}")
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.caption("Métricas macro calculam a média simples entre as cinco classes. O resultado não demonstra o atingimento da meta financeira.")
        la, lb = st.columns([1.15, 1])
        with la:
            st.write("**Evolução da perda durante o aprendizado**")
            dfcurva = mod["trilha"].set_index("Época")
            st.line_chart(dfcurva[["Loss treino", "Loss validação"]])
            st.caption("A rede preserva o ponto com menor loss de validação. A curva ajuda a discutir generalização e sobreajuste.")
        with lb:
            st.write("**Matriz de confusão — rede neural no teste**")
            cm = confusion_matrix(mod["yt"], mod["yp"], labels=PECAS)
            fig, ax = plt.subplots(figsize=(5.5, 4.4))
            ax.imshow(cm, cmap="Blues", interpolation="nearest")
            ax.set_xticks(range(5), PECAS, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(5), PECAS, fontsize=8)
            ax.set_xlabel("Peça prevista", fontsize=9)
            ax.set_ylabel("Peça real", fontsize=9)
            ax.tick_params(axis="both", length=0)
            threshold_color = cm.max() * .52
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if cm[i, j] > threshold_color else "#163657")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        st.write("**Desempenho da rede neural por peça**")
        pp = mod["por_peca"].copy()
        pp["Precisão"] = pp["Precisão"].map(lambda x: f"{x:.1%}")
        pp["Recall"] = pp["Recall"].map(lambda x: f"{x:.1%}")
        st.dataframe(pp, use_container_width=True, hide_index=True)
    else:
        st.info("Clique em **Treinar e avaliar modelos** para liberar as previsões e a decisão executiva.")

with t3:
    st.subheader("Quais 20 máquinas a equipe deve inspecionar primeiro?")
    if "pacote_modelo" not in st.session_state:
        st.info("Treine os modelos na aba 2 antes de priorizar as inspeções.")
    else:
        prev = prever_atual(st.session_state["pacote_modelo"], atuais)
        st.markdown("**Atenção:** uma previsão com maior escore não significa que a máquina possui maior probabilidade de falhar. "
                    "A priorização deve considerar **criticidade da máquina, severidade dos sensores e custo potencial**.")
        st.write("**Critério sugerido de triagem (não é saída da IA):** 20 × criticidade da máquina + "
                 "0,35 × índice de severidade dos sensores. Use esse ranking como apoio, não como ordem automática de manutenção.")
        st.caption("Custos de parada por componente são hipotéticos e servem apenas para discussão gerencial: "
                   + "; ".join([f"{p}: R$ {CUSTO_PECAS[p]:,.0f}" for p in PECAS]).replace(",", ".") + ".")
        out = prev.copy()
        out["custo_parada_peca"] = out["peca_predita"].map(CUSTO_PECAS)
        out["maior_escore"] = (out["maior_escore"] * 100).round(1)
        cols_vis = ["id_maquina", "peca_predita", "maior_escore", "criticidade_maquina",
                    "severidade_sensores", "custo_parada_peca", "prioridade_operacional"]
        st.dataframe(out[cols_vis].rename(columns={"maior_escore": "Escore maior (%)"}),
                     use_container_width=True, hide_index=True, height=380)
        sugestoes = out.head(20)["id_maquina"].tolist()
        selecionadas = st.multiselect("Selecione até 20 máquinas para inspeção (a sugestão inicial usa a triagem operacional)",
                                      options=out["id_maquina"].tolist(), default=sugestoes,
                                      key="selecionadas")
        if len(selecionadas) > 20:
            st.error("Selecione no máximo 20 máquinas.")
        else:
            result_sel = out[out["id_maquina"].isin(selecionadas)].copy()
            ca, cb, cc = st.columns(3)
            ca.metric("Máquinas selecionadas", len(selecionadas))
            cb.metric("Maior criticidade (4–5)", int(result_sel["criticidade_maquina"].ge(4).sum()))
            cc.metric("Casos com escore < 55%", int(result_sel["maior_escore"].lt(55).sum()))
            st.download_button("Baixar plano de inspeção (CSV)", csv_bytes(result_sel[cols_vis]),
                               "metalf_prioridade_inspecoes.csv", "text/csv")
            st.write("**Examine uma máquina e as cinco saídas do modelo**")
            example_id = st.selectbox("Máquina", out["id_maquina"].tolist(), index=0)
            linha = prev.loc[prev["id_maquina"] == example_id].iloc[0]
            scores = pd.DataFrame({"Peça": PECAS, "Escore": [linha[f"escore_peca_{i}"] for i in range(1, 6)]})
            st.bar_chart(scores.set_index("Peça"))
            st.caption("Os escores somam 100%, mas não são probabilidades de falha calibradas. "
                       "O sistema não demonstra qual peça causará falha em uma máquina que não foi triada.")

with t4:
    st.subheader("Comitê Executivo — parecer final")
    st.write("A equipe deve justificar a decisão considerando **desempenho, custo dos erros, explicabilidade e governança**.")
    if "pacote_modelo" not in st.session_state:
        st.info("Treine os modelos na aba 2 para realizar a decisão final.")
    else:
        st.markdown("""**Perguntas para discussão em equipe**
        1. Por que esse é um problema supervisionado e multiclasse? Qual é o alvo, a unidade e o horizonte?
        2. A rede neural superou o modelo simples no teste? O ganho justifica maior complexidade?
        3. Que peça apresentou o menor recall? Qual consequência isso pode gerar na operação?
        4. Quais máquinas a equipe inspecionará primeiro? Por que o maior escore Softmax não basta?
        5. Quais controles de explicabilidade, validação e supervisão seriam exigidos antes de um piloto?
        """)
        decisao = st.radio("Decisão recomendada", ["Autorizar somente um piloto supervisionado",
                                                      "Solicitar melhorias antes de testar na fábrica",
                                                      "Não recomendar a implantação neste momento"], index=0)
        justificativa = st.text_area("Justificativa executiva (3–6 linhas)", height=115,
                                     placeholder="Considere indicadores técnicos, impactos empresariais e limites do estudo...")
        salvaguardas = st.text_area("Duas salvaguardas obrigatórias", height=90,
                                    placeholder="Ex.: inspeção humana para casos críticos; monitorar recall por peça...")
        texto = f"OFICINA FINAL — METALFORT\nDECISÃO: {decisao}\n\nJUSTIFICATIVA:\n{justificativa}\n\nSALVAGUARDAS:\n{salvaguardas}\n"
        st.download_button("Baixar parecer executivo (TXT)", texto.encode("utf-8"),
                           "metalf_parecer_executivo.txt", "text/plain")
        st.caption("A aprovação de um piloto não equivale à autorização para manutenção autônoma nem confirma economia de 25%.")

st.divider()
st.caption("Uso didático. Observações, rótulos, indicadores, custos e classes são simulados. "
           "O resultado do treinamento não comprova causalidade, confiabilidade operacional nem conformidade para uso industrial real.")
