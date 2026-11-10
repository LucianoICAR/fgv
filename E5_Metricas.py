"""Oficina de Métricas — SoftPlus | FGV Educação Executiva.

Publicação: streamlit run metrica.py
Base sintética interna, sem arquivos adicionais.
Este aplicativo simula ESCORES de risco, não treina um classificador.
"""

import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SoftPlus | Oficina de Métricas",
    page_icon="🎯",
    layout="wide",
)

AZUL = "#164A7B"
AZUL_CLARO = "#EAF3FA"
VERDE = "#177A58"
LARANJA = "#BA6A1C"


@st.cache_data

def criar_lote(qtd: int, seed: int, prefixo: str) -> pd.DataFrame:
    """Casos hipotéticos, com cancelamento observado e escore associado.

    Os escores NÃO são probabilidades calibradas e NÃO vieram de modelo treinado.
    """
    rng = np.random.default_rng(seed)
    n_positivos = int(qtd * 0.20)
    n_negativos = qtd - n_positivos
    positivos = rng.beta(3.8, 2.8, size=n_positivos)
    negativos = rng.beta(1.8, 4.5, size=n_negativos)
    dados = pd.DataFrame({
        "Cliente": [f"{prefixo}-{i:03d}" for i in range(1, qtd + 1)],
        "Cancelou (real)": np.concatenate([
            np.ones(n_positivos, dtype=int),
            np.zeros(n_negativos, dtype=int),
        ]),
        "Escore de risco": np.concatenate([positivos, negativos]).round(4),
    })
    return dados.sample(frac=1, random_state=seed).reset_index(drop=True)


def calcular_metricas(base: pd.DataFrame, limiar: float) -> dict:
    real = base["Cancelou (real)"].to_numpy(dtype=int)
    alertado = (base["Escore de risco"].to_numpy() >= limiar).astype(int)
    vp = int(((real == 1) & (alertado == 1)).sum())
    fp = int(((real == 0) & (alertado == 1)).sum())
    fn = int(((real == 1) & (alertado == 0)).sum())
    vn = int(((real == 0) & (alertado == 0)).sum())
    return {
        "VP": vp, "FP": fp, "FN": fn, "VN": vn,
        "alertas": vp + fp,
        "precisao": (vp / (vp + fp)) if (vp + fp) else None,
        "recall": (vp / (vp + fn)) if (vp + fn) else None,
        "acuracia": (vp + vn) / len(base),
        "previsto": alertado,
    }


def pct(valor, casas=1):
    if valor is None or pd.isna(valor):
        return "N/D"
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def moeda(valor):
    return "R$ " + f"{valor:,.0f}".replace(",", ".")


def resumir_politica(base: pd.DataFrame, limiar: float, limite_contatos: int,
                    custo_acao: float, margem_evitada: float, eficacia: float) -> dict:
    """Métricas de alerta ≠ resultado das ações. Contatos: escores mais altos."""
    metricas = calcular_metricas(base, limiar)
    todos = base.copy()
    todos["Alerta"] = metricas["previsto"]
    aptos = todos.loc[todos["Alerta"].eq(1)].sort_values(
        ["Escore de risco", "Cliente"], ascending=[False, True]
    )
    atendidos = aptos.head(limite_contatos)
    n_contatos = len(atendidos)
    canceladores_contatados = int(atendidos["Cancelou (real)"].sum())
    n_canceladores = int(todos["Cancelou (real)"].sum())
    gasto = n_contatos * custo_acao
    valor_potencial = canceladores_contatados * eficacia * margem_evitada
    liquido = valor_potencial - gasto

    # Comparador: selecionar a mesma quantidade de pessoas sem informação de risco.
    # É uma expectativa matemática, NÃO uma campanha randomizada executada.
    taxa_base = n_canceladores / len(base)
    acertos_aleatorios_esperados = n_contatos * taxa_base
    valor_aleatorio = acertos_aleatorios_esperados * eficacia * margem_evitada
    liquido_aleatorio = valor_aleatorio - gasto

    todos["Selecionado para contato"] = todos["Cliente"].isin(atendidos["Cliente"])
    todos["Resultado da classificação"] = np.select(
        [
            (todos["Alerta"] == 1) & (todos["Cancelou (real)"] == 1),
            (todos["Alerta"] == 1) & (todos["Cancelou (real)"] == 0),
            (todos["Alerta"] == 0) & (todos["Cancelou (real)"] == 1),
        ],
        ["VP", "FP", "FN"],
        default="VN",
    )
    return {
        **metricas,
        "contatos": n_contatos,
        "nao_atendidos": int(metricas["alertas"] - n_contatos),
        "canceladores_contatados": canceladores_contatados,
        "gasto": gasto,
        "valor_potencial": valor_potencial,
        "liquido": liquido,
        "valor_aleatorio": valor_aleatorio,
        "liquido_aleatorio": liquido_aleatorio,
        "taxa_base": taxa_base,
        "clientes": todos,
    }


def grafico_matriz(metricas: dict):
    """Eixo vertical: alerta da IA; horizontal: resultado real."""
    valores = np.array([[metricas["VP"], metricas["FP"]],
                        [metricas["FN"], metricas["VN"]]])
    # Rótulos menores e abreviados para evitar sobreposição em telas pequenas.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.imshow([[0, 1], [1, 0]], cmap="RdYlGn_r", vmin=0, vmax=1)
    titulos = [["VP — correto", "FP — indevido"],
               ["FN — não detectado", "VN — correto"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{titulos[i][j]}\n{valores[i, j]} clientes",
                    fontsize=8.5, ha="center", va="center", color="#162F42", fontweight="bold")
    ax.set_xticks([0, 1], ["Cancelaria", "Permaneceria"])
    ax.set_yticks([0, 1], ["Alerta", "Sem alerta"])
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlabel("Resultado real observado", fontsize=9)
    ax.set_ylabel("Classificação pelo limiar", fontsize=9)
    fig.tight_layout()
    return fig


def grafico_curvas(base: pd.DataFrame, escolhido: float):
    limiares = np.round(np.arange(0, 1.001, 0.01), 2)
    resultados = [calcular_metricas(base, float(t)) for t in limiares]
    fig, ax = plt.subplots(figsize=(10.3, 3.3))
    for chave, nome, cor, estilo in [
        ("precisao", "Precisão", AZUL, "-"),
        ("recall", "Recall", VERDE, "-"),
        ("acuracia", "Acurácia", LARANJA, "--"),
    ]:
        ax.plot(limiares, [v[chave] if v[chave] is not None else np.nan for v in resultados],
                label=nome, color=cor, lw=2.2, ls=estilo)
    ax.axvline(escolhido, color="#6B7785", ls=":", lw=1.6,
               label=f"Limiar atual ({escolhido:.2f})")
    ax.set(xlabel="Limiar de decisão", ylabel="Proporção", xlim=(0, 1), ylim=(0, 1.04))
    ax.grid(alpha=0.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9)
    fig.tight_layout()
    return fig


BASE_VALIDACAO = criar_lote(qtd=300, seed=2026, prefixo="VAL")
BASE_TESTE = criar_lote(qtd=200, seed=2027, prefixo="TES")

st.title("🎯 Oficina de Métricas — SoftPlus")
st.markdown("**Desafio:** decidir quais clientes receberão ações de retenção com base em escores de risco e em um orçamento limitado.")
st.info(
    "**Experimento inteiramente fictício:** 500 clientes, sendo **300 registros históricos para a oficina** "
    "e **200 registros reservados para o teste final**. Em cada conjunto, 20% cancelariam. "
    "Os escores são *simulados*, não foram produzidos por um modelo realmente treinado "
    "e não devem ser interpretados como probabilidades calibradas."
)

st.sidebar.header("1. Política de decisão")
limiar = st.sidebar.slider(
    "Threshold — limiar para emitir alerta", min_value=0.0,
    max_value=1.0, value=0.50, step=0.01,
    help="Alerta quando o escore de risco é maior ou igual ao limiar."
)
st.sidebar.caption("Limiar mais baixo tende a emitir mais alertas. Aqui você modifica a política de decisão, NÃO treina o modelo.")

st.sidebar.header("2. Recursos da campanha")
orçamento = st.sidebar.number_input("Orçamento total (R$)", min_value=0, value=5000, step=500)
custo = st.sidebar.number_input("Custo por cliente contatado (R$)", min_value=0, value=200, step=50)
capacidade = st.sidebar.number_input("Capacidade da equipe (clientes)", min_value=0, value=25, step=5)
max_por_orcamento = (orçamento // custo) if custo > 0 else len(BASE_VALIDACAO)
limite_contatos = int(min(capacidade, max_por_orcamento))

st.sidebar.header("3. Hipóteses econômicas")
valor = st.sidebar.number_input(
    "Margem futura estimada por cancelamento evitado (R$)",
    min_value=0, value=4000, step=500,
)
eficacia_pct = st.sidebar.slider(
    "Eficácia incremental hipotética da retenção (%)", 0, 100, 20, 5,
    help="Se um cliente que cancelaria for contatado, chance hipotética de o contato evitar o cancelamento. NÃO foi medida por experimento."
)
eficacia = eficacia_pct / 100.0
st.sidebar.caption("Essas hipóteses NÃO comprovam o efeito causal de uma campanha. A eficácia exigiria avaliação experimental.")

res = resumir_politica(BASE_VALIDACAO, limiar, limite_contatos, custo, valor, eficacia)

st.header("1. Diagnóstico do limiar selecionado")
a, b, c, d = st.columns(4)
a.metric("Precisão", pct(res["precisao"]), help="VP / (VP + FP): percentual de alertas corretos.")
b.metric("Recall", pct(res["recall"]), help="VP / (VP + FN): proporção dos canceladores identificados.")
c.metric("Acurácia", pct(res["acuracia"]), help="(VP + VN) / 300.")
d.metric("Alertas emitidos", res["alertas"])

col1, col2 = st.columns([1.02, 1], gap="large")
with col1:
    st.subheader("Matriz de confusão — alertas")
    fig = grafico_matriz(res)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with col2:
    st.subheader("Interpretação empresarial")
    st.write(
        f"- **Precisão:** entre **{res['alertas']} alertas**, **{res['VP']}** se referem a clientes que cancelariam "
        f"e **{res['FP']}** a clientes que permaneceriam."
    )
    st.write(
        f"- **Recall:** dos **{res['VP'] + res['FN']} clientes que cancelariam**, "
        f"foram identificados **{res['VP']}**; ficaram sem alerta **{res['FN']}**."
    )
    st.write(
        f"- **Acurácia:** acertou **{res['VP'] + res['VN']} de {len(BASE_VALIDACAO)}** classificações."
    )
    st.caption("A matriz mede a **classificação**, não os resultados de ações de retenção.")

st.subheader("Efeito do threshold nas métricas")
fig = grafico_curvas(BASE_VALIDACAO, limiar)
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("O recall não aumenta ao elevar o threshold; a precisão pode oscilar. N/D indica que não houve alertas para calcular precisão.")

st.header("2. Acurácia alta basta? Política sem alertas")
base_rate = BASE_VALIDACAO["Cancelou (real)"].mean()
sem_ia_acuracia = 1 - base_rate
m1, m2, m3 = st.columns(3)
m1.metric("Acurácia: prever todos como 'não cancela'", pct(sem_ia_acuracia))
m2.metric("Recall dessa política", "0,0%")
m3.metric("Canceladores não identificados", int(BASE_VALIDACAO["Cancelou (real)"].sum()))
st.info(
    "Essa política acerta os clientes que permanecem, mas **não identifica nenhum cancelamento**. "
    "É uma referência simples para mostrar por que a acurácia isolada pode ser enganosa; "
    "não representa um sistema de IA nem uma política de retenção recomendada."
)

st.header("3. Restrições de orçamento e capacidade")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Limite de contatos", limite_contatos)
c2.metric("Alertas emitidos", res["alertas"])
c3.metric("Clientes que receberiam contato", res["contatos"])
c4.metric("Alertados sem atendimento", res["nao_atendidos"])
if res["nao_atendidos"]:
    st.warning(
        f"Há **{res['alertas']} alertas**, mas só **{limite_contatos} contatos** são possíveis. "
        f"O simulador seleciona os **{res['contatos']} maiores escores** dentre os alertas. "
        "As métricas acima continuam se referindo a TODOS os alertas, e não apenas aos clientes contatados."
    )
else:
    st.success("Todos os alertas podem ser atendidos com o orçamento e a equipe disponíveis.")

st.subheader("Comparação de limiares — mesma base de 300 clientes")
comparacoes = []
for t in [0.30, 0.50, 0.70]:
    v = resumir_politica(BASE_VALIDACAO, t, limite_contatos, custo, valor, eficacia)
    comparacoes.append({
        "Threshold": f"{t:.2f}".replace(".", ","),
        "Alertas": v["alertas"], "Contatados": v["contatos"],
        "Precisão": pct(v["precisao"]), "Recall": pct(v["recall"]),
        "Acurácia": pct(v["acuracia"]),
        "Custo das ações": moeda(v["gasto"]),
        "Líquido hipotético": moeda(v["liquido"]),
    })
st.dataframe(pd.DataFrame(comparacoes), hide_index=True, use_container_width=True)

st.header("4. Simulação econômica — hipótese, não promessa")
a, b, c, d = st.columns(4)
a.metric("Gasto em ações", moeda(res["gasto"]))
b.metric("Canceladores contatados (histórico)", res["canceladores_contatados"])
c.metric("Valor potencial evitado (hipotético)", moeda(res["valor_potencial"]))
d.metric("Saldo incremental hipotético", moeda(res["liquido"]))

st.markdown(
    f"**Como a conta funciona:** {res['canceladores_contatados']} clientes contatados cancelariam "
    f"× **{eficacia_pct}%** de eficácia *hipotética* × **{moeda(valor)}** de margem preservada "
    f"− **{moeda(res['gasto'])}** de custos = **{moeda(res['liquido'])}**."
)
col_r, col_s = st.columns(2)
with col_r:
    st.metric("Sem campanha — benefício incremental", moeda(0))
    st.caption("Referência financeira: não realizar nenhum contato de retenção.")
with col_s:
    st.metric("Contatos aleatórios — saldo esperado", moeda(res["liquido_aleatorio"]))
    st.caption("Mesma quantidade e custo de contatos, mas sem priorização por risco. Valor esperado matemático, não experimento realizado.")
st.warning(
    "**Limite da simulação:** os dados históricos permitem saber quem cancelou, mas não demonstram "
    "quem teria sido salvo por uma abordagem. A taxa de eficácia é uma hipótese uniforme; "
    "o efeito real deve ser medido em um piloto controlado (por exemplo, com grupo de controle). "
    "Não se pode concluir, apenas com esse gráfico, que a campanha gera lucro."
)

with st.expander("Ver registros de VALIDAÇÃO e baixar CSV de estudo"):
    st.caption("Os resultados reais são históricos e aparecem para que a turma estude os erros.")
    st.dataframe(res["clientes"], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Baixar resultados da base de validação",
        data=res["clientes"].to_csv(index=False).encode("utf-8-sig"),
        file_name="softplus_validacao.csv", mime="text/csv",
    )

st.header("5. Decisão do Comitê Executivo")
st.markdown(
    "**Discuta em equipe:** (1) Por que a maior acurácia não basta? "
    "(2) Qual é o custo de FP e FN? "
    "(3) Qual threshold é defensável com recursos limitados? "
    "(4) Quais evidências faltam para aprovar um piloto?"
)

if "decisao_registrada" not in st.session_state:
    st.session_state.decisao_registrada = None

if st.session_state.decisao_registrada is None:
    with st.form("parecer_executivo"):
        equipe = st.text_input("Identificação da equipe", placeholder="Ex.: Equipe 3")
        justificativa = st.text_area(
            "Por que esse threshold? Utilize métricas, alertas, orçamento e capacidade.",
            height=110, placeholder="Descreva os trade-offs..."
        )
        riscos = st.text_area(
            "Riscos dos falsos positivos e dos falsos negativos para a SoftPlus",
            height=90,
        )
        evidencias = st.text_area(
            "Duas evidências adicionais necessárias antes da implantação",
            height=90, placeholder="Ex.: resultado de teste independente; eficácia da campanha com grupo de controle."
        )
        recomendacao = st.selectbox(
            "Recomendação executiva",
            ["Autorizar piloto controlado", "Solicitar mais evidências",
             "Não aprovar a implantação", "Recomendar implantação plena"],
        )
        enviar = st.form_submit_button("🔒 Registrar decisão e abrir teste reservado", type="primary")
        if enviar:
            if not all(x.strip() for x in [equipe, justificativa, riscos, evidencias]):
                st.error("Preencha a equipe e os três campos de argumentação antes de abrir o teste.")
            else:
                st.session_state.decisao_registrada = {
                    "equipe": equipe.strip(),
                    "threshold": float(limiar),
                    "orçamento": float(orçamento),
                    "custo": float(custo),
                    "capacidade": int(capacidade),
                    "limite_contatos": limite_contatos,
                    "margem": float(valor),
                    "eficacia": float(eficacia),
                    "justificativa": justificativa.strip(),
                    "riscos": riscos.strip(),
                    "evidencias": evidencias.strip(),
                    "recomendacao": recomendacao,
                    "registrado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
                }
                st.rerun()

if st.session_state.decisao_registrada is not None:
    d = st.session_state.decisao_registrada
    st.success(
        f"Decisão registrada para **{d['equipe']}**: threshold **{d['threshold']:.2f}**, "
        f"recomendação **{d['recomendacao']}**. A avaliação final utiliza EXATAMENTE essa configuração."
    )
    st.subheader("Teste final independente — 200 clientes reservados")
    teste = resumir_politica(
        BASE_TESTE, d["threshold"], d["limite_contatos"],
        d["custo"], d["margem"], d["eficacia"],
    )
    ref = calcular_metricas(BASE_VALIDACAO, d["threshold"])
    tabela = pd.DataFrame([
        {"Base": "Validação (300)", "Precisão": pct(ref["precisao"]),
         "Recall": pct(ref["recall"]), "Acurácia": pct(ref["acuracia"]),
         "Alertas": ref["alertas"]},
        {"Base": "Teste reservado (200)", "Precisão": pct(teste["precisao"]),
         "Recall": pct(teste["recall"]), "Acurácia": pct(teste["acuracia"]),
         "Alertas": teste["alertas"]},
    ])
    st.dataframe(tabela, hide_index=True, use_container_width=True)
    st.markdown(
        "**Pergunta final:** sua recomendação inicial se mantém ao observar novos dados? "
        "Que evidências justificariam novos experimentos?"
    )
    st.warning(
        "**Regra metodológica:** não use o conjunto de teste final para escolher repetidamente outros limiares. "
        "A decisão foi fixada antes da abertura. Alterações nos controles laterais não alteram este teste registrado."
    )

    parecer = f"""FGV — OFICINA DE MÉTRICAS | SOFTPLUS

Equipe: {d['equipe']}
Registro: {d['registrado_em']}
Recomendação: {d['recomendacao']}

POLÍTICA REGISTRADA
Threshold: {d['threshold']:.2f}
Orçamento: {moeda(d['orçamento'])}
Custo por contato: {moeda(d['custo'])}
Capacidade: {d['capacidade']} clientes
Limite efetivo: {d['limite_contatos']} contatos
Eficácia HIPOTÉTICA da ação: {pct(d['eficacia'])}
Margem futura estimada por cancelamento evitado: {moeda(d['margem'])}

VALIDAÇÃO (300 clientes)
Precisão: {pct(ref['precisao'])}
Recall: {pct(ref['recall'])}
Acurácia: {pct(ref['acuracia'])}
VP={ref['VP']}; FP={ref['FP']}; FN={ref['FN']}; VN={ref['VN']}

TESTE RESERVADO (200 clientes)
Precisão: {pct(teste['precisao'])}
Recall: {pct(teste['recall'])}
Acurácia: {pct(teste['acuracia'])}
VP={teste['VP']}; FP={teste['FP']}; FN={teste['FN']}; VN={teste['VN']}

JUSTIFICATIVA
{d['justificativa']}

RISCOS
{d['riscos']}

EVIDÊNCIAS NECESSÁRIAS
{d['evidencias']}

OBSERVAÇÕES METODOLÓGICAS
Base sintética. Escores simulados, sem treinamento real e sem calibração.
A eficácia da retenção não é demonstrada pelos rótulos de churn.
Este exercício é apenas didático; não recomenda decisões para clientes reais.
"""
    st.download_button(
        "⬇️ Baixar parecer executivo da equipe (.txt)",
        data=parecer.encode("utf-8-sig"),
        file_name="softplus_parecer_executivo.txt", mime="text/plain",
        type="primary",
    )
    st.caption("A decisão e a configuração do teste ficam fixas nesta sessão da aplicação.")
