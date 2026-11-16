"""FGV — Atividade 9: AgroVerde, aprendizado por reforço com Q-Learning real.

Publicação Streamlit Cloud: arquivo principal app.py; manter simulador.py no mesmo diretório.
"""

import numpy as np
import pandas as pd
import streamlit as st

from simulador import (
    ACOES_MM, Pesos, estado, treinar_q_learning,
    avaliar_politicas, avaliar_meta,
)

st.set_page_config(page_title="FGV | Atividade 9 — AgroVerde", page_icon="🌱", layout="wide")

st.title("🌱 AgroVerde | Oficina de Aprendizado por Reforço")
st.markdown("**Atividade 9 — Formação Executiva em IA (FGV)** · Q-Learning aplicado à irrigação simulada")
st.info(
    "**Problema:** a irrigação por horários fixos aumenta o consumo de água e energia e pode "
    "prejudicar a cultura. **Objetivo empresarial:** reduzir em **20%** a água utilizada por talhão "
    "ao longo de uma safra de **180 dias**, sem piorar o indicador simulado de produtividade "
    "e controlando o risco de estresse hídrico grave."
)
st.caption(
    "Atenção: é um simulador DIDÁTICO com clima, solo, recompensas e resultados sintéticos. "
    "Não estima produtividade agrícola real nem autoriza controle físico de irrigação."
)

with st.expander("📘 Como o agente aprende (leitura de 1 minuto)"):
    st.markdown(
        "**Agente:** controlador da irrigação. **Ambiente:** talhão e condições meteorológicas simuladas. "
        "**Estado:** umidade do solo, previsão de chuva e demanda de água. "
        "**Ações:** não irrigar, aplicar 6 mm ou aplicar 12 mm. "
        "**Recompensa:** penaliza água consumida e estresse da cultura, com penalidade extra para estresse grave. "
        "**Política:** ação escolhida para cada estado após o treinamento.\n\n"
        "O sistema utiliza **Q-Learning tabular de verdade**: durante centenas de episódios, atualiza "
        "valores Q(estado, ação), experimenta alternativas (**exploração**) e passa gradualmente a "
        "usar ações mais promissoras (**explotação**). A comparação final utiliza **12 novas safras simuladas**, "
        "não utilizadas no treinamento. Como os alunos podem reusar essas safras ao ajustar estratégias, "
        "elas não equivalem a um teste final totalmente independente."
    )

st.sidebar.header("1. Prioridades da diretoria")
opcoes = (
    "Equilíbrio inicial",
    "Economia a qualquer custo",
    "Proteção da cultura",
    "Personalizado",
)
modo = st.sidebar.radio("Estratégia de recompensa", opcoes, index=0)
if modo == "Equilíbrio inicial":
    pesos = Pesos(agua=4, estresse=12)
elif modo == "Economia a qualquer custo":
    pesos = Pesos(agua=20, estresse=2)
elif modo == "Proteção da cultura":
    pesos = Pesos(agua=2, estresse=18)
else:
    agua = st.sidebar.slider("Penalidade pelo consumo de água", 1, 20, 4, 1)
    estresse = st.sidebar.slider("Penalidade pelo estresse hídrico", 1, 25, 12, 1)
    pesos = Pesos(agua=float(agua), estresse=float(estresse))
st.sidebar.caption(
    f"Peso relativo da água: **{pesos.agua:.0f}** · "
    f"Peso relativo do estresse: **{pesos.estresse:.0f}**. "
    "São escolhas gerenciais da função de recompensa, não valores agronômicos."
)

with st.sidebar.expander("Configuração de treinamento (opcional)"):
    episodios = st.slider("Episódios de treinamento", 120, 440, 260, 20)
    opcao_exp = st.select_slider(
        "Exploração inicial", options=["Baixa", "Equilibrada", "Alta"], value="Equilibrada"
    )
    exploracoes = {"Baixa": 0.20, "Equilibrada": 0.55, "Alta": 0.85}
    epsi = exploracoes[opcao_exp]
    st.caption("A exploração cai gradualmente durante o treino. No teste final, o agente usa sua política aprendida.")

st.sidebar.divider()
st.sidebar.markdown("**Política tradicional de referência**: irrigar 8 mm a cada 2 dias, sem consultar o estado do solo.")
st.sidebar.caption("As duas políticas recebem o mesmo clima em cada safra de teste.")

st.subheader("1. Treinar uma política e comparar as alternativas")
st.write(
    "Escolha uma estratégia de recompensa na lateral e clique no botão. "
    "O sistema treina um agente e o compara com a irrigação por calendário fixo."
)

config = (float(pesos.agua), float(pesos.estresse), int(episodios), float(epsi))
if st.button("▶️ Treinar o agente e testar em 12 safras", type="primary", use_container_width=True):
    with st.spinner("Treinando a política Q-Learning e avaliando novas safras simuladas..."):
        Q, curva = treinar_q_learning(pesos, episodios=episodios,
                                     exploracao_inicial=epsi, semente=2026)
        resultados, historico = avaliar_politicas(Q, pesos, safras=12, semente=7331)
        checagem = avaliar_meta(resultados)
    st.session_state.agro_resultado = {
        "Q": Q, "curva": curva, "resultados": resultados,
        "historico": historico, "checagem": checagem,
        "config": config, "modo": modo,
    }

if "agro_resultado" not in st.session_state:
    st.markdown("**Comece treinando uma das estratégias.** Após o treino, aparecerão os indicadores e a decisão executiva.")
    st.stop()

info = st.session_state.agro_resultado
if config != info["config"]:
    st.warning("Os controles foram alterados. Os resultados abaixo pertencem ao treinamento anterior; clique em **Treinar** para atualizá-los.")

val = info["checagem"]
base, rl = val["base"], val["rl"]

st.subheader("2. Resultados em 12 safras simuladas não vistas no treinamento")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Economia de água", f"{val['economia_pct']:.1f}%",
            help="Redução média relativa ao calendário fixo nos mesmos cenários meteorológicos.")
col2.metric("Água — Q-Learning", f"{rl['agua_m3_ha']:,.0f} m³/ha",
            delta=f"{rl['agua_m3_ha']-base['agua_m3_ha']:,.0f} m³/ha vs. calendário",
            delta_color="inverse")
col3.metric("Índice da cultura — Q-Learning", f"{rl['indice_cultura']:.1f}/100",
            delta=f"{rl['indice_cultura']-base['indice_cultura']:+.1f} pontos vs. calendário",
            delta_color="normal")
col4.metric("Dias de estresse grave", f"{rl['pct_criticos']:.1f}%",
            help="Percentual médio dos 180 dias com umidade do solo inferior a 20 mm.")

resumo = pd.DataFrame({
    "Indicador": ["Água por safra (m³/ha)", "Índice simulado da cultura (0–100)",
                  "Dias com estresse por safra", "Dias com estresse grave por safra"],
    "Calendário fixo": [base["agua_m3_ha"], base["indice_cultura"],
                        base["dias_estresse"], base["dias_criticos"]],
    "Q-Learning": [rl["agua_m3_ha"], rl["indice_cultura"],
                   rl["dias_estresse"], rl["dias_criticos"]],
}).round(1)
st.dataframe(resumo, hide_index=True, use_container_width=True)
st.caption(
    "Água = volume equivalente por hectare (1 mm de lâmina = 10 m³/ha). "
    "O índice da cultura é apenas uma PROXY ARTIFICIAL construída a partir de dias de estresse, "
    "não uma estimativa de produtividade real."
)

st.markdown("**Critérios de avaliação definidos no caso**")
criterios = [
    ("Reduzir água em pelo menos 20%", val["agua_ok"]),
    ("Não reduzir o índice simulado da cultura", val["cultura_ok"]),
    ("Manter estresse grave em até 5% dos dias E não pior que a referência", val["grave_ok"]),
]
for nome, ok in criterios:
    st.write(("✅ " if ok else "❌ ") + nome)
if val["meta_total"]:
    st.success("A estratégia atende aos três critérios DIDÁTICOS no teste simulado. Isso permite **considerar** um piloto controlado, não uma implantação automática.")
else:
    st.warning("A estratégia **não atende a todos os critérios**. Revise prioridades, recompensas e segurança antes de recomendar qualquer piloto.")

st.subheader("3. O que o agente aprendeu?")
aba1, aba2, aba3 = st.tabs(["Evolução do treinamento", "Comportamento da irrigação", "Política de decisão aprendida"])
with aba1:
    curva = info["curva"].copy()
    curva["Recompensa média móvel"] = curva["Recompensa do episódio"].rolling(20, min_periods=1).mean()
    st.line_chart(curva.set_index("Episódio")[["Recompensa média móvel"]], height=270)
    st.caption("Quanto maior (menos negativa) a recompensa acumulada, melhor o resultado segundo os pesos escolhidos. A evolução não precisa ser monotônica.")
    ini, fim = curva["Exploração (%)"].iloc[[0,-1]].to_list()
    st.write(f"Exploração (decisões experimentais): **{ini:.0f}%** no início → **{fim:.0f}%** ao final do treinamento.")
with aba2:
    his = info["historico"]
    umidade = his.pivot(index="Dia", columns="Política", values="Umidade final (mm)")
    st.line_chart(umidade, height=270)
    st.caption("Primeira safra do teste: umidade final por dia. A linha de estresse começa abaixo de 35 mm, e o estresse grave abaixo de 20 mm.")
    volume = his.pivot(index="Dia", columns="Política", values="Irrigação (mm)")
    st.line_chart(volume, height=220)
    st.caption("Irrigação diária (mm): o calendário segue uma programação fixa; o agente seleciona ações segundo o estado observado.")
with aba3:
    Q = info["Q"]
    exemplos = [(18,0,1),(30,0,1),(42,0,1),(42,1,0),(62,0,0),(75,1,1)]
    linhas = []
    for u,p,d in exemplos:
        acao = ACOES_MM[int(np.argmax(Q[estado(u,p,d)]))]
        linhas.append({"Umidade inicial (mm)":u, "Previsão de chuva": "Sim" if p else "Não",
                       "Demanda hídrica": "Alta" if d else "Normal", "Ação aprendida": f"{acao:g} mm"})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
    st.caption("Esta tabela é extraída da política Q efetivamente treinada; os resultados dependem das prioridades escolhidas.")

st.subheader("4. Comitê Executivo — decisão da equipe")
st.markdown(
    "**Respondam como executivos:** qual foi a relação entre economia de água e proteção da cultura? "
    "A função de recompensa promoveu o comportamento desejado? "
    "Quando explorar é aceitável e por que a exploração irrestrita seria perigosa numa fazenda real?"
)
with st.form("decisao_executiva"):
    equipe = st.text_input("Identificação da equipe (opcional)", placeholder="Equipe 1")
    decisao = st.radio("Recomendação", ["Solicitar ajustes e novos testes",
                                    "Autorizar somente um piloto controlado",
                                    "Não autorizar o uso operacional"], index=0)
    justificativa = st.text_area("Justificativa com dois indicadores da simulação",
                                  placeholder="Nossa equipe recomenda... porque o consumo de água foi..., e o índice da cultura foi...")
    cuidados = st.text_area("Duas condições de segurança e governança",
                            placeholder="Ex.: testar sensores no campo; exigir supervisão humana e limite de irrigação...")
    enviado = st.form_submit_button("Preparar parecer para download")

if enviado:
    parecer = (
        "FGV — ATIVIDADE 9 — PARECER EXECUTIVO | AGROVERDE\n"
        f"Equipe: {equipe or 'Não informada'}\n"
        f"Estratégia: {info['modo']}\n"
        f"Redução média de água: {val['economia_pct']:.1f}%\n"
        f"Água: calendário {base['agua_m3_ha']:.0f} / Q-Learning {rl['agua_m3_ha']:.0f} m3/ha\n"
        f"Índice da cultura (proxy): calendário {base['indice_cultura']:.1f} / Q-Learning {rl['indice_cultura']:.1f}\n"
        f"Estresse grave: calendário {base['pct_criticos']:.1f}% / Q-Learning {rl['pct_criticos']:.1f}% dos dias\n"
        f"Atendeu critérios didáticos: {'sim' if val['meta_total'] else 'não'}\n"
        f"Decisão: {decisao}\n"
        f"Justificativa: {justificativa}\n"
        f"Condições: {cuidados}\n"
        "Observação: resultados sintéticos, não constituem validação agronômica ou recomendação de campo.\n"
    )
    st.download_button("📄 Baixar parecer da equipe (.txt)", data=parecer.encode("utf-8-sig"),
                       file_name="agroverde_parecer_equipe.txt", mime="text/plain")
    st.info("Parecer preparado. A equipe continua responsável por defender as escolhas durante a discussão em sala.")

st.divider()
st.caption("Material didático fictício para ensino de aprendizado por reforço; parâmetros, custos e resultados não são dados agrícolas reais.")
