"""Laboratório de métricas de classificação — SoftPlus.
Execute: streamlit run metrica.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Laboratório | Precisão e Recall", page_icon="🎯", layout="wide")

@st.cache_data
def criar_base():
    # Base didática fixa: 20 clientes cancelariam; 80 permaneceriam.
    # No limiar 0,50: VP=15, FP=10, FN=5, VN=70.
    positivos = np.array([.96,.91,.88,.85,.82,.80,.77,.74,.70,.67,.64,.61,.58,.54,.51,.48,.41,.34,.25,.12])
    negativos = np.array([.89,.79,.73,.68,.63,.60,.57,.55,.52,.50] + list(np.linspace(.49,.01,70)))
    df = pd.DataFrame({"Cliente": [f"CLI-{n:03}" for n in range(1,101)],
                       "Cancelou (real)": [1]*20 + [0]*80,
                       "Score de risco": np.concatenate([positivos, negativos])})
    return df.sample(frac=1, random_state=19).reset_index(drop=True)

def calcular(df, limiar):
    real = df["Cancelou (real)"].to_numpy()
    previsto = (df["Score de risco"].to_numpy() >= limiar).astype(int)
    vp=int(np.sum((real==1)&(previsto==1)))
    fp=int(np.sum((real==0)&(previsto==1)))
    fn=int(np.sum((real==1)&(previsto==0)))
    vn=int(np.sum((real==0)&(previsto==0)))
    precisao=vp/(vp+fp) if (vp+fp)>0 else None
    recall=vp/(vp+fn) if (vp+fn)>0 else None
    acuracia=(vp+vn)/len(df)
    return dict(VP=vp,FP=fp,FN=fn,VN=vn,precisao=precisao,recall=recall,acuracia=acuracia,previsto=previsto)

def pct(valor):
    return f"{valor:.0%}" if valor is not None else "N/D"

def brl(valor):
    return "R$ " + f"{valor:,.0f}".replace(",", ".")

st.title("🎯 Precisão x Recall: laboratório de decisão")
st.write("**Caso SoftPlus:** prever quais clientes cancelarão a assinatura no próximo mês e decidir quem receberá ações de retenção.")
st.info("**Base didática simulada:** 100 clientes; 20 cancelariam e 80 permaneceriam. Os scores são fictícios e NÃO representam um modelo treinado em dados reais.")

base = criar_base()
st.sidebar.header("1. Limiar de decisão")
limiar = st.sidebar.slider("Limiar de alerta", min_value=0.0,max_value=1.0,value=0.50,step=0.01, help="Um cliente é sinalizado quando seu score de risco é maior ou igual ao limiar.")
st.sidebar.caption("Limiar menor → mais clientes sinalizados. Limiar maior → menos clientes sinalizados.")
st.sidebar.header("2. Impacto no negócio")
custo_acao=st.sidebar.number_input("Custo da ação de retenção por cliente (R$)",min_value=0, value=200,step=50)
perda_cliente=st.sidebar.number_input("Perda estimada por cliente não abordado que cancela (R$)",min_value=0,value=10000,step=500)
st.sidebar.caption("O simulador de custos é ilustrativo: receber um alerta não significa que o cancelamento será evitado.")

r=calcular(base,limiar)
st.subheader("Indicadores para o limiar escolhido")
a,b,c,d=st.columns(4)
a.metric("Precisão",pct(r["precisao"]),help="VP / (VP + FP): dos clientes alertados, quantos realmente cancelariam?")
b.metric("Recall",pct(r["recall"]),help="VP / (VP + FN): de todos os canceladores reais, quantos encontramos?")
c.metric("Acurácia",pct(r["acuracia"]),help="(VP + VN) / total de clientes")
d.metric("Alertas emitidos",r["VP"]+r["FP"])
st.caption("**N/D** significa que não houve nenhum cliente sinalizado, portanto a precisão não possui denominador.")

left,right=st.columns([1.05,1],gap="large")
with left:
    st.subheader("Matriz de confusão")
    matriz=np.array([[r["VP"],r["FP"]],[r["FN"],r["VN"]]])
    fig,ax=plt.subplots(figsize=(6.0,3.6))
    ax.imshow(np.array([[.25,.85],[.85,.25]]),cmap="RdYlGn_r",vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):
            key=[["VP","FP"],["FN","VN"]][i][j]
            ax.text(j,i,f"{key}\n{matriz[i,j]}",ha="center",va="center",fontsize=10,fontweight="bold",color="#1c2936")
    ax.set_xticks([0,1],["Cancelaria","Permaneceria"])
    ax.set_yticks([0,1],["Alerta","Sem alerta"])
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlabel("Resultado real", fontsize=9)
    ax.set_ylabel("Previsão da IA", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)
with right:
    st.subheader("O significado na prática")
    st.write(f"**Precisão:** dos **{r['VP']+r['FP']} clientes alertados**, **{r['VP']} realmente cancelariam** e **{r['FP']} permaneceriam**.")
    st.write(f"**Recall:** dos **{r['VP']+r['FN']} canceladores reais**, a IA **encontrou {r['VP']}** e **deixou escapar {r['FN']}**.")
    st.write(f"**Acurácia:** a IA acertou **{r['VP']+r['VN']} de 100** resultados.")
    st.markdown("**Memorize:** Precisão = confiança nos alertas. Recall = cobertura dos casos reais.")

st.subheader("Como o limiar modifica precisão e recall")
limiares=np.round(np.linspace(0,1,101),2)
met=pd.DataFrame([{"Limiar":t, "Precisão":calcular(base,t)["precisao"],"Recall":calcular(base,t)["recall"],"Acurácia":calcular(base,t)["acuracia"]} for t in limiares])
fig,ax=plt.subplots(figsize=(10,3.5))
ax.plot(met["Limiar"],met["Precisão"],label="Precisão",linewidth=2.4)
ax.plot(met["Limiar"],met["Recall"],label="Recall",linewidth=2.4)
ax.plot(met["Limiar"],met["Acurácia"],label="Acurácia",linestyle="--",alpha=.65)
ax.axvline(limiar,color="gray",linestyle=":",label=f"Limiar escolhido ({limiar:.2f})")
ax.set(xlabel="Limiar de classificação",ylabel="Valor da métrica",xlim=(0,1),ylim=(0,1.05))
ax.grid(alpha=.2)
ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(.5,1.01))
fig.tight_layout()
st.pyplot(fig,use_container_width=True)
plt.close(fig)
st.caption("A precisão pode oscilar ao variar o limiar; o recall não aumenta quando elevamos o limiar.")

st.subheader("Impacto econômico (simulação)")
alertados = r["VP"]+r["FP"]
gasto=alertados*custo_acao
perdas_nao_abordados=r["FN"]*perda_cliente
c1,c2=st.columns(2)
c1.metric("Custo de contatar os clientes sinalizados",brl(gasto))
c2.metric("Valor associado aos cancelamentos não alertados (hipótese)",brl(perdas_nao_abordados))
st.caption("O segundo valor considera os FN × perda estimada por cliente e **não mede uma perda evitável**. Um alerta não garante retenção. Use esses números apenas para discutir o custo potencial dos erros.")

st.subheader("Atividade em equipe — 4 perguntas")
st.markdown("""1. Com limiar de **50%**, quais são os valores de **VP, FP, FN e VN**? O que significam **precisão** e **recall** neste caso?
2. Compare os limiares de **30%, 50% e 70%**. Como mudam a precisão, o recall e a quantidade de alertas?
3. O que é mais prejudicial à SoftPlus: um **falso positivo** ou um **falso negativo**? Explique considerando custos e perda de clientes.
4. **Qual limiar você recomendaria inicialmente?** Justifique com as métricas e as limitações da simulação.""")

with st.expander("Ver a base de clientes e exportar os dados"):
    exibicao=base.copy()
    exibicao["Alerta da IA"] = r["previsto"]
    exibicao["Situação"] = np.select([
        (exibicao["Alerta da IA"]==1)&(exibicao["Cancelou (real)"]==1),
        (exibicao["Alerta da IA"]==1)&(exibicao["Cancelou (real)"]==0),
        (exibicao["Alerta da IA"]==0)&(exibicao["Cancelou (real)"]==1)
    ],["VP","FP","FN"],default="VN")
    st.dataframe(exibicao,hide_index=True,use_container_width=True)
    st.download_button("Baixar CSV com as previsões",exibicao.to_csv(index=False).encode("utf-8-sig"),file_name="softplus_previsoes.csv",mime="text/csv")
