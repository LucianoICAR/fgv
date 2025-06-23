
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

st.title("Classificação de Frutas para Exportação - KNN com Dados Balanceados")

st.markdown("""
Este app utiliza um dataset de frutas com variáveis balanceadas para aplicar **KNN (K-Nearest Neighbors)** e classificar frutas como:
- 🍎 **Aprovada para exportação**
- ❌ **Reprovada para exportação**

As variáveis são:
- Peso da fruta (g)
- Diâmetro da fruta (cm)
- Cor da casca (1 a 10)

Os dados são carregados de um arquivo CSV balanceado.
""")

# Carregar o dataset
df = pd.read_csv("frutas_exportacao_balanceada.csv")
st.write("### Exemplo dos dados")
st.dataframe(df.head())

# K escolhido pelo usuário
k = st.slider("Escolha o valor de K", min_value=1, max_value=15, value=5)

# Modelo
X = df[["Peso (g)", "Diâmetro (cm)", "Cor (1-10)"]]
y = df["Aprovada"]

modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X, y)

# Gráfico: Peso vs Diâmetro
st.write("### Distribuição (Peso x Diâmetro)")
fig1, ax1 = plt.subplots()
cores = ['red' if c == 0 else 'green' for c in y]
ax1.scatter(df["Peso (g)"], df["Diâmetro (cm)"], c=cores, alpha=0.6)
ax1.set_xlabel("Peso (g)")
ax1.set_ylabel("Diâmetro (cm)")
st.pyplot(fig1)

# Gráfico: Peso vs Cor
st.write("### Distribuição (Peso x Cor)")
fig2, ax2 = plt.subplots()
ax2.scatter(df["Peso (g)"], df["Cor (1-10)"], c=cores, alpha=0.6)
ax2.set_xlabel("Peso (g)")
ax2.set_ylabel("Cor (1-10)")
st.pyplot(fig2)

# Entrada do usuário
st.write("### Teste com nova fruta")
peso_input = st.number_input("Peso (g)", 50.0, 500.0, 200.0)
diametro_input = st.number_input("Diâmetro (cm)", 3.0, 15.0, 8.0)
cor_input = st.number_input("Cor da casca (1 a 10)", 1.0, 10.0, 7.0)

nova_amostra = np.array([[peso_input, diametro_input, cor_input]])
pred = modelo.predict(nova_amostra)[0]

if pred == 1:
    st.success("🍎 Esta fruta foi **Aprovada para exportação**!")
else:
    st.error("❌ Esta fruta foi **Reprovada para exportação**.")
