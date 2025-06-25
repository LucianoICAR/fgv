
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rede Neural - Previsão de Compras", layout="centered")

# Título
st.title("🧠 Previsão de Valor de Compra com Rede Neural")
st.subheader("Case: E-commerce de Eletrônicos")

# Explicação do case
st.markdown("""
Você é gestor de um e-commerce que vende eletrônicos.  
Seu objetivo é prever **quanto um cliente gastará** com base em características anteriores de compra:

- Frequência de compras anteriores  
- Valor médio das compras passadas  
- Tempo desde a última compra (em dias)  
- Número de itens comprados  
- Engajamento em campanhas anteriores
""")

# Gerar dados fictícios
np.random.seed(42)
n_samples = 200
data = pd.DataFrame({
    "frequencia_compras": np.random.randint(1, 10, size=n_samples),
    "valor_medio": np.random.uniform(50, 1000, size=n_samples),
    "dias_ultima_compra": np.random.randint(1, 120, size=n_samples),
    "num_itens": np.random.randint(1, 5, size=n_samples),
    "engajamento": np.random.uniform(0, 1, size=n_samples)
})

# Variável alvo (valor gasto)
data["valor_gasto"] = (
    20 * data["frequencia_compras"]
    + 0.5 * data["valor_medio"]
    - 0.3 * data["dias_ultima_compra"]
    + 15 * data["num_itens"]
    + 200 * data["engajamento"]
    + np.random.normal(0, 30, size=n_samples)
)

# Sidebar para parâmetros
st.sidebar.header("🔧 Parâmetros do Modelo")

# Camadas
num_camadas = st.sidebar.selectbox("Número de camadas ocultas", [1, 2, 3])
neuronios = st.sidebar.slider("Neurônios por camada", 1, 50, 10)
hidden_layer_sizes = tuple([neuronios] * num_camadas)

# Ativação e solver
activation = st.sidebar.selectbox("Função de ativação", ["relu", "tanh", "logistic"])
solver = st.sidebar.selectbox("Algoritmo de otimização (solver)", ["adam", "sgd", "lbfgs"])
learning_rate_init = st.sidebar.slider("Taxa de aprendizado", 0.0001, 0.1, 0.001, step=0.0001)

# Separar dados
X = data.drop("valor_gasto", axis=1)
y = data["valor_gasto"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Treinar modelo
model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    learning_rate_init=learning_rate_init,
    max_iter=2000,
    random_state=1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Gráfico com cores distintas
st.markdown("### 📈 Comparação entre valores reais e previstos")
fig, ax = plt.subplots()
ax.scatter(range(len(y_test)), y_test, color='blue', label='Valor Real', alpha=0.7)
ax.scatter(range(len(y_pred)), y_pred, color='orange', label='Valor Previsto', alpha=0.7)
ax.set_xlabel("Amostras (Clientes)")
ax.set_ylabel("Valor de Compra (R$)")
ax.set_title("Valores Reais vs Valores Previstos")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Métricas
st.markdown(f"**Erro Médio Absoluto (MAE):** R$ {mae:.2f}")
st.markdown(f"**R² Score:** {r2:.2f}")

# Mostrar parte dos dados
with st.expander("🔍 Ver dados de exemplo"):
    st.write(data.head(10))
