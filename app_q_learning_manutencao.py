import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Q-Learning para Manutenção Preventiva
# App único para Streamlit Cloud
# ============================================================

st.set_page_config(
    page_title="IA de Reforço para Manutenção Preventiva",
    page_icon="🛠️",
    layout="wide",
)

# -----------------------------
# Definições do problema
# -----------------------------
STATE_NAMES = {
    0: "Normal",
    1: "Alerta leve",
    2: "Desgaste moderado",
    3: "Risco alto",
    4: "Falha",
}

ACTION_NAMES = {
    0: "Continuar operando",
    1: "Reduzir carga",
    2: "Manutenção preventiva",
}

STATE_DESCRIPTIONS = {
    0: "Máquina estável e operando dentro dos padrões.",
    1: "Há pequenos sinais de degradação, mas ainda sem criticidade.",
    2: "Os sinais de desgaste já são relevantes e exigem atenção.",
    3: "A chance de falha é alta caso a operação continue sem intervenção.",
    4: "A máquina falhou e a operação foi interrompida.",
}


@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool


class MaintenanceEnv:
    """
    Ambiente discreto para Q-Learning.
    Estados:
        0 = Normal
        1 = Alerta leve
        2 = Desgaste moderado
        3 = Risco alto
        4 = Falha (terminal)
    Ações:
        0 = Continuar operando
        1 = Reduzir carga
        2 = Manutenção preventiva
    """

    def __init__(self, max_steps: int = 20, seed: int = 42):
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.current_state = 0
        self.steps = 0

    def reset(self, start_state: int = 0) -> int:
        self.current_state = start_state
        self.steps = 0
        return self.current_state

    def step(self, action: int) -> StepResult:
        state = self.current_state
        self.steps += 1

        # Estado terminal
        if state == 4:
            return StepResult(next_state=4, reward=0.0, done=True)

        # Tabelas de recompensa imediata
        reward_matrix = {
            0: {0: 8, 1: 4, 2: -4},   # Normal
            1: {0: 5, 1: 3, 2: 2},    # Alerta leve
            2: {0: 0, 1: 2, 2: 6},    # Desgaste moderado
            3: {0: -12, 1: 1, 2: 10}, # Risco alto
        }

        reward = reward_matrix[state][action]

        # Transições probabilísticas por estado e ação
        transition_probs = self._get_transition_probs(state, action)
        next_state = self._sample_next_state(transition_probs)

        # Penalidade adicional para falha
        if next_state == 4:
            reward -= 50

        done = next_state == 4 or self.steps >= self.max_steps
        self.current_state = next_state
        return StepResult(next_state=next_state, reward=reward, done=done)

    def _sample_next_state(self, probs: Dict[int, float]) -> int:
        states = list(probs.keys())
        weights = list(probs.values())
        return self.rng.choices(states, weights=weights, k=1)[0]

    def _get_transition_probs(self, state: int, action: int) -> Dict[int, float]:
        """
        Regras simples, didáticas e realistas.
        """
        # Ação 0 = Continuar operando
        if action == 0:
            if state == 0:
                return {0: 0.70, 1: 0.25, 2: 0.05}
            if state == 1:
                return {1: 0.50, 2: 0.35, 3: 0.10, 0: 0.05}
            if state == 2:
                return {2: 0.35, 3: 0.40, 4: 0.20, 1: 0.05}
            if state == 3:
                return {3: 0.25, 4: 0.60, 2: 0.15}

        # Ação 1 = Reduzir carga
        if action == 1:
            if state == 0:
                return {0: 0.85, 1: 0.10, 2: 0.05}
            if state == 1:
                return {0: 0.20, 1: 0.60, 2: 0.15, 3: 0.05}
            if state == 2:
                return {1: 0.25, 2: 0.55, 3: 0.15, 4: 0.05}
            if state == 3:
                return {2: 0.35, 3: 0.45, 4: 0.20}

        # Ação 2 = Manutenção preventiva
        if action == 2:
            if state == 0:
                return {0: 1.00}
            if state == 1:
                return {0: 0.90, 1: 0.10}
            if state == 2:
                return {0: 0.85, 1: 0.15}
            if state == 3:
                return {0: 0.80, 1: 0.20}

        return {state: 1.0}


def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(q_table.shape[1])
    return int(np.argmax(q_table[state]))


def train_q_learning(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    max_steps: int,
    start_state_mode: str,
    seed: int,
) -> Tuple[np.ndarray, List[float], List[int], List[int], List[float]]:
    np.random.seed(seed)
    env = MaintenanceEnv(max_steps=max_steps, seed=seed)

    n_states = len(STATE_NAMES)
    n_actions = len(ACTION_NAMES)
    q_table = np.zeros((n_states, n_actions))

    rewards_history = []
    failures_history = []
    maintenance_history = []
    epsilon_history = []

    possible_start_states = [0, 1, 2, 3]

    for _ in range(episodes):
        if start_state_mode == "Sempre Normal":
            state = env.reset(start_state=0)
        else:
            state = env.reset(start_state=random.choice(possible_start_states))

        total_reward = 0.0
        failures = 0
        maintenances = 0

        for _step in range(max_steps):
            action = epsilon_greedy_action(q_table, state, epsilon)
            if action == 2:
                maintenances += 1

            result = env.step(action)
            next_state, reward, done = result.next_state, result.reward, result.done

            best_next_q = np.max(q_table[next_state])
            old_q = q_table[state, action]

            q_table[state, action] = old_q + alpha * (
                reward + gamma * best_next_q - old_q
            )

            total_reward += reward
            if next_state == 4:
                failures += 1

            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(total_reward)
        failures_history.append(failures)
        maintenance_history.append(maintenances)
        epsilon_history.append(epsilon)

    return q_table, rewards_history, failures_history, maintenance_history, epsilon_history


def simulate_policy(
    q_table: np.ndarray,
    max_steps: int,
    start_state: int,
    seed: int,
    policy_name: str = "aprendida",
) -> pd.DataFrame:
    env = MaintenanceEnv(max_steps=max_steps, seed=seed)
    state = env.reset(start_state=start_state)
    rows = []

    for step in range(1, max_steps + 1):
        if state == 4:
            break

        if policy_name == "aleatória":
            action = np.random.randint(len(ACTION_NAMES))
        else:
            action = int(np.argmax(q_table[state]))

        result = env.step(action)

        rows.append(
            {
                "Passo": step,
                "Estado atual": STATE_NAMES[state],
                "Ação escolhida": ACTION_NAMES[action],
                "Recompensa": result.reward,
                "Próximo estado": STATE_NAMES[result.next_state],
                "Fim?": "Sim" if result.done else "Não",
            }
        )

        state = result.next_state
        if result.done:
            break

    return pd.DataFrame(rows)


def q_table_to_dataframe(q_table: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        q_table,
        index=[STATE_NAMES[i] for i in range(len(STATE_NAMES))],
        columns=[ACTION_NAMES[i] for i in range(len(ACTION_NAMES))],
    )
    df["Melhor ação"] = df.idxmax(axis=1)
    return df.round(2)


def policy_dataframe(q_table: np.ndarray) -> pd.DataFrame:
    rows = []
    for state in range(len(STATE_NAMES)):
        best_action = int(np.argmax(q_table[state]))
        rows.append(
            {
                "Estado": STATE_NAMES[state],
                "Melhor ação": ACTION_NAMES[best_action],
                "Descrição": STATE_DESCRIPTIONS[state],
            }
        )
    return pd.DataFrame(rows)


def moving_average(values: List[float], window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def compare_policies(
    q_table: np.ndarray,
    episodes: int,
    max_steps: int,
    start_state_mode: str,
    seed: int,
) -> pd.DataFrame:
    possible_start_states = [0, 1, 2, 3]

    def run(policy_name: str) -> Tuple[float, float, float]:
        total_rewards = []
        total_failures = []
        total_maintenances = []

        for i in range(episodes):
            env = MaintenanceEnv(max_steps=max_steps, seed=seed + i)
            if start_state_mode == "Sempre Normal":
                state = env.reset(start_state=0)
            else:
                state = env.reset(start_state=random.choice(possible_start_states))

            ep_reward = 0.0
            ep_failures = 0
            ep_maintenances = 0

            for _ in range(max_steps):
                if policy_name == "aleatória":
                    action = np.random.randint(len(ACTION_NAMES))
                else:
                    action = int(np.argmax(q_table[state]))

                if action == 2:
                    ep_maintenances += 1

                result = env.step(action)
                ep_reward += result.reward
                if result.next_state == 4:
                    ep_failures += 1
                state = result.next_state

                if result.done:
                    break

            total_rewards.append(ep_reward)
            total_failures.append(ep_failures)
            total_maintenances.append(ep_maintenances)

        return (
            float(np.mean(total_rewards)),
            float(np.mean(total_failures)),
            float(np.mean(total_maintenances)),
        )

    learned = run("aprendida")
    random_policy = run("aleatória")

    return pd.DataFrame(
        [
            {
                "Estratégia": "Q-Learning",
                "Recompensa média": round(learned[0], 2),
                "Falhas médias": round(learned[1], 2),
                "Manutenções médias": round(learned[2], 2),
            },
            {
                "Estratégia": "Aleatória",
                "Recompensa média": round(random_policy[0], 2),
                "Falhas médias": round(random_policy[1], 2),
                "Manutenções médias": round(random_policy[2], 2),
            },
        ]
    )


# -----------------------------
# Interface
# -----------------------------
st.title("🛠️ IA de Reforço para Manutenção Preventiva")
st.markdown(
    """
Este app demonstra **Aprendizado por Reforço com Q-Learning** em um problema de negócio:
**decidir a melhor ação para uma máquina industrial em diferentes estados operacionais**.

A IA aprende por **tentativa e erro**, maximizando a recompensa acumulada ao longo do tempo.
"""
)

with st.expander("📌 Entenda o problema de negócio", expanded=True):
    st.markdown(
        """
### Objetivo gerencial
Encontrar a melhor decisão para cada estado da máquina, equilibrando:

- continuidade da produção;
- prevenção de falhas;
- custo de manutenção;
- visão de curto e longo prazo.

### Estados
- **Normal**
- **Alerta leve**
- **Desgaste moderado**
- **Risco alto**
- **Falha**

### Ações
- **Continuar operando**
- **Reduzir carga**
- **Manutenção preventiva**
"""
    )

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Configuração do treinamento")

    episodes = st.slider("Número de episódios", 100, 5000, 1500, step=100)
    max_steps = st.slider("Máximo de passos por episódio", 5, 50, 20)
    alpha = st.slider("Alpha (taxa de aprendizado)", 0.01, 1.0, 0.10, step=0.01)
    gamma = st.slider("Gamma (valor do futuro)", 0.01, 0.99, 0.90, step=0.01)
    epsilon = st.slider("Epsilon inicial (exploração)", 0.01, 1.0, 0.30, step=0.01)
    epsilon_decay = st.slider("Decaimento do epsilon", 0.900, 0.999, 0.995, step=0.001)
    epsilon_min = st.slider("Epsilon mínimo", 0.00, 0.30, 0.05, step=0.01)
    start_state_mode = st.selectbox(
        "Estado inicial dos episódios",
        ["Sempre Normal", "Aleatório entre estados operacionais"],
    )
    seed = st.number_input("Semente aleatória", min_value=1, max_value=9999, value=42)

    train_button = st.button("Treinar IA", type="primary", use_container_width=True)

with col_right:
    st.subheader("📚 Interpretação da Tabela Q")
    st.markdown(
        """
A **Tabela Q** relaciona:

- **linhas** → estados da máquina;
- **colunas** → ações possíveis;
- **células** → valor esperado de cada ação em cada estado.

Ao final do treinamento, a política escolhida é:

> **em cada estado, selecionar a ação com maior valor Q**
"""
    )

if "trained" not in st.session_state:
    st.session_state["trained"] = False

if train_button:
    with st.spinner("Treinando agente com Q-Learning..."):
        (
            q_table,
            rewards_history,
            failures_history,
            maintenance_history,
            epsilon_history,
        ) = train_q_learning(
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            max_steps=max_steps,
            start_state_mode=start_state_mode,
            seed=int(seed),
        )

        st.session_state["trained"] = True
        st.session_state["q_table"] = q_table
        st.session_state["rewards_history"] = rewards_history
        st.session_state["failures_history"] = failures_history
        st.session_state["maintenance_history"] = maintenance_history
        st.session_state["epsilon_history"] = epsilon_history
        st.session_state["max_steps"] = max_steps
        st.session_state["start_state_mode"] = start_state_mode
        st.session_state["seed"] = int(seed)

if st.session_state["trained"]:
    q_table = st.session_state["q_table"]
    rewards_history = st.session_state["rewards_history"]
    failures_history = st.session_state["failures_history"]
    maintenance_history = st.session_state["maintenance_history"]
    epsilon_history = st.session_state["epsilon_history"]
    max_steps = st.session_state["max_steps"]
    start_state_mode = st.session_state["start_state_mode"]
    seed = st.session_state["seed"]

    st.success("Treinamento concluído.")

    # Métricas principais
    st.subheader("📊 Indicadores do treinamento")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recompensa média final (últimos 100)", f"{np.mean(rewards_history[-100:]):.2f}")
    m2.metric("Falhas médias (últimos 100)", f"{np.mean(failures_history[-100:]):.2f}")
    m3.metric("Manutenções médias (últimos 100)", f"{np.mean(maintenance_history[-100:]):.2f}")
    m4.metric("Epsilon final", f"{epsilon_history[-1]:.3f}")

    # Tabela Q
    st.subheader("🧠 Tabela Q aprendida")
    q_df = q_table_to_dataframe(q_table)
    st.dataframe(q_df, use_container_width=True)

    # Política ótima
    st.subheader("✅ Política aprendida")
    st.dataframe(policy_dataframe(q_table), use_container_width=True)

    # Gráficos
    st.subheader("📈 Evolução do aprendizado")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(rewards_history, label="Recompensa por episódio")
        ma_rewards = moving_average(rewards_history, window=20)
        if len(ma_rewards) > 0:
            start_idx = len(rewards_history) - len(ma_rewards)
            ax1.plot(range(start_idx, len(rewards_history)), ma_rewards, linewidth=2, label="Média móvel (20)")
        ax1.set_title("Evolução da recompensa")
        ax1.set_xlabel("Episódio")
        ax1.set_ylabel("Recompensa")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(epsilon_history)
        ax2.set_title("Decaimento da exploração (epsilon)")
        ax2.set_xlabel("Episódio")
        ax2.set_ylabel("Epsilon")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(failures_history)
        ax3.set_title("Falhas por episódio")
        ax3.set_xlabel("Episódio")
        ax3.set_ylabel("Quantidade de falhas")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.plot(maintenance_history)
        ax4.set_title("Manutenções preventivas por episódio")
        ax4.set_xlabel("Episódio")
        ax4.set_ylabel("Quantidade de manutenções")
        st.pyplot(fig4)

    # Comparativo
    st.subheader("⚖️ Comparativo: política aprendida x política aleatória")
    comparison_df = compare_policies(
        q_table=q_table,
        episodes=200,
        max_steps=max_steps,
        start_state_mode=start_state_mode,
        seed=seed,
    )
    st.dataframe(comparison_df, use_container_width=True)

    # Simulação
    st.subheader("🎮 Simulação da política")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        start_state_sim = st.selectbox(
            "Estado inicial da simulação",
            options=list(STATE_NAMES.keys())[:-1],
            format_func=lambda x: STATE_NAMES[x],
        )
    with sim_col2:
        sim_policy = st.selectbox("Tipo de política", ["aprendida", "aleatória"])
    with sim_col3:
        sim_steps = st.slider("Máximo de passos da simulação", 5, 30, min(12, max_steps))

    if st.button("Rodar simulação", use_container_width=True):
        sim_df = simulate_policy(
            q_table=q_table,
            max_steps=sim_steps,
            start_state=start_state_sim,
            seed=seed + 1000,
            policy_name=sim_policy,
        )
        st.dataframe(sim_df, use_container_width=True)

        if not sim_df.empty:
            total_sim_reward = sim_df["Recompensa"].sum()
            st.info(f"Recompensa total da simulação: {total_sim_reward:.2f}")

    # Exportação
    st.subheader("💾 Exportar resultados")
    csv_q = q_df.to_csv(index=True).encode("utf-8-sig")
    csv_policy = policy_dataframe(q_table).to_csv(index=False).encode("utf-8-sig")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Baixar Tabela Q em CSV",
            data=csv_q,
            file_name="tabela_q_manutencao.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Baixar política aprendida em CSV",
            data=csv_policy,
            file_name="politica_aprendida_q_learning.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("Ajuste os parâmetros e clique em **Treinar IA** para iniciar o aprendizado.")

st.markdown("---")
st.caption(
    "Aplicação didática de Q-Learning para Formação Executiva em IA. "
    "O objetivo é tornar visível a lógica da Tabela Q em um problema de negócio."
)
