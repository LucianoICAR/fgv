
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Q-Learning para Manutenção Preventiva (VERSÃO FINAL)
# Com comparação: Modelo Original x Modelo Corrigido
# ============================================================

st.set_page_config(
    page_title="IA de Reforço para Manutenção Preventiva",
    page_icon="🛠️",
    layout="wide",
)

STATE_NAMES = {
    0: "Normal",
    1: "Alerta leve",
    2: "Desgaste moderado",
    3: "Risco alto",
    4: "Falha",
}

ACTION_NAMES_ORIGINAL = {
    0: "Continuar operando",
    1: "Reduzir carga",
    2: "Manutenção preventiva",
}

ACTION_NAMES_CORRIGIDO = {
    0: "Continuar operando",
    1: "Reduzir carga",
    2: "Manutenção preventiva",
    3: "Manutenção corretiva",
}


@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool


class MaintenanceEnv:

    def __init__(self, mode="original", max_steps=20, seed=42):
        self.mode = mode
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.current_state = 0
        self.steps = 0

    def action_names(self):
        return ACTION_NAMES_ORIGINAL if self.mode == "original" else ACTION_NAMES_CORRIGIDO

    def reset(self, start_state=0):
        self.current_state = start_state
        self.steps = 0
        return self.current_state

    def step(self, action):

        state = self.current_state
        self.steps += 1

        # ===============================
        # MODELO ORIGINAL
        # ===============================
        if self.mode == "original":

            if state == 4:
                return StepResult(4, 0.0, True)

            reward_matrix = {
                0: {0: 8, 1: 4, 2: -4},
                1: {0: 5, 1: 3, 2: 2},
                2: {0: 0, 1: 2, 2: 6},
                3: {0: -12, 1: 1, 2: 10},
            }

            reward = reward_matrix[state][action]

            transition = self.get_transition(state, action)
            next_state = self.rng.choices(list(transition.keys()), weights=list(transition.values()))[0]

            if next_state == 4:
                reward -= 50

            done = next_state == 4 or self.steps >= self.max_steps
            self.current_state = next_state

            return StepResult(next_state, reward, done)

        # ===============================
        # MODELO CORRIGIDO
        # ===============================
        else:

            # 🔴 TRATAMENTO CORRETO DA FALHA
            if state == 4:

                reward_matrix = {
                    0: -120,
                    1: -40,
                    2: -20,
                    3: 25,
                }

                transition = {
                    0: {4: 1.0},
                    1: {4: 1.0},
                    2: {4: 0.9, 3: 0.1},
                    3: {0: 0.7, 1: 0.3},
                }

                reward = reward_matrix[action]
                next_state = self.rng.choices(list(transition[action].keys()), weights=list(transition[action].values()))[0]

                done = self.steps >= self.max_steps
                self.current_state = next_state

                return StepResult(next_state, reward, done)

            reward_matrix = {
                0: {0: 8, 1: 4, 2: -4, 3: -30},
                1: {0: 5, 1: 3, 2: 2, 3: -25},
                2: {0: 0, 1: 2, 2: 6, 3: -15},
                3: {0: -12, 1: 1, 2: 10, 3: -10},
            }

            reward = reward_matrix[state][action]

            transition = self.get_transition(state, action)
            next_state = self.rng.choices(list(transition.keys()), weights=list(transition.values()))[0]

            if next_state == 4:
                reward -= 50

            done = self.steps >= self.max_steps
            self.current_state = next_state

            return StepResult(next_state, reward, done)

    def get_transition(self, state, action):

        if action == 0:
            return {0:0.7,1:0.25,2:0.05} if state==0 else {1:0.5,2:0.35,3:0.1,0:0.05} if state==1 else {2:0.35,3:0.4,4:0.2,1:0.05} if state==2 else {3:0.25,4:0.6,2:0.15}

        if action == 1:
            return {0:0.85,1:0.1,2:0.05} if state==0 else {0:0.2,1:0.6,2:0.15,3:0.05} if state==1 else {1:0.25,2:0.55,3:0.15,4:0.05} if state==2 else {2:0.35,3:0.45,4:0.2}

        if action == 2:
            return {0:1.0} if state==0 else {0:0.9,1:0.1} if state==1 else {0:0.85,1:0.15} if state==2 else {0:0.8,1:0.2}

        if action == 3:
            return {0:1.0} if state==0 else {0:0.95,1:0.05} if state==1 else {0:0.9,1:0.1} if state==2 else {0:0.85,1:0.15}

        return {state:1.0}


def train(mode, episodes=1000):
    env = MaintenanceEnv(mode=mode)
    n_states = 5
    n_actions = len(env.action_names())

    q = np.zeros((n_states, n_actions))

    alpha, gamma, epsilon = 0.1, 0.9, 0.3

    for _ in range(episodes):
        state = env.reset()
        for _ in range(20):

            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(q[state])

            res = env.step(action)

            q[state, action] += alpha * (
                res.reward + gamma * np.max(q[res.next_state]) - q[state, action]
            )

            state = res.next_state

            if res.done:
                break

    return q


def q_df(mode, q):
    names = ACTION_NAMES_ORIGINAL if mode=="original" else ACTION_NAMES_CORRIGIDO
    df = pd.DataFrame(q, index=STATE_NAMES.values(), columns=names.values())
    df["Melhor ação"] = df.idxmax(axis=1)
    return df.round(2)


# ===============================
# INTERFACE
# ===============================

st.title("🧠 Q-Learning: Manutenção Preventiva")

mode = st.radio("Escolha o modelo", ["original", "corrigido"],
                format_func=lambda x: "Modelo Original" if x=="original" else "Modelo Corrigido")

if st.button("Treinar IA"):
    q = train(mode, 1500)
    st.session_state["q"] = q
    st.session_state["mode"] = mode

if "q" in st.session_state:

    q = st.session_state["q"]
    mode = st.session_state["mode"]

    st.subheader("Tabela Q")
    st.dataframe(q_df(mode, q))

    if mode == "original":
        st.warning("Observe: em 'Falha', o modelo pode indicar ação incorreta.")
    else:
        st.success("Observe: em 'Falha', a ação correta é 'Manutenção corretiva'.")
