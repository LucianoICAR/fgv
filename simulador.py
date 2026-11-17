"""Motor didático para a Atividade 9 da FGV.

Q-Learning tabular efetivo em um ambiente sintético de irrigação.
Não é um modelo agronômico calibrado nem deve controlar equipamentos reais.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

DIAS = 180
ACOES_MM = (0.0, 6.0, 12.0)
CAPACIDADE_SOLO_MM = 100.0
LIMIAR_ESTRESSE = 35.0
LIMIAR_CRITICO = 20.0


@dataclass(frozen=True)
class Pesos:
    agua: float = 4.0
    estresse: float = 12.0


def gerar_clima(semente: int, dias: int = DIAS) -> pd.DataFrame:
    """Gera clima exógeno e previsões imperfeitas; reuso para comparação justa."""
    rng = np.random.default_rng(semente)
    linhas = []
    for dia in range(dias):
        # Pequena sazonalidade: possibilidade de chuvas de 14% a 37%.
        chance_chuva = 0.255 + 0.115 * np.sin(2 * np.pi * (dia + 25) / 180)
        choveu = bool(rng.random() < chance_chuva)
        chuva_mm = float(rng.gamma(2.1, 3.8)) if choveu else 0.0
        # Previsão do dia: sensibilidade ~78%, falso alarme ~16%.
        previsão_chuva = bool(rng.random() < (0.78 if choveu else 0.16))
        demanda_alta = bool(rng.random() < 0.48)
        evap_mm = float(np.clip(3.7 + 1.3 * demanda_alta + rng.normal(0, 0.45), 2.5, 6.4))
        linhas.append((dia + 1, chuva_mm, int(previsão_chuva), int(demanda_alta), evap_mm))
    return pd.DataFrame(linhas, columns=["dia", "chuva_mm", "prev_chuva", "demanda_alta", "evap_mm"])


def estado(umidade_mm: float, prev_chuva: int, demanda_alta: int) -> tuple[int, int, int]:
    baldes = (20, 35, 50, 70)
    faixa = int(np.searchsorted(baldes, umidade_mm, side="right"))
    return faixa, int(prev_chuva), int(demanda_alta)


def transicao(umidade_mm: float, acao_mm: float, clima_dia, pesos: Pesos) -> dict:
    """Ambiente com limite de 100 mm, evaporação e estresse hídrico."""
    disponivel = umidade_mm + acao_mm * 0.95 + float(clima_dia.chuva_mm)
    excesso = max(0.0, disponivel - CAPACIDADE_SOLO_MM)
    seguinte = max(0.0, min(CAPACIDADE_SOLO_MM, disponivel) - float(clima_dia.evap_mm))
    estresse = seguinte < LIMIAR_ESTRESSE
    critico = seguinte < LIMIAR_CRITICO
    deficit = max(0.0, LIMIAR_ESTRESSE - seguinte)
    # Penalidade de estresse cresce à medida que solo seca. Penalidade grave extra.
    recompensa = (
        - pesos.agua * (acao_mm / 8.0)
        - pesos.estresse * (deficit / 12.0)
        - pesos.estresse * 2.6 * float(critico)
        - 0.1 * excesso
    )
    return {"umidade": float(seguinte), "estresse": bool(estresse),
            "critico": bool(critico), "recompensa": float(recompensa)}


def treinar_q_learning(pesos: Pesos, episodios: int = 260, exploracao_inicial: float = 0.55,
                        semente: int = 2026) -> tuple[np.ndarray, pd.DataFrame]:
    """Aprende política Q(s,a) com exploração epsilon-greedy e TD-Q-learning."""
    rng = np.random.default_rng(semente)
    Q = np.zeros((5, 2, 2, len(ACOES_MM)), dtype=float)
    curva = []
    alpha = 0.16
    gamma = 0.94
    for ep in range(episodios):
        # Decaimento gradativo, para testar estratégias novas antes de explotá-las.
        frac = ep / max(1, episodios - 1)
        epsilon = max(0.05, exploracao_inicial * ((0.05 / exploracao_inicial) ** frac))
        clima = gerar_clima(semente + 50_000 + ep)
        umidade = float(rng.uniform(49, 61))
        recompensa_total = 0.0
        for t in range(DIAS):
            dia = clima.iloc[t]
            s = estado(umidade, int(dia.prev_chuva), int(dia.demanda_alta))
            if rng.random() < epsilon:
                a = int(rng.integers(len(ACOES_MM)))
            else:
                a = int(np.argmax(Q[s]))
            passo = transicao(umidade, ACOES_MM[a], dia, pesos)
            umidade = passo["umidade"]
            r = passo["recompensa"]
            if t + 1 < DIAS:
                dia_seguinte = clima.iloc[t + 1]
                s_linha = estado(umidade, int(dia_seguinte.prev_chuva), int(dia_seguinte.demanda_alta))
                alvo = r + gamma * float(np.max(Q[s_linha]))
            else:
                alvo = r
            Q[s][a] += alpha * (alvo - Q[s][a])
            recompensa_total += r
        curva.append({"Episódio": ep + 1, "Recompensa do episódio": recompensa_total,
                      "Exploração (%)": 100 * epsilon})
    return Q, pd.DataFrame(curva)


def simular_safra(clima: pd.DataFrame, pesos: Pesos, politica: str,
                   Q: np.ndarray | None = None, umidade_inicial: float = 55.0) -> tuple[dict, pd.DataFrame]:
    umidade = float(umidade_inicial)
    linha_dia = []
    recompensas = []
    for t in range(len(clima)):
        dia = clima.iloc[t]
        if politica == "calendario":
            irrigacao = 8.0 if t % 2 == 0 else 0.0
        elif politica == "qlearning":
            if Q is None:
                raise ValueError("Política de Q-Learning precisa de tabela Q treinada.")
            s = estado(umidade, int(dia.prev_chuva), int(dia.demanda_alta))
            irrigacao = ACOES_MM[int(np.argmax(Q[s]))]
        else:
            raise ValueError("Política desconhecida")
        umidade_antes = umidade
        trans = transicao(umidade, irrigacao, dia, pesos)
        umidade = trans["umidade"]
        recompensas.append(trans["recompensa"])
        linha_dia.append({"Dia": int(dia.dia), "Irrigação (mm)": irrigacao,
                          "Chuva (mm)": float(dia.chuva_mm),
                          "Previsão de chuva": "Sim" if int(dia.prev_chuva) else "Não",
                          "Demanda hídrica": "Alta" if int(dia.demanda_alta) else "Normal",
                          "Umidade inicial (mm)": umidade_antes,
                          "Umidade final (mm)": umidade, "Estresse": int(trans["estresse"]),
                          "Estresse grave": int(trans["critico"])})
    historico = pd.DataFrame(linha_dia)
    dias_estresse = int(historico["Estresse"].sum())
    dias_criticos = int(historico["Estresse grave"].sum())
    # ÍNDICE MERAMENTE DIDÁTICO, não é produtividade agronômica medida.
    indice = max(0.0, 100.0 - 0.28 * dias_estresse - 0.95 * dias_criticos)
    resumo = {"agua_mm": float(historico["Irrigação (mm)"].sum()),
              "agua_m3_ha": float(historico["Irrigação (mm)"].sum() * 10),
              "indice_cultura": float(indice), "dias_estresse": dias_estresse,
              "dias_criticos": dias_criticos, "pct_criticos": 100 * dias_criticos / len(clima),
              "recompensa": float(np.sum(recompensas)),
              "dias_0mm": int((historico["Irrigação (mm)"] == 0).sum()),
              "dias_6mm": int((historico["Irrigação (mm)"] == 6).sum()),
              "dias_8mm": int((historico["Irrigação (mm)"] == 8).sum()),
              "dias_12mm": int((historico["Irrigação (mm)"] == 12).sum())}
    return resumo, historico


def avaliar_politicas(Q: np.ndarray, pesos: Pesos, safras: int = 12,
                       semente: int = 7331) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas = []
    historico_primeiro = []
    for n in range(safras):
        clima = gerar_clima(semente + n * 13)
        ini = 55.0  # Mesma condição inicial para ambas as alternativas.
        for politica, titulo in (("calendario", "Calendário fixo"), ("qlearning", "Q-Learning")):
            resumo, historico = simular_safra(clima, pesos, politica, Q, ini)
            linhas.append({"Safra": n + 1, "Política": titulo, **resumo})
            if n == 0:
                historico["Política"] = titulo
                historico_primeiro.append(historico)
    return pd.DataFrame(linhas), pd.concat(historico_primeiro, ignore_index=True)


def avaliar_meta(df: pd.DataFrame) -> dict:
    medias = df.groupby("Política").mean(numeric_only=True)
    base = medias.loc["Calendário fixo"]
    rl = medias.loc["Q-Learning"]
    economia_pct = 100 * (base["agua_mm"] - rl["agua_mm"]) / base["agua_mm"]
    criterio_agua = economia_pct >= 20
    criterio_cultura = rl["indice_cultura"] >= base["indice_cultura"] - 1e-9
    criterio_grave = (rl["pct_criticos"] <= 5.0 and
                      rl["pct_criticos"] <= base["pct_criticos"] + 1e-9)
    return {"base": base, "rl": rl, "economia_pct": float(economia_pct),
            "agua_ok": bool(criterio_agua), "cultura_ok": bool(criterio_cultura),
            "grave_ok": bool(criterio_grave),
            "meta_total": bool(criterio_agua and criterio_cultura and criterio_grave)}


if __name__ == "__main__":
    for nome, pesos in (("Equilíbrio", Pesos(4,12)), ("Economia máxima",Pesos(20,2)),
                        ("Proteção",Pesos(2,18))):
        Q, curva = treinar_q_learning(pesos)
        df, hist = avaliar_politicas(Q,pesos)
        avaliacao = avaliar_meta(df)
        print(nome, "economia",round(avaliacao['economia_pct'],1),
              "índice", round(avaliacao['base']['indice_cultura'],1),round(avaliacao['rl']['indice_cultura'],1),
              "graves",round(avaliacao['base']['pct_criticos'],1),round(avaliacao['rl']['pct_criticos'],1),
              "meta",avaliacao['meta_total'])
