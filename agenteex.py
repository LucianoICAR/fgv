import streamlit as st
from datetime import datetime
import random

st.set_page_config(page_title="Agente de IA - Demo Didática", layout="wide")

st.title("🤖 Agente de IA Simples")
st.subheader("Demonstração didática: perceber → decidir → agir")

st.markdown("""
Este agente simula um atendimento de TI.  
Ele:
1. lê a solicitação,
2. identifica a intenção,
3. escolhe uma ferramenta,
4. executa a ação,
5. responde ao usuário.
""")

# -----------------------------
# Ferramentas simuladas
# -----------------------------
def abrir_chamado(descricao):
    numero = random.randint(1000, 9999)
    return f"Chamado aberto com sucesso. Número do chamado: {numero}. Descrição registrada: {descricao}"

def consultar_base_conhecimento(tema):
    base = {
        "senha": "Procedimento padrão: acesse o portal corporativo e clique em 'Esqueci minha senha'.",
        "teams": "Para instalar o Teams, abra a Central de Softwares e selecione 'Microsoft Teams'.",
        "notebook": "Verifique energia, carregador e luz indicadora. Se não houver sinal, escale para suporte.",
        "email": "Se o e-mail não sincroniza, validar conexão e credenciais antes de reconfigurar."
    }

    for chave, valor in base.items():
        if chave in tema.lower():
            return valor

    return "Nenhum artigo específico encontrado. Recomenda-se abrir chamado ou escalar para suporte."

def consultar_status_chamado(numero):
    status_possiveis = ["Aberto", "Em atendimento", "Aguardando usuário", "Resolvido"]
    status = random.choice(status_possiveis)
    return f"O chamado {numero} está com status: {status}."

def escalar_para_humano(motivo):
    return f"Solicitação escalada para atendimento humano. Motivo: {motivo}"

# -----------------------------
# Motor de decisão do agente
# -----------------------------
def classificar_intencao(texto):
    t = texto.lower()

    if "status" in t and any(char.isdigit() for char in t):
        return "consultar_status"
    elif "senha" in t or "resetar senha" in t:
        return "base_conhecimento"
    elif "instalar" in t or "teams" in t or "software" in t:
        return "base_conhecimento"
    elif "não liga" in t or "nao liga" in t or "quebrou" in t or "erro grave" in t:
        return "escalar_humano"
    elif "ajuda" in t or "problema" in t or "falha" in t:
        return "abrir_chamado"
    else:
        return "desconhecida"

def extrair_numero_chamado(texto):
    numeros = "".join([c if c.isdigit() else " " for c in texto]).split()
    return numeros[0] if numeros else None

def agente(solicitacao):
    log = []
    memoria = {}

    log.append("1. Percepção: solicitação recebida.")
    memoria["solicitacao"] = solicitacao

    intencao = classificar_intencao(solicitacao)
    memoria["intencao"] = intencao
    log.append(f"2. Classificação da intenção: {intencao}")

    if intencao == "consultar_status":
        numero = extrair_numero_chamado(solicitacao)
        log.append(f"3. Ferramenta escolhida: consultar_status_chamado({numero})")
        resposta = consultar_status_chamado(numero)

    elif intencao == "base_conhecimento":
        log.append("3. Ferramenta escolhida: consultar_base_conhecimento()")
        resposta = consultar_base_conhecimento(solicitacao)

    elif intencao == "abrir_chamado":
        log.append("3. Ferramenta escolhida: abrir_chamado()")
        resposta = abrir_chamado(solicitacao)

    elif intencao == "escalar_humano":
        log.append("3. Ferramenta escolhida: escalar_para_humano()")
        resposta = escalar_para_humano("Possível incidente crítico ou problema físico")

    else:
        log.append("3. Nenhuma ferramenta adequada encontrada.")
        resposta = "Não consegui classificar com segurança sua solicitação. Vou encaminhar para atendimento humano."
        log.append("4. Ação de contingência: escalar para humano.")

    log.append("5. Resposta final gerada.")
    memoria["resposta"] = resposta
    memoria["timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    return resposta, log, memoria

# -----------------------------
# Interface
# -----------------------------
exemplos = [
    "Quero resetar minha senha de rede",
    "Qual o status do chamado 1234?",
    "Meu notebook não liga",
    "Preciso instalar o Teams",
    "Estou com problema no e-mail corporativo"
]

solicitacao = st.text_area(
    "Digite uma solicitação do usuário:",
    value=exemplos[0],
    height=120
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Executar agente"):
        resposta, log, memoria = agente(solicitacao)

        st.success("Resposta do agente")
        st.write(resposta)

        st.markdown("### Log de execução do agente")
        for item in log:
            st.write(f"- {item}")

        st.markdown("### Memória do agente")
        st.json(memoria)

with col2:
    st.markdown("### Exemplos para teste")
    for ex in exemplos:
        st.write(f"- {ex}")
