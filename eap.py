"""Editor didático de EAP — aplicativo único para Streamlit Cloud.

Dependências: streamlit, Pillow. Não requer Graphviz, arquivos de dados,
importação JSON, banco de dados, nem configurações externas.

Para executar: streamlit run eap.py
"""
from __future__ import annotations

import csv
import html
import io
import math
import os
import textwrap
from collections import defaultdict
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

# ===========================================================================
# MODELO E REGRAS DA EAP — independentes do Streamlit
# ===========================================================================


def projeto_vazio() -> dict:
    return {"nome": "Meu projeto", "proximo_id": 1, "componentes": []}


def indexar(documento: dict) -> dict:
    return {c["id"]: c for c in documento["componentes"]}


def filhos_por_pai(documento: dict) -> dict:
    filhos = defaultdict(list)
    for comp in documento["componentes"]:
        filhos[comp["pai"]].append(comp)
    return filhos


def adicionar_componente(documento: dict, pai: int, tipo: str, nome: str,
                         horas: float = 0.0, custo: float = 0.0) -> dict:
    if tipo not in ("entrega", "pacote"):
        raise ValueError("Selecione entrega ou pacote de trabalho.")
    nome = str(nome).strip()
    if not nome:
        raise ValueError("Informe o nome do componente.")
    if len(nome) > 140:
        raise ValueError("O nome deve ter até 140 caracteres.")
    mapa = indexar(documento)
    if pai != 0 and (pai not in mapa or mapa[pai]["tipo"] != "entrega"):
        raise ValueError("O componente pai precisa ser o projeto ou uma entrega.")
    if not math.isfinite(float(horas)) or not math.isfinite(float(custo)):
        raise ValueError("Horas e custo devem ser números finitos.")
    if float(horas) < 0 or float(custo) < 0:
        raise ValueError("Horas e custo não podem ser negativos.")
    novo = deepcopy(documento)
    novo["componentes"].append({
        "id": novo["proximo_id"], "pai": pai, "tipo": tipo, "nome": nome,
        "horas": round(float(horas), 2) if tipo == "pacote" else 0.0,
        "custo": round(float(custo), 2) if tipo == "pacote" else 0.0,
    })
    novo["proximo_id"] += 1
    return novo


def excluir_componente(documento: dict, componente_id: int) -> tuple[dict, int]:
    mapa = indexar(documento)
    if componente_id not in mapa:
        raise ValueError("Selecione uma entrega ou um pacote existente.")
    filhos = filhos_por_pai(documento)
    remover = set()

    def percorrer(cid):
        remover.add(cid)
        for filho in filhos.get(cid, []):
            percorrer(filho["id"])

    percorrer(componente_id)
    novo = deepcopy(documento)
    novo["componentes"] = [c for c in novo["componentes"] if c["id"] not in remover]
    return novo, len(remover)


def linhas_eap(documento: dict) -> list[dict]:
    filhos = filhos_por_pai(documento)
    mapa = indexar(documento)
    custos = {}

    def total_do_no(cid):
        if cid in custos:
            return custos[cid]
        valor = mapa[cid]["custo"] if cid != 0 else 0.0
        for filho in filhos.get(cid, []):
            valor += total_do_no(filho["id"])
        custos[cid] = round(valor, 2)
        return custos[cid]

    total_do_no(0)
    resultado = [{
        "id": 0, "pai": None, "codigo": "0", "tipo": "projeto",
        "nome": documento["nome"], "profundidade": 0,
        "horas": None, "custo_direto": None, "custo_total": custos[0],
    }]

    def visitar(pai, prefixo, profundidade):
        for numero, componente in enumerate(filhos.get(pai, []), start=1):
            cid = componente["id"]
            codigo = str(numero) if pai == 0 else f"{prefixo}.{numero}"
            resultado.append({
                "id": cid, "pai": pai, "codigo": codigo, "tipo": componente["tipo"],
                "nome": componente["nome"], "profundidade": profundidade,
                "horas": componente["horas"] if componente["tipo"] == "pacote" else None,
                "custo_direto": componente["custo"] if componente["tipo"] == "pacote" else None,
                "custo_total": custos[cid],
            })
            visitar(cid, codigo, profundidade + 1)

    visitar(0, "", 1)
    return resultado


def moeda(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def numero(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def exportar_csv(documento: dict) -> bytes:
    buffer = io.StringIO(newline="")
    escritor = csv.writer(buffer, delimiter=";")
    escritor.writerow(["Código", "Tipo", "Componente", "Horas do pacote", "Custo do pacote (R$)", "Custo consolidado (R$)"])
    for r in linhas_eap(documento):
        escritor.writerow([
            r["codigo"], r["tipo"], r["nome"],
            "" if r["horas"] is None else numero(r["horas"]),
            "" if r["custo_direto"] is None else numero(r["custo_direto"]),
            numero(r["custo_total"]),
        ])
    return buffer.getvalue().encode("utf-8-sig")


def exportar_texto(documento: dict) -> bytes:
    partes = []
    for r in linhas_eap(documento):
        esquerda = "    " * r["profundidade"] + r["codigo"] + "  " + r["nome"]
        if r["tipo"] == "pacote":
            direita = f"{numero(r['horas'])} h | {moeda(r['custo_direto'])}"
        else:
            direita = f"Custo consolidado: {moeda(r['custo_total'])}"
        partes.append(f"{esquerda} — {direita}")
    return ("\n".join(partes) + "\n").encode("utf-8-sig")

# ===========================================================================
# DESENHO: PACOTES DE TRABALHO EMPILHADOS VERTICALMENTE
# ===========================================================================

CARD_W = 258
CARD_H = 102
GAP_X = 30
GAP_Y = 82
GAP_STACK = 16
MARGIN = 44
SPINE_GUTTER = 27


def calcular_layout(documento: dict) -> dict:
    """Organograma em colunas: entregas lado a lado, pacotes irmãos empilhados."""
    linhas = linhas_eap(documento)
    por_id = {r["id"]: r for r in linhas}
    filhos = filhos_por_pai(documento)
    larguras = {}

    def medir(cid):
        if cid in larguras:
            return larguras[cid]
        grupos = []
        entregas = [x for x in filhos.get(cid, []) if x["tipo"] == "entrega"]
        pacotes = [x for x in filhos.get(cid, []) if x["tipo"] == "pacote"]
        for sub in entregas:
            grupos.append(medir(sub["id"]))
        if pacotes:
            grupos.append(CARD_W + SPINE_GUTTER)
        largura_inferior = sum(grupos) + GAP_X * max(0, len(grupos) - 1)
        larguras[cid] = max(CARD_W, largura_inferior)
        return larguras[cid]

    medir(0)
    caixas = {}
    linhas_conexao = []

    def linha(x1, y1, x2, y2):
        linhas_conexao.append((x1, y1, x2, y2))

    def posicionar(cid, esquerda, topo):
        largura = larguras[cid]
        centro = esquerda + largura / 2
        caixas[cid] = (centro - CARD_W / 2, topo, CARD_W, CARD_H)
        entregas = [x for x in filhos.get(cid, []) if x["tipo"] == "entrega"]
        pacotes = [x for x in filhos.get(cid, []) if x["tipo"] == "pacote"]
        if not entregas and not pacotes:
            return
        grupos = [larguras[x["id"]] for x in entregas]
        if pacotes:
            grupos.append(CARD_W + SPINE_GUTTER)
        largura_grupos = sum(grupos) + GAP_X * (len(grupos) - 1)
        inicio = esquerda + (largura - largura_grupos) / 2
        filho_topo = topo + CARD_H + GAP_Y
        barra_y = topo + CARD_H + 33
        destinos = []
        cursor = inicio
        for entrega in entregas:
            largura_entrega = larguras[entrega["id"]]
            cx = cursor + largura_entrega / 2
            destinos.append(cx)
            posicionar(entrega["id"], cursor, filho_topo)
            cursor += largura_entrega + GAP_X
        if pacotes:
            caixa_x = cursor + SPINE_GUTTER
            espinha = caixa_x - 15
            destinos.append(espinha)
            for i, pacote in enumerate(pacotes):
                y = filho_topo + i * (CARD_H + GAP_STACK)
                caixas[pacote["id"]] = (caixa_x, y, CARD_W, CARD_H)
                meio = y + CARD_H / 2
                linha(espinha, meio, caixa_x, meio)
            linha(espinha, barra_y, espinha,
                  filho_topo + (len(pacotes) - 1) * (CARD_H + GAP_STACK) + CARD_H / 2)
        linha(centro, topo + CARD_H, centro, barra_y)
        if destinos:
            linha(min([centro] + destinos), barra_y, max([centro] + destinos), barra_y)
        for cx in destinos[:len(entregas)]:
            linha(cx, barra_y, cx, filho_topo)

    posicionar(0, MARGIN, MARGIN)
    max_x = max(x + w for x, y, w, h in caixas.values())
    max_y = max(y + h for x, y, w, h in caixas.values())
    return {
        "caixas": caixas,
        "linhas": linhas_conexao,
        "por_id": por_id,
        "largura": int(math.ceil(max_x + MARGIN)),
        "altura": int(math.ceil(max_y + MARGIN)),
    }


def estilo(tipo: str) -> tuple[str, str, str, str]:
    if tipo == "projeto":
        return ("#173B67", "#EAF1F9", "#173B67", "#FFFFFF")
    if tipo == "entrega":
        return ("#4175A8", "#F2F7FD", "#DCEBFA", "#244E79")
    return ("#C29211", "#FFFBEA", "#FFE6A4", "#674800")


def cortar_texto(t: str, max_chars: int = 31, max_lines: int = 2) -> list[str]:
    """Quebra pelo tamanho *real* da fonte, evitando títulos fora da caixa."""
    largura_maxima = CARD_W - 22
    fonte = _fonte(14, True)

    def largura(txt):
        return fonte.getlength(txt) if hasattr(fonte, "getlength") else len(txt) * 8

    palavras = t.strip().split()
    linhas = []
    atual = ""
    for palavra in palavras:
        candidato = (atual + " " + palavra).strip()
        if largura(candidato) <= largura_maxima:
            atual = candidato
            continue
        if atual:
            linhas.append(atual)
            atual = ""
        # Palavras muito longas são divididas para não transbordar.
        while palavra and largura(palavra) > largura_maxima:
            pedaco = ""
            for c in palavra:
                if largura(pedaco + c) > largura_maxima:
                    break
                pedaco += c
            if not pedaco:
                break
            linhas.append(pedaco)
            palavra = palavra[len(pedaco):]
        atual = palavra
    if atual:
        linhas.append(atual)
    if not linhas:
        linhas = [""]
    if len(linhas) > max_lines:
        linhas = linhas[:max_lines]
        ultima = linhas[-1]
        while ultima and largura(ultima + "…") > largura_maxima:
            ultima = ultima[:-1]
        linhas[-1] = ultima.rstrip() + "…"
    return linhas


def rotulo_rodape(linha: dict) -> str:
    if linha["tipo"] == "pacote":
        return f"{numero(linha['horas'])} h  |  {moeda(linha['custo_direto'])}"
    return f"Custo: {moeda(linha['custo_total'])}"


def desenhar_svg(documento: dict) -> bytes:
    layout = calcular_layout(documento)
    partes = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{layout["largura"]}" '
              f'height="{layout["altura"]}" viewBox="0 0 {layout["largura"]} {layout["altura"]}">',
              '<rect width="100%" height="100%" fill="#FFFFFF"/>']
    for x1, y1, x2, y2 in layout["linhas"]:
        partes.append(f'<path d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" '
                      'stroke="#8D9DB1" stroke-width="2" fill="none"/>')
    for cid, (x, y, w, h) in layout["caixas"].items():
        r = layout["por_id"][cid]
        borda, fundo, faixa, texto_faixa = estilo(r["tipo"])
        partes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" '
                      f'rx="7" fill="{fundo}" stroke="{borda}" stroke-width="2"/>')
        partes.append(f'<path d="M {x+1:.1f} {y+7:.1f} Q {x+1:.1f} {y+1:.1f} {x+7:.1f} {y+1:.1f} '
                      f'L {x+w-7:.1f} {y+1:.1f} Q {x+w-1:.1f} {y+1:.1f} {x+w-1:.1f} {y+7:.1f} '
                      f'L {x+w-1:.1f} {y+25:.1f} L {x+1:.1f} {y+25:.1f} Z" fill="{faixa}"/>')
        tipo_nome = {"projeto": "PROJETO", "entrega": "ENTREGA", "pacote": "PACOTE DE TRABALHO"}[r["tipo"]]
        partes.append(f'<text x="{x+11:.1f}" y="{y+17:.1f}" font-size="11" font-weight="bold" '
                      f'fill="{texto_faixa}" font-family="DejaVu Sans, Arial, sans-serif">'
                      f'{html.escape(r["codigo"] + "  " + tipo_nome)}</text>')
        for i, trecho in enumerate(cortar_texto(r["nome"])):
            partes.append(f'<text x="{x+11:.1f}" y="{y+44+i*17:.1f}" font-size="14" '
                          f'font-weight="bold" fill="#192D42" font-family="DejaVu Sans, Arial, sans-serif">'
                          f'{html.escape(trecho)}</text>')
        partes.append(f'<text x="{x+11:.1f}" y="{y+89:.1f}" font-size="12" '
                      f'fill="#465A70" font-family="DejaVu Sans, Arial, sans-serif">'
                      f'{html.escape(rotulo_rodape(r))}</text>')
    partes.append("</svg>")
    return "".join(partes).encode("utf-8")


def _fonte(tamanho: int, negrito: bool = False):
    local = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
             if negrito else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(local, tamanho)
    except OSError:
        return ImageFont.load_default()


def desenhar_png(documento: dict) -> bytes:
    layout = calcular_layout(documento)
    escala = 2  # Exportação em alta resolução para aulas e slides.
    canvas = Image.new("RGB", (layout["largura"] * escala,
                               layout["altura"] * escala), "white")
    d = ImageDraw.Draw(canvas)
    f11b = _fonte(11 * escala, True)
    f14b = _fonte(14 * escala, True)
    f12 = _fonte(12 * escala)

    def ponto(*valores):
        return tuple(round(v * escala) for v in valores)

    for x1, y1, x2, y2 in layout["linhas"]:
        d.line(ponto(x1, y1, x2, y2), fill="#8D9DB1", width=2 * escala)
    for cid, (x, y, w, h) in layout["caixas"].items():
        r = layout["por_id"][cid]
        borda, fundo, faixa, cor_titulo = estilo(r["tipo"])
        d.rounded_rectangle(ponto(x, y, x+w, y+h), radius=7*escala,
                            fill=fundo, outline=borda, width=2*escala)
        d.rectangle(ponto(x+1, y+8, x+w-1, y+25), fill=faixa)
        d.rounded_rectangle(ponto(x+1, y+1, x+w-1, y+13), radius=6*escala, fill=faixa)
        d.rectangle(ponto(x+1, y+12, x+w-1, y+25), fill=faixa)
        tipo_nome = {"projeto": "PROJETO", "entrega": "ENTREGA", "pacote": "PACOTE DE TRABALHO"}[r["tipo"]]
        d.text(ponto(x+11, y+4), r["codigo"] + "  " + tipo_nome, font=f11b, fill=cor_titulo)
        for i, trecho in enumerate(cortar_texto(r["nome"])):
            d.text(ponto(x+11, y+30+i*17), trecho, font=f14b, fill="#192D42")
        d.text(ponto(x+11, y+75), rotulo_rodape(r), font=f12, fill="#465A70")
    b = io.BytesIO()
    canvas.save(b, format="PNG", optimize=True)
    return b.getvalue()

# ===========================================================================
# INTERFACE STREAMLIT — sem alterações de estado de chaves de widgets
# ===========================================================================


def iniciar_interface() -> None:
    import streamlit as st

    st.set_page_config(page_title="Editor de EAP", page_icon="🧩", layout="wide")
    st.markdown("""
    <style>
      .block-container {padding-top: 1.3rem;}
      [data-testid="stMetric"] {border: 1px solid #dce5ef; border-radius: 10px;
                                  padding: 12px; background: #f8fbfe;}
    </style>
    """, unsafe_allow_html=True)

    # Apenas o documento (não é um widget) é gravado no session_state.
    # Nunca escrever em chaves usadas por st.text_input, st.selectbox etc.
    if "documento_eap_v3" not in st.session_state:
        st.session_state["documento_eap_v3"] = projeto_vazio()

    documento = st.session_state["documento_eap_v3"]
    linhas = linhas_eap(documento)
    tipos = {r["tipo"]: sum(x["tipo"] == r["tipo"] for x in linhas) for r in linhas}

    st.title("Editor de Estrutura Analítica do Projeto (EAP)")
    st.caption("Construa a EAP por entregas e decomponha-as até os pacotes de trabalho. "
               "Para corrigir um componente, exclua-o e cadastre-o novamente.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Entregas e subentregas", tipos.get("entrega", 0))
    c2.metric("Pacotes de trabalho", tipos.get("pacote", 0))
    c3.metric("Custo total da EAP", moeda(linhas[0]["custo_total"]))

    tab_construcao, tab_diagrama, tab_lista = st.tabs([
        "➕ Construir EAP", "🌳 Diagrama hierárquico", "📋 Lista da EAP"
    ])

    with tab_construcao:
        st.subheader("1. Nome do projeto")
        with st.form("nome_projeto_form"):
            nome_projeto = st.text_input("Nome do projeto (componente principal)",
                                         value=documento["nome"], max_chars=140)
            renomear = st.form_submit_button("Definir nome do projeto")
        if renomear:
            if not nome_projeto.strip():
                st.error("O nome do projeto não pode ficar vazio.")
            else:
                novo = deepcopy(documento)
                novo["nome"] = nome_projeto.strip()
                st.session_state["documento_eap_v3"] = novo
                st.rerun()

        st.divider()
        st.subheader("2. Adicionar entrega ou pacote de trabalho")
        tipo = st.radio("O que deseja adicionar?", options=("entrega", "pacote"),
                        format_func=lambda v: "Entrega / subentrega" if v == "entrega" else "Pacote de trabalho",
                        horizontal=True)
        permissoes = [r for r in linhas if r["tipo"] != "pacote"]
        opcoes = [r["id"] for r in permissoes]
        labels = {r["id"]: f"{r['codigo']} — {r['nome']}" for r in permissoes}
        with st.form("adicionar_form", clear_on_submit=True):
            pai = st.selectbox("Dentro de qual componente?", options=opcoes,
                               format_func=lambda cid: labels[cid])
            nome = st.text_input("Nome da entrega ou pacote", max_chars=140,
                                 placeholder="Ex.: Equipamentos instalados")
            horas = custo = 0.0
            if tipo == "pacote":
                a, b = st.columns(2)
                horas = a.number_input("Tempo estimado (horas)", min_value=0.0,
                                       max_value=1000000.0, value=8.0, step=1.0,
                                       format="%.2f")
                custo = b.number_input("Custo estimado (R$)", min_value=0.0,
                                       max_value=1000000000.0, value=0.0,
                                       step=100.0, format="%.2f")
            adicionar = st.form_submit_button("Adicionar componente", type="primary")
        if adicionar:
            try:
                st.session_state["documento_eap_v3"] = adicionar_componente(
                    documento, pai, tipo, nome, horas, custo)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("3. Excluir um componente para corrigir")
        if not documento["componentes"]:
            st.info("Adicione uma entrega ou um pacote para habilitar a exclusão.")
        else:
            candidatos = [r for r in linhas if r["id"] != 0]
            nomes_candidatos = {r["id"]: f"{r['codigo']} — {r['nome']}" for r in candidatos}
            with st.form("excluir_form"):
                excluir_id = st.selectbox("Componente a excluir", options=list(nomes_candidatos),
                                          format_func=lambda cid: nomes_candidatos[cid])
                confirmar = st.checkbox("Confirmo a exclusão deste componente e de todas as suas subentregas/pacotes")
                executar = st.form_submit_button("Excluir componente")
            if executar:
                if not confirmar:
                    st.warning("Marque a confirmação antes de excluir.")
                else:
                    novo, _ = excluir_componente(documento, excluir_id)
                    st.session_state["documento_eap_v3"] = novo
                    st.rerun()
            st.caption("A exclusão de uma entrega também elimina todos os componentes subordinados a ela.")

        with st.expander("Reiniciar a EAP"):
            st.warning("Reiniciar apaga a EAP da sessão atual.")
            with st.form("reiniciar_form"):
                confirmar_reset = st.checkbox("Quero apagar toda a EAP")
                reiniciar = st.form_submit_button("Criar uma EAP em branco")
            if reiniciar and confirmar_reset:
                st.session_state["documento_eap_v3"] = projeto_vazio()
                st.rerun()

    with tab_diagrama:
        st.subheader("Organograma da EAP")
        st.caption("O projeto aparece no topo. As entregas podem ter subentregas. "
                   "Os pacotes de trabalho são apresentados verticalmente, um abaixo do outro, "
                   "sem linhas de dependência entre si. Os valores das entregas representam "
                   "a soma dos custos dos pacotes subordinados.")
        try:
            png = desenhar_png(documento)
            st.image(png, caption="EAP hierárquica (horas e custos nos pacotes)",
                     use_container_width=True)
            st.download_button("⬇️ Baixar organograma em PNG", data=png,
                               file_name="eap_organograma.png", mime="image/png")
            st.download_button("⬇️ Baixar organograma em SVG", data=desenhar_svg(documento),
                               file_name="eap_organograma.svg", mime="image/svg+xml")
        except Exception as exc:
            st.error(f"Não foi possível desenhar a EAP: {exc}")

    with tab_lista:
        st.subheader("Lista hierárquica da EAP")
        st.caption("A numeração é automática. Custos de entregas são consolidados; "
                   "o tempo, em horas, pertence apenas aos pacotes de trabalho.")
        tabela = []
        for r in linhas:
            tabela.append({
                "Código": r["codigo"],
                "Componente": "    " * r["profundidade"] + r["nome"],
                "Tipo": {"projeto": "Projeto", "entrega": "Entrega", "pacote": "Pacote de trabalho"}[r["tipo"]],
                "Tempo (h)": "—" if r["horas"] is None else numero(r["horas"]),
                "Custo do pacote": "—" if r["custo_direto"] is None else moeda(r["custo_direto"]),
                "Custo consolidado": moeda(r["custo_total"]),
            })
        st.dataframe(tabela, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Baixar lista em CSV", data=exportar_csv(documento),
                           file_name="eap_lista.csv", mime="text/csv")
        st.download_button("⬇️ Baixar lista em TXT", data=exportar_texto(documento),
                           file_name="eap_lista.txt", mime="text/plain")
        st.caption("As horas dos pacotes não foram somadas como prazo do projeto, pois "
                   "os trabalhos podem ocorrer simultaneamente.")

    with st.sidebar:
        st.markdown("### Como utilizar")
        st.markdown("1. Informe o nome do projeto.\n"
                    "2. Adicione entregas e subentregas.\n"
                    "3. Selecione uma entrega e inclua seus pacotes de trabalho.\n"
                    "4. Informe horas e custo de cada pacote.\n"
                    "5. Para corrigir, exclua e cadastre novamente.\n"
                    "6. Consulte o organograma ou a lista e baixe o resultado.")
        st.info("Os dados permanecem apenas enquanto a sessão do aplicativo estiver ativa. "
                "Não há importação de arquivos nem armazenamento permanente.")
        st.caption("Ferramenta didática: EAP orientada a entregas. "
                   "Os pacotes são componentes terminais, não atividades de cronograma.")


if __name__ == "__main__":
    iniciar_interface()
