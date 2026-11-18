"""Motor de dados, hierarquia e exportação do editor didático de EAP.

O documento é um JSON simples, sem dependência de banco de dados.
A EAP possui uma raiz (nível 0), entregas e pacotes de trabalho.
"""

from __future__ import annotations

import copy
import csv
import html
import io
import json
import math
import textwrap
import uuid
from collections import defaultdict
from typing import Any

from graphviz import Digraph

VERSAO = 1
LIMITE_COMPONENTES = 500
TIPOS = {"projeto", "entrega", "pacote"}


def criar_projeto(nome: str = "Novo projeto") -> dict[str, Any]:
    return {
        "versao": VERSAO,
        "nodes": [
            {
                "id": "root", "parent_id": None, "tipo": "projeto",
                "nome": nome, "descricao": "", "duracao_dias": 0.0,
                "custo_reais": 0.0,
            }
        ],
    }


def indice(documento: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["id"]: n for n in documento["nodes"]}


def filhos_por_pai(documento: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mapa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for no in documento["nodes"]:
        if no["parent_id"] is not None:
            mapa[no["parent_id"]].append(no)
    return mapa


def validar(documento: Any) -> dict[str, Any]:
    """Valida e normaliza conteúdo de JSON importado (sem executá-lo)."""
    if not isinstance(documento, dict) or type(documento.get("versao")) is not int or documento.get("versao") != VERSAO:
        raise ValueError("Arquivo incompatível: versão do projeto não reconhecida.")
    nos = documento.get("nodes")
    if not isinstance(nos, list) or not (1 <= len(nos) <= LIMITE_COMPONENTES):
        raise ValueError(f"A EAP deve conter de 1 a {LIMITE_COMPONENTES} componentes.")

    limpos = []
    ids = set()
    for pos, no in enumerate(nos):
        if not isinstance(no, dict):
            raise ValueError(f"Componente {pos + 1} inválido.")
        ident = no.get("id")
        parent = no.get("parent_id")
        tipo = no.get("tipo")
        nome = no.get("nome")
        descricao = no.get("descricao", "")
        if not isinstance(ident, str) or not ident or len(ident) > 80 or ident in ids:
            raise ValueError("Há identificadores ausentes ou repetidos no arquivo.")
        ids.add(ident)
        if parent is not None and (not isinstance(parent, str) or len(parent) > 80):
            raise ValueError("O vínculo com o componente pai é inválido.")
        if not isinstance(tipo, str) or tipo not in TIPOS:
            raise ValueError("Há um componente com tipo inválido.")
        if not isinstance(nome, str) or not nome.strip() or len(nome) > 140:
            raise ValueError("O nome de cada componente deve ter de 1 a 140 caracteres.")
        if not isinstance(descricao, str) or len(descricao) > 1000:
            raise ValueError("A descrição de um componente é inválida ou muito longa.")
        duracao = no.get("duracao_dias", 0.0)
        custo = no.get("custo_reais", 0.0)
        for campo, valor in (("duração", duracao), ("custo", custo)):
            if isinstance(valor, bool) or not isinstance(valor, (int, float)):
                raise ValueError(f"O valor de {campo} deve ser numérico.")
            if valor < 0 or valor > 1_000_000_000 or not math.isfinite(valor):
                raise ValueError(f"O valor de {campo} está fora do intervalo permitido.")
        limpos.append({
            "id": ident, "parent_id": parent, "tipo": tipo,
            "nome": nome.strip(), "descricao": descricao.strip(),
            "duracao_dias": float(duracao) if tipo == "pacote" else 0.0,
            "custo_reais": float(custo) if tipo == "pacote" else 0.0,
        })

    mapa = {n["id"]: n for n in limpos}
    raiz = mapa.get("root")
    if raiz is None or raiz["parent_id"] is not None or raiz["tipo"] != "projeto":
        raise ValueError("O arquivo deve conter exatamente uma raiz de projeto (nível 0).")
    if sum(n["tipo"] == "projeto" for n in limpos) != 1:
        raise ValueError("Só pode existir um componente do tipo projeto.")
    for no in limpos:
        if no["id"] == "root":
            continue
        if no["parent_id"] not in mapa:
            raise ValueError(f"'{no['nome']}' não possui pai válido.")
        if mapa[no["parent_id"]]["tipo"] == "pacote":
            raise ValueError("Um pacote de trabalho não pode possuir componentes filhos.")

    # Confirma aciclicidade e vinculação de todos os componentes à raiz.
    filhos: dict[str, list[str]] = defaultdict(list)
    for no in limpos:
        if no["parent_id"] is not None:
            filhos[no["parent_id"]].append(no["id"])
    visitados: set[str] = set()
    pilha = ["root"]
    while pilha:
        atual = pilha.pop()
        if atual in visitados:
            raise ValueError("O arquivo contém um ciclo na hierarquia.")
        visitados.add(atual)
        pilha.extend(filhos[atual])
    if len(visitados) != len(limpos):
        raise ValueError("Existem componentes desconectados do projeto.")
    return {"versao": VERSAO, "nodes": limpos}


def carregar_json(conteudo: bytes) -> dict[str, Any]:
    if len(conteudo) > 2_000_000:
        raise ValueError("Arquivo muito grande (limite de 2 MB).")
    try:
        objeto = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Não foi possível interpretar o JSON enviado.") from exc
    return validar(objeto)


def exportar_json(documento: dict[str, Any]) -> bytes:
    return json.dumps(validar(documento), indent=2, ensure_ascii=False).encode("utf-8")


def agregar(documento: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    """Agrega custo e quantidade de pacotes, sem somar prazos como cronograma."""
    filhos = filhos_por_pai(documento)
    resultado: dict[str, dict[str, float | int]] = {}

    def visitar(ident: str) -> tuple[float, int]:
        no = indice_local[ident]
        if no["tipo"] == "pacote":
            custo, quant = no["custo_reais"], 1
        else:
            valores = [visitar(f["id"]) for f in filhos.get(ident, [])]
            custo = sum(v[0] for v in valores)
            quant = sum(v[1] for v in valores)
        resultado[ident] = {"custo": float(custo), "pacotes": quant}
        return float(custo), quant

    indice_local = indice(documento)
    visitar("root")
    return resultado


def linhas(documento: dict[str, Any]) -> list[dict[str, Any]]:
    """Percurso em profundidade que determina os códigos EAP pela ordem dos irmãos."""
    filhos = filhos_por_pai(documento)
    totais = agregar(documento)
    resultado = []

    def visitar(no: dict[str, Any], codigo: str, nivel: int) -> None:
        resultado.append({
            "id": no["id"], "parent_id": no["parent_id"],
            "codigo": codigo, "nivel": nivel, "tipo": no["tipo"],
            "nome": no["nome"], "descricao": no["descricao"],
            "duracao_dias": no["duracao_dias"] if no["tipo"] == "pacote" else None,
            "custo_pacote": no["custo_reais"] if no["tipo"] == "pacote" else None,
            "custo_consolidado": totais[no["id"]]["custo"],
            "pacotes_subarvore": totais[no["id"]]["pacotes"],
        })
        for i, filho in enumerate(filhos.get(no["id"], []), start=1):
            visitar(filho, f"{codigo}.{i}", nivel + 1)

    visitar(indice(documento)["root"], "1", 0)
    return resultado


def descendentes(documento: dict[str, Any], ident: str) -> set[str]:
    filhos = filhos_por_pai(documento)
    encontrados = set()
    pilha = [ident]
    while pilha:
        atual = pilha.pop()
        if atual in encontrados:
            continue
        encontrados.add(atual)
        pilha.extend(f["id"] for f in filhos.get(atual, []))
    return encontrados


def adicionar(
    documento: dict[str, Any], parent_id: str, tipo: str, nome: str,
    descricao: str = "", duracao_dias: float = 0.0, custo_reais: float = 0.0,
) -> tuple[dict[str, Any], str]:
    projeto = copy.deepcopy(documento)
    novos_ids = set(indice(projeto))
    if len(novos_ids) >= LIMITE_COMPONENTES:
        raise ValueError(f"A EAP chegou ao limite de {LIMITE_COMPONENTES} componentes.")
    if parent_id not in novos_ids or indice(projeto)[parent_id]["tipo"] == "pacote":
        raise ValueError("Selecione o projeto ou uma entrega como componente pai.")
    if tipo not in {"entrega", "pacote"}:
        raise ValueError("Tipo inválido para um componente subordinado.")
    identificador = f"n_{uuid.uuid4().hex}"
    projeto["nodes"].append({
        "id": identificador, "parent_id": parent_id, "tipo": tipo,
        "nome": nome, "descricao": descricao,
        "duracao_dias": duracao_dias if tipo == "pacote" else 0,
        "custo_reais": custo_reais if tipo == "pacote" else 0,
    })
    return validar(projeto), identificador


def editar(
    documento: dict[str, Any], ident: str, nome: str, descricao: str,
    tipo: str | None = None, parent_id: str | None = None,
    duracao_dias: float = 0.0, custo_reais: float = 0.0,
) -> dict[str, Any]:
    projeto = copy.deepcopy(documento)
    mapa = indice(projeto)
    if ident not in mapa:
        raise ValueError("O componente selecionado não existe.")
    no = mapa[ident]
    if ident == "root":
        tipo, parent_id = "projeto", None
    else:
        if tipo not in {"entrega", "pacote"}:
            raise ValueError("Tipo de componente inválido.")
        if tipo == "pacote" and len(descendentes(projeto, ident)) > 1:
            raise ValueError("Para converter em pacote, mova ou exclua primeiro os componentes filhos.")
        if parent_id not in mapa or mapa[parent_id]["tipo"] == "pacote":
            raise ValueError("O novo componente pai deve ser o projeto ou uma entrega.")
        if parent_id in descendentes(projeto, ident):
            raise ValueError("Não é permitido mover um componente para dentro de si próprio.")
    mudou_pai = no["parent_id"] != parent_id
    no.update({
        "nome": nome, "descricao": descricao, "tipo": tipo,
        "parent_id": parent_id,
        "duracao_dias": duracao_dias if tipo == "pacote" else 0,
        "custo_reais": custo_reais if tipo == "pacote" else 0,
    })
    if mudou_pai:
        # Move para o final da ordem do novo conjunto de irmãos.
        projeto["nodes"] = [n for n in projeto["nodes"] if n["id"] != ident] + [no]
    return validar(projeto)


def excluir(documento: dict[str, Any], ident: str) -> tuple[dict[str, Any], int]:
    if ident == "root":
        raise ValueError("O projeto não pode ser excluído; use 'Nova EAP'.")
    if ident not in indice(documento):
        raise ValueError("Componente inexistente.")
    remover = descendentes(documento, ident)
    projeto = copy.deepcopy(documento)
    projeto["nodes"] = [n for n in projeto["nodes"] if n["id"] not in remover]
    return validar(projeto), len(remover)


def reordenar(documento: dict[str, Any], ident: str, direcao: int) -> dict[str, Any]:
    if ident == "root" or direcao not in {-1, 1}:
        raise ValueError("Operação de ordenação inválida.")
    projeto = copy.deepcopy(documento)
    mapa = indice(projeto)
    if ident not in mapa:
        raise ValueError("Componente inexistente.")
    parent_id = mapa[ident]["parent_id"]
    ids_irmaos = [n["id"] for n in projeto["nodes"] if n["parent_id"] == parent_id]
    pos = ids_irmaos.index(ident)
    destino = pos + direcao
    if not 0 <= destino < len(ids_irmaos):
        return projeto
    outro = ids_irmaos[destino]
    local_a = next(i for i, n in enumerate(projeto["nodes"]) if n["id"] == ident)
    local_b = next(i for i, n in enumerate(projeto["nodes"]) if n["id"] == outro)
    projeto["nodes"][local_a], projeto["nodes"][local_b] = (
        projeto["nodes"][local_b], projeto["nodes"][local_a]
    )
    return validar(projeto)


def dinheiro_br(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def numero_br(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def exportar_csv(documento: dict[str, Any]) -> bytes:
    destino = io.StringIO(newline="")
    escritor = csv.writer(destino, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    escritor.writerow([
        "Código EAP", "Nível", "Tipo", "Componente", "Descrição",
        "Duração do pacote (dias úteis)", "Custo do pacote (R$)",
        "Custo consolidado (R$)",
    ])
    for linha in linhas(documento):
        escritor.writerow([
            linha["codigo"], linha["nivel"], linha["tipo"], linha["nome"],
            linha["descricao"],
            "" if linha["duracao_dias"] is None else numero_br(linha["duracao_dias"]),
            "" if linha["custo_pacote"] is None else numero_br(linha["custo_pacote"]),
            numero_br(linha["custo_consolidado"]),
        ])
    return destino.getvalue().encode("utf-8-sig")


def texto_hierarquico(documento: dict[str, Any]) -> str:
    partes = []
    for linha in linhas(documento):
        recuo = "    " * linha["nivel"]
        titulo = f"{linha['codigo']}  {linha['nome']}"
        if linha["tipo"] == "pacote":
            detalhe = (
                f"  [pacote | {numero_br(linha['duracao_dias'])} dias úteis | "
                f"{dinheiro_br(linha['custo_pacote'])}]"
            )
        else:
            detalhe = f"  [custo consolidado: {dinheiro_br(linha['custo_consolidado'])}]"
        partes.append(recuo + titulo + detalhe)
    return "\n".join(partes)


def _quebrar(texto: str, largura: int = 23, linhas_max: int = 4) -> str:
    partes = textwrap.wrap(texto, width=largura, break_long_words=True,
                           break_on_hyphens=False) or [""]
    if len(partes) > linhas_max:
        partes = partes[:linhas_max]
        partes[-1] = partes[-1].rstrip(" .") + "…"
    return '<BR ALIGN="LEFT"/>'.join(html.escape(parte, quote=True) for parte in partes)


def gerar_grafo(documento: dict[str, Any], orientacao: str = "vertical") -> Digraph:
    """Grafo baseado nas caixas clássicas da ferramenta de referência."""
    if orientacao not in {"vertical", "horizontal"}:
        raise ValueError("Orientação inválida.")
    grafo = Digraph("EAP", engine="dot")
    grafo.attr(
        rankdir="TB" if orientacao == "vertical" else "LR",
        bgcolor="white", splines="ortho", nodesep="0.30", ranksep="0.73",
        pad="0.26", dpi="155", outputorder="edgesfirst",
    )
    grafo.attr("node", shape="plain", fontname="DejaVu Sans")
    grafo.attr("edge", color="#8B9BB0", penwidth="1.35", arrowhead="none")
    ordem = linhas(documento)
    for item in ordem:
        tipo = item["tipo"]
        if tipo == "projeto":
            borda, faixa, texto_faixa, corpo = "#204C80", "#173B67", "#FFFFFF", "#EDF3FA"
            titulo_tipo = "PROJETO · NÍVEL 0"
        elif tipo == "entrega":
            borda, faixa, texto_faixa, corpo = "#4778AB", "#E2EDFA", "#244A79", "#FFFFFF"
            titulo_tipo = f"ENTREGA · NÍVEL {item['nivel']}"
        else:
            borda, faixa, texto_faixa, corpo = "#C8991A", "#FFF0B4", "#755410", "#FFFBEA"
            titulo_tipo = f"PACOTE · NÍVEL {item['nivel']}"

        cabecalho = html.escape(f"{item['codigo']}  {titulo_tipo}")
        titulo = _quebrar(item["nome"])
        if tipo == "pacote":
            informacao = (
                f"{numero_br(item['duracao_dias'])} dias úteis"
                f"  |  {dinheiro_br(item['custo_pacote'])}"
            )
        else:
            informacao = f"Custo consolidado: {dinheiro_br(item['custo_consolidado'])}"
        informacao = html.escape(informacao)
        rotulo = f'''<
        <TABLE BORDER="1" COLOR="{borda}" CELLBORDER="0" CELLSPACING="0"
               CELLPADDING="6" BGCOLOR="{corpo}">
          <TR><TD ALIGN="LEFT" BGCOLOR="{faixa}">
            <FONT FACE="DejaVu Sans" POINT-SIZE="9" COLOR="{texto_faixa}"><B>{cabecalho}</B></FONT>
          </TD></TR>
          <TR><TD ALIGN="LEFT" WIDTH="215">
            <FONT FACE="DejaVu Sans" POINT-SIZE="12" COLOR="#192B40"><B>{titulo}</B></FONT>
          </TD></TR>
          <TR><TD ALIGN="LEFT">
            <FONT FACE="DejaVu Sans" POINT-SIZE="9" COLOR="#44576E">{informacao}</FONT>
          </TD></TR>
        </TABLE>
        >'''
        grafo.node(item["id"], label=rotulo)
        if item["parent_id"]:
            grafo.edge(item["parent_id"], item["id"])
    return grafo


def criar_exemplo() -> dict[str, Any]:
    """Exemplo fictício para demonstração, não é carregado por padrão."""
    p = criar_projeto("Implantação de Laboratório de IA")
    p, infra = adicionar(p, "root", "entrega", "Infraestrutura preparada")
    p, fisico = adicionar(p, infra, "entrega", "Ambiente físico adequado")
    p, _ = adicionar(p, fisico, "pacote", "Espaço reformado", duracao_dias=15, custo_reais=28000)
    p, _ = adicionar(p, fisico, "pacote", "Mobiliário instalado", duracao_dias=5, custo_reais=12000)
    p, rede = adicionar(p, infra, "entrega", "Rede e energia disponíveis")
    p, _ = adicionar(p, rede, "pacote", "Rede configurada", duracao_dias=4, custo_reais=8500)
    p, _ = adicionar(p, rede, "pacote", "Instalação elétrica certificada", duracao_dias=6, custo_reais=15000)
    p, digital = adicionar(p, "root", "entrega", "Ambiente digital disponível")
    p, _ = adicionar(p, digital, "pacote", "Servidores provisionados", duracao_dias=8, custo_reais=45000)
    p, _ = adicionar(p, digital, "pacote", "Plataforma de ML implantada", duracao_dias=10, custo_reais=30000)
    p, pessoas = adicionar(p, "root", "entrega", "Equipe capacitada")
    p, _ = adicionar(p, pessoas, "pacote", "Material didático aprovado", duracao_dias=5, custo_reais=4500)
    p, _ = adicionar(p, pessoas, "pacote", "Treinamento concluído", duracao_dias=3, custo_reais=9000)
    return p
