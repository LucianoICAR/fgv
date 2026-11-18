"""Testes do motor de EAP: python -m unittest testes_eap.py -v"""

import json
import unittest

from eap_core import (
    adicionar, agregar, carregar_json, criar_projeto, descendentes, editar, excluir,
    exportar_csv, exportar_json, gerar_grafo, indice, linhas, reordenar, validar,
)


class TestesEAP(unittest.TestCase):
    def setUp(self):
        p = criar_projeto("Minha EAP")
        p, self.a = adicionar(p, "root", "entrega", "Entrega A")
        p, self.b = adicionar(p, "root", "entrega", "Entrega B")
        p, self.wp = adicionar(p, self.a, "pacote", "Produto A1", duracao_dias=3, custo_reais=1200)
        self.doc = p

    def test_estrutura_e_codigos(self):
        codigos = {r["id"]: r["codigo"] for r in linhas(self.doc)}
        self.assertEqual(codigos["root"], "1")
        self.assertEqual(codigos[self.a], "1.1")
        self.assertEqual(codigos[self.b], "1.2")
        self.assertEqual(codigos[self.wp], "1.1.1")
        self.assertEqual(linhas(self.doc)[0]["nivel"], 0)

    def test_custo_consolidado(self):
        self.assertEqual(agregar(self.doc)["root"]["custo"], 1200)
        self.assertEqual(agregar(self.doc)[self.b]["custo"], 0)
        self.assertEqual(agregar(self.doc)["root"]["pacotes"], 1)

    def test_editar_pacote(self):
        doc = editar(self.doc, self.wp, "Produto aprovado", "Conforme aceite", "pacote", self.a, 8, 3750)
        self.assertEqual(indice(doc)[self.wp]["duracao_dias"], 8)
        self.assertEqual(indice(doc)[self.wp]["nome"], "Produto aprovado")
        self.assertEqual(agregar(doc)["root"]["custo"], 3750)

    def test_mover_e_reordenar(self):
        doc = editar(self.doc, self.wp, "Produto A1", "", "pacote", self.b, 3, 1200)
        codigos = {r["id"]: r["codigo"] for r in linhas(doc)}
        self.assertEqual(codigos[self.wp], "1.2.1")
        doc = reordenar(doc, self.b, -1)
        codigos = {r["id"]: r["codigo"] for r in linhas(doc)}
        self.assertEqual(codigos[self.b], "1.1")
        self.assertEqual(codigos[self.wp], "1.1.1")
        self.assertEqual(codigos[self.a], "1.2")

    def test_bloquear_pacote_com_filhos(self):
        with self.assertRaises(ValueError):
            adicionar(self.doc, self.wp, "pacote", "Inadequado")
        with self.assertRaises(ValueError):
            editar(self.doc, self.a, "A", "", "pacote", "root", 0, 0)
        with self.assertRaises(ValueError):
            editar(self.doc, self.a, "A", "", "entrega", self.wp, 0, 0)
        with self.assertRaises(ValueError):
            editar(self.doc, self.a, "A", "", "entrega", self.a, 0, 0)

    def test_exclusao_de_subarvore(self):
        doc, removidos = excluir(self.doc, self.a)
        self.assertEqual(removidos, 2)
        self.assertEqual(len(doc["nodes"]), 2)
        self.assertNotIn(self.wp, indice(doc))
        with self.assertRaises(ValueError):
            excluir(self.doc, "root")

    def test_salvar_reabrir_e_csv(self):
        restaurado = carregar_json(exportar_json(self.doc))
        self.assertEqual(restaurado, self.doc)
        csv = exportar_csv(self.doc)
        self.assertTrue(csv.startswith(b"\xef\xbb\xbf"))
        self.assertIn("Custo consolidado", csv.decode("utf-8-sig"))

    def test_rejeitar_documento_malformado(self):
        obj = json.loads(exportar_json(self.doc))
        obj["nodes"][1]["parent_id"] = self.wp
        with self.assertRaises(ValueError):
            validar(obj)
        obj = json.loads(exportar_json(self.doc))
        obj["nodes"][1]["parent_id"] = self.a
        with self.assertRaises(ValueError):
            validar(obj)

    def test_renderizacao_e_escapes(self):
        doc, ident = adicionar(self.doc, self.b, "pacote", "Risco <alto> & garantia", custo_reais=4)
        dot = gerar_grafo(doc).source
        self.assertIn("&lt;alto&gt; &amp; garantia", dot)
        imagem = gerar_grafo(doc).pipe(format="png")
        self.assertTrue(imagem.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
