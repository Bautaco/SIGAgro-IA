"""Varios modelos locales: el análisis usa el que cubre el cultivo del lote."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import cargar_config
from app.main import crear_app
from app.modelo import GestorModelo

from .test_api import ClasificadorFalso, inferir


class ExtensivosFalso:
    id = "modelos/sigagro-extensivos"
    version = "local-test"
    etiquetas = ["Groundnut___Healthy", "Groundnut___Late_Leaf_Spot", "Soybean___Frog_Eye_Leaf_Spot", "Soybean___Healthy"]

    def __init__(self, probs):
        self.probs = probs

    def clasificar(self, img):
        return sorted(((e, self.probs.get(e, 0.0)) for e in self.etiquetas), key=lambda x: x[1], reverse=True)


def cliente(adicional=None) -> TestClient:
    gestor = GestorModelo()
    gestor.clasificador = ClasificadorFalso({"Corn___Common_Rust": 0.9, "Corn___Healthy": 0.1})
    if adicional:
        gestor.adicionales = [adicional]
    return TestClient(crear_app(cargar_config(), gestor, cargar_modelo=False, en_linea=None))


def test_soja_usa_el_modelo_de_extensivos():
    c = cliente(ExtensivosFalso({"Soybean___Frog_Eye_Leaf_Spot": 0.88, "Soybean___Healthy": 0.12}))
    r = inferir(c, cultivo="Soja 2da").json()
    assert r["estado"] == "OK"
    assert r["afeccion"] == "Mancha ojo de rana de la soja"
    assert r["modelo"] == {"id": "modelos/sigagro-extensivos", "version": "local-test"}
    assert all("l/ha" not in t for t in r["alternativasControl"])


def test_maiz_sigue_usando_el_modelo_principal():
    c = cliente(ExtensivosFalso({"Soybean___Healthy": 1.0}))
    r = inferir(c, cultivo="Maíz").json()
    assert r["afeccion"] == "Roya común del maíz" and r["modelo"]["id"] == "falso/modelo"


def test_foto_de_mani_en_lote_de_soja_se_analiza_y_avisa():
    c = cliente(ExtensivosFalso({"Groundnut___Late_Leaf_Spot": 0.9, "Soybean___Healthy": 0.1}))
    r = inferir(c, cultivo="Soja").json()
    assert r["estado"] == "OK" and r["afeccion"] == "Viruela tardía del maní"
    assert r["cultivoNoCoincide"] is True


def test_sin_modelo_de_extensivos_la_soja_se_analiza_igual_con_aviso():
    r = inferir(cliente(), cultivo="Soja").json()
    assert r["estado"] == "OK" and r["cultivoNoCoincide"] is True


def test_sin_cultivo_usa_el_modelo_mas_seguro():
    c = cliente(ExtensivosFalso({"Soybean___Frog_Eye_Leaf_Spot": 0.97, "Soybean___Healthy": 0.03}))
    r = inferir(c).json()  # sin cultivo: se prueban todos los modelos (principal 0.9 vs extensivos 0.97)
    assert r["afeccion"] == "Mancha ojo de rana de la soja"


def test_modelo_informa_los_cultivos_de_todos_los_modelos():
    c = cliente(ExtensivosFalso({}))
    m = c.get("/modelo").json()
    assert {"Maíz", "Soja", "Maní"} <= set(m["cultivosModeloLocal"])
    assert c.get("/salud").json()["modelosAdicionales"][0]["cultivos"] == ["Maní", "Soja"]
