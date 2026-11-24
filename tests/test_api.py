"""Tests de la API con un clasificador falso (no descargan el modelo)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import cargar_config
from app.main import crear_app
from app.modelo import GestorModelo

ETIQUETAS = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Invalid",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
]


class ClasificadorFalso:
    id = "falso/modelo"
    version = "abc123"
    etiquetas = ETIQUETAS

    def __init__(self, probs: dict[str, float]):
        self.probs = probs

    def clasificar(self, img):
        todas = {e: self.probs.get(e, 0.0) for e in self.etiquetas}
        return sorted(todas.items(), key=lambda x: x[1], reverse=True)


def imagen(ancho=400, alto=300, ruido=True, valor=120, formato="JPEG") -> str:
    rng = np.random.default_rng(0)
    if ruido:
        arr = rng.integers(40, 220, size=(alto, ancho, 3), dtype=np.uint8)
    else:
        arr = np.full((alto, ancho, 3), valor, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=formato)
    mime = {"JPEG": "jpeg", "PNG": "png", "GIF": "gif"}[formato]
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def cliente(probs: dict[str, float], token: str = "", en_linea=None, **cfg) -> TestClient:
    config = cargar_config()
    config = config.__class__(**{**config.__dict__, "token_servicio": token, **cfg})
    gestor = GestorModelo()
    gestor.clasificador = ClasificadorFalso(probs)
    # en_linea explícito: los tests nunca llaman a un servicio externo
    return TestClient(crear_app(config, gestor, cargar_modelo=False, en_linea=en_linea))


def inferir(c: TestClient, **body):
    return c.post("/inferencia", json={"imagen": imagen(), **body})


def test_resultado_ok_con_informacion_de_referencia_y_sin_gravedad():
    r = inferir(cliente({"Corn___Common_Rust": 0.91, "Corn___Gray_Leaf_Spot": 0.06, "Corn___Healthy": 0.03}), cultivo="Maíz")
    assert r.status_code == 200
    d = r.json()
    assert d["estado"] == "OK"
    assert d["afeccion"] == "Roya común del maíz"
    assert d["confianza"] == 0.91  # probabilidad del modelo, sin ajustes
    assert d["severidad"] is None  # el modelo no estima gravedad
    assert d["cultivo"] == "Maíz"
    assert d["sintomas"] and d["alternativasControl"]
    assert d["alternativas"][0]["afeccion"] == "Mancha gris del maíz"
    assert d["modelo"] == {"id": "falso/modelo", "version": "abc123"}
    assert any("no es un diagnóstico confirmado" in o for o in d["observaciones"])


def test_ninguna_recomendacion_incluye_dosis():
    d = inferir(cliente({"Corn___Common_Rust": 0.95, "Corn___Healthy": 0.05})).json()
    texto = " ".join(d["alternativasControl"]).lower()
    for unidad in ("l/ha", "cc/ha", "kg/ha", "g/ha", "ml/ha"):
        assert unidad not in texto


def test_baja_confianza_es_inconcluso_y_no_presenta_afeccion():
    d = inferir(cliente({"Corn___Common_Rust": 0.45, "Corn___Gray_Leaf_Spot": 0.40, "Corn___Healthy": 0.15})).json()
    assert d["estado"] == "INCONCLUSO"
    assert d["afeccion"] is None and d["sintomas"] == [] and d["recomendaciones"] == ""
    assert "nueva fotografía" in d["mensaje"]
    assert d["alternativas"][0]["afeccion"] == "Roya común del maíz"


def test_clases_parejas_es_inconcluso_aunque_supere_el_umbral():
    d = inferir(cliente({"Corn___Common_Rust": 0.62, "Corn___Gray_Leaf_Spot": 0.55}, umbral_confianza=0.5)).json()
    assert d["estado"] == "INCONCLUSO"


def test_cultivo_distinto_al_del_lote_se_analiza_igual_y_avisa():
    d = inferir(cliente({"Wheat___Brown_Rust": 0.93, "Wheat___Healthy": 0.07}), cultivo="Maíz").json()
    assert d["estado"] == "OK"  # la foto se analiza siempre
    assert d["afeccion"] == "Roya anaranjada del trigo" or "trigo" in d["afeccion"].lower()
    assert d["cultivoNoCoincide"] is True and d["cultivoIndicado"] == "Maíz"
    assert "Maíz" in d["observaciones"][0] and "trigo" in d["observaciones"][0]


def test_cultivo_coincidente_no_avisa():
    d = inferir(cliente({"Corn___Common_Rust": 0.95, "Corn___Healthy": 0.05}), cultivo="Maíz 2da").json()
    assert d["estado"] == "OK" and d["cultivoNoCoincide"] is False and d["cultivoIndicado"] == "Maíz 2da"


def test_cultivo_no_cubierto_se_analiza_con_los_modelos_locales_y_avisa():
    d = inferir(cliente({"Corn___Common_Rust": 0.99}), cultivo="Soja").json()
    assert d["estado"] == "OK"
    assert d["cultivoNoCoincide"] is True
    assert any("no está entre los que reconocen los modelos locales" in o for o in d["observaciones"])


def test_muestra_de_suelo_no_soportada():
    d = inferir(cliente({"Corn___Common_Rust": 0.99}), tipoMuestra="SUELO").json()
    assert d["estado"] == "NO_SOPORTADO"


def test_clase_invalid_del_modelo_es_imagen_invalida():
    d = inferir(cliente({"Invalid": 0.97, "Corn___Healthy": 0.03})).json()
    assert d["estado"] == "IMAGEN_INVALIDA"


def test_contenido_que_no_es_imagen():
    c = cliente({"Corn___Healthy": 1.0})
    d = c.post("/inferencia", json={"imagen": "data:image/jpeg;base64," + base64.b64encode(b"hola").decode()}).json()
    assert d["estado"] == "IMAGEN_INVALIDA"


def test_formato_no_admitido():
    c = cliente({"Corn___Healthy": 1.0})
    d = c.post("/inferencia", json={"imagen": imagen(formato="GIF")}).json()
    assert d["estado"] == "IMAGEN_INVALIDA" and "Formato no admitido" in d["mensaje"]


def test_no_descarga_urls_remotas():
    c = cliente({"Corn___Healthy": 1.0})
    d = c.post("/inferencia", json={"imagen": "http://169.254.169.254/latest"}).json()
    assert d["estado"] == "IMAGEN_INVALIDA"


@pytest.mark.parametrize(
    "img,problema",
    [
        (imagen(ancho=100, alto=80), "Resolución insuficiente"),
        (imagen(ruido=False, valor=10), "oscura"),
        (imagen(ruido=False, valor=250), "sobreexpuesta"),
        (imagen(ruido=False, valor=120), "desenfocada"),
    ],
)
def test_imagen_ilegible(img, problema):
    c = cliente({"Corn___Healthy": 1.0})
    d = c.post("/inferencia", json={"imagen": img}).json()
    assert d["estado"] == "IMAGEN_ILEGIBLE"
    assert problema in d["mensaje"]


def test_imagen_demasiado_grande_413():
    c = cliente({"Corn___Healthy": 1.0}, max_bytes=1000)
    assert inferir(c).status_code == 413


def test_token_de_servicio():
    c = cliente({"Corn___Healthy": 1.0}, token="secreto")
    assert inferir(c).status_code == 401
    r = c.post("/inferencia", json={"imagen": imagen()}, headers={"Authorization": "Bearer secreto"})
    assert r.status_code == 200
    assert c.get("/salud").status_code == 200  # la salud no requiere token


def test_modelo_no_cargado_503():
    config = cargar_config()
    app = crear_app(config, GestorModelo(), cargar_modelo=False, en_linea=None)
    c = TestClient(app)
    assert inferir(c).status_code == 503
    assert c.get("/salud").json()["estado"] == "MODELO_NO_DISPONIBLE"


def test_info_del_modelo():
    d = cliente({"Corn___Healthy": 1.0}).get("/modelo").json()
    assert d["cultivosCubiertos"] == ["Maíz", "Trigo"]
    assert d["estimaGravedad"] is False
