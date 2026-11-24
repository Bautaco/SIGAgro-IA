"""Análisis en línea (Claude) con un analizador falso: sin red, sin clave y sin costo."""

from __future__ import annotations

from app.en_linea import EvaluacionEnLinea, limpiar_texto

from .test_api import cliente, inferir


class EnLineaFalso:
    id = "claude-opus-5-5"

    def __init__(self, respuesta: dict | None):
        self.respuesta = respuesta
        self.llamadas: list[dict] = []

    def evaluar(self, img, cultivo, tipo_muestra, observaciones):
        self.llamadas.append({"cultivo": cultivo, "tipo": tipo_muestra, "obs": observaciones})
        return EvaluacionEnLinea.model_validate(self.respuesta) if self.respuesta else None


def evaluacion(**cambios) -> dict:
    base = {
        "esImagenAnalizable": True,
        "cultivoReconocido": "Soja",
        "coincideConCultivoIndicado": True,
        "resultado": "ENFERMEDAD",
        "afeccion": "Mancha ojo de rana",
        "agente": "Cercospora sojina (hongo)",
        "confianza": 0.82,
        "descripcion": "Manchas circulares con centro gris y borde castaño rojizo.",
        "sintomas": ["Manchas circulares de 1 a 5 mm con borde rojizo"],
        "condiciones": ["Alta humedad y temperaturas templadas a cálidas"],
        "consecuencias": ["Defoliación anticipada y pérdida de rendimiento"],
        "alternativasControl": [
            "Monitorear el lote y estimar incidencia antes de decidir",
            "Aplicar 0,5 l/ha de un fungicida",  # trae dosis: se tiene que descartar
            "Usar cultivares resistentes",
        ],
        "diagnosticosAlternativos": [{"afeccion": "Mancha marrón (Septoria glycines)", "confianza": 0.1}],
        "motivoSiNoIdentifica": None,
    }
    return {**base, **cambios}


def test_cultivo_no_cubierto_usa_el_analisis_en_linea():
    falso = EnLineaFalso(evaluacion())
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=falso), cultivo="Soja", observaciones="Manchas en hojas medias").json()
    assert r["estado"] == "OK"
    assert r["afeccion"] == "Mancha ojo de rana"
    assert r["origenAnalisis"] == "EN_LINEA" and r["tipoConfianza"] == "ESTIMADA"
    assert r["modelo"]["id"] == "claude-opus-5-5"
    assert r["severidad"] is None
    assert falso.llamadas[0] == {"cultivo": "Soja", "tipo": "PLANTA", "obs": "Manchas en hojas medias"}


def test_nunca_devuelve_dosis():
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=EnLineaFalso(evaluacion())), cultivo="Soja").json()
    assert all("l/ha" not in t for t in r["alternativasControl"])
    assert len(r["alternativasControl"]) == 2
    assert limpiar_texto(["Dosis 500 cc/ha", "Rotar cultivos", "2 kg por ha"]) == ["Rotar cultivos"]


def test_baja_confianza_en_linea_es_inconcluso():
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=EnLineaFalso(evaluacion(confianza=0.4))), cultivo="Soja").json()
    assert r["estado"] == "INCONCLUSO"
    assert r["afeccion"] is None and r["alternativasControl"] == []
    assert r["alternativas"][0]["afeccion"] == "Mancha ojo de rana"


def test_foto_de_otro_cultivo_se_analiza_y_avisa():
    falso = EnLineaFalso(evaluacion(cultivoReconocido="Maíz", coincideConCultivoIndicado=False, afeccion="Roya común del maíz"))
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=falso), cultivo="Girasol").json()
    assert r["estado"] == "OK" and r["afeccion"] == "Roya común del maíz"
    assert r["cultivoNoCoincide"] is True and r["cultivoIndicado"] == "Girasol"
    assert "Girasol" in r["observaciones"][0]


def test_no_identificable():
    falso = EnLineaFalso(evaluacion(resultado="NO_IDENTIFICABLE", afeccion=None, motivoSiNoIdentifica="La foto está muy lejos"))
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=falso), cultivo="Maní").json()
    assert r["estado"] == "INCONCLUSO"
    assert "muy lejos" in r["mensaje"]


def test_planta_sana():
    falso = EnLineaFalso(evaluacion(resultado="SANO", afeccion=None, agente=None))
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=falso), cultivo="Soja").json()
    assert r["estado"] == "OK" and r["tipoDeteccion"] == "SANO"
    assert r["afeccion"] == "Soja sin síntomas visibles"


def test_sin_conexion_analiza_con_los_modelos_locales():
    r = inferir(cliente({"Corn___Healthy": 1.0}, en_linea=EnLineaFalso(None)), cultivo="Soja").json()
    assert r["estado"] == "OK" and r["origenAnalisis"] == "LOCAL"
    assert r["cultivoNoCoincide"] is True


def test_cultivo_cubierto_y_concluyente_no_gasta_en_linea():
    falso = EnLineaFalso(evaluacion())
    r = inferir(cliente({"Corn___Common_Rust": 0.95, "Corn___Healthy": 0.05}, en_linea=falso), cultivo="Maíz").json()
    assert r["estado"] == "OK" and r["origenAnalisis"] == "LOCAL" and r["tipoConfianza"] == "PROBABILIDAD"
    assert falso.llamadas == []


def test_inconcluso_local_pide_segunda_opinion_en_linea():
    falso = EnLineaFalso(evaluacion(cultivoReconocido="Maíz", afeccion="Roya común del maíz", agente="Puccinia sorghi"))
    r = inferir(cliente({"Corn___Common_Rust": 0.5, "Corn___Gray_Leaf_Spot": 0.45}, en_linea=falso), cultivo="Maíz").json()
    assert r["estado"] == "OK" and r["origenAnalisis"] == "EN_LINEA"
    assert any("modelo local no fue concluyente" in o for o in r["observaciones"])


def test_suelo_y_fotos_ilegibles_no_se_envian():
    falso = EnLineaFalso(evaluacion())
    c = cliente({"Corn___Healthy": 1.0}, en_linea=falso)
    assert inferir(c, cultivo="Soja", tipoMuestra="SUELO").json()["estado"] == "NO_SOPORTADO"
    assert falso.llamadas == []


def test_modelo_informa_cultivos_en_linea():
    c = cliente({"Corn___Healthy": 1.0}, en_linea=EnLineaFalso(evaluacion()))
    m = c.get("/modelo").json()
    assert "Soja" in m["cultivosCubiertos"] and "Girasol" in m["cultivosCubiertos"]
    assert m["analisisEnLinea"] == {"habilitado": True, "modelo": "claude-opus-5-5"}
    assert c.get("/salud").json()["enLinea"]["habilitado"] is True
