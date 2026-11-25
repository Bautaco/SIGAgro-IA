"""Calibración de la confianza (temperatura) y umbral mínimo por cultivo."""

from __future__ import annotations

import json
import math

import torch

from app.calibracion import (
    Calibracion,
    UMBRAL_MAXIMO,
    ajustar_temperatura,
    cargar,
    error_calibracion,
    softmax,
    umbral_para_precision,
)

from .test_modelos_adicionales import ExtensivosFalso, cliente
from .test_api import inferir


def test_la_temperatura_no_cambia_la_clase_ganadora_y_suaviza_la_confianza():
    logits = [4.0, 1.0, 0.5]
    crudo, suave = softmax(logits), softmax(logits, 2.0)
    assert crudo.index(max(crudo)) == suave.index(max(suave)) == 0
    assert max(suave) < max(crudo)
    assert math.isclose(sum(suave), 1.0)


def test_ajusta_temperatura_mayor_a_uno_para_un_modelo_sobreconfiado():
    torch.manual_seed(0)
    n, c = 2000, 5
    y = torch.randint(0, c, (n,))
    # Logits muy grandes y con 30 % de errores: el modelo dice ~100 % pero acierta ~70 %
    pred = torch.where(torch.rand(n) < 0.7, y, torch.randint(0, c, (n,)))
    logits = torch.nn.functional.one_hot(pred, c).float() * 12
    t = ajustar_temperatura(logits, y)
    assert t > 2
    conf_cruda = torch.softmax(logits, -1).max(-1).values.tolist()
    conf_cal = torch.softmax(logits / t, -1).max(-1).values.tolist()
    aciertos = (logits.argmax(-1) == y).tolist()
    assert error_calibracion(conf_cal, aciertos) < error_calibracion(conf_cruda, aciertos)


def test_umbral_para_precision():
    # 30 predicciones de alta confianza correctas y 30 de baja confianza con la mitad de errores
    conf = [0.9] * 30 + [0.5] * 30
    aciertos = [True] * 30 + [True, False] * 15
    u = umbral_para_precision(conf, aciertos, objetivo=0.85, minimo=20)
    assert u == 0.9  # bajar a 0,5 deja la precisión en 75 %
    assert umbral_para_precision([0.9] * 5, [True] * 5, minimo=20) is None  # pocas fotos: sin umbral propio
    assert umbral_para_precision([0.6] * 30, [False] * 30, minimo=20) == UMBRAL_MAXIMO  # nunca alcanza el objetivo


def test_carga_calibracion_y_descarta_valores_invalidos(tmp_path):
    (tmp_path / "calibracion.json").write_text(json.dumps({"temperatura": 1.7, "umbralesPorCultivo": {"Sorgo": 0.8}}), encoding="utf-8")
    c = cargar(tmp_path)
    assert c.temperatura == 1.7 and c.umbrales_por_cultivo == {"Sorgo": 0.8} and c.activa
    (tmp_path / "calibracion.json").write_text(json.dumps({"temperatura": -3}), encoding="utf-8")
    assert cargar(tmp_path) == Calibracion()
    assert cargar(tmp_path / "no-existe") == Calibracion()


class ExtensivosCalibrado(ExtensivosFalso):
    calibracion = Calibracion(1.0, {"Soja": 0.9})


def test_el_servicio_exige_el_umbral_del_cultivo_cuando_es_mayor_al_general():
    # 88 % supera el 60 % general, pero no el 90 % que exige la calibración para soja: queda inconcluso
    r = inferir(cliente(ExtensivosCalibrado({"Soybean___Frog_Eye_Leaf_Spot": 0.88, "Soybean___Healthy": 0.12})), cultivo="Soja").json()
    assert r["estado"] == "INCONCLUSO"
    assert "90%" in r["mensaje"]


def test_sin_calibracion_rige_el_umbral_general():
    r = inferir(cliente(ExtensivosFalso({"Soybean___Frog_Eye_Leaf_Spot": 0.88, "Soybean___Healthy": 0.12})), cultivo="Soja").json()
    assert r["estado"] == "OK"
