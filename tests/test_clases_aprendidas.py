"""La IA aprende de lo que escribe el Ingeniero: afecciones nuevas a partir de diagnósticos validados."""

import json

from PIL import Image

from app import conocimiento
from entrenamiento import importar_validadas as imp


def _foto(ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), (90, 140, 60)).save(ruta, format="JPEG")


def _preparar(tmp_path, monkeypatch, n, afeccion="Mancha anillada (Corynespora)", cultivo="Soja"):
    monkeypatch.setattr(imp, "DATOS", tmp_path / "datos")
    monkeypatch.setattr(imp, "SIN_CLASE", tmp_path / "sin_clase")
    monkeypatch.setattr(imp, "CLASES_APRENDIDAS", tmp_path / "clases_aprendidas.json")
    manifiesto = []
    for i in range(n):
        _foto(tmp_path / "sin_clase" / f"sigagro_d{i}.jpg")
        # Variantes de escritura de la misma afección se agrupan
        texto = afeccion if i % 4 else afeccion.upper().replace("A", "Á", 1)
        manifiesto.append({"id": f"d{i}", "afeccion": texto, "cultivo": cultivo})
    return manifiesto


def test_con_suficientes_fotos_crea_la_clase_con_el_nombre_del_ingeniero(tmp_path, monkeypatch):
    manifiesto = _preparar(tmp_path, monkeypatch, 4)
    resto, creadas = imp.crear_clases_aprendidas(manifiesto, minimo=3)
    assert creadas == ["Soybean___Ing_mancha_anillada_corynespora"]
    assert resto == []
    assert len(list((tmp_path / "datos" / creadas[0]).glob("*.jpg"))) == 4
    registro = json.loads((tmp_path / "clases_aprendidas.json").read_text(encoding="utf-8"))
    assert registro[creadas[0]]["cultivo"] == "Soja"
    assert registro[creadas[0]]["nombre"] == "Mancha anillada (Corynespora)"
    assert registro[creadas[0]]["fotos"] == 4


def test_con_pocas_fotos_quedan_esperando(tmp_path, monkeypatch):
    manifiesto = _preparar(tmp_path, monkeypatch, 2)
    resto, creadas = imp.crear_clases_aprendidas(manifiesto, minimo=3)
    assert creadas == [] and len(resto) == 2
    assert not (tmp_path / "datos").exists()


def test_la_etiqueta_respeta_el_formato_que_acepta_el_backend(tmp_path, monkeypatch):
    manifiesto = _preparar(tmp_path, monkeypatch, 3, afeccion="Carbón 2 (raro)", cultivo="Caña de azúcar")
    _, creadas = imp.crear_clases_aprendidas(manifiesto, minimo=3)
    import re

    assert re.fullmatch(r"[A-Za-z]+___[A-Za-z_]+", creadas[0]), creadas[0]
    assert creadas[0].startswith("Sugarcane___Ing_")


def test_el_servicio_carga_la_clase_aprendida_sin_inventar_informacion(tmp_path, monkeypatch):
    carpeta = tmp_path / "modelos" / "sigagro-extensivos"
    carpeta.mkdir(parents=True)
    (carpeta / "clases_aprendidas.json").write_text(
        json.dumps({"Soybean___Ing_mancha_anillada": {"nombre": "Mancha anillada", "cultivo": "Soja", "fotos": 31}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(conocimiento, "CLASES", dict(conocimiento.CLASES))
    assert conocimiento.cargar_clases_aprendidas(tmp_path / "modelos") == 1
    info = conocimiento.info_de("Soybean___Ing_mancha_anillada")
    assert info.cultivo == "Soja" and info.tipo == "APRENDIDA" and info.nombre == "Mancha anillada"
    assert info.sintomas == [] and info.agente is None and info.condiciones == []
    assert "31 diagnósticos validados" in info.descripcion


def test_no_reemplaza_una_clase_con_ficha_tecnica(tmp_path, monkeypatch):
    carpeta = tmp_path / "modelos" / "m"
    carpeta.mkdir(parents=True)
    (carpeta / "clases_aprendidas.json").write_text(json.dumps({"Soybean___Rust": {"nombre": "Otra cosa", "cultivo": "Soja"}}), encoding="utf-8")
    monkeypatch.setattr(conocimiento, "CLASES", dict(conocimiento.CLASES))
    assert conocimiento.cargar_clases_aprendidas(tmp_path / "modelos") == 0
    assert conocimiento.info_de("Soybean___Rust").agente
