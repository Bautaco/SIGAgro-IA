"""
Lógica del análisis: calidad de imagen -> clasificación -> interpretación prudente del resultado.

Principios:
- La confianza es la probabilidad (softmax) que asigna el modelo a la clase: nunca se ajusta ni se inventa.
- El modelo NO estima la gravedad: `severidad` se devuelve null y la determina el Ingeniero Agrónomo.
- Ante baja confianza, clases muy parejas, imagen ilegible, cultivo no cubierto o incoherencia con el cultivo
  del lote, el resultado es INCONCLUSO / NO_SOPORTADO y se pide otra foto o la revisión profesional.
- Un resultado OK es un diagnóstico PROBABLE, no una confirmación.
"""

from __future__ import annotations

from .config import Config
from .conocimiento import FUENTE, InfoClase, info_de, normalizar_cultivo
from .en_linea import AnalizadorEnLinea, EvaluacionEnLinea, limpiar_texto
from .imagenes import ImagenDemasiadoGrande, ImagenInvalida, decodificar, evaluar_calidad
from .modelo import Clasificador

VERSION_CONTRATO = 2
AVISO_PROBABLE = (
    "Resultado probable generado por un modelo de IA: no es un diagnóstico confirmado. "
    "Debe revisarlo y validarlo un Ingeniero Agrónomo."
)
AVISO_GRAVEDAD = "El modelo no estima la gravedad de la afección: la determina el Ingeniero Agrónomo en la revisión."
AVISO_EN_LINEA = (
    "Análisis realizado por un modelo multimodal en línea: la confianza es una estimación del propio modelo, "
    "no una probabilidad calibrada. La información es de referencia general."
)
FUENTE_EN_LINEA = "Análisis de un modelo de IA multimodal en línea; referencia general que debe validar el Ingeniero Agrónomo."


def cultivos_cubiertos(clasificador: Clasificador) -> list[str]:
    cultivos = {info.cultivo for e in clasificador.etiquetas if (info := info_de(e)) and info.cultivo}
    return sorted(cultivos)


def _info(etiqueta: str) -> InfoClase:
    return info_de(etiqueta) or InfoClase(cultivo=None, tipo="DESCONOCIDO", nombre=etiqueta)


def _base(estado: str, mensaje: str, clasificador: Clasificador | None, calidad=None) -> dict:
    return {
        "versionContrato": VERSION_CONTRATO,
        "estado": estado,
        "mensaje": mensaje,
        "afeccion": None,
        "confianza": None,
        "severidad": None,
        "tipoDeteccion": None,
        "cultivo": None,
        "especie": None,
        "estadoPlanta": None,
        "agente": None,
        "descripcion": None,
        "sintomas": [],
        "condiciones": [],
        "consecuencias": [],
        "alternativasControl": [],
        "recomendaciones": "",
        "alternativas": [],
        "observaciones": [],
        "etiquetaModelo": None,
        "modelo": {"id": clasificador.id, "version": clasificador.version} if clasificador else None,
        "calidadImagen": calidad.a_dict() if calidad else None,
        "fuenteInformacion": None,
        "origenAnalisis": "LOCAL",
        "tipoConfianza": "PROBABILIDAD",
        # La foto se analiza siempre; si el cultivo reconocido no es el sembrado en el lote, se avisa (no se descarta)
        "cultivoIndicado": None,
        "cultivoNoCoincide": False,
    }


class SolicitudDemasiadoGrande(Exception):
    pass


def analizar(
    imagen: str,
    clasificador: Clasificador,
    config: Config,
    *,
    cultivo: str | None = None,
    tipo_muestra: str | None = None,
    en_linea: AnalizadorEnLinea | None = None,
    observaciones: str | None = None,
    adicionales: list[Clasificador] | None = None,
) -> dict:
    tipo = (tipo_muestra or "PLANTA").upper()
    if tipo == "SUELO":
        return _base(
            "NO_SOPORTADO",
            "El modelo de IA analiza hojas de cultivos; no analiza muestras de suelo. "
            "Registre la muestra para que la evalúe el Ingeniero Agrónomo o envíela a un laboratorio.",
            clasificador,
        )

    try:
        img, formato = decodificar(imagen, config)
    except ImagenDemasiadoGrande as e:
        raise SolicitudDemasiadoGrande(str(e)) from e
    except ImagenInvalida as e:
        return _base("IMAGEN_INVALIDA", f"{e} Cargue otra fotografía.", clasificador)

    calidad = evaluar_calidad(img, formato, config)
    if not calidad.legible:
        r = _base(
            "IMAGEN_ILEGIBLE",
            "La fotografía no tiene calidad suficiente para analizarla: "
            + " ".join(calidad.problemas)
            + " Tome una nueva foto, cerca de la hoja, con buena luz y enfocada.",
            clasificador,
            calidad,
        )
        r["observaciones"] = calidad.problemas
        return r

    modelos = [clasificador, *(adicionales or [])]
    cubiertos = sorted({c for m in modelos for c in cultivos_cubiertos(m)})
    cultivo_lote = normalizar_cultivo(cultivo)
    if cultivo_lote and cultivo_lote not in cubiertos:
        cultivo_lote = None  # el cultivo es conocido pero ningún modelo local lo cubre
    no_cubierto = bool(cultivo and not cultivo_lote)

    # Cultivo que los modelos locales no cubren: primero el análisis en línea (reconoce cualquier cultivo)
    en_linea_probado = False
    if no_cubierto and en_linea:
        en_linea_probado = True
        r = _analizar_en_linea(img, calidad, en_linea, config, cultivo, tipo, observaciones)
        if r:
            return r  # el modelo en línea analizó la foto: su resultado vale más que el de un modelo que no conoce el cultivo

    # La foto se analiza SIEMPRE: con el modelo del cultivo sembrado o, si no hay uno, con todos los modelos
    # locales (se queda el resultado más seguro). Si lo reconocido no coincide con el lote, se avisa.
    candidatos = [m for m in modelos if cultivo_lote and cultivo_lote in cultivos_cubiertos(m)] or modelos
    local = _mejor_resultado([_clasificar_local(img, calidad, m, config, cultivo) for m in candidatos])
    if no_cubierto:
        local["observaciones"].append(
            f"El cultivo indicado ({cultivo.strip()}) no está entre los que reconocen los modelos locales "
            f"({', '.join(cubiertos)}): el resultado corresponde a lo que el modelo reconoce en la foto."
        )

    # Segunda opinión en línea cuando el modelo local no llega a un resultado (inconcluso o "no es una hoja")
    if en_linea and not en_linea_probado and local["estado"] in ("INCONCLUSO", "IMAGEN_INVALIDA"):
        r = _analizar_en_linea(img, calidad, en_linea, config, cultivo, tipo, observaciones)
        if r and r["estado"] == "OK":
            r["observaciones"].append(
                "El modelo local no fue concluyente (" + (local.get("mensaje") or local["estado"]).rstrip(".") + "); se usó el análisis en línea."
            )
            return r
    return local


def _mejor_resultado(resultados: list[dict]) -> dict:
    """Entre varios modelos, el resultado concluyente de mayor confianza (o el de mayor confianza si ninguno lo es)."""
    return max(resultados, key=lambda r: (r["estado"] == "OK", r.get("confianza") or 0.0))


def _avisar_cultivo(r: dict, indicado: str | None, reconocido: str | None) -> None:
    """Marca y explica la diferencia entre el cultivo sembrado en el lote y el reconocido en la foto."""
    r["cultivoIndicado"] = indicado.strip() if indicado else None
    if indicado and reconocido and not _mismo_cultivo(indicado, reconocido):
        r["cultivoNoCoincide"] = True
        r["observaciones"].insert(
            0,
            f"Atención: en el lote figura sembrado {indicado.strip()}, pero en la foto se reconoce {reconocido.lower()}. "
            "Verificá que la muestra corresponda a este lote; si no, cambiá el lote de la muestra.",
        )


def _clasificar_local(img, calidad, clasificador: Clasificador, config: Config, cultivo_indicado: str | None) -> dict:
    probabilidades = clasificador.clasificar(img)
    etiqueta, prob = probabilidades[0]
    info = _info(etiqueta)
    segunda = next((p for e, p in probabilidades[1:]), 0.0)
    alternativas = [
        {"etiqueta": e, "afeccion": _info(e).nombre, "cultivo": _info(e).cultivo, "confianza": round(p, 4)}
        for e, p in probabilidades[1:]
        if _info(e).tipo != "INVALIDA" and p >= 0.01
    ][: config.alternativas]

    if info.tipo == "INVALIDA" and prob >= config.umbral_confianza:
        r = _base(
            "IMAGEN_INVALIDA",
            "La imagen no parece una hoja de un cultivo que el modelo pueda analizar. "
            "Fotografíe de cerca una hoja con síntomas, ocupando la mayor parte de la imagen.",
            clasificador,
            calidad,
        )
        r["confianza"] = round(prob, 4)
        r["etiquetaModelo"] = etiqueta
        return r

    r = _base("OK", "", clasificador, calidad)
    r.update(
        {
            "afeccion": info.nombre,
            "confianza": round(prob, 4),
            "tipoDeteccion": info.tipo,
            "cultivo": info.cultivo,
            "agente": info.agente,
            "descripcion": info.descripcion,
            "sintomas": list(info.sintomas),
            "condiciones": list(info.condiciones),
            "consecuencias": list(info.consecuencias),
            "alternativasControl": list(info.control),
            "recomendaciones": "\n".join(info.control),
            "alternativas": alternativas,
            "etiquetaModelo": etiqueta,
            "fuenteInformacion": FUENTE if info_de(etiqueta) else None,
        }
    )
    observaciones = [AVISO_PROBABLE]
    if info.tipo == "SANO":
        r["estadoPlanta"] = "Sin síntomas visibles de las enfermedades que reconoce el modelo."
        observaciones.append(
            "El modelo solo reconoce algunas enfermedades foliares; no detecta plagas, malezas ni deficiencias. "
            "Si observa otros síntomas, consulte al Ingeniero Agrónomo."
        )
    elif info.tipo == "ENFERMEDAD":
        r["estadoPlanta"] = f"Con síntomas compatibles con {info.nombre.lower()}."
        observaciones.append(AVISO_GRAVEDAD)
    elif info.tipo == "DESCONOCIDO":
        observaciones.append("La clase devuelta por el modelo no tiene información de referencia en el servicio.")
    if not info_de(etiqueta):
        r["afeccion"] = etiqueta

    # Motivos para no dar el resultado como probable
    motivos: list[str] = []
    if prob < config.umbral_confianza:
        motivos.append(
            f"la confianza del modelo ({prob:.0%}) es menor al mínimo requerido ({config.umbral_confianza:.0%})"
        )
    elif prob - segunda < config.margen_minimo:
        motivos.append("hay más de un diagnóstico posible con probabilidades similares")
    if info.tipo == "INVALIDA":
        motivos.append("no se pudo confirmar que la imagen sea una hoja analizable")

    if motivos:
        r["estado"] = "INCONCLUSO"
        r["mensaje"] = (
            "Diagnóstico inconcluso: "
            + "; ".join(motivos)
            + ". Tome una nueva fotografía (hoja con síntomas, de cerca, con buena luz) o solicite la revisión del Ingeniero Agrónomo."
        )
        # En un resultado inconcluso la clase más probable se informa solo como alternativa
        r["alternativas"] = [
            {"etiqueta": etiqueta, "afeccion": r["afeccion"], "cultivo": info.cultivo, "confianza": round(prob, 4)},
            *alternativas,
        ][: config.alternativas + 1] if info.tipo != "INVALIDA" else alternativas
        # No se presenta información de una afección que no quedó identificada
        r.update(
            {
                "afeccion": None,
                "tipoDeteccion": None,
                "cultivo": None,
                "estadoPlanta": None,
                "agente": None,
                "descripcion": None,
                "sintomas": [],
                "condiciones": [],
                "consecuencias": [],
                "alternativasControl": [],
                "recomendaciones": "",
                "fuenteInformacion": None,
            }
        )
        observaciones = [r["mensaje"]]
    r["observaciones"] = observaciones
    _avisar_cultivo(r, cultivo_indicado, info.cultivo if info.tipo != "INVALIDA" else None)
    return r


def _mismo_cultivo(a: str | None, b: str | None) -> bool:
    """'Soja 2da' ≈ 'soja', 'Maíz tardío' ≈ 'maiz': se compara la especie (primera palabra, sin tildes)."""
    if not a or not b:
        return False
    n = lambda t: t.strip().lower().translate(str.maketrans("áéíóúü", "aeiouu")).split()[0]  # noqa: E731
    return n(a) == n(b)


def _analizar_en_linea(
    img,
    calidad,
    en_linea: AnalizadorEnLinea,
    config: Config,
    cultivo: str | None,
    tipo_muestra: str,
    observaciones: str | None,
) -> dict | None:
    """Análisis con el modelo en línea. None si no está disponible (se usa el resultado local)."""
    ev: EvaluacionEnLinea | None = en_linea.evaluar(img, cultivo, tipo_muestra, observaciones)
    if ev is None:
        return None
    conf = min(max(float(ev.confianza or 0), 0.0), 1.0)
    alternativas = [
        {"etiqueta": None, "afeccion": d.afeccion, "cultivo": ev.cultivoReconocido, "confianza": round(min(max(d.confianza, 0.0), 1.0), 4)}
        for d in ev.diagnosticosAlternativos
        if d.afeccion and d.afeccion.strip()
    ][: config.alternativas]

    r = _base("OK", "", None, calidad)
    r["modelo"] = {"id": en_linea.id, "version": None}
    r["origenAnalisis"] = "EN_LINEA"
    r["tipoConfianza"] = "ESTIMADA"

    if not ev.esImagenAnalizable:
        r["estado"] = "IMAGEN_INVALIDA"
        r["mensaje"] = (
            (ev.motivoSiNoIdentifica or "La imagen no permite analizar el cultivo.")
            + " Fotografíe de cerca una hoja o planta con síntomas, con buena luz."
        )
        r["observaciones"] = [r["mensaje"]]
        return r

    motivos: list[str] = []
    if ev.resultado == "NO_IDENTIFICABLE":
        motivos.append((ev.motivoSiNoIdentifica or "no se pudo identificar la afección con esta foto").rstrip(".").lower())
    elif conf < config.umbral_en_linea:
        motivos.append(f"la confianza estimada ({conf:.0%}) es menor al mínimo requerido ({config.umbral_en_linea:.0%})")

    if motivos:
        r["estado"] = "INCONCLUSO"
        r["mensaje"] = (
            "Diagnóstico inconcluso: " + "; ".join(motivos)
            + ". Tome una nueva fotografía (hoja con síntomas, de cerca, con buena luz) o solicite la revisión del Ingeniero Agrónomo."
        )
        principal = (
            [{"etiqueta": None, "afeccion": ev.afeccion, "cultivo": ev.cultivoReconocido, "confianza": round(conf, 4)}]
            if ev.afeccion and ev.resultado != "NO_IDENTIFICABLE"
            else []
        )
        r["alternativas"] = (principal + alternativas)[: config.alternativas + 1]
        r["observaciones"] = [r["mensaje"], AVISO_EN_LINEA]
        _avisar_cultivo(r, cultivo, ev.cultivoReconocido or ("otro cultivo" if ev.coincideConCultivoIndicado is False else None))
        return r

    cultivo_final = ev.cultivoReconocido or (cultivo.strip() if cultivo else None)
    control = limpiar_texto(ev.alternativasControl)
    sano = ev.resultado == "SANO"
    r.update(
        {
            "afeccion": (f"{cultivo_final} sin síntomas visibles" if cultivo_final else "Sin síntomas visibles") if sano else ev.afeccion,
            "confianza": round(conf, 4),
            "tipoDeteccion": ev.resultado,
            "cultivo": cultivo_final,
            "agente": ev.agente,
            "descripcion": ev.descripcion,
            "sintomas": limpiar_texto(ev.sintomas),
            "condiciones": limpiar_texto(ev.condiciones),
            "consecuencias": limpiar_texto(ev.consecuencias),
            "alternativasControl": control,
            "recomendaciones": "\n".join(control),
            "alternativas": alternativas,
            "estadoPlanta": "Sin síntomas visibles." if sano else f"Con síntomas compatibles con {(ev.afeccion or 'una afección').lower()}.",
            "fuenteInformacion": FUENTE_EN_LINEA,
        }
    )
    obs = [AVISO_PROBABLE]
    if not sano:
        obs.append(AVISO_GRAVEDAD)
    obs.append(AVISO_EN_LINEA)
    r["observaciones"] = obs
    if not r["afeccion"]:
        return None  # respuesta incoherente: mejor quedarse con el resultado local
    _avisar_cultivo(r, cultivo, ev.cultivoReconocido or ("otro cultivo" if ev.coincideConCultivoIndicado is False else None))
    return r
