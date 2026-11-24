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
from .imagenes import ImagenDemasiadoGrande, ImagenInvalida, decodificar, evaluar_calidad
from .modelo import Clasificador

VERSION_CONTRATO = 2
AVISO_PROBABLE = (
    "Resultado probable generado por un modelo de IA: no es un diagnóstico confirmado. "
    "Debe revisarlo y validarlo un Ingeniero Agrónomo."
)
AVISO_GRAVEDAD = "El modelo no estima la gravedad de la afección: la determina el Ingeniero Agrónomo en la revisión."


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

    cubiertos = cultivos_cubiertos(clasificador)
    cultivo_lote = normalizar_cultivo(cultivo)
    if cultivo and not cultivo_lote:
        return _base(
            "NO_SOPORTADO",
            f"El modelo de IA no está entrenado para el cultivo '{cultivo}'. "
            f"Cultivos que puede analizar: {', '.join(cubiertos) or 'ninguno conocido'}. "
            "La muestra queda registrada para que la evalúe el Ingeniero Agrónomo.",
            clasificador,
            calidad,
        )

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
    if cultivo_lote and info.cultivo and info.cultivo != cultivo_lote:
        motivos.append(f"el modelo reconoce {info.cultivo.lower()}, pero el cultivo indicado es {cultivo_lote.lower()}")

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
    return r
