"""
Análisis en línea con un modelo multimodal (Claude, de Anthropic) para los cultivos que el modelo local no cubre
(soja, girasol, maní, sorgo, cebada...) o cuando el modelo local no llega a un resultado concluyente.

Reglas (las mismas del servicio):
- El resultado es un diagnóstico PROBABLE: lo valida el Ingeniero Agrónomo en SIGAgro.
- No estima la gravedad (`severidad` siempre null).
- Nunca nombra productos comerciales, principios activos ni dosis: esas salen del catálogo fitosanitario oficial.
  Además del pedido al modelo, `limpiar_texto` descarta cualquier frase que traiga una dosis.
- La `confianza` es una ESTIMACIÓN del propio modelo (no una probabilidad calibrada como la del modelo local):
  se informa como tal en `tipoConfianza` y en las observaciones.
- Si no hay clave de API, no hay internet o el servicio falla, se devuelve None y queda el resultado local.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Literal, Protocol

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("sigagro-ia.en-linea")

# Cultivos extensivos y regionales de Argentina que se ofrecen cuando el análisis en línea está habilitado
CULTIVOS_ARGENTINA = [
    "Soja", "Maíz", "Trigo", "Girasol", "Maní", "Sorgo", "Cebada", "Avena", "Centeno", "Arroz", "Papa",
    "Poroto", "Garbanzo", "Lenteja", "Arveja", "Algodón", "Caña de azúcar", "Colza", "Alfalfa", "Tabaco",
    "Yerba mate", "Vid", "Cítricos", "Olivo", "Tomate", "Pimiento", "Cebolla", "Ajo",
]


class DiagnosticoAlternativo(BaseModel):
    afeccion: str
    confianza: float


class EvaluacionEnLinea(BaseModel):
    """Lo que devuelve el modelo (validado contra el esquema)."""

    esImagenAnalizable: bool
    cultivoReconocido: str | None
    coincideConCultivoIndicado: bool | None
    resultado: Literal["ENFERMEDAD", "PLAGA", "DEFICIENCIA", "SANO", "NO_IDENTIFICABLE"]
    afeccion: str | None
    agente: str | None
    confianza: float = Field(description="0 a 1")
    descripcion: str | None
    sintomas: list[str]
    condiciones: list[str]
    consecuencias: list[str]
    alternativasControl: list[str]
    diagnosticosAlternativos: list[DiagnosticoAlternativo]
    motivoSiNoIdentifica: str | None


def _nulo(tipo: dict) -> dict:
    return {"anyOf": [tipo, {"type": "null"}]}


_TEXTO = {"type": "string"}
_LISTA = {"type": "array", "items": _TEXTO}

ESQUEMA = {
    "type": "object",
    "properties": {
        "esImagenAnalizable": {"type": "boolean"},
        "cultivoReconocido": _nulo(_TEXTO),
        "coincideConCultivoIndicado": _nulo({"type": "boolean"}),
        "resultado": {"type": "string", "enum": ["ENFERMEDAD", "PLAGA", "DEFICIENCIA", "SANO", "NO_IDENTIFICABLE"]},
        "afeccion": _nulo(_TEXTO),
        "agente": _nulo(_TEXTO),
        "confianza": {"type": "number"},
        "descripcion": _nulo(_TEXTO),
        "sintomas": _LISTA,
        "condiciones": _LISTA,
        "consecuencias": _LISTA,
        "alternativasControl": _LISTA,
        "diagnosticosAlternativos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"afeccion": _TEXTO, "confianza": {"type": "number"}},
                "required": ["afeccion", "confianza"],
                "additionalProperties": False,
            },
        },
        "motivoSiNoIdentifica": _nulo(_TEXTO),
    },
    "required": [
        "esImagenAnalizable", "cultivoReconocido", "coincideConCultivoIndicado", "resultado", "afeccion", "agente",
        "confianza", "descripcion", "sintomas", "condiciones", "consecuencias", "alternativasControl",
        "diagnosticosAlternativos", "motivoSiNoIdentifica",
    ],
    "additionalProperties": False,
}

SISTEMA = """Sos un asistente de fitopatología para productores y técnicos agrícolas de Argentina, integrado en SIGAgro.
Analizás una fotografía tomada en un lote y proponés un diagnóstico PROBABLE de lo que se ve, que después revisa y valida un Ingeniero Agrónomo.

Reglas obligatorias:
1. Prudencia: si la foto no permite identificar con razonable seguridad (borrosa, lejana, varias plantas, síntomas inespecíficos), respondé resultado NO_IDENTIFICABLE y explicá en motivoSiNoIdentifica qué foto haría falta. No adivines.
2. Cultivo: identificá el cultivo que ves en la foto (cultivoReconocido). El usuario indica el cultivo sembrado en el lote; si lo que ves no corresponde a ese cultivo, indicá coincideConCultivoIndicado=false.
3. Confianza: un número entre 0 y 1 que refleje honestamente tu seguridad en la afección (no en el cultivo). Usá valores bajos cuando haya varias causas posibles; listalas en diagnosticosAlternativos.
4. Priorizá las enfermedades, plagas y deficiencias relevantes en Argentina para ese cultivo, con el nombre común usado en Argentina y el agente causal (nombre científico) cuando corresponda.
5. NUNCA nombres productos comerciales, principios activos ni dosis. En alternativasControl describí solo medidas generales de manejo (monitoreo, umbrales de daño, rotación, genética resistente, manejo de rastrojo, consultar el catálogo fitosanitario y la receta del Ingeniero Agrónomo).
6. No estimes la gravedad ni la incidencia en el lote: eso lo determina el Ingeniero a campo.
7. Si no hay afección visible y la planta se ve sana, resultado SANO.
8. Escribí todo en español, en frases breves."""

# Frases con dosis o concentraciones (p. ej. "2 l/ha", "500 cc/ha", "0,5 %"): se descartan siempre
_PATRON_DOSIS = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:l|lt|lts|litros?|cc|ml|cm3|kg|g|gr|gramos?)\s*(?:/|por)\s*(?:ha|hect[aá]rea|hl|100\s*l)"
    r"|\d+(?:[.,]\d+)?\s*%\s*(?:p/v|v/v|de\s+concentraci[oó]n)",
    re.IGNORECASE,
)


def limpiar_texto(lista: list[str]) -> list[str]:
    return [t.strip() for t in lista if isinstance(t, str) and t.strip() and not _PATRON_DOSIS.search(t)]


def imagen_a_jpeg_base64(img: Image.Image, lado_maximo: int = 1568) -> str:
    """Re-codifica la imagen ya validada a JPEG (máximo 1568 px por lado) para enviarla al modelo."""
    copia = img.convert("RGB")
    copia.thumbnail((lado_maximo, lado_maximo))
    buffer = io.BytesIO()
    copia.save(buffer, format="JPEG", quality=85)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _pedido(cultivo: str | None, tipo_muestra: str | None, observaciones: str | None) -> str:
    partes = [
        f"Cultivo sembrado en el lote (indicado por el usuario): {cultivo or 'no indicado'}.",
        f"Tipo de muestra: {'planta u hoja individual' if (tipo_muestra or 'PLANTA') == 'PLANTA' else 'vista general del cultivo'}.",
    ]
    if observaciones:
        partes.append(f"Observaciones de campo del usuario: {observaciones.strip()[:600]}")
    partes.append("Analizá la fotografía y respondé con el esquema JSON pedido.")
    return "\n".join(partes)


class AnalizadorEnLinea(Protocol):
    id: str

    def evaluar(
        self, img: Image.Image, cultivo: str | None, tipo_muestra: str | None, observaciones: str | None
    ) -> EvaluacionEnLinea | None: ...


class AnalizadorClaude:
    """Cliente de Claude (SDK oficial `anthropic`). La clave se toma de ANTHROPIC_API_KEY."""

    def __init__(self, modelo: str, timeout_s: float, esfuerzo: str = "medium"):
        import anthropic  # importación diferida: el servicio funciona sin el paquete si no se usa

        self.id = modelo
        self._anthropic = anthropic
        self._cliente = anthropic.Anthropic(timeout=timeout_s, max_retries=1)
        self._esfuerzo = esfuerzo

    def evaluar(self, img, cultivo, tipo_muestra, observaciones):
        a = self._anthropic
        try:
            respuesta = self._cliente.beta.messages.create(
                model=self.id,
                max_tokens=16000,
                # Si el modelo declinara el pedido, la API lo reintenta en un modelo alternativo en la misma llamada
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SISTEMA,
                output_config={"effort": self._esfuerzo, "format": {"type": "json_schema", "schema": ESQUEMA}},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": imagen_a_jpeg_base64(img)},
                            },
                            {"type": "text", "text": _pedido(cultivo, tipo_muestra, observaciones)},
                        ],
                    }
                ],
            )
        except a.AuthenticationError:
            log.error("Análisis en línea: la clave ANTHROPIC_API_KEY no es válida.")
            return None
        except a.RateLimitError:
            log.warning("Análisis en línea: límite de uso alcanzado; se usa el resultado local.")
            return None
        except a.APIStatusError as e:
            log.warning("Análisis en línea: error %s del servicio (%s).", e.status_code, getattr(e, "message", e))
            return None
        except a.APIConnectionError:
            log.warning("Análisis en línea: sin conexión a internet; se usa el resultado local.")
            return None

        if respuesta.stop_reason == "refusal":
            log.warning("Análisis en línea: el modelo declinó el pedido.")
            return None
        if respuesta.stop_reason == "max_tokens":
            log.warning("Análisis en línea: respuesta incompleta (max_tokens).")
            return None
        texto = next((b.text for b in respuesta.content if b.type == "text"), None)
        if not texto:
            return None
        try:
            evaluacion = EvaluacionEnLinea.model_validate(json.loads(texto))
        except (json.JSONDecodeError, ValidationError) as e:
            log.warning("Análisis en línea: respuesta fuera de esquema (%s).", e)
            return None
        log.info("Análisis en línea %s: %s / %s (%.2f) [req %s]", self.id, evaluacion.resultado, evaluacion.afeccion,
                 evaluacion.confianza, getattr(respuesta, "_request_id", "-"))
        return evaluacion


def crear_analizador(habilitado: bool, modelo: str, timeout_s: float, esfuerzo: str) -> AnalizadorEnLinea | None:
    if not habilitado:
        return None
    try:
        return AnalizadorClaude(modelo, timeout_s, esfuerzo)
    except Exception as e:  # noqa: BLE001 - paquete ausente o configuración inválida: queda solo el modelo local
        log.error("No se pudo iniciar el análisis en línea: %s", e)
        return None
