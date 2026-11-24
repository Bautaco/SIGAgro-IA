"""Configuración del servicio de IA (variables de entorno, ver .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _float(nombre: str, defecto: float) -> float:
    try:
        return float(os.environ.get(nombre, defecto))
    except ValueError:
        return defecto


def _int(nombre: str, defecto: int) -> int:
    try:
        return int(os.environ.get(nombre, defecto))
    except ValueError:
        return defecto


@dataclass(frozen=True)
class Config:
    # Modelo de clasificación de Hugging Face (image-classification). Se puede reemplazar por uno propio.
    modelo_id: str
    # Revisión fija del modelo (commit de Hugging Face) para que los resultados sean reproducibles.
    modelo_revision: str | None
    # Por debajo de esta probabilidad el diagnóstico es INCONCLUSO (se pide otra foto).
    umbral_confianza: float
    # Si la 1ª y 2ª clase están más cerca que esto, el diagnóstico es INCONCLUSO.
    margen_minimo: float
    # Token compartido con el backend (Authorization: Bearer). Vacío = sin autenticación (solo desarrollo).
    token_servicio: str
    # Tamaño máximo de la imagen decodificada desde base64.
    max_bytes: int
    # Calidad mínima de la imagen.
    lado_minimo_px: int
    brillo_minimo: float
    brillo_maximo: float
    nitidez_minima: float
    # Cantidad de diagnósticos alternativos que se devuelven.
    alternativas: int
    # Análisis en línea (Claude) para cultivos que el modelo local no cubre o resultados no concluyentes.
    # EN_LINEA: auto (si hay ANTHROPIC_API_KEY) | si | no
    # Modelos locales adicionales (rutas o ids de Hugging Face) para otros cultivos
    modelos_adicionales: tuple[str, ...] = ()
    en_linea_habilitado: bool = False
    modelo_en_linea: str = "claude-opus-5-5"
    timeout_en_linea: float = 45.0
    esfuerzo_en_linea: str = "medium"
    umbral_en_linea: float = 0.6


def cargar_config() -> Config:
    return Config(
        modelo_id=os.environ.get("MODELO_ID", "wambugu71/crop_leaf_diseases_vit").strip(),
        modelo_revision=(os.environ.get("MODELO_REVISION") or "").strip() or None,
        umbral_confianza=_float("UMBRAL_CONFIANZA", 0.60),
        margen_minimo=_float("MARGEN_MINIMO", 0.15),
        token_servicio=(os.environ.get("TOKEN_SERVICIO") or "").strip(),
        max_bytes=_int("MAX_BYTES", 8 * 1024 * 1024),
        lado_minimo_px=_int("LADO_MINIMO_PX", 160),
        brillo_minimo=_float("BRILLO_MINIMO", 35),
        brillo_maximo=_float("BRILLO_MAXIMO", 235),
        nitidez_minima=_float("NITIDEZ_MINIMA", 25),
        alternativas=_int("ALTERNATIVAS", 3),
        modelos_adicionales=_modelos_adicionales(),
        en_linea_habilitado=_en_linea_habilitado(),
        modelo_en_linea=(os.environ.get("MODELO_EN_LINEA") or "claude-opus-5-5").strip(),
        timeout_en_linea=_float("TIMEOUT_EN_LINEA", 45.0),
        esfuerzo_en_linea=(os.environ.get("ESFUERZO_EN_LINEA") or "medium").strip(),
        umbral_en_linea=_float("UMBRAL_EN_LINEA", 0.6),
    )


def _modelos_adicionales() -> tuple[str, ...]:
    """MODELOS_ADICIONALES=ruta1,ruta2. Sin la variable, se usa modelos/sigagro-extensivos si existe."""
    from pathlib import Path

    texto = os.environ.get("MODELOS_ADICIONALES")
    if texto is None:
        por_defecto = Path(__file__).resolve().parent.parent / "modelos" / "sigagro-extensivos"
        return (str(por_defecto),) if (por_defecto / "config.json").exists() else ()
    return tuple(x.strip() for x in texto.split(",") if x.strip())


def _en_linea_habilitado() -> bool:
    modo = (os.environ.get("EN_LINEA") or "auto").strip().lower()
    if modo in ("no", "false", "0"):
        return False
    tiene_clave = bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
    return tiene_clave if modo == "auto" else tiene_clave or modo in ("si", "sí", "true", "1")
