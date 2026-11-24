"""Decodificación y control de calidad de la fotografía antes de pasarla al modelo."""

from __future__ import annotations

import base64
import binascii
import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Config

FORMATOS_ADMITIDOS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(?P<datos>.+)$", re.DOTALL)


class ImagenInvalida(Exception):
    """El contenido no es una imagen admitida (formato, codificación o tamaño)."""


class ImagenDemasiadoGrande(ImagenInvalida):
    pass


@dataclass
class CalidadImagen:
    ancho: int
    alto: int
    formato: str
    brillo: float
    nitidez: float
    problemas: list[str]

    @property
    def legible(self) -> bool:
        return not self.problemas

    def a_dict(self) -> dict:
        return {
            "ancho": self.ancho,
            "alto": self.alto,
            "formato": self.formato,
            "brillo": round(self.brillo, 1),
            "nitidez": round(self.nitidez, 1),
            "legible": self.legible,
            "problemas": self.problemas,
        }


def decodificar(imagen: str, config: Config) -> tuple[Image.Image, str]:
    """Acepta una data URL (data:image/...;base64,...) o base64 plano. No descarga URLs remotas."""
    texto = (imagen or "").strip()
    if texto.lower().startswith(("http://", "https://")):
        raise ImagenInvalida(
            "El servicio solo recibe la imagen embebida (data URL base64); no descarga URLs remotas."
        )
    m = _DATA_URL.match(texto)
    datos_b64 = m.group("datos") if m else texto
    # Rechaza antes de decodificar si el base64 ya excede el máximo (4 caracteres = 3 bytes)
    if len(datos_b64) * 3 // 4 > config.max_bytes:
        raise ImagenDemasiadoGrande(f"La imagen supera el máximo de {config.max_bytes // (1024 * 1024)} MB.")
    try:
        crudo = base64.b64decode(datos_b64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ImagenInvalida("La imagen no está codificada en base64 válido.") from e
    if not crudo:
        raise ImagenInvalida("La imagen está vacía.")
    if len(crudo) > config.max_bytes:
        raise ImagenDemasiadoGrande(f"La imagen supera el máximo de {config.max_bytes // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(crudo))
        formato = img.format or ""
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ImagenInvalida("El archivo no es una imagen legible.") from e
    if formato not in FORMATOS_ADMITIDOS:
        raise ImagenInvalida(f"Formato no admitido ({formato or 'desconocido'}). Se admiten JPG, PNG y WEBP.")
    # Respeta la orientación EXIF de las fotos de celular
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img, FORMATOS_ADMITIDOS[formato]


def _varianza_laplaciano(gris: np.ndarray) -> float:
    """Medida clásica de nitidez: varianza del laplaciano (baja = imagen borrosa)."""
    g = gris.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def evaluar_calidad(img: Image.Image, formato: str, config: Config) -> CalidadImagen:
    ancho, alto = img.size
    problemas: list[str] = []
    if min(ancho, alto) < config.lado_minimo_px:
        problemas.append(
            f"Resolución insuficiente ({ancho}×{alto} px; mínimo {config.lado_minimo_px} px por lado)."
        )
    # La nitidez se mide sobre una copia reducida a 512 px para que no dependa del tamaño original
    reducida = img.copy()
    reducida.thumbnail((512, 512))
    gris = np.asarray(reducida.convert("L"))
    brillo = float(gris.mean())
    nitidez = _varianza_laplaciano(gris) if min(gris.shape) >= 3 else 0.0
    if brillo < config.brillo_minimo:
        problemas.append("La foto está demasiado oscura.")
    elif brillo > config.brillo_maximo:
        problemas.append("La foto está sobreexpuesta (demasiado clara).")
    if nitidez < config.nitidez_minima:
        problemas.append("La foto está desenfocada o movida.")
    return CalidadImagen(ancho, alto, formato, brillo, nitidez, problemas)
