"""Carga del clasificador de imágenes y obtención de probabilidades por clase."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Protocol

from PIL import Image

log = logging.getLogger("sigagro-ia")


class Clasificador(Protocol):
    id: str
    version: str | None
    etiquetas: list[str]

    def clasificar(self, img: Image.Image) -> list[tuple[str, float]]:
        """Probabilidad (softmax) de TODAS las clases, ordenadas de mayor a menor."""
        ...


class ClasificadorHuggingFace:
    """Modelo `image-classification` de Hugging Face (transformers + PyTorch, CPU)."""

    def __init__(self, modelo_id: str, revision: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._torch = torch
        # Modelos locales (carpetas): se identifican por el nombre de la carpeta, sin exponer rutas del equipo
        self.id = Path(modelo_id).name if Path(modelo_id).is_dir() else modelo_id
        self.procesador = AutoImageProcessor.from_pretrained(modelo_id, revision=revision)
        self.modelo = AutoModelForImageClassification.from_pretrained(modelo_id, revision=revision)
        self.modelo.eval()
        # Commit exacto del modelo descargado (o versión del entrenamiento local): identifica el modelo usado
        self.version = revision or getattr(self.modelo.config, "_commit_hash", None) or _version_local(modelo_id)
        id2label = self.modelo.config.id2label
        self.etiquetas = [id2label[i] for i in sorted(id2label)]
        self._lock = threading.Lock()

    def clasificar(self, img: Image.Image) -> list[tuple[str, float]]:
        entradas = self.procesador(images=img, return_tensors="pt")
        with self._lock, self._torch.no_grad():
            logits = self.modelo(**entradas).logits[0]
        probs = self._torch.softmax(logits, dim=-1).tolist()
        return sorted(zip(self.etiquetas, probs), key=lambda x: x[1], reverse=True)


def _version_local(ruta: str) -> str | None:
    """Modelos entrenados localmente (entrenamiento/): la versión sale de metricas.json."""
    try:
        m = json.loads((Path(ruta) / "metricas.json").read_text(encoding="utf-8"))
        return m.get("version") or "local"
    except (OSError, ValueError):
        return "local" if Path(ruta).is_dir() else None


class GestorModelo:
    """Carga los modelos una sola vez (al iniciar) y recuerda el error si no se pudieron cargar.

    `clasificador` es el modelo principal; `adicionales` cubren otros cultivos (p. ej. sigagro-extensivos
    para soja, girasol y maní). El análisis elige el modelo que cubre el cultivo del lote.
    """

    def __init__(self):
        self.clasificador: Clasificador | None = None
        self.error: str | None = None
        self.adicionales: list[Clasificador] = []
        self.errores_adicionales: list[str] = []

    def cargar_adicional(self, ruta: str) -> None:
        try:
            log.info("Cargando modelo adicional %s...", ruta)
            c = ClasificadorHuggingFace(ruta)
            self.adicionales.append(c)
            log.info("Modelo adicional cargado: %d clases, versión %s", len(c.etiquetas), c.version)
        except Exception as e:  # noqa: BLE001 - el servicio sigue con el modelo principal
            self.errores_adicionales.append(f"No se pudo cargar el modelo adicional '{ruta}': {e}")
            log.exception(self.errores_adicionales[-1])

    def cargar(self, modelo_id: str, revision: str | None) -> None:
        try:
            log.info("Cargando modelo %s (revisión %s)...", modelo_id, revision or "última")
            self.clasificador = ClasificadorHuggingFace(modelo_id, revision)
            self.error = None
            log.info("Modelo cargado: %d clases, versión %s", len(self.clasificador.etiquetas), self.clasificador.version)
        except Exception as e:  # noqa: BLE001 - se informa por /salud y la API responde 503
            self.clasificador = None
            self.error = f"No se pudo cargar el modelo '{modelo_id}': {e}"
            log.exception(self.error)
