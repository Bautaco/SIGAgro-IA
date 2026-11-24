"""Carga del clasificador de imágenes y obtención de probabilidades por clase."""

from __future__ import annotations

import logging
import threading
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
        self.id = modelo_id
        self.procesador = AutoImageProcessor.from_pretrained(modelo_id, revision=revision)
        self.modelo = AutoModelForImageClassification.from_pretrained(modelo_id, revision=revision)
        self.modelo.eval()
        # Commit exacto del modelo descargado: identifica la versión usada en cada diagnóstico
        self.version = revision or getattr(self.modelo.config, "_commit_hash", None)
        id2label = self.modelo.config.id2label
        self.etiquetas = [id2label[i] for i in sorted(id2label)]
        self._lock = threading.Lock()

    def clasificar(self, img: Image.Image) -> list[tuple[str, float]]:
        entradas = self.procesador(images=img, return_tensors="pt")
        with self._lock, self._torch.no_grad():
            logits = self.modelo(**entradas).logits[0]
        probs = self._torch.softmax(logits, dim=-1).tolist()
        return sorted(zip(self.etiquetas, probs), key=lambda x: x[1], reverse=True)


class GestorModelo:
    """Carga el modelo una sola vez (al iniciar) y recuerda el error si no se pudo cargar."""

    def __init__(self):
        self.clasificador: Clasificador | None = None
        self.error: str | None = None

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
