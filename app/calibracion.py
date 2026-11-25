"""
Calibración de la confianza del modelo.

La probabilidad cruda (softmax) de una red suele no coincidir con el acierto real: un «90 %» puede acertar
bastante menos. Se corrige con *temperature scaling* (Guo et al., 2017): los logits se dividen por una temperatura
T ajustada con el conjunto de VALIDACIÓN (nunca con el de prueba). No cambia qué clase gana; solo hace que la
confianza informada se parezca al acierto observado.

Además, por cultivo se calcula el umbral mínimo de confianza (ya calibrada) desde el que las predicciones de
validación alcanzan la precisión objetivo. El servicio usa el MAYOR entre ese umbral y `UMBRAL_CONFIANZA`: si el
modelo es menos confiable en un cultivo (p. ej. sorgo, sin fotos de plantas sanas), pide otra foto con más
frecuencia en vez de dar un diagnóstico dudoso.

El resultado se guarda junto al modelo en `calibracion.json` (lo genera `entrenamiento/calibrar.py`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ARCHIVO = "calibracion.json"
PRECISION_OBJETIVO = 0.85
MINIMO_FOTOS_CULTIVO = 20
UMBRAL_MAXIMO = 0.95


@dataclass(frozen=True)
class Calibracion:
    temperatura: float = 1.0
    umbrales_por_cultivo: dict[str, float] = field(default_factory=dict)

    @property
    def activa(self) -> bool:
        return self.temperatura != 1.0 or bool(self.umbrales_por_cultivo)


def cargar(carpeta: str | Path) -> Calibracion:
    """Calibración guardada junto a un modelo local; sin archivo (o inválido), la identidad."""
    try:
        datos = json.loads((Path(carpeta) / ARCHIVO).read_text(encoding="utf-8"))
        t = float(datos.get("temperatura", 1.0))
        if not (0.05 <= t <= 20) or not math.isfinite(t):
            return Calibracion()
        umbrales = {str(k): float(v) for k, v in (datos.get("umbralesPorCultivo") or {}).items() if 0 <= float(v) <= 1}
        return Calibracion(t, umbrales)
    except (OSError, ValueError, TypeError, AttributeError):
        return Calibracion()


def softmax(logits: list[float], temperatura: float = 1.0) -> list[float]:
    escalados = [x / temperatura for x in logits]
    m = max(escalados)
    exps = [math.exp(x - m) for x in escalados]
    s = sum(exps)
    return [e / s for e in exps]


def ajustar_temperatura(logits, etiquetas, iteraciones: int = 200) -> float:
    """Temperatura que minimiza la log-verosimilitud negativa (logits: tensor N×C, etiquetas: tensor N)."""
    import torch

    logits = torch.as_tensor(logits, dtype=torch.float32)
    etiquetas = torch.as_tensor(etiquetas, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iteraciones)
    perdida = torch.nn.CrossEntropyLoss()

    def paso():
        opt.zero_grad()
        l = perdida(logits / log_t.exp(), etiquetas)
        l.backward()
        return l

    opt.step(paso)
    return float(log_t.exp().clamp(0.05, 20).item())


def error_calibracion(confianzas: list[float], aciertos: list[bool], bins: int = 15) -> float:
    """ECE: diferencia promedio (ponderada) entre la confianza informada y el acierto real, por tramos."""
    n = len(confianzas)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confianzas) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            conf = sum(confianzas[i] for i in idx) / len(idx)
            acc = sum(1 for i in idx if aciertos[i]) / len(idx)
            total += len(idx) / n * abs(conf - acc)
    return total


def umbral_para_precision(
    confianzas: list[float], aciertos: list[bool], objetivo: float = PRECISION_OBJETIVO, minimo: int = MINIMO_FOTOS_CULTIVO
) -> float | None:
    """
    Menor umbral t tal que, entre las predicciones con confianza ≥ t, la precisión sea ≥ objetivo (y queden al
    menos `minimo` fotos). None si hay pocas fotos; UMBRAL_MAXIMO si nunca se alcanza el objetivo.
    """
    if len(confianzas) < minimo:
        return None
    pares = sorted(zip(confianzas, aciertos), key=lambda p: p[0], reverse=True)
    mejor: float | None = None
    bien = 0
    for k, (c, a) in enumerate(pares, start=1):
        bien += 1 if a else 0
        # Un umbral t incluye TODAS las predicciones con confianza ≥ t: se evalúa solo al cerrar cada valor
        cierra_grupo = k == len(pares) or pares[k][0] < c
        if cierra_grupo and k >= minimo and bien / k >= objetivo:
            mejor = c  # bajando el umbral hasta acá la precisión sigue cumpliendo
    return round(min(mejor, UMBRAL_MAXIMO), 4) if mejor is not None else UMBRAL_MAXIMO
