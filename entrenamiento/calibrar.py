"""
Calibra la confianza del modelo `sigagro-extensivos` (ver app/calibracion.py) y elige el umbral por cultivo.

1. Misma división que `entrenar.py` / `ajuste_fino.py` (semilla 42, 80/10/10): la temperatura y los umbrales se
   ajustan SOLO con validación.
2. Se mide en el conjunto de PRUEBA: error de calibración (ECE) antes y después, y por cultivo cuántas fotos
   reciben diagnóstico con su umbral (cobertura) y con qué acierto.
3. Guarda `modelos/sigagro-extensivos/calibracion.json` (el servicio lo aplica al reiniciarse) y
   `METRICAS_CALIBRACION.md`.

Correrlo cada vez que se reentrena el modelo. Uso:  .venv\\Scripts\\python.exe -m entrenamiento.calibrar
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(RAIZ / ".cache"))
sys.path.insert(0, str(RAIZ.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

from app.calibracion import (  # noqa: E402
    ARCHIVO,
    MINIMO_FOTOS_CULTIVO,
    PRECISION_OBJETIVO,
    ajustar_temperatura,
    error_calibracion,
    umbral_para_precision,
)
from app.conocimiento import info_de  # noqa: E402

DATOS = RAIZ / "datos"
MODELO = RAIZ.parent / "modelos" / "sigagro-extensivos"
SEMILLA = 42
LOTE = 32


def division() -> tuple[list[str], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Validación y prueba con la MISMA división que entrenar.py (mismas fotos, misma semilla)."""
    etiquetas = sorted(p.name for p in DATOS.iterdir() if p.is_dir() and any(p.glob("*.jpg")))
    a_id = {e: i for i, e in enumerate(etiquetas)}
    items = [(ruta, a_id[e]) for e in etiquetas for ruta in sorted((DATOS / e).glob("*.jpg"))]
    _, resto = train_test_split(items, test_size=0.2, stratify=[i for _, i in items], random_state=SEMILLA)
    valid, prueba = train_test_split(resto, test_size=0.5, stratify=[i for _, i in resto], random_state=SEMILLA)
    return etiquetas, valid, prueba


@torch.no_grad()
def logits_de(modelo, procesador, datos: list[tuple[Path, int]], nombre: str) -> torch.Tensor:
    """Logits con la foto original y el procesador del modelo (igual que el servicio)."""
    salida = []
    inicio = time.time()
    for i in range(0, len(datos), LOTE):
        imgs = [Image.open(r).convert("RGB") for r, _ in datos[i : i + LOTE]]
        salida.append(modelo(**procesador(images=imgs, return_tensors="pt")).logits)
        hechas = min(i + LOTE, len(datos))
        if (i // LOTE) % 20 == 0 or hechas == len(datos):
            print(f"  {nombre}: {hechas}/{len(datos)} · {hechas / max(time.time() - inicio, 1e-6):.1f} fotos/s", flush=True)
    return torch.cat(salida)


def cultivo(etiqueta: str) -> str:
    info = info_de(etiqueta)
    return (info.cultivo if info and info.cultivo else etiqueta.split("___")[0]) or "Sin cultivo"


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    procesador = AutoImageProcessor.from_pretrained(MODELO)
    modelo = AutoModelForImageClassification.from_pretrained(MODELO).eval()
    id2label = modelo.config.id2label
    etiquetas_modelo = [id2label[i] for i in sorted(id2label)]
    etiquetas, valid, prueba = division()
    if etiquetas != etiquetas_modelo:
        raise SystemExit("Las clases de entrenamiento/datos no coinciden con las del modelo: reentrenar antes de calibrar.")
    print(f"{len(etiquetas)} clases · {len(valid)} fotos de validación · {len(prueba)} de prueba", flush=True)

    lv, yv = logits_de(modelo, procesador, valid, "validación"), torch.tensor([y for _, y in valid])
    lp, yp = logits_de(modelo, procesador, prueba, "prueba"), torch.tensor([y for _, y in prueba])

    t = ajustar_temperatura(lv, yv)
    print(f"Temperatura: {t:.3f}", flush=True)

    def resumen(logits: torch.Tensor, y: torch.Tensor, temp: float):
        probs = torch.softmax(logits / temp, dim=-1)
        conf, pred = probs.max(dim=-1)
        return conf.tolist(), pred.tolist(), (pred == y).tolist()

    # Umbrales por cultivo (de la clase PREDICHA, como los aplica el servicio), con validación calibrada
    cv, pv, av = resumen(lv, yv, t)
    por_cultivo_v: dict[str, tuple[list[float], list[bool]]] = defaultdict(lambda: ([], []))
    for c, p, a in zip(cv, pv, av):
        por_cultivo_v[cultivo(etiquetas[p])][0].append(c)
        por_cultivo_v[cultivo(etiquetas[p])][1].append(a)
    umbrales = {k: u for k, (cs, as_) in sorted(por_cultivo_v.items()) if (u := umbral_para_precision(cs, as_)) is not None}

    # Medición en prueba
    c0, _, a0 = resumen(lp, yp, 1.0)
    c1, p1, a1 = resumen(lp, yp, t)
    ece_antes, ece_despues = error_calibracion(c0, a0), error_calibracion(c1, a1)
    filas = []
    por_cultivo_p: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for c, p, a in zip(c1, p1, a1):
        por_cultivo_p[cultivo(etiquetas[p])].append((c, a))
    for k, pares in sorted(por_cultivo_p.items()):
        u = max(umbrales.get(k, 0.0), 0.0)
        dados = [a for c, a in pares if c >= u]
        filas.append({
            "cultivo": k,
            "fotos": len(pares),
            "aciertoSinUmbral": round(sum(a for _, a in pares) / len(pares), 4),
            "umbral": umbrales.get(k),
            "cobertura": round(len(dados) / len(pares), 4),
            "aciertoConUmbral": round(sum(dados) / len(dados), 4) if dados else None,
        })

    datos = {
        "version": time.strftime("local-%Y%m%d-%H%M"),
        "metodo": "temperature scaling (Guo et al., 2017) ajustado con validación",
        "temperatura": round(t, 4),
        "precisionObjetivo": PRECISION_OBJETIVO,
        "minimoFotosCultivo": MINIMO_FOTOS_CULTIVO,
        "umbralesPorCultivo": umbrales,
        "prueba": {"eceAntes": round(ece_antes, 4), "eceDespues": round(ece_despues, 4), "porCultivo": filas},
    }
    (MODELO / ARCHIVO).write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# Modelo `sigagro-extensivos` — calibración de la confianza",
        "",
        f"- Versión: `{datos['version']}` · método: {datos['metodo']}",
        f"- Temperatura: **{t:.3f}** ({'el modelo estaba sobreconfiado' if t > 1 else 'el modelo estaba subconfiado' if t < 1 else 'sin cambio'})",
        f"- Error de calibración en prueba (ECE): {ece_antes:.1%} → **{ece_despues:.1%}** (cuánto se aleja la confianza informada del acierto real)",
        f"- Umbral por cultivo: menor confianza desde la que la precisión en validación es ≥ {PRECISION_OBJETIVO:.0%} "
        f"(mín. {MINIMO_FOTOS_CULTIVO} fotos). El servicio usa el mayor entre este y `UMBRAL_CONFIANZA`.",
        "",
        "| Cultivo | Fotos de prueba | Acierto (todas) | Umbral | Reciben diagnóstico | Acierto de esas |",
        "|---|---|---|---|---|---|",
    ]
    for f in filas:
        lineas.append(
            f"| {f['cultivo']} | {f['fotos']} | {f['aciertoSinUmbral']:.1%} | {f['umbral']:.0%} | {f['cobertura']:.1%} | "
            f"{f['aciertoConUmbral']:.1%} |" if f["umbral"] is not None and f["aciertoConUmbral"] is not None
            else f"| {f['cultivo']} | {f['fotos']} | {f['aciertoSinUmbral']:.1%} | — | — | — |"
        )
    lineas += [
        "",
        "Las fotos que no alcanzan el umbral se informan como INCONCLUSO (se pide otra foto o la revisión del",
        "Ingeniero Agrónomo). Las fotos de prueba son de conjuntos públicos de otros países: para la precisión con",
        "fotos argentinas ver `METRICAS_ARGENTINA.md` (`entrenamiento/evaluar_argentina.py`).",
    ]
    (MODELO / "METRICAS_CALIBRACION.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nECE en prueba: {ece_antes:.1%} -> {ece_despues:.1%}. Guardado {MODELO / ARCHIVO}", flush=True)


if __name__ == "__main__":
    main()
