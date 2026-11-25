"""
Entrena el modelo de cultivos extensivos (soja, girasol, maní) a partir de entrenamiento/datos/<Etiqueta>/*.jpg.

Técnica (apta para equipos sin placa de video): se parte del ViT liviano `wambugu71/crop_leaf_diseases_vit`
(MIT, ya entrenado con hojas de cultivos), se mantiene fijo su extractor de características y se entrena solo la
capa de clasificación con las etiquetas nuevas ("linear probing"):

1. Se calcula una vez el vector de características de cada foto con el mismo preprocesamiento que usa el servicio
   (para entrenamiento, además, variantes con giro y recorte: el clasificador aprende a ser robusto).
2. Se entrena la capa final (regresión logística multiclase con pesos por clase) y se elige la regularización con
   el conjunto de validación.
3. Se mide UNA vez en el conjunto de prueba (fotos nunca vistas) y se exporta el modelo completo en formato
   Hugging Face a `modelos/sigagro-extensivos/`, que el servicio carga como modelo adicional.

División estratificada y reproducible: 80 % entrenamiento · 10 % validación · 10 % prueba.
Las características se guardan en `entrenamiento/salida/caracteristicas.pt`: volver a entrenar (p. ej. con otra
regularización) no requiere recalcularlas, salvo que cambien las fotos.

Uso:  .venv\\Scripts\\python.exe -m entrenamiento.entrenar [--variantes 2] [--recalcular]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(RAIZ / ".cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

DATOS = RAIZ / "datos"
SALIDA = RAIZ / "salida"
CACHE = SALIDA / "caracteristicas.pt"
DESTINO = RAIZ.parent / "modelos" / "sigagro-extensivos"
BASE = "wambugu71/crop_leaf_diseases_vit"
BASE_REVISION = "7d5b32bcd6f83a2f57e7e0346358fad276296877"
SEMILLA = 42
LOTE = 32


def _variante(img: Image.Image, rng: random.Random) -> Image.Image:
    """Variante para entrenamiento: espejo, giro y recorte leves (simulan otra toma de la misma hoja)."""
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.5:
        img = ImageOps.flip(img)
    img = img.rotate(rng.uniform(-20, 20), expand=False, fillcolor=(0, 0, 0))
    w, h = img.size
    escala = rng.uniform(0.75, 1.0)
    cw, ch = int(w * escala), int(h * escala)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return img.crop((x, y, x + cw, y + ch))


def _caracteristicas(modelo, procesador, rutas: list[Path], variantes: int, rng: random.Random, etiqueta: str) -> torch.Tensor:
    """Vector [CLS] normalizado de la última capa (lo que recibe la capa de clasificación del ViT)."""
    salida: list[torch.Tensor] = []
    inicio = time.time()
    for i in range(0, len(rutas), LOTE):
        imgs = [Image.open(r).convert("RGB") for r in rutas[i : i + LOTE]]
        lote = imgs if variantes == 0 else [_variante(im, rng) for im in imgs]
        with torch.no_grad():
            x = procesador(images=lote, return_tensors="pt")["pixel_values"]
            salida.append(modelo.vit(pixel_values=x).last_hidden_state[:, 0, :])
        hechas = min(i + LOTE, len(rutas))
        if (i // LOTE) % 20 == 0 or hechas == len(rutas):
            vel = hechas / max(time.time() - inicio, 1e-6)
            print(f"  {etiqueta}: {hechas}/{len(rutas)} fotos · {vel:.1f} fotos/s", flush=True)
    return torch.cat(salida)


def _entrenar_cabeza(x: torch.Tensor, y: torch.Tensor, n: int, peso: torch.Tensor, decaimiento: float, epocas: int = 300) -> torch.nn.Linear:
    torch.manual_seed(SEMILLA)
    cabeza = torch.nn.Linear(x.shape[1], n)
    opt = torch.optim.AdamW(cabeza.parameters(), lr=1e-2, weight_decay=decaimiento)
    perdida = torch.nn.CrossEntropyLoss(weight=peso, label_smoothing=0.05)
    for _ in range(epocas):
        opt.zero_grad()
        l = perdida(cabeza(x), y)
        l.backward()
        opt.step()
    return cabeza


def _acierto(cabeza: torch.nn.Linear, x: torch.Tensor, y: torch.Tensor) -> tuple[float, list[int]]:
    with torch.no_grad():
        pred = cabeza(x).argmax(1)
    return float((pred == y).float().mean()), pred.tolist()


def copiar_clases_aprendidas(etiquetas: list[str], destino: Path) -> None:
    """Copia junto al modelo la ficha mínima de las afecciones aprendidas de los Ingenieros que el modelo conoce."""
    origen = Path(__file__).resolve().parent / "clases_aprendidas.json"
    archivo = destino / "clases_aprendidas.json"
    registro = json.loads(origen.read_text(encoding="utf-8")) if origen.exists() else {}
    propias = {e: c for e, c in registro.items() if e in etiquetas}
    if propias:
        archivo.write_text(json.dumps(propias, indent=2, ensure_ascii=False), encoding="utf-8")
    elif archivo.exists():
        archivo.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variantes", type=int, default=2, help="Variantes con giro/recorte por foto de entrenamiento")
    ap.add_argument("--recalcular", action="store_true", help="Recalcular las características aunque estén guardadas")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    etiquetas = sorted(p.name for p in DATOS.iterdir() if p.is_dir() and any(p.glob("*.jpg")))
    if not etiquetas:
        raise SystemExit(f"No hay datos en {DATOS}. Ejecutar antes: python -m entrenamiento.preparar_datos")
    a_id = {e: i for i, e in enumerate(etiquetas)}
    items = [(ruta, a_id[e]) for e in etiquetas for ruta in sorted((DATOS / e).glob("*.jpg"))]
    entr, resto = train_test_split(items, test_size=0.2, stratify=[i for _, i in items], random_state=SEMILLA)
    valid, prueba = train_test_split(resto, test_size=0.5, stratify=[i for _, i in resto], random_state=SEMILLA)
    print(f"{len(etiquetas)} etiquetas · {len(entr)} entrenamiento / {len(valid)} validación / {len(prueba)} prueba", flush=True)

    procesador = AutoImageProcessor.from_pretrained(BASE, revision=BASE_REVISION)
    modelo = AutoModelForImageClassification.from_pretrained(
        BASE, revision=BASE_REVISION, num_labels=len(etiquetas),
        id2label={i: e for e, i in a_id.items()}, label2id=a_id, ignore_mismatched_sizes=True,
    )
    modelo.eval()

    huella = {"etiquetas": etiquetas, "n": len(items), "variantes": args.variantes}
    if CACHE.exists() and not args.recalcular and torch.load(CACHE)["huella"] == huella:
        print("Usando características guardadas.", flush=True)
        c = torch.load(CACHE)
    else:
        rng = random.Random(SEMILLA)
        rutas = lambda datos: [r for r, _ in datos]  # noqa: E731
        c = {"huella": huella}
        c["valid"] = _caracteristicas(modelo, procesador, rutas(valid), 0, rng, "validación")
        c["prueba"] = _caracteristicas(modelo, procesador, rutas(prueba), 0, rng, "prueba")
        partes = [_caracteristicas(modelo, procesador, rutas(entr), 0, rng, "entrenamiento")]
        for v in range(args.variantes):
            partes.append(_caracteristicas(modelo, procesador, rutas(entr), 1, rng, f"entrenamiento (variante {v + 1})"))
        c["entr"] = torch.cat(partes)
        SALIDA.mkdir(exist_ok=True)
        torch.save(c, CACHE)

    y_entr = torch.tensor([i for _, i in entr] * (1 + args.variantes))
    y_valid = torch.tensor([i for _, i in valid])
    y_prueba = torch.tensor([i for _, i in prueba])
    frec = np.bincount(y_entr.numpy(), minlength=len(etiquetas)).astype(float)
    peso = torch.tensor(frec.sum() / (len(frec) * frec), dtype=torch.float32)

    # Regularización elegida con validación
    mejor = None
    for dec in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
        cabeza = _entrenar_cabeza(c["entr"], y_entr, len(etiquetas), peso, dec)
        acc, _ = _acierto(cabeza, c["valid"], y_valid)
        print(f"  regularización {dec:g}: validación {acc:.1%}", flush=True)
        if mejor is None or acc > mejor[0]:
            mejor = (acc, dec, cabeza)
    acc_valid, decaimiento, cabeza = mejor

    # Prueba final (fotos nunca vistas)
    acierto, pred = _acierto(cabeza, c["prueba"], y_prueba)
    reales = y_prueba.tolist()

    # Exportar: el ViT original con la capa de clasificación entrenada
    modelo.classifier.weight.data = cabeza.weight.data.clone()
    modelo.classifier.bias.data = cabeza.bias.data.clone()
    DESTINO.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(DESTINO)
    procesador.save_pretrained(DESTINO)
    copiar_clases_aprendidas(etiquetas, DESTINO)

    reporte = classification_report(reales, pred, labels=list(range(len(etiquetas))), target_names=etiquetas, output_dict=True, zero_division=0)
    matriz = confusion_matrix(reales, pred, labels=list(range(len(etiquetas)))).tolist()
    metricas = {
        "version": time.strftime("local-%Y%m%d-%H%M"),
        "base": f"{BASE}@{BASE_REVISION}",
        "tecnica": "linear probing (extractor fijo + capa de clasificación entrenada)",
        "etiquetas": etiquetas,
        "imagenes": {"entrenamiento": len(entr), "validacion": len(valid), "prueba": len(prueba), "variantesPorFoto": args.variantes},
        "aciertoPrueba": round(acierto, 4),
        "aciertoValidacion": round(acc_valid, 4),
        "regularizacion": decaimiento,
        "porClase": {e: {k: round(v, 4) for k, v in reporte[e].items()} for e in etiquetas},
        "macroF1": round(reporte["macro avg"]["f1-score"], 4),
        "matrizConfusion": matriz,
    }
    (DESTINO / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# Modelo `sigagro-extensivos` — métricas",
        "",
        f"- Versión: `{metricas['version']}` · base: `{BASE}` (MIT) · técnica: extractor fijo + capa de clasificación entrenada",
        f"- Datos: {len(items)} fotos de 4 conjuntos públicos CC BY 4.0 (ver `entrenamiento/FUENTES.md`) · semilla {SEMILLA}",
        f"- División: {len(entr)} entrenamiento ({args.variantes} variantes por foto) · {len(valid)} validación · {len(prueba)} prueba",
        "",
        f"**Acierto en el conjunto de prueba ({len(prueba)} fotos no vistas): {acierto:.1%}** · F1 macro: {reporte['macro avg']['f1-score']:.1%}",
        "",
        "| Etiqueta | Precisión | Sensibilidad | F1 | Fotos de prueba |",
        "|---|---|---|---|---|",
    ]
    for e in etiquetas:
        r = reporte[e]
        lineas.append(f"| {e} | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1-score']:.1%} | {int(r['support'])} |")
    lineas += [
        "",
        "**Precisión:** de las fotos que el modelo asignó a la clase, cuántas eran correctas. "
        "**Sensibilidad:** de las fotos de la clase, cuántas reconoció.",
        "",
        "Las fotos provienen de conjuntos públicos de otros países: la precisión con fotos de lotes argentinos puede ser",
        "menor. Conviene validarlo con fotos propias y reentrenar agregándolas a `entrenamiento/datos/`.",
    ]
    (DESTINO / "METRICAS.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nAcierto en prueba: {acierto:.1%} (F1 macro {reporte['macro avg']['f1-score']:.1%}). Modelo en {DESTINO}", flush=True)


if __name__ == "__main__":
    main()
