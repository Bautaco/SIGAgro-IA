"""
Segunda etapa de entrenamiento del modelo `sigagro-extensivos`: ajuste fino de las últimas capas del ViT.

`entrenar.py` entrena solo la capa de clasificación con el extractor fijo (rápido, pero limitado). Acá se parte
de ese modelo y se ajustan también los últimos bloques del transformer, con fotos variadas en cada época (recorte,
espejo, giro y cambios de luz/color, como en una toma a campo). Es apto para equipos sin placa de video:

1. Las fotos se reducen UNA vez a 256x256 y se guardan en `entrenamiento/salida/imagenes_256.npy` (uint8): leer
   y decodificar los JPG era lo más lento.
2. Los primeros bloques quedan fijos (no se calcula su gradiente); se entrenan los últimos `--bloques`, la
   normalización final y el clasificador, con pesos por clase (las clases con pocas fotos pesan más).
3. En cada época se mide con validación y se guarda el mejor estado (F1 macro). Se puede cortar y retomar
   (`--reanudar`): el avance queda en `entrenamiento/salida/ajuste_fino.pt`.
4. Se mide UNA vez en el conjunto de prueba (las MISMAS fotos que `entrenar.py`: misma división y semilla) y el
   modelo nuevo reemplaza al anterior solo si mejora el F1 macro; el anterior queda como copia de respaldo.

Uso:  .venv\\Scripts\\python.exe -m entrenamiento.ajuste_fino [--epocas 6] [--bloques 3] [--reanudar]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(RAIZ / ".cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torchvision.transforms import v2  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

from entrenamiento.entrenar import BASE, BASE_REVISION, DATOS, DESTINO, SALIDA, SEMILLA, copiar_clases_aprendidas  # noqa: E402

LADO = 256
IMAGENES = SALIDA / "imagenes_256.npy"
INDICE = SALIDA / "imagenes_256.json"
AVANCE = SALIDA / "ajuste_fino.pt"


def _cachear_imagenes(rutas: list[Path]) -> np.ndarray:
    """Fotos reducidas a 256x256 (como el servicio: se estira la foto entera, sin recortar)."""
    claves = [str(r.relative_to(DATOS)) for r in rutas]
    if IMAGENES.exists() and INDICE.exists() and json.loads(INDICE.read_text(encoding="utf-8")) == claves:
        return np.load(IMAGENES, mmap_mode="r")
    SALIDA.mkdir(exist_ok=True)
    arr = np.lib.format.open_memmap(IMAGENES, mode="w+", dtype=np.uint8, shape=(len(rutas), LADO, LADO, 3))
    inicio = time.time()
    for i, r in enumerate(rutas):
        arr[i] = np.asarray(Image.open(r).convert("RGB").resize((LADO, LADO), Image.BILINEAR))
        if i % 1000 == 0 or i == len(rutas) - 1:
            print(f"  reduciendo fotos: {i + 1}/{len(rutas)} · {(i + 1) / max(time.time() - inicio, 1e-6):.0f} fotos/s", flush=True)
    arr.flush()
    INDICE.write_text(json.dumps(claves), encoding="utf-8")
    return np.load(IMAGENES, mmap_mode="r")


# Variaciones de una toma a campo: encuadre, orientación, luz y color
AUMENTO = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.8, 1.25), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomApply([v2.RandomRotation(25)], p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
])


def _a_tensor(lote: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(lote, copy=True)).permute(0, 3, 1, 2)  # uint8 NCHW


def _normalizar(x: torch.Tensor) -> torch.Tensor:
    """Igual que el procesador del modelo: escala a [0,1] y normaliza con media y desvío 0.5."""
    return (x.float() / 255.0 - 0.5) / 0.5


def _entrada_eval(lote: np.ndarray) -> torch.Tensor:
    x = _a_tensor(lote).float()
    return _normalizar(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True))


def _entrada_entr(lote: np.ndarray) -> torch.Tensor:
    x = _a_tensor(lote)
    return _normalizar(torch.stack([AUMENTO(im) for im in x]))


@torch.no_grad()
def _predecir(modelo, imgs: np.ndarray, indices: list[int], lote: int = 64) -> list[int]:
    modelo.eval()
    pred: list[int] = []
    for i in range(0, len(indices), lote):
        idx = indices[i : i + lote]
        pred += modelo(pixel_values=_entrada_eval(imgs[idx])).logits.argmax(1).tolist()
    return pred


@torch.no_grad()
def _predecir_como_el_servicio(modelo, procesador, rutas: list[Path], lote: int = 32) -> list[int]:
    """Prueba final con la foto ORIGINAL y el procesador del modelo, igual que el servicio y que entrenar.py."""
    modelo.eval()
    pred: list[int] = []
    for i in range(0, len(rutas), lote):
        x = procesador(images=[Image.open(r).convert("RGB") for r in rutas[i : i + lote]], return_tensors="pt")["pixel_values"]
        pred += modelo(pixel_values=x).logits.argmax(1).tolist()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--bloques", type=int, default=3, help="Bloques finales del transformer que se ajustan")
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--reanudar", action="store_true", help="Seguir desde el último avance guardado")
    ap.add_argument("--limite", type=int, default=0, help="Solo para probar la velocidad: N fotos de entrenamiento")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)

    # Misma división que entrenar.py (mismas fotos de prueba)
    etiquetas = sorted(p.name for p in DATOS.iterdir() if p.is_dir() and any(p.glob("*.jpg")))
    a_id = {e: i for i, e in enumerate(etiquetas)}
    items = [(ruta, a_id[e]) for e in etiquetas for ruta in sorted((DATOS / e).glob("*.jpg"))]
    pos = {r: i for i, (r, _) in enumerate(items)}
    entr, resto = train_test_split(items, test_size=0.2, stratify=[i for _, i in items], random_state=SEMILLA)
    valid, prueba = train_test_split(resto, test_size=0.5, stratify=[i for _, i in resto], random_state=SEMILLA)
    if args.limite:
        entr = entr[: args.limite]
    print(f"{len(etiquetas)} etiquetas · {len(entr)} entrenamiento / {len(valid)} validación / {len(prueba)} prueba", flush=True)

    imgs = _cachear_imagenes([r for r, _ in items])
    i_entr = np.array([pos[r] for r, _ in entr])
    y_entr = torch.tensor([y for _, y in entr])
    i_valid, y_valid = [pos[r] for r, _ in valid], [y for _, y in valid]
    i_prueba, y_prueba = [pos[r] for r, _ in prueba], [y for _, y in prueba]

    # Punto de partida: el modelo actual (extractor + capa de clasificación ya entrenada)
    origen = DESTINO if (DESTINO / "config.json").exists() else None
    modelo = AutoModelForImageClassification.from_pretrained(
        str(origen) if origen else BASE,
        **({} if origen else {"revision": BASE_REVISION, "ignore_mismatched_sizes": True}),
        num_labels=len(etiquetas), id2label={i: e for e, i in a_id.items()}, label2id=a_id,
    )
    procesador = AutoImageProcessor.from_pretrained(str(origen) if origen else BASE, **({} if origen else {"revision": BASE_REVISION}))
    if list(modelo.config.id2label.values()) != etiquetas:
        raise SystemExit("Las etiquetas del modelo actual no coinciden con las carpetas de datos: correr antes entrenar.py")

    capas = modelo.vit.encoder.layer
    k = max(1, min(args.bloques, len(capas)))
    for p in modelo.parameters():
        p.requires_grad = False
    entrenables_backbone = [p for b in capas[-k:] for p in b.parameters()] + list(modelo.vit.layernorm.parameters())
    for p in entrenables_backbone + list(modelo.classifier.parameters()):
        p.requires_grad = True
    opt = torch.optim.AdamW(
        [{"params": entrenables_backbone, "lr": 2e-4}, {"params": modelo.classifier.parameters(), "lr": 1e-3}],
        weight_decay=0.05,
    )
    pasos_epoca = math.ceil(len(entr) / args.lote)
    total = pasos_epoca * args.epocas
    calentamiento = max(1, pasos_epoca // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / calentamiento) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    frec = np.bincount(y_entr.numpy(), minlength=len(etiquetas)).astype(float)
    peso = torch.tensor(np.sqrt(frec.sum() / (len(frec) * frec)), dtype=torch.float32)  # clases chicas pesan más
    perdida = torch.nn.CrossEntropyLoss(weight=peso, label_smoothing=0.1)

    epoca0, mejor_f1, mejor_estado = 0, -1.0, None
    if args.reanudar and AVANCE.exists():
        a = torch.load(AVANCE, weights_only=False)
        modelo.load_state_dict(a["modelo"])
        opt.load_state_dict(a["opt"])
        sched.load_state_dict(a["sched"])
        epoca0, mejor_f1, mejor_estado = a["epoca"], a["mejor_f1"], a["mejor_estado"]
        print(f"Reanudando desde la época {epoca0} (mejor F1 validación {mejor_f1:.1%})", flush=True)
    else:
        f1_ini = f1_score(y_valid, _predecir(modelo, imgs, i_valid), average="macro")
        print(f"Modelo actual · F1 macro en validación: {f1_ini:.1%}", flush=True)
        mejor_f1, mejor_estado = f1_ini, {n: t.clone() for n, t in modelo.state_dict().items()}

    rng = np.random.default_rng(SEMILLA)
    for epoca in range(epoca0, args.epocas):
        modelo.train()
        for n in range(len(capas) - k):  # los bloques fijos siguen en modo evaluación
            capas[n].eval()
        orden = rng.permutation(len(i_entr))
        inicio, suma = time.time(), 0.0
        for paso in range(pasos_epoca):
            sel = orden[paso * args.lote : (paso + 1) * args.lote]
            idx = np.sort(i_entr[sel])  # lectura ordenada del archivo de fotos
            y = y_entr[sel][np.argsort(i_entr[sel])]
            loss = perdida(modelo(pixel_values=_entrada_entr(imgs[idx])).logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            suma += loss.item()
            if paso % 25 == 0 or paso == pasos_epoca - 1:
                vel = (paso + 1) * args.lote / max(time.time() - inicio, 1e-6)
                resta = (pasos_epoca - paso - 1) * args.lote / max(vel, 1e-6) / 60
                print(f"  época {epoca + 1}/{args.epocas} · paso {paso + 1}/{pasos_epoca} · pérdida {suma / (paso + 1):.3f} · {vel:.1f} fotos/s · faltan ~{resta:.0f} min de esta época", flush=True)
        pred = _predecir(modelo, imgs, i_valid)
        f1 = f1_score(y_valid, pred, average="macro")
        acc = float(np.mean(np.array(pred) == np.array(y_valid)))
        print(f"Época {epoca + 1}: validación acierto {acc:.1%} · F1 macro {f1:.1%} ({(time.time() - inicio) / 60:.0f} min)", flush=True)
        if f1 > mejor_f1:
            mejor_f1, mejor_estado = f1, {n: t.clone() for n, t in modelo.state_dict().items()}
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoca": epoca + 1,
                    "mejor_f1": mejor_f1, "mejor_estado": mejor_estado}, AVANCE)

    if args.limite:
        print("Prueba de velocidad terminada (no se exporta el modelo).", flush=True)
        return

    # Prueba final con el mejor estado (fotos nunca vistas)
    modelo.load_state_dict(mejor_estado)
    pred = _predecir_como_el_servicio(modelo, procesador, [r for r, _ in prueba])
    acierto = float(np.mean(np.array(pred) == np.array(y_prueba)))
    reporte = classification_report(y_prueba, pred, labels=list(range(len(etiquetas))), target_names=etiquetas, output_dict=True, zero_division=0)
    macro = reporte["macro avg"]["f1-score"]
    anterior = json.loads((DESTINO / "metricas.json").read_text(encoding="utf-8")) if (DESTINO / "metricas.json").exists() else {}
    print(f"\nPrueba: acierto {acierto:.1%} · F1 macro {macro:.1%} (antes {anterior.get('aciertoPrueba', 0):.1%} · {anterior.get('macroF1', 0):.1%})", flush=True)

    if anterior and macro <= anterior.get("macroF1", 0):
        print("No mejoró el F1 macro en prueba: se mantiene el modelo anterior.", flush=True)
        return

    if (DESTINO / "config.json").exists():
        respaldo = DESTINO.parent / f"{DESTINO.name}.anterior-{anterior.get('version', 'sin-version')}"
        if not respaldo.exists():
            shutil.copytree(DESTINO, respaldo)
            print(f"Modelo anterior respaldado en {respaldo}", flush=True)
    modelo.save_pretrained(DESTINO)
    procesador.save_pretrained(DESTINO)
    copiar_clases_aprendidas(etiquetas, DESTINO)

    version = time.strftime("local-%Y%m%d-%H%M")
    metricas = {
        "version": version,
        "base": f"{BASE}@{BASE_REVISION}",
        "tecnica": f"ajuste fino de los últimos {k} bloques + clasificador, con aumento de datos ({args.epocas} épocas)",
        "etiquetas": etiquetas,
        "imagenes": {"entrenamiento": len(entr), "validacion": len(valid), "prueba": len(prueba)},
        "aciertoPrueba": round(acierto, 4),
        "aciertoValidacion": None,
        "macroF1": round(macro, 4),
        "f1Validacion": round(mejor_f1, 4),
        "anterior": {"version": anterior.get("version"), "aciertoPrueba": anterior.get("aciertoPrueba"), "macroF1": anterior.get("macroF1")},
        "porClase": {e: {kk: round(v, 4) for kk, v in reporte[e].items()} for e in etiquetas},
        "matrizConfusion": confusion_matrix(y_prueba, pred, labels=list(range(len(etiquetas)))).tolist(),
    }
    (DESTINO / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    ant = anterior.get("porClase", {})
    lineas = [
        "# Modelo `sigagro-extensivos` — métricas",
        "",
        f"- Versión: `{version}` · base: `{BASE}` (MIT) · técnica: {metricas['tecnica']}",
        f"- Datos: {len(items)} fotos de 4 conjuntos públicos CC BY 4.0 (ver `entrenamiento/FUENTES.md`) · semilla {SEMILLA}",
        f"- División: {len(entr)} entrenamiento · {len(valid)} validación · {len(prueba)} prueba (las mismas fotos de prueba que la versión anterior)",
        "",
        f"**Acierto en el conjunto de prueba ({len(prueba)} fotos no vistas): {acierto:.1%}** · F1 macro: {macro:.1%}",
        f"(versión anterior `{anterior.get('version', '—')}`: {anterior.get('aciertoPrueba', 0):.1%} · F1 macro {anterior.get('macroF1', 0):.1%})",
        "",
        "| Etiqueta | Precisión | Sensibilidad | F1 | F1 anterior | Fotos de prueba |",
        "|---|---|---|---|---|---|",
    ]
    for e in etiquetas:
        r = reporte[e]
        f1_ant = ant.get(e, {}).get("f1-score")
        lineas.append(f"| {e} | {r['precision']:.1%} | {r['recall']:.1%} | {r['f1-score']:.1%} | {'—' if f1_ant is None else f'{f1_ant:.1%}'} | {int(r['support'])} |")
    lineas += [
        "",
        "**Precisión:** de las fotos que el modelo asignó a la clase, cuántas eran correctas. "
        "**Sensibilidad:** de las fotos de la clase, cuántas reconoció.",
        "",
        "Las fotos provienen de conjuntos públicos de otros países: la precisión con fotos de lotes argentinos puede ser",
        "menor. Conviene validarlo con fotos propias y reentrenar agregándolas a `entrenamiento/datos/`.",
    ]
    (DESTINO / "METRICAS.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Modelo nuevo guardado en {DESTINO}", flush=True)


if __name__ == "__main__":
    main()
