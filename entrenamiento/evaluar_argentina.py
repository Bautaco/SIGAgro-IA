"""
Mide la IA con fotos ARGENTINAS validadas por un Ingeniero Agrónomo (conjunto de prueba propio).

Las métricas de `METRICAS.md` salen de fotos de conjuntos públicos de otros países (las mismas fuentes con las que
se entrena), así que suelen ser optimistas. Este script mide lo que importa: cómo le va al servicio con fotos de
lotes argentinos que el modelo NUNCA vio.

- Fotos en `entrenamiento/prueba_argentina/<Etiqueta>/*.jpg` (misma etiqueta que `entrenamiento/datos/`). Esta
  carpeta no la lee ningún script de entrenamiento. La llena `importar_validadas.py` (reserva una parte de los
  diagnósticos validados, `--fraccion-prueba`) o se cargan a mano con fotos identificadas por un Ingeniero.
- Cada foto pasa por el MISMO análisis que el servicio (control de calidad, modelo según el cultivo, calibración
  y umbrales), con el cultivo de su etiqueta como cultivo del lote.
- Resultado por cultivo y por afección: cuántas reciben diagnóstico (el resto queda INCONCLUSO o ilegible), cuántas
  de esas son correctas, y las confusiones más frecuentes. Se guarda en `modelos/sigagro-extensivos/
  METRICAS_ARGENTINA.md` y `metricas_argentina.json`.

Uso:  .venv\\Scripts\\python.exe -m entrenamiento.evaluar_argentina [--minimo 1]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(RAIZ / ".cache"))
sys.path.insert(0, str(RAIZ.parent))

from PIL import Image  # noqa: E402

from app.analisis import analizar  # noqa: E402
from app.config import cargar_config  # noqa: E402
from app.conocimiento import info_de  # noqa: E402
from app.modelo import GestorModelo  # noqa: E402

PRUEBA = RAIZ / "prueba_argentina"
DESTINO = RAIZ.parent / "modelos" / "sigagro-extensivos"
EXT = (".jpg", ".jpeg", ".png", ".webp")


def fotos() -> dict[str, list[Path]]:
    if not PRUEBA.exists():
        return {}
    return {
        d.name: sorted(f for f in d.iterdir() if f.suffix.lower() in EXT)
        for d in sorted(PRUEBA.iterdir())
        if d.is_dir() and any(f.suffix.lower() in EXT for f in d.iterdir())
    }


def data_url(ruta: Path) -> str:
    img = Image.open(ruta).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def resumen(filas: list[dict]) -> dict:
    n = len(filas)
    dados = [f for f in filas if f["estado"] == "OK"]
    bien = [f for f in dados if f["correcto"]]
    return {
        "fotos": n,
        "conDiagnostico": len(dados),
        "cobertura": round(len(dados) / n, 4) if n else None,
        "aciertoDeLosDiagnosticos": round(len(bien) / len(dados), 4) if dados else None,
        "aciertoSobreTotal": round(len(bien) / n, 4) if n else None,
    }


def pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minimo", type=int, default=1, help="Mínimo de fotos de una afección para incluirla")
    args = ap.parse_args()

    conjunto = {e: fs for e, fs in fotos().items() if len(fs) >= args.minimo}
    if not conjunto:
        PRUEBA.mkdir(exist_ok=True)
        raise SystemExit(
            f"No hay fotos en {PRUEBA}.\n"
            "Cargalas en entrenamiento/prueba_argentina/<Etiqueta>/ (p. ej. Corn___Common_Rust/) con fotos de lotes\n"
            "argentinos identificadas por un Ingeniero Agrónomo, o importá diagnósticos validados con\n"
            "  .venv\\Scripts\\python.exe -m entrenamiento.importar_validadas --fraccion-prueba 0.25"
        )

    config = cargar_config()
    gestor = GestorModelo()
    gestor.cargar(config.modelo_id, config.modelo_revision)
    for ruta in config.modelos_adicionales:
        gestor.cargar_adicional(ruta)
    if gestor.clasificador is None:
        raise SystemExit(gestor.error or "No se pudo cargar el modelo.")

    filas: list[dict] = []
    inicio = time.time()
    total = sum(len(v) for v in conjunto.values())
    for etiqueta, rutas in conjunto.items():
        info = info_de(etiqueta)
        cultivo = info.cultivo if info else None
        for ruta in rutas:
            r = analizar(data_url(ruta), gestor.clasificador, config, cultivo=cultivo, adicionales=gestor.adicionales)
            pred = r.get("etiquetaModelo")
            filas.append({
                "foto": ruta.name,
                "real": etiqueta,
                "cultivo": cultivo or "Sin cultivo",
                "estado": r["estado"],
                "prediccion": pred,
                "confianza": r.get("confianza"),
                "correcto": r["estado"] == "OK" and pred == etiqueta,
            })
            if len(filas) % 25 == 0 or len(filas) == total:
                print(f"  {len(filas)}/{total} fotos · {len(filas) / max(time.time() - inicio, 1e-6):.1f} fotos/s", flush=True)

    por_cultivo = defaultdict(list)
    por_clase = defaultdict(list)
    for f in filas:
        por_cultivo[f["cultivo"]].append(f)
        por_clase[f["real"]].append(f)
    confusiones = Counter((f["real"], f["prediccion"]) for f in filas if f["estado"] == "OK" and not f["correcto"])

    datos = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "total": resumen(filas),
        "porCultivo": {k: resumen(v) for k, v in sorted(por_cultivo.items())},
        "porAfeccion": {k: resumen(v) for k, v in sorted(por_clase.items())},
        "confusiones": [{"real": a, "prediccion": b, "veces": n} for (a, b), n in confusiones.most_common(15)],
        "estados": dict(Counter(f["estado"] for f in filas)),
    }
    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / "metricas_argentina.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    t = datos["total"]
    lineas = [
        "# IA de SIGAgro — precisión con fotos argentinas",
        "",
        f"- Fecha: {datos['fecha']} · {t['fotos']} fotos de lotes argentinos validadas por Ingeniero Agrónomo, nunca usadas para entrenar",
        f"- **Reciben diagnóstico: {pct(t['cobertura'])}** (el resto queda inconcluso: se pide otra foto) · "
        f"**acierto de esos diagnósticos: {pct(t['aciertoDeLosDiagnosticos'])}** · acierto sobre el total: {pct(t['aciertoSobreTotal'])}",
        "",
        "## Por cultivo",
        "",
        "| Cultivo | Fotos | Reciben diagnóstico | Acierto de esos | Acierto sobre el total |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {v['fotos']} | {pct(v['cobertura'])} | {pct(v['aciertoDeLosDiagnosticos'])} | {pct(v['aciertoSobreTotal'])} |"
            for k, v in datos["porCultivo"].items()
        ],
        "",
        "## Por afección",
        "",
        "| Afección | Fotos | Reciben diagnóstico | Acierto de esos |",
        "|---|---|---|---|",
        *[
            f"| {(info_de(k).nombre if info_de(k) else k)} (`{k}`) | {v['fotos']} | {pct(v['cobertura'])} | {pct(v['aciertoDeLosDiagnosticos'])} |"
            for k, v in datos["porAfeccion"].items()
        ],
    ]
    if datos["confusiones"]:
        lineas += ["", "## Confusiones más frecuentes", "", "| Real | El modelo dijo | Veces |", "|---|---|---|"]
        lineas += [f"| `{c['real']}` | `{c['prediccion']}` | {c['veces']} |" for c in datos["confusiones"]]
    lineas += [
        "",
        "Con pocas fotos por afección los porcentajes varían mucho: conviene juntar al menos 20–30 por afección",
        "prioritaria antes de sacar conclusiones.",
    ]
    (DESTINO / "METRICAS_ARGENTINA.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nReciben diagnóstico {pct(t['cobertura'])} · acierto {pct(t['aciertoDeLosDiagnosticos'])}. Ver {DESTINO / 'METRICAS_ARGENTINA.md'}")


if __name__ == "__main__":
    main()
