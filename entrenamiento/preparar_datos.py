"""
Descarga y prepara las fotos para entrenar el modelo de cultivos extensivos (soja, girasol, maní).

Fuentes (todas CC BY 4.0, publicadas por Project-AgML en Hugging Face; ver FUENTES.md):
  - MH_SoyaHealthVision_disease_classification_leaf  (soja, India, a campo)
  - soybean_leaf_disease_classification_brazil        (soja, Brasil)
  - sunflower_disease_classification                  (girasol)
  - groundnut_leaf_disease_classification_2           (maní)

Cada parquet se procesa y las imágenes se guardan reducidas (lado mayor 320 px, JPEG) en
entrenamiento/datos/<Etiqueta>/<fuente>_<n>.jpg. Las clases de cada fuente se unifican en etiquetas
<Cultivo>___<Clase> (las mismas que usa app/conocimiento.py). Las clases ambiguas se descartan.

Uso:  .venv\\Scripts\\python.exe -m entrenamiento.preparar_datos [--maximo 1500]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
# Caché de Hugging Face en D: (junto al proyecto): el disco del sistema tiene poco espacio
os.environ.setdefault("HF_HOME", str(RAIZ / ".cache"))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ORG = "Project-AgML"

# fuente -> {clase original: etiqueta unificada | None (se descarta)}
FUENTES: dict[str, dict[str, str | None]] = {
    "MH_SoyaHealthVision_disease_classification_leaf": {
        "Frog_Leaf_Eye": "Soybean___Frog_Eye_Leaf_Spot",
        "Rust": "Soybean___Rust",
        "Spectoria_Brown_Spot": "Soybean___Septoria_Brown_Spot",
        "Mosaic": "Soybean___Mosaic",
        "Caterpillar_Semilooper_Pest": "Soybean___Defoliating_Caterpillars",
        "Healthy": "Soybean___Healthy",
    },
    "soybean_leaf_disease_classification_brazil": {
        "Caterpillar": "Soybean___Defoliating_Caterpillars",
        "Diabrotica speciosa": "Soybean___Diabrotica_speciosa",
        "Healthy": "Soybean___Healthy",
    },
    "sunflower_disease_classification": {
        "Downy_mildew": "Sunflower___Downy_Mildew",
        "Gray_mold": "Sunflower___Gray_Mold",
        "Fresh_leaf": "Sunflower___Healthy",
        "Leaf_scars": None,  # clase ambigua (daño de origen no especificado): no se usa
    },
    "groundnut_leaf_disease_classification_2": {
        "early_leaf_spot": "Groundnut___Early_Leaf_Spot",
        "late leaf spot": "Groundnut___Late_Leaf_Spot",
        "rust": "Groundnut___Rust",
        "nutrition deficiency": "Groundnut___Nutrient_Deficiency",
        "healthy leaf": "Groundnut___Healthy",
    },
}

LADO = 320


def _nombres_clase(parquet: Path) -> list[str] | None:
    """Nombres de las clases desde los metadatos de Hugging Face guardados en el parquet."""
    meta = pq.read_schema(parquet).metadata or {}
    crudo = meta.get(b"huggingface")
    if not crudo:
        return None
    feats = json.loads(crudo).get("info", {}).get("features", {})
    for f in feats.values():
        if isinstance(f, dict) and f.get("_type") == "ClassLabel":
            return f.get("names")
    return None


def _guardar(img_bytes: bytes, destino: Path) -> bool:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img.thumbnail((LADO, LADO))
        destino.parent.mkdir(parents=True, exist_ok=True)
        img.save(destino, format="JPEG", quality=90)
        return True
    except Exception:  # noqa: BLE001 - imagen dañada: se omite
        return False


def preparar(maximo: int) -> Counter:
    api = HfApi()
    conteo: Counter = Counter()
    for fuente, mapa in FUENTES.items():
        repo = f"{ORG}/{fuente}"
        archivos = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        print(f"\n[{fuente}] {len(archivos)} archivo(s)", flush=True)
        for nombre in archivos:
            ruta = Path(hf_hub_download(repo, nombre, repo_type="dataset"))
            clases = _nombres_clase(ruta)
            pf = pq.ParquetFile(ruta)
            columnas = pf.schema_arrow.names
            col_img = "image" if "image" in columnas else next(c for c in columnas if "image" in c.lower())
            col_lab = "label" if "label" in columnas else next(c for c in columnas if c != col_img)
            for lote in pf.iter_batches(batch_size=64, columns=[col_img, col_lab]):
                for fila in lote.to_pylist():
                    lab = fila[col_lab]
                    original = clases[lab] if clases and isinstance(lab, int) else str(lab)
                    etiqueta = mapa.get(original, "__desconocida__")
                    if etiqueta == "__desconocida__":
                        print(f"  ! clase sin mapear: {original!r}", flush=True)
                        mapa[original] = None
                        continue
                    if etiqueta is None or conteo[etiqueta] >= maximo:
                        continue
                    dato = fila[col_img]
                    img_bytes = dato.get("bytes") if isinstance(dato, dict) else dato
                    if not img_bytes:
                        continue
                    n = conteo[etiqueta]
                    if _guardar(img_bytes, DATOS / etiqueta / f"{fuente[:12]}_{n:05d}.jpg"):
                        conteo[etiqueta] += 1
            print(f"  {nombre}: listo ({sum(conteo.values())} imágenes en total)", flush=True)
    return conteo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maximo", type=int, default=1500, help="Máximo de imágenes por etiqueta (equilibra las clases)")
    args = ap.parse_args()
    if DATOS.exists() and any(DATOS.iterdir()):
        print(f"Ya existen datos en {DATOS}. Borrar la carpeta para volver a prepararlos.")
        sys.exit(0)
    conteo = preparar(args.maximo)
    print("\nImágenes por etiqueta:")
    for etiqueta, n in sorted(conteo.items()):
        print(f"  {etiqueta:40s} {n}")
    (RAIZ / "salida").mkdir(exist_ok=True)
    (RAIZ / "salida" / "datos.json").write_text(json.dumps(dict(sorted(conteo.items())), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
