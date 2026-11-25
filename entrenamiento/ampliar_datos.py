"""
Amplía los datos de entrenamiento con más cultivos y enfermedades presentes en Argentina.

Agrega a `entrenamiento/datos/<Etiqueta>/` fotos de conjuntos públicos con licencia abierta (ver FUENTES.md):
maíz, trigo, sorgo, papa, poroto, algodón y caña de azúcar, y más fotos de clases débiles de soja y maní.
Solo se usan clases de problemas presentes en el país; las demás se descartan (p. ej. MSV del maíz, roseta del
maní, virus del enrollamiento del algodón) y también las clases ambiguas.

- Tope por etiqueta (`--maximo`, contando lo que ya hay) y por fuente (`--por-fuente`), así ninguna fuente domina.
- Las fotos se guardan reducidas (lado mayor 320 px, JPEG) con un prefijo propio de cada fuente.
- Es reanudable: las fuentes terminadas se anotan en `entrenamiento/salida/fuentes_ampliadas.json`.
- Una fuente que falla (red, formato) se informa y se sigue con las demás.

Uso:  .venv\\Scripts\\python.exe -m entrenamiento.ampliar_datos [--maximo 900] [--por-fuente 500] [--solo nombre]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(RAIZ / ".cache"))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

from entrenamiento.preparar_datos import _nombres_clase  # noqa: E402

DATOS = RAIZ / "datos"
DESCARGAS = RAIZ / ".cache" / "descargas"
HECHAS = RAIZ / "salida" / "fuentes_ampliadas.json"
LADO = 320
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ───────── Hugging Face (Project-AgML): fuente -> (prefijo, {clase original: etiqueta | None}) ─────────
HF: dict[str, tuple[str, dict[str, str | None]]] = {
    "plant_village_classification": ("pvill", {
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Gray_Leaf_Spot",
        "Corn___Common_rust": "Corn___Common_Rust",
        "Corn___Northern_Leaf_Blight": "Corn___Northern_Leaf_Blight",
        "Corn___healthy": "Corn___Healthy",
        "Potato___Early_blight": "Potato___Early_Blight",
        "Potato___Late_blight": "Potato___Late_Blight",
        "Potato___healthy": "Potato___Healthy",
        "Soybean___healthy": "Soybean___Healthy",
    }),
    "plant_doc_classification": ("pdoc", {
        "Corn Gray leaf spot": "Corn___Gray_Leaf_Spot",
        "Corn leaf blight": "Corn___Northern_Leaf_Blight",
        "Corn rust leaf": "Corn___Common_Rust",
        "Potato leaf early blight": "Potato___Early_Blight",
        "Potato leaf late blight": "Potato___Late_Blight",
        "Soyabean leaf": "Soybean___Healthy",
    }),
    "corn_leaf_pest_classification": ("cpest", {
        "healthy": "Corn___Healthy",
        "helminthosporium_leaf_blight": "Corn___Northern_Leaf_Blight",
        "spodoptera_frugiperda": "Corn___Fall_Armyworm",
        "zonocerus_variegatus": None,  # langosta africana: no está en Argentina
    }),
    "soybean_leaf_disease_classification": ("soyld", {
        "Bacterial leaf Blight": "Soybean___Bacterial_Blight",
        "Septoria_Brown_Spot": "Soybean___Septoria_Brown_Spot",
        "Healthy": "Soybean___Healthy",
        "Vein Necrosis": None, "Dry_leaf": None, "Root_images": None,
    }),
    "groundnut_leaf_disease_classification": ("gnut1", {
        "ALTERNARIA LEAF SPOT": "Groundnut___Alternaria_Leaf_Spot",
        "RUST": "Groundnut___Rust",
        "HEALTHY": "Groundnut___Healthy",
        "LEAF SPOT": None,  # no distingue viruela temprana de tardía
        "ROSETTE": None,  # virosis africana: no está en Argentina
    }),
    "potato_leaf_disease_classification": ("potld", {
        "Phytopthora": "Potato___Late_Blight",
        "Virus": "Potato___Virus",
        "Healthy": "Potato___Healthy",
        "Bacteria": None, "Fungi": None, "Nematode": None, "Pest": None,  # categorías demasiado generales
    }),
    "bean_disease_uganda": ("beanu", {
        "angular_leaf_spot": "Bean___Angular_Leaf_Spot",
        "bean_rust": "Bean___Rust",
        "healthy": "Bean___Healthy",
    }),
    "bean_disease_classification_tanzania": ("beant", {
        "anthracnose": "Bean___Anthracnose",
        "rust": "Bean___Rust",
        "healthy": "Bean___Healthy",
    }),
    "cotton_leaf_disease_classification_bangladesh_2": ("cotbd", {
        "Bacterial Blight": "Cotton___Bacterial_Blight",
        "Herbicide Growth Damage": "Cotton___Herbicide_Damage",
        "Healthy Leaf": "Cotton___Healthy",
        "Curl Virus": None,  # no está en Argentina
        "Leaf Hopper Jassids": None, "Leaf Redding": None, "Leaf Variegation": None,
    }),
    "sugarcane_leaf_disease_classification": ("sugar", {
        "BrownRust": "Sugarcane___Brown_Rust",
        "Smut": "Sugarcane___Smut",
        "Yellow Leaf": "Sugarcane___Yellow_Leaf",
        "Pokkah Boeng": "Sugarcane___Pokkah_Boeng",
        "Brown Spot": "Sugarcane___Brown_Spot",
        "Healthy": "Sugarcane___Healthy",
        "Banded Chlorosis": None, "Dried": None, "Grassy shoot": None, "Sett Rot": None, "Viral Disease": None,
    }),
}

# ───────── Mendeley Data (CC BY 4.0) ─────────
MENDELEY_ARCHIVOS = {
    # nombre: (prefijo, URL del archivo, reglas [(palabra en la carpeta, etiqueta)], la ruta debe contener)
    "wheat_leaf_etiopia": (
        "wheet",
        "https://data.mendeley.com/public-files/datasets/wgd66f8n6h/files/4ec9266d-d7c1-4fd5-85c9-e5624c09a847/file_downloaded",
        [("stripe", "Wheat___Yellow_Rust"), ("septoria", "Wheat___Septoria"), ("healthy", "Wheat___Healthy")],
        None,
        "wheat_leaf.rar",
    ),
    "wheat_disease_bangladesh": (
        "whebd",
        "https://data.mendeley.com/public-files/datasets/5gc7hwydwg/files/5c163bc0-311f-4b10-bd09-d5cb949237cc/file_downloaded",
        [("blight", "Wheat___Leaf_Blight"), ("blast", "Wheat___Blast"), ("healthy", "Wheat___Healthy")],
        "original",  # el conjunto trae también copias aumentadas y balanceadas: solo se usan las originales
        "wheat_disease.zip",
    ),
}
SORGO_DATASET = "fgb3cmfrg2"
SORGO_CARPETAS = {
    "Anthracnose and Red Rot": "Sorghum___Anthracnose",
    "Rust": "Sorghum___Rust",
    "Head Smut": "Sorghum___Head_Smut",
    "Cereal Grain molds": "Sorghum___Grain_Mold",
    # Carbón cubierto y carbón volador: fotos de grano/panoja muy parecidas entre sí; se omiten
}


class Contador:
    """Lleva cuántas fotos tiene cada etiqueta (incluidas las que ya estaban) y cuántas aportó cada fuente."""

    def __init__(self, maximo: int, por_fuente: int):
        self.maximo, self.por_fuente = maximo, por_fuente
        self.total: Counter = Counter({p.name: sum(1 for f in p.iterdir() if f.suffix.lower() in EXT) for p in DATOS.iterdir() if p.is_dir()} if DATOS.exists() else {})
        self.fuente: Counter = Counter()

    def admite(self, prefijo: str, etiqueta: str) -> bool:
        return self.total[etiqueta] < self.maximo and self.fuente[(prefijo, etiqueta)] < self.por_fuente

    def guardar(self, prefijo: str, etiqueta: str, img: Image.Image) -> bool:
        if not self.admite(prefijo, etiqueta):
            return False
        try:
            img = img.convert("RGB")
            img.thumbnail((LADO, LADO))
            destino = DATOS / etiqueta / f"{prefijo}_{self.fuente[(prefijo, etiqueta)]:05d}.jpg"
            destino.parent.mkdir(parents=True, exist_ok=True)
            img.save(destino, format="JPEG", quality=90)
        except Exception:  # noqa: BLE001 - imagen dañada: se omite
            return False
        self.fuente[(prefijo, etiqueta)] += 1
        self.total[etiqueta] += 1
        return True

    def completa(self, prefijo: str, etiquetas: set[str]) -> bool:
        return all(not self.admite(prefijo, e) for e in etiquetas)


def _hf(fuente: str, prefijo: str, mapa: dict[str, str | None], c: Contador) -> None:
    repo = f"Project-AgML/{fuente}"
    archivos = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    destinos = {e for e in mapa.values() if e}
    print(f"\n[{fuente}] {len(archivos)} archivo(s) parquet", flush=True)
    for nombre in archivos:
        if c.completa(prefijo, destinos):
            print("  todas sus clases llegaron al tope: no se descarga el resto", flush=True)
            break
        ruta = Path(hf_hub_download(repo, nombre, repo_type="dataset"))
        clases = _nombres_clase(ruta) or _nombres_desde_hub(repo)
        pf = pq.ParquetFile(ruta)
        cols = pf.schema_arrow.names
        col_img = "image" if "image" in cols else next(x for x in cols if "image" in x.lower())
        col_lab = "label" if "label" in cols else next(x for x in cols if x != col_img and "label" in x.lower())
        for lote in pf.iter_batches(batch_size=64, columns=[col_img, col_lab]):
            for fila in lote.to_pylist():
                lab = fila[col_lab]
                original = clases[lab] if clases and isinstance(lab, int) else str(lab)
                if original not in mapa:
                    print(f"  ! clase sin mapear: {original!r} (se omite)", flush=True)
                    mapa[original] = None
                etiqueta = mapa[original]
                dato = fila[col_img]
                b = dato.get("bytes") if isinstance(dato, dict) else dato
                if etiqueta and b and c.admite(prefijo, etiqueta):
                    try:
                        c.guardar(prefijo, etiqueta, Image.open(io.BytesIO(b)))
                    except Exception:  # noqa: BLE001
                        pass
        print(f"  {nombre}: listo", flush=True)


def _nombres_desde_hub(repo: str) -> list[str] | None:
    """Nombres de las clases desde la ficha del conjunto (algunos parquet traen solo el número de clase)."""
    try:
        info = _curl_json(f"https://datasets-server.huggingface.co/info?dataset={repo}")["dataset_info"]
        for f in next(iter(info.values()))["features"].values():
            if isinstance(f, dict) and f.get("_type") == "ClassLabel":
                return f["names"]
    except Exception:  # noqa: BLE001
        return None
    return None


def _descargar(url: str, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0 and not destino.with_suffix(destino.suffix + ".parcial").exists():
        return destino
    marca = destino.with_suffix(destino.suffix + ".parcial")
    marca.touch()
    print(f"  descargando {destino.name}...", flush=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(destino), url], check=True)
    marca.unlink()
    return destino


def _etiqueta_por_carpeta(ruta: str, reglas, debe_contener: str | None) -> str | None:
    partes = [p.lower() for p in Path(ruta).parts[:-1]]
    if debe_contener and not any(debe_contener in p for p in partes):
        return None
    carpeta = partes[-1] if partes else ""
    for palabra, etiqueta in reglas:
        if palabra in carpeta:
            return etiqueta
    return None


def _mendeley_archivo(nombre: str, c: Contador) -> None:
    prefijo, url, reglas, debe_contener, archivo = MENDELEY_ARCHIVOS[nombre]
    print(f"\n[{nombre}]", flush=True)
    ruta = _descargar(url, DESCARGAS / archivo)
    if archivo.endswith(".zip"):
        with zipfile.ZipFile(ruta) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(EXT):
                    continue
                etiqueta = _etiqueta_por_carpeta(info.filename, reglas, debe_contener)
                if etiqueta and c.admite(prefijo, etiqueta):
                    with z.open(info) as f:
                        try:
                            c.guardar(prefijo, etiqueta, Image.open(io.BytesIO(f.read())))
                        except Exception:  # noqa: BLE001
                            pass
    else:
        # RAR: lo extrae el tar de Windows (libarchive)
        destino = DESCARGAS / (archivo + ".extraido")
        if not destino.exists():
            destino.mkdir(parents=True)
            subprocess.run(["tar", "-xf", str(ruta), "-C", str(destino)], check=True)
        for f in sorted(destino.rglob("*")):
            if f.suffix.lower() in EXT:
                etiqueta = _etiqueta_por_carpeta(str(f.relative_to(destino)), reglas, debe_contener)
                if etiqueta and c.admite(prefijo, etiqueta):
                    try:
                        c.guardar(prefijo, etiqueta, Image.open(f))
                    except Exception:  # noqa: BLE001
                        pass


def _curl_json(url: str):
    r = subprocess.run(["curl", "-s", "--fail", "--retry", "3", url], capture_output=True, check=True)
    return json.loads(r.stdout)


def _sorgo(c: Contador) -> None:
    print("\n[sorgo_mendeley]", flush=True)
    base = f"https://data.mendeley.com/public-api/datasets/{SORGO_DATASET}"
    for carpeta in _curl_json(f"{base}/folders/1"):
        etiqueta = SORGO_CARPETAS.get(carpeta["name"].strip())
        if not etiqueta:
            continue
        archivos = _curl_json(f"{base}/files?folder_id={carpeta['id']}&version=1")
        print(f"  {carpeta['name'].strip()}: {len(archivos)} fotos disponibles", flush=True)
        for a in archivos:
            if not c.admite("sorgo", etiqueta):
                break
            url = (a.get("content_details") or {}).get("download_url")
            if not url:
                continue
            r = subprocess.run(["curl", "-s", "-L", "--fail", "--retry", "3", url], capture_output=True)
            if r.returncode == 0 and r.stdout:
                try:
                    c.guardar("sorgo", etiqueta, Image.open(io.BytesIO(r.stdout)))
                except Exception:  # noqa: BLE001
                    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maximo", type=int, default=900, help="Máximo de fotos por etiqueta (contando las que ya hay)")
    ap.add_argument("--por-fuente", type=int, default=500, help="Máximo de fotos que aporta cada fuente a una etiqueta")
    ap.add_argument("--solo", help="Procesar solo esta fuente")
    args = ap.parse_args()

    hechas = set(json.loads(HECHAS.read_text(encoding="utf-8"))) if HECHAS.exists() else set()
    c = Contador(args.maximo, args.por_fuente)
    tareas = [(n, lambda n=n: _hf(n, *HF[n], c)) for n in HF]
    tareas += [(n, lambda n=n: _mendeley_archivo(n, c)) for n in MENDELEY_ARCHIVOS]
    tareas += [("sorgo_mendeley", lambda: _sorgo(c))]

    inicio = time.time()
    for nombre, tarea in tareas:
        if (args.solo and nombre != args.solo) or nombre in hechas:
            continue
        try:
            tarea()
            hechas.add(nombre)
            HECHAS.parent.mkdir(exist_ok=True)
            HECHAS.write_text(json.dumps(sorted(hechas), indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - se sigue con las demás fuentes
            print(f"  ✗ {nombre}: {e}", flush=True)
        print(f"  (llevamos {(time.time() - inicio) / 60:.0f} min)", flush=True)

    print("\nFotos por etiqueta:")
    for etiqueta, n in sorted(Contador(args.maximo, args.por_fuente).total.items()):
        print(f"  {etiqueta:40s} {n}")


if __name__ == "__main__":
    main()
