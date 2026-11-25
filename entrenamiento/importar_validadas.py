"""
Importa al dataset las fotos de diagnósticos VALIDADOS por el Ingeniero Agrónomo en SIGAgro (aprendizaje activo).

Son las mejores fotos para entrenar: tomadas en lotes argentinos, con celular, y con la afección confirmada por
un profesional. El backend las expone en `GET /api/diagnostico-ia/diagnosticos/dataset` (solo Administrador /
Dueño), sin datos personales ni ubicación.

- Si el Ingeniero ACEPTÓ el resultado, la clase es la que devolvió la IA; si lo RECTIFICÓ eligiendo una clase de
  la lista, es esa. Esas fotos van a `entrenamiento/datos/<Clase>/sigagro_<id>.jpg`.
- Si la afección no es una clase del modelo («Otra»), la foto va a `entrenamiento/sin_clase/` con su afección en
  `sin_clase/manifiesto.json`. **La IA aprende también de esas afecciones nuevas**: cuando junta `--minimo-clase`
  fotos (25 por defecto) de la misma afección en el mismo cultivo, crea la clase `<Cultivo>___Ing_<afeccion>`,
  mueve las fotos a `datos/` y la anota en `entrenamiento/clases_aprendidas.json` (nombre y cultivo tal como los
  escribió el Ingeniero; nunca se inventan síntomas). Al reentrenar, el modelo la reconoce y el Ingeniero la ve
  en la lista para elegirla en las próximas correcciones. Si después se le carga una ficha técnica en
  `app/conocimiento.py`, esa ficha reemplaza a la mínima.
- Es incremental: recuerda la última fecha de validación importada (`entrenamiento/salida/importadas.json`).

Las fotos con prefijo `sigagro_` se informan aparte en las métricas (acierto con fotos propias).

Uso (la contraseña se pide por teclado si no está en la variable de entorno):
    $env:SIGAGRO_API = "http://localhost:3000/api"     # opcional
    $env:SIGAGRO_USUARIO = "sigagro@gmail.com"
    .venv\\Scripts\\python.exe -m entrenamiento.importar_validadas [--desde-cero]
"""

from __future__ import annotations

import argparse
import base64
import getpass
import io
import json
import os
import time
from pathlib import Path

import re
import shutil
import unicodedata
from collections import defaultdict

import httpx
from PIL import Image

from app.conocimiento import CLASES, normalizar_cultivo

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
SIN_CLASE = RAIZ / "sin_clase"
ESTADO = RAIZ / "salida" / "importadas.json"
CLASES_APRENDIDAS = RAIZ / "clases_aprendidas.json"
# Prefijo de la etiqueta según el cultivo (mismo estilo que las demás clases del modelo)
PREFIJO_CULTIVO = {
    "Soja": "Soybean", "Maíz": "Corn", "Trigo": "Wheat", "Girasol": "Sunflower", "Maní": "Groundnut", "Papa": "Potato",
    "Arroz": "Rice", "Sorgo": "Sorghum", "Poroto": "Bean", "Algodón": "Cotton", "Caña de azúcar": "Sugarcane",
}
LADO = 320
PREFIJO = "sigagro"


def _imagen_de_data_url(valor: str) -> Image.Image | None:
    if not isinstance(valor, str) or not valor.startswith("data:image/") or "," not in valor:
        return None  # el servicio nunca descarga URLs remotas
    try:
        return Image.open(io.BytesIO(base64.b64decode(valor.split(",", 1)[1]))).convert("RGB")
    except Exception:  # noqa: BLE001 - foto dañada: se omite
        return None


def _guardar(img: Image.Image, destino: Path) -> None:
    img.thumbnail((LADO, LADO))
    destino.parent.mkdir(parents=True, exist_ok=True)
    img.save(destino, format="JPEG", quality=90)


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def _clave_afeccion(texto: str) -> str:
    """'Mancha anillada (Corynespora)' -> 'mancha_anillada_corynespora' (agrupa variantes de mayúsculas y tildes)."""
    return re.sub(r"[^a-z]+", "_", _sin_tildes(texto).lower()).strip("_")[:50]


def crear_clases_aprendidas(manifiesto: list[dict], minimo: int) -> tuple[list[dict], list[str]]:
    """Agrupa las fotos sin clase por cultivo y afección; con `minimo` fotos o más, crea la clase nueva.

    Devuelve el manifiesto con las fotos que siguen sin clase y las etiquetas creadas.
    """
    grupos: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in manifiesto:
        cultivo = normalizar_cultivo(m.get("cultivo"))
        clave = _clave_afeccion(m.get("afeccion") or "")
        if cultivo and clave:
            grupos[(cultivo, clave)].append(m)
    registro = json.loads(CLASES_APRENDIDAS.read_text(encoding="utf-8")) if CLASES_APRENDIDAS.exists() else {}
    creadas, usadas = [], set()
    for (cultivo, clave), fotos in grupos.items():
        prefijo = PREFIJO_CULTIVO.get(cultivo) or re.sub(r"[^A-Za-z]", "", _sin_tildes(cultivo)) or "Cultivo"
        etiqueta = f"{prefijo}___Ing_{clave}"
        if len(fotos) < minimo and etiqueta not in registro:
            continue
        # El nombre visible: el texto más usado por los Ingenieros para esa afección
        nombres = defaultdict(int)
        for f in fotos:
            nombres[(f.get("afeccion") or "").strip()] += 1
        for f in fotos:
            origen = SIN_CLASE / f"{PREFIJO}_{f['id']}.jpg"
            if origen.exists():
                destino = DATOS / etiqueta / origen.name
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(origen), str(destino))
            usadas.add(f["id"])
        anterior = registro.get(etiqueta, {})
        registro[etiqueta] = {
            "nombre": anterior.get("nombre") or max(nombres, key=nombres.get),
            "cultivo": cultivo,
            "fotos": len(list((DATOS / etiqueta).glob("*.jpg"))),
            "creada": anterior.get("creada") or time.strftime("%Y-%m-%d"),
        }
        if etiqueta not in anterior and not anterior:
            creadas.append(etiqueta)
    if registro:
        CLASES_APRENDIDAS.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    return [m for m in manifiesto if m.get("id") not in usadas], creadas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desde-cero", action="store_true", help="Volver a revisar todas las validaciones")
    ap.add_argument("--minimo-clase", type=int, default=25, help="Fotos validadas necesarias para crear una afección nueva")
    args = ap.parse_args()

    api = os.environ.get("SIGAGRO_API", "http://localhost:3000/api").rstrip("/")
    usuario = os.environ.get("SIGAGRO_USUARIO") or input("Usuario o correo (Administrador / Dueño): ").strip()
    password = os.environ.get("SIGAGRO_PASSWORD") or getpass.getpass("Contraseña: ")
    estado = {} if args.desde_cero or not ESTADO.exists() else json.loads(ESTADO.read_text(encoding="utf-8"))
    desde = estado.get("ultimaValidacion")
    clases_validas = {e for e, i in CLASES.items() if i.cultivo}

    with httpx.Client(base_url=api, timeout=60) as http:
        r = http.post("/seguridad/auth/login", json={"usernameOrEmail": usuario, "password": password})
        if r.status_code != 200 and r.status_code != 201:
            raise SystemExit(f"No se pudo iniciar sesión ({r.status_code}): {r.text[:200]}")
        token = r.json().get("accessToken")
        http.headers.update({"Authorization": f"Bearer {token}", "x-session-token": token})
        try:
            nuevas, sin_clase, omitidas, pagina = 0, 0, 0, 1
            manifiesto_ruta = SIN_CLASE / "manifiesto.json"
            manifiesto = json.loads(manifiesto_ruta.read_text(encoding="utf-8")) if manifiesto_ruta.exists() else []
            ultima = desde
            while True:
                params = {"pagina": pagina, "limite": 20, **({"desde": desde} if desde else {})}
                r = http.get("/diagnostico-ia/diagnosticos/dataset", params=params)
                if r.status_code == 403:
                    raise SystemExit("El usuario no tiene permiso: la exportación es solo para Administrador / Dueño.")
                r.raise_for_status()
                datos = r.json()
                items = datos.get("items", [])
                if not items:
                    break
                for it in items:
                    ultima = max(filter(None, [ultima, it.get("fechaValidacion")]), default=ultima)
                    img = _imagen_de_data_url(it.get("foto"))
                    if img is None:
                        omitidas += 1
                        continue
                    etiqueta = it.get("etiqueta")
                    if etiqueta in clases_validas:
                        destino = DATOS / etiqueta / f"{PREFIJO}_{it['id']}.jpg"
                        if not destino.exists():
                            _guardar(img, destino)
                            nuevas += 1
                    else:
                        destino = SIN_CLASE / f"{PREFIJO}_{it['id']}.jpg"
                        if not destino.exists():
                            _guardar(img, destino)
                            manifiesto.append({k: it.get(k) for k in ("id", "afeccion", "cultivo", "tipoMuestra", "origen", "etiquetaIa", "fechaValidacion")})
                            sin_clase += 1
                print(f"  página {pagina}: {len(items)} diagnósticos (de {datos.get('total')})", flush=True)
                pagina += 1
        finally:
            http.post("/seguridad/auth/logout")

    # Afecciones nuevas escritas por los Ingenieros: con suficientes fotos, se convierten en clases del modelo
    manifiesto, creadas = crear_clases_aprendidas(manifiesto, args.minimo_clase)
    if sin_clase or creadas or manifiesto_ruta.exists():
        SIN_CLASE.mkdir(parents=True, exist_ok=True)
        manifiesto_ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    for e in creadas:
        print(f"Nueva afección aprendida de los Ingenieros: {e}")
    ESTADO.parent.mkdir(exist_ok=True)
    ESTADO.write_text(json.dumps({"ultimaValidacion": ultima}, indent=2), encoding="utf-8")
    print(f"\nImportadas al dataset: {nuevas} · sin clase del modelo (a revisar en {SIN_CLASE.name}/): {sin_clase} · sin foto: {omitidas}")
    if nuevas or creadas:
        print("Para reentrenar: .venv\\Scripts\\python.exe -m entrenamiento.entrenar y luego -m entrenamiento.ajuste_fino")


if __name__ == "__main__":
    main()
