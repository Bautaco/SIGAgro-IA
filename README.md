# SIGAgro IA — servicio de diagnóstico fitosanitario por imagen

Servicio Python (FastAPI) independiente del backend de SIGAgro. El backend lo consume solo a través de
`ClienteInferenciaIA` (`SIGAgro-BackEnd/src/common/inferencia-ia/`).

## Puesta en marcha

```powershell
cd SIGAgro-IA
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m pytest          # tests (clasificador falso, no descargan el modelo)
```

La primera vez descarga el modelo de Hugging Face (queda en caché). En el backend: `AI_SERVICE_URL=http://localhost:8000`.

Forma recomendada de iniciarlo (copia `.env.example` a `.env` la primera vez): `.\iniciar.ps1` carga `.env` y levanta
uvicorn en `127.0.0.1:8000` (`-Puerto`, `-Direccion 0.0.0.0` solo si el backend corre en otra máquina). `start-dev.ps1` y
`start-prod.ps1` de la carpeta raíz ya lo lanzan.

Despliegue en Linux/nube con Docker (el modelo queda dentro de la imagen, no necesita internet al ejecutar):
```bash
docker build -t sigagro-ia .
docker run -d --name sigagro-ia --restart unless-stopped -p 127.0.0.1:8000:8000 -e TOKEN_SERVICIO=<token> sigagro-ia
```
Ver `DESPLIEGUE.md` en la carpeta raíz del sistema.

## Configuración (variables de entorno)

Se leen del entorno del proceso (el servicio no carga un archivo `.env`; `.env.example` sirve de referencia).

| Variable | Defecto | Uso |
|---|---|---|
| `MODELO_ID` | `wambugu71/crop_leaf_diseases_vit` | Modelo `image-classification` de Hugging Face |
| `MODELO_REVISION` | (última) | Commit fijo del modelo, para resultados reproducibles |
| `UMBRAL_CONFIANZA` | `0.60` | Por debajo, el resultado es INCONCLUSO |
| `MARGEN_MINIMO` | `0.15` | Si la 1ª y 2ª clase están más cerca, INCONCLUSO |
| `TOKEN_SERVICIO` | (vacío) | Token compartido; el backend lo envía con `AI_SERVICE_TOKEN`. Vacío = sin autenticación (solo desarrollo) |
| `MAX_BYTES` | 8 MB | Tamaño máximo de imagen (413 si se supera) |
| `LADO_MINIMO_PX`, `BRILLO_MINIMO`, `BRILLO_MAXIMO`, `NITIDEZ_MINIMA` | 160, 35, 235, 25 | Control de calidad (IMAGEN_ILEGIBLE) |

## API (contrato versión 2)

- `GET /salud` (sin token): `{ estado: OK | MODELO_NO_DISPONIBLE, modelo, error, versionContrato }`
- `GET /modelo`: id, versión (commit), etiquetas, `cultivosCubiertos`, `umbralConfianza`, `estimaGravedad: false`
- `POST /inferencia` (`Authorization: Bearer <TOKEN_SERVICIO>` si está configurado)

Pedido:
```json
{ "imagen": "data:image/jpeg;base64,...", "formato": "jpg", "cultivo": "Maíz",
  "tipoMuestra": "PLANTA | CULTIVO | SUELO", "origen": "MUESTRA_MONITOREO | EVIDENCIA_LABOR" }
```
Solo se acepta la imagen embebida (data URL o base64). **No se descargan URLs remotas** (evita SSRF).

Respuesta 200 (siempre la misma forma):
```json
{ "versionContrato": 2, "estado": "OK | INCONCLUSO | IMAGEN_INVALIDA | IMAGEN_ILEGIBLE | NO_SOPORTADO",
  "mensaje": "motivo y qué hacer (vacío si OK)",
  "afeccion": "Roya común del maíz", "confianza": 0.851, "severidad": null,
  "tipoDeteccion": "ENFERMEDAD | SANO", "cultivo": "Maíz", "especie": null, "estadoPlanta": "...",
  "agente": "Puccinia sorghi (hongo)", "descripcion": "...", "sintomas": [], "condiciones": [],
  "consecuencias": [], "alternativasControl": [], "recomendaciones": "...",
  "alternativas": [{ "etiqueta": "...", "afeccion": "Mancha gris del maíz", "cultivo": "Maíz", "confianza": 0.13 }],
  "observaciones": [], "etiquetaModelo": "Corn___Common_Rust",
  "modelo": { "id": "...", "version": "7d5b32b..." }, "calidadImagen": { "ancho": 960, "alto": 720, "brillo": 0, "nitidez": 0, "legible": true, "problemas": [] },
  "fuenteInformacion": "...", "tiempoProcesamientoMs": 301 }
```
Errores HTTP: `401` token inválido · `413` imagen demasiado grande · `422` pedido mal formado · `503` modelo no cargado.

### Cuándo NO hay diagnóstico

| Estado | Cuándo |
|---|---|
| `IMAGEN_INVALIDA` | No es una imagen JPG/PNG/WEBP válida, es una URL remota, o el modelo la clasifica como "no es una hoja" |
| `IMAGEN_ILEGIBLE` | Resolución insuficiente, muy oscura, sobreexpuesta o desenfocada |
| `NO_SOPORTADO` | Muestra de suelo, o el cultivo indicado no está entre los que cubre el modelo |
| `INCONCLUSO` | Confianza < umbral, dos clases con probabilidades parejas, o el cultivo reconocido no coincide con el indicado |

En esos casos no se informa afección, síntomas ni recomendaciones: se pide otra foto o la revisión del Ingeniero.

## Reglas de seguridad agronómica

- `confianza` es la probabilidad softmax del modelo, sin ajustes. **No es la gravedad.**
- El modelo **no estima la gravedad**: `severidad` es siempre `null`; la determina el Ingeniero al validar.
- Un `OK` es un diagnóstico **probable**, nunca confirmado (lo valida el Ingeniero en SIGAgro).
- La información de referencia (`app/conocimiento.py`) describe el problema y alternativas de control
  **generales**. No nombra productos ni dosis: salen del catálogo fitosanitario de SIGAgro (cargado desde
  etiquetas oficiales) y de la receta del Ingeniero.

## Limitaciones conocidas del modelo por defecto

- Cubre **maíz** (roya común, mancha gris, sano), **trigo** (roya anaranjada, roya amarilla, sano), **papa**
  (tizón temprano, tizón tardío, sano) y **arroz** (mancha parda, piricularia, sano). **No cubre soja, girasol
  ni otros cultivos**, ni plagas, malezas o deficiencias.
- Entrenado con imágenes de datasets públicos: en fotos de campo puede confundir cultivos con confianza alta
  (en pruebas: una hoja de maíz sana clasificada como "trigo sano" al 92 %, roya de trigo como "roya de maíz"
  al 95 %). Por eso es importante indicar el cultivo: si no coincide, el resultado es INCONCLUSO.
- Para uso productivo conviene entrenar un modelo propio con fotos locales y reemplazarlo con `MODELO_ID`
  (agregando sus etiquetas en `app/conocimiento.py`).
