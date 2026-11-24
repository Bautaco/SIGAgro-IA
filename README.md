# SIGAgro IA — Diagnóstico fitosanitario por imagen

Servicio de inteligencia artificial de **SIGAgro (Sistema Integral de Gestión Agronómica)**. Recibe la fotografía de
una hoja tomada en el lote y devuelve un **diagnóstico probable** de enfermedad foliar, con su nivel de confianza e
información de referencia agronómica, para que luego lo **valide un Ingeniero Agrónomo** dentro de SIGAgro.

- **Tecnología:** Python 3.13 · FastAPI · Hugging Face Transformers · PyTorch (CPU)
- **Modelo por defecto:** [`wambugu71/crop_leaf_diseases_vit`](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) (Vision Transformer)
- **Cultivos cubiertos:** Maíz, Trigo, Papa y Arroz
- **Puerto:** `8000` · **Contrato de API:** versión 2

> Este servicio **no reemplaza al profesional**: un resultado `OK` es un diagnóstico *probable*, nunca confirmado.
> No estima la gravedad ni recomienda productos o dosis.

---

## Índice

1. [Cómo encaja en SIGAgro](#cómo-encaja-en-sigagro)
2. [Requisitos](#requisitos)
3. [Instalación y puesta en marcha](#instalación-y-puesta-en-marcha)
4. [Configuración](#configuración)
5. [Conexión con el backend](#conexión-con-el-backend)
6. [API](#api)
7. [Cómo se analiza una imagen](#cómo-se-analiza-una-imagen)
8. [Enfermedades que reconoce](#enfermedades-que-reconoce)
9. [Reglas de seguridad agronómica](#reglas-de-seguridad-agronómica)
10. [Limitaciones del modelo](#limitaciones-del-modelo)
11. [Despliegue con Docker](#despliegue-con-docker)
12. [Tests](#tests)
13. [Estructura del proyecto](#estructura-del-proyecto)
14. [Cambiar el modelo](#cambiar-el-modelo)
15. [Solución de problemas](#solución-de-problemas)

---

## Cómo encaja en SIGAgro

```
 Navegador (React)  ──►  BackEnd NestJS :3000  ──►  SIGAgro IA :8000  (este servicio)
                              │   ClienteInferenciaIA
                              └──►  PostgreSQL + PostGIS
```

- El **navegador nunca llama a este servicio**: solo lo consume el backend, a través de `ClienteInferenciaIA`
  (`SIGAgro-BackEnd/src/common/inferencia-ia/`). Por eso escucha en `127.0.0.1` y no se expone a internet.
- Flujo en SIGAgro: el usuario carga una **muestra** (foto, campo, lote, tipo de muestra y cultivo) → el backend pide
  la **inferencia** → el diagnóstico queda `PENDIENTE_VALIDACION` → el **Ingeniero Agrónomo** lo acepta, rectifica o
  rechaza y define la gravedad → se puede iniciar la **receta fitosanitaria** con productos del catálogo oficial.
- Si el servicio no está disponible, SIGAgro **sigue funcionando**: la muestra se guarda con estado
  `ERROR_SERVICIO` ("IA no disponible") y el análisis se puede reintentar más tarde. Nunca se inventa un resultado.

## Requisitos

| Recurso | Mínimo |
|---|---|
| Python | 3.13 |
| RAM | 2 GB libres (el modelo ocupa ~1 GB en memoria) |
| Disco | ~2 GB (PyTorch CPU + modelo) |
| Internet | Solo la primera vez, para descargar el modelo (~350 MB, queda en caché) |
| GPU | No necesaria |

## Instalación y puesta en marcha

En Windows (PowerShell), desde la carpeta `SIGAgro-IA`:

```powershell
# 1. Instalación completa (una sola vez): crea .venv, instala dependencias, crea .env y descarga el modelo
powershell -ExecutionPolicy Bypass -File .\instalar.ps1        # agregar -Dev para instalar pytest y correr los tests

# 2. Iniciar el servicio
powershell -ExecutionPolicy Bypass -File .\iniciar.ps1
```

`instalar.ps1` se puede volver a ejecutar sin problema (saltea lo que ya está hecho). Si no encuentra Python 3.13,
indica cómo instalarlo (`winget install Python.Python.3.13`).

<details>
<summary>Instalación manual (equivalente al script)</summary>

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
copy .env.example .env
.\iniciar.ps1
```
</details>

`iniciar.ps1` carga las variables de `.env` (o de `.env.example` si no existe) y levanta uvicorn en
`http://127.0.0.1:8000`. Opciones: `-Puerto 8001` y `-Direccion 0.0.0.0` (esta última solo si el backend corre en
otra máquina). Los scripts `start-dev.ps1` y `start-prod.ps1` de la carpeta raíz del sistema ya lo lanzan.

Para comprobar que está listo, abrir <http://127.0.0.1:8000/salud>: debe responder `"estado": "OK"`.
La documentación interactiva (Swagger) queda en <http://127.0.0.1:8000/docs>.

<details>
<summary>Iniciarlo sin el script (Linux/macOS o manualmente)</summary>

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export MODELO_REVISION=7d5b32bcd6f83a2f57e7e0346358fad276296877
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
El servicio lee **variables de entorno**; no carga `.env` por sí mismo (eso lo hace `iniciar.ps1`).
</details>

## Configuración

| Variable | Defecto | Uso |
|---|---|---|
| `MODELO_ID` | `wambugu71/crop_leaf_diseases_vit` | Modelo `image-classification` de Hugging Face |
| `MODELO_REVISION` | (última) | Commit fijo del modelo, para resultados reproducibles. `.env.example` trae el probado |
| `UMBRAL_CONFIANZA` | `0.60` | Por debajo, el resultado es `INCONCLUSO` |
| `MARGEN_MINIMO` | `0.15` | Si la 1ª y la 2ª clase están más cerca que esto, `INCONCLUSO` |
| `TOKEN_SERVICIO` | (vacío) | Token compartido con el backend (`Authorization: Bearer`). Vacío = sin autenticación, solo desarrollo |
| `MAX_BYTES` | `8388608` (8 MB) | Tamaño máximo de la imagen (413 si se supera) |
| `LADO_MINIMO_PX` | `160` | Resolución mínima del lado menor |
| `BRILLO_MINIMO` / `BRILLO_MAXIMO` | `35` / `235` | Rango de brillo medio aceptado (0–255) |
| `NITIDEZ_MINIMA` | `25` | Varianza del laplaciano mínima (detecta fotos movidas o desenfocadas) |
| `ALTERNATIVAS` | `3` | Cantidad de diagnósticos alternativos devueltos |

## Conexión con el backend

En `SIGAgro-BackEnd/.env`:

```env
AI_SERVICE_URL=http://127.0.0.1:8000
AI_SERVICE_TOKEN=          # el MISMO valor que TOKEN_SERVICIO (vacío si no se usa token)
AI_SERVICE_TIMEOUT_MS=15000
```

Con eso, la pantalla **Diagnóstico IA** de SIGAgro muestra la tarjeta "Servicio de IA: **Disponible**" y la lista
de cultivos cubiertos (`GET /api/diagnostico-ia/diagnosticos/servicio-ia` del backend consulta `/salud` y `/modelo`).

## API

| Método | Ruta | Token | Descripción |
|---|---|---|---|
| `GET` | `/salud` | No | Estado del servicio: `{ estado: OK \| MODELO_NO_DISPONIBLE, modelo, error, versionContrato }` |
| `GET` | `/modelo` | Sí* | Id y versión del modelo, etiquetas, `cultivosCubiertos`, `umbralConfianza`, `estimaGravedad: false` |
| `POST` | `/inferencia` | Sí* | Analiza una imagen |

\* Solo si `TOKEN_SERVICIO` está configurado: `Authorization: Bearer <TOKEN_SERVICIO>`.

### `POST /inferencia`

Pedido:

```json
{
  "imagen": "data:image/jpeg;base64,/9j/4AAQ...",
  "formato": "jpg",
  "cultivo": "Maíz",
  "tipoMuestra": "PLANTA",
  "origen": "MUESTRA_MONITOREO"
}
```

| Campo | Obligatorio | Valores |
|---|---|---|
| `imagen` | Sí | Data URL o base64 de una imagen JPG, PNG o WEBP. **No se aceptan URLs remotas** (evita SSRF) |
| `formato` | No | Texto informativo (`jpg`, `png`…) |
| `cultivo` | No (recomendado) | Cultivo del lote. Si el modelo reconoce otro cultivo, el resultado es `INCONCLUSO` |
| `tipoMuestra` | No | `PLANTA` (defecto), `CULTIVO`, `SUELO` |
| `origen` | No | `MUESTRA_MONITOREO` (defecto), `EVIDENCIA_LABOR` |

Respuesta `200` (siempre la misma forma; los campos de la afección van vacíos si no hay diagnóstico):

```json
{
  "versionContrato": 2,
  "estado": "OK",
  "mensaje": "",
  "afeccion": "Roya común del maíz",
  "confianza": 0.851,
  "severidad": null,
  "tipoDeteccion": "ENFERMEDAD",
  "cultivo": "Maíz",
  "estadoPlanta": "Con síntomas compatibles con roya común del maíz.",
  "agente": "Puccinia sorghi (hongo)",
  "descripcion": "...",
  "sintomas": ["..."],
  "condiciones": ["..."],
  "consecuencias": ["..."],
  "alternativasControl": ["..."],
  "recomendaciones": "...",
  "alternativas": [{ "etiqueta": "Corn___Gray_Leaf_Spot", "afeccion": "Mancha gris del maíz", "cultivo": "Maíz", "confianza": 0.13 }],
  "observaciones": ["Resultado probable generado por un modelo de IA: no es un diagnóstico confirmado. ..."],
  "etiquetaModelo": "Corn___Common_Rust",
  "modelo": { "id": "wambugu71/crop_leaf_diseases_vit", "version": "7d5b32b..." },
  "calidadImagen": { "ancho": 960, "alto": 720, "formato": "JPEG", "brillo": 128.4, "nitidez": 310.2, "legible": true, "problemas": [] },
  "fuenteInformacion": "...",
  "tiempoProcesamientoMs": 301
}
```

Errores HTTP: `401` token inválido · `413` imagen demasiado grande · `422` pedido mal formado · `503` modelo todavía
no cargado o con error.

Prueba rápida desde PowerShell:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\fotos\hoja-maiz.jpg"))
$body = @{ imagen = "data:image/jpeg;base64,$b64"; cultivo = "Maíz"; tipoMuestra = "PLANTA" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/inferencia -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

## Cómo se analiza una imagen

```
Pedido ─► ¿SUELO? ─sí─► NO_SOPORTADO
            │no
            ▼
       Decodificar ─error─► IMAGEN_INVALIDA      (no es JPG/PNG/WEBP, es una URL, está dañada)
            │                413                 (supera MAX_BYTES)
            ▼
     Control de calidad ─falla─► IMAGEN_ILEGIBLE (resolución, oscura, sobreexpuesta, desenfocada)
            │
            ▼
   ¿Cultivo cubierto? ─no─► NO_SOPORTADO         (p. ej. soja, girasol)
            │sí
            ▼
     Clasificación (ViT, softmax)
            │
            ├─ clase "Invalid" con confianza alta ─► IMAGEN_INVALIDA (no parece una hoja)
            ├─ confianza < UMBRAL_CONFIANZA ───────► INCONCLUSO
            ├─ 1ª − 2ª clase < MARGEN_MINIMO ──────► INCONCLUSO
            ├─ cultivo reconocido ≠ indicado ──────► INCONCLUSO
            ▼
           OK  (diagnóstico probable + información de referencia + alternativas)
```

| Estado | Significado | Qué hace SIGAgro |
|---|---|---|
| `OK` | Diagnóstico probable | Queda `PENDIENTE_VALIDACION` con gravedad `SIN_DETERMINAR` |
| `INCONCLUSO` | El modelo respondió pero no identificó con seguridad | Pide una nueva foto o la revisión del Ingeniero; las clases consideradas se muestran solo como alternativas |
| `IMAGEN_INVALIDA` | No es una imagen válida o no es una hoja | Pide otra foto |
| `IMAGEN_ILEGIBLE` | Calidad insuficiente (se indica el motivo en `calidadImagen.problemas`) | Pide otra foto, de cerca, con luz y enfocada |
| `NO_SOPORTADO` | Muestra de suelo o cultivo no cubierto | Queda para evaluación del Ingeniero o laboratorio |

## Enfermedades que reconoce

| Cultivo | Clases |
|---|---|
| Maíz | Roya común (*Puccinia sorghi*) · Mancha gris · Sano |
| Trigo | Roya anaranjada · Roya amarilla · Sano |
| Papa | Tizón temprano · Tizón tardío · Sano |
| Arroz | Mancha parda · Piricularia (quemado) · Sano |
| — | `Invalid`: la imagen no es una hoja analizable |

La información de referencia de cada clase (agente, síntomas, condiciones predisponentes, consecuencias y
alternativas de control generales) está en `app/conocimiento.py`.

## Reglas de seguridad agronómica

- `confianza` es la **probabilidad softmax del modelo**, sin ajustes. **No es la gravedad.**
- El modelo **no estima la gravedad**: `severidad` es siempre `null`; la determina el Ingeniero Agrónomo al validar.
- Un `OK` es un diagnóstico **probable**; solo pasa a confirmado cuando lo valida el Ingeniero en SIGAgro.
- La información de referencia **no nombra productos ni dosis**. Los productos y dosis salen del catálogo
  fitosanitario de SIGAgro (cargado desde etiquetas oficiales SENASA) y de la receta del Ingeniero. Hay un test que
  lo verifica (`test_ninguna_recomendacion_incluye_dosis`).
- En un resultado no concluyente no se informa afección, síntomas ni recomendaciones.

## Limitaciones del modelo

- **No cubre soja, girasol ni otros cultivos**, ni plagas, malezas o deficiencias nutricionales.
- Fue entrenado con imágenes de datasets públicos (fotos de laboratorio). En fotos de campo puede confundir cultivos
  con confianza alta: en pruebas, una hoja de maíz sana salió como "trigo sano" al 92 % y una roya de trigo como
  "roya de maíz" al 95 %. Por eso conviene **indicar siempre el cultivo**: si no coincide, el resultado es
  `INCONCLUSO`.
- Para uso productivo conviene entrenar un modelo propio con fotos locales y reemplazarlo
  (ver [Cambiar el modelo](#cambiar-el-modelo)).

## Despliegue con Docker

Para un servidor Linux o en la nube. El modelo se descarga **al construir la imagen**, así que en ejecución no
necesita internet (`HF_HUB_OFFLINE=1`).

```bash
docker build -t sigagro-ia .
docker run -d --name sigagro-ia --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e TOKEN_SERVICIO=<token-largo-y-secreto> \
  sigagro-ia
curl http://127.0.0.1:8000/salud
```

- Publicar el puerto solo en `127.0.0.1` (o en la red interna de Docker): la IA no debe quedar expuesta.
- Reservar al menos **2 GB de RAM**; en planes de 512 MB el modelo no carga.
- La guía completa del sistema (base de datos, backend, frontend, Nginx y HTTPS) está en `DESPLIEGUE.md`, en la
  carpeta raíz de SIGAgro.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Usan un **clasificador falso**, así que no descargan el modelo y corren en segundos. Cubren: resultado `OK` sin
gravedad, ausencia de dosis, baja confianza y clases parejas → `INCONCLUSO`, cultivo distinto o no cubierto, muestra
de suelo, imagen inválida / no imagen / formato no admitido, rechazo de URLs remotas, imágenes ilegibles (pequeña,
oscura, sobreexpuesta, desenfocada), límite de tamaño (413), token de servicio (401), modelo no cargado (503) e
información del modelo.

## Estructura del proyecto

```
SIGAgro-IA/
├── app/
│   ├── main.py          # API FastAPI: /salud, /modelo, /inferencia, token y ciclo de vida (carga del modelo)
│   ├── analisis.py      # Reglas de decisión: estados, umbral, margen, cultivo, alternativas
│   ├── imagenes.py      # Decodificación segura de la imagen y control de calidad (tamaño, brillo, nitidez)
│   ├── modelo.py        # Carga del clasificador de Hugging Face (con revisión fija)
│   ├── conocimiento.py  # Información de referencia por clase y normalización de cultivos
│   └── config.py        # Variables de entorno
├── tests/test_api.py    # Tests con clasificador falso
├── instalar.ps1         # Instalación del entorno en Windows (venv, dependencias, .env, modelo)
├── iniciar.ps1          # Inicio en Windows (carga .env)
├── Dockerfile           # Imagen para despliegue (modelo incluido)
├── .env.example         # Configuración de referencia
├── requirements.txt     # Dependencias de ejecución
└── requirements-dev.txt # + pytest y httpx
```

## Cambiar el modelo

1. Entrenar o elegir un modelo `image-classification` de Hugging Face (o una carpeta local con el modelo).
2. Configurar `MODELO_ID` (y `MODELO_REVISION` con el commit, para que los resultados sean reproducibles).
3. Agregar en `app/conocimiento.py` una entrada `InfoClase` por cada etiqueta nueva (cultivo, tipo `ENFERMEDAD` /
   `SANO` / `INVALIDA`, nombre en español e información de referencia **sin productos ni dosis**). Una etiqueta sin
   entrada se informa con tipo `DESCONOCIDO` y sin información de referencia.
4. Si se agrega un cultivo, sumarlo a la normalización de cultivos del mismo archivo.
5. Correr los tests y verificar `GET /modelo` (`cultivosCubiertos`).

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| SIGAgro muestra "IA no disponible" (`SIN_CONEXION`) | El servicio no está corriendo o `AI_SERVICE_URL` es incorrecta | Iniciar con `.\iniciar.ps1`; revisar `.env` del backend y reiniciarlo |
| `AUTENTICACION` / 401 | `AI_SERVICE_TOKEN` del backend ≠ `TOKEN_SERVICIO` | Poner el mismo valor en ambos y reiniciar los dos |
| `TIEMPO_AGOTADO` | Equipo lento (la primera inferencia es la más lenta) | Subir `AI_SERVICE_TIMEOUT_MS` a 30000–60000 |
| `/salud` responde `MODELO_NO_DISPONIBLE` o 503 | Sin internet en el primer arranque, poca RAM o todavía cargando | Esperar unos segundos; dar internet la primera vez; ≥2 GB de RAM; ver el campo `error` |
| Siempre `NO_SOPORTADO` | El cultivo del lote no está cubierto | Limitación del modelo: solo maíz, trigo, papa y arroz |
| Muchos `INCONCLUSO` | Fotos lejanas, con varias hojas o fondo | Fotografiar una sola hoja con síntomas, de cerca, con luz natural |
| `No existe el entorno virtual` al iniciar | Falta la instalación | Ejecutar `.\instalar.ps1` |

---

Proyecto académico — Seminario Integrador. Modelo de clasificación de terceros publicado en Hugging Face
([`wambugu71/crop_leaf_diseases_vit`](https://huggingface.co/wambugu71/crop_leaf_diseases_vit)); revisar su licencia
antes de cualquier uso comercial.
