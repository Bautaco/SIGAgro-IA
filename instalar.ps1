# ============================================================
# SIGAgro IA - Instalación del entorno de Python (Windows)
# Crea .venv, instala dependencias (PyTorch CPU), crea .env, descarga el modelo y verifica.
# Se puede ejecutar más de una vez: lo que ya está hecho se saltea.
#   .\instalar.ps1            -> instalación normal
#   .\instalar.ps1 -Dev       -> además pytest/httpx y corre los tests
# ============================================================
param([switch]$Dev)

$ErrorActionPreference = 'Stop'
$dir = $PSScriptRoot
Set-Location $dir
$python = Join-Path $dir '.venv\Scripts\python.exe'

function Paso($texto) { Write-Host "`n[IA] $texto" -ForegroundColor Magenta }

# 1. Python 3.13
Paso 'Buscando Python 3.13...'
$tienePy = [bool](Get-Command py -ErrorAction SilentlyContinue)
if ($tienePy) { & py -3.13 --version 2>$null | Out-Null; $tienePy = ($LASTEXITCODE -eq 0) }
if (-not $tienePy -and -not (Test-Path $python)) {
  Write-Host 'No se encontró Python 3.13.' -ForegroundColor Red
  Write-Host 'Instalalo desde https://www.python.org/downloads/ (marcar "Add python.exe to PATH" y "py launcher")' -ForegroundColor Yellow
  Write-Host 'o con:  winget install Python.Python.3.13' -ForegroundColor Yellow
  exit 1
}

# 2. Entorno virtual
if (-not (Test-Path $python)) {
  Paso 'Creando el entorno virtual (.venv)...'
  & py -3.13 -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual.' }
} else {
  Paso 'El entorno virtual ya existe.'
}

# 3. Dependencias
Paso 'Instalando dependencias (la primera vez descarga ~1 GB: PyTorch CPU, Transformers, FastAPI)...'
& $python -m pip install --upgrade pip --quiet
$requisitos = if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' }
& $python -m pip install -r $requisitos --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Falló la instalación de dependencias (revisá la conexión a internet).' }

# 4. Configuración
if (-not (Test-Path (Join-Path $dir '.env'))) {
  Paso 'Creando .env a partir de .env.example...'
  Copy-Item (Join-Path $dir '.env.example') (Join-Path $dir '.env')
} else {
  Paso '.env ya existe (no se modifica).'
}

# 5. Descarga anticipada del modelo (queda en la caché de Hugging Face)
$modelo = 'wambugu71/crop_leaf_diseases_vit'
$revision = ''
foreach ($l in Get-Content (Join-Path $dir '.env') -Encoding UTF8) {
  if ($l -match '^\s*MODELO_ID\s*=\s*(.+)$') { $modelo = $Matches[1].Trim() }
  if ($l -match '^\s*MODELO_REVISION\s*=\s*(.+)$') { $revision = $Matches[1].Trim() }
}
Paso "Descargando el modelo $modelo (~350 MB, solo la primera vez)..."
$env:SIGAGRO_MODELO = $modelo
$env:SIGAGRO_REVISION = $revision
& $python -c "import os; from transformers import AutoImageProcessor as P, AutoModelForImageClassification as M; i=os.environ['SIGAGRO_MODELO']; r=os.environ['SIGAGRO_REVISION'] or None; P.from_pretrained(i, revision=r); m=M.from_pretrained(i, revision=r); print('Modelo listo:', len(m.config.id2label), 'clases')"
if ($LASTEXITCODE -ne 0) { throw 'No se pudo descargar el modelo.' }

# 6. Tests (opcional)
if ($Dev) {
  Paso 'Corriendo los tests...'
  & $python -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw 'Fallaron los tests.' }
}

Write-Host "`n[IA] Instalación completa." -ForegroundColor Green
Write-Host '  Iniciar el servicio:  .\iniciar.ps1   (o .\start-dev.ps1 desde la carpeta raíz)' -ForegroundColor Green
Write-Host '  Verificar:            http://127.0.0.1:8000/salud  ->  "estado": "OK"' -ForegroundColor Green
Write-Host '  En SIGAgro-BackEnd\.env:  AI_SERVICE_URL=http://127.0.0.1:8000' -ForegroundColor Green
