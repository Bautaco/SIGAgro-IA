# ============================================================
# SIGAgro IA - Inicio del servicio de diagnóstico por imagen
# Carga las variables de .env (si existe; si no, de .env.example) y levanta uvicorn.
#   .\iniciar.ps1                       -> http://127.0.0.1:8000
#   .\iniciar.ps1 -Puerto 8000 -Direccion 0.0.0.0   (solo si el backend está en OTRA máquina)
# ============================================================
param(
  [string]$Direccion = '127.0.0.1',
  [int]$Puerto = 8000
)

$dir = $PSScriptRoot
$python = Join-Path $dir '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
  Write-Host '[IA] No existe el entorno virtual (.venv). Instalalo con:' -ForegroundColor Red
  Write-Host '  py -3.13 -m venv .venv' -ForegroundColor Yellow
  Write-Host '  .\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu' -ForegroundColor Yellow
  exit 1
}

$archivoEnv = Join-Path $dir '.env'
if (-not (Test-Path $archivoEnv)) { $archivoEnv = Join-Path $dir '.env.example' }

foreach ($linea in Get-Content $archivoEnv -Encoding UTF8) {
  $l = $linea.Trim()
  if ($l -eq '' -or $l.StartsWith('#') -or -not $l.Contains('=')) { continue }
  $i = $l.IndexOf('=')
  $nombre = $l.Substring(0, $i).Trim()
  $valor = $l.Substring($i + 1).Trim().Trim('"').Trim("'")
  # Una variable ya definida en el entorno tiene prioridad sobre el archivo
  if (-not [Environment]::GetEnvironmentVariable($nombre)) {
    [Environment]::SetEnvironmentVariable($nombre, $valor, 'Process')
  }
}

Write-Host "[IA] Configuración: $archivoEnv" -ForegroundColor DarkGray
Write-Host "[IA] Modelo: $env:MODELO_ID @ $env:MODELO_REVISION" -ForegroundColor DarkGray
Write-Host "[IA] Iniciando en http://${Direccion}:$Puerto (la primera vez descarga el modelo)..." -ForegroundColor Magenta

Set-Location $dir
& $python -m uvicorn app.main:app --host $Direccion --port $Puerto
