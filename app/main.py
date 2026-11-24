"""
API del servicio de IA de SIGAgro (FastAPI).

  GET  /salud       estado del servicio y del modelo (sin autenticación)
  GET  /modelo      modelo, versión, clases y cultivos que cubre
  POST /inferencia  análisis de una fotografía (contrato documentado en README.md)

Ejecutar:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .analisis import VERSION_CONTRATO, SolicitudDemasiadoGrande, analizar, cultivos_cubiertos
from .config import Config, cargar_config
from .modelo import GestorModelo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sigagro-ia")


class SolicitudInferencia(BaseModel):
    imagen: str = Field(..., min_length=1, description="Data URL base64 (data:image/jpeg;base64,...) o base64 plano")
    formato: str | None = Field(None, max_length=20)
    cultivo: str | None = Field(None, max_length=100, description="Cultivo del lote indicado por el usuario")
    tipoMuestra: Literal["PLANTA", "CULTIVO", "SUELO"] | None = None
    origen: Literal["MUESTRA_MONITOREO", "EVIDENCIA_LABOR"] = "MUESTRA_MONITOREO"


def crear_app(config: Config | None = None, gestor: GestorModelo | None = None, cargar_modelo: bool = True) -> FastAPI:
    config = config or cargar_config()
    gestor = gestor or GestorModelo()

    @asynccontextmanager
    async def ciclo_de_vida(_: FastAPI):
        if cargar_modelo and gestor.clasificador is None:
            await run_in_threadpool(gestor.cargar, config.modelo_id, config.modelo_revision)
        if not config.token_servicio:
            log.warning("TOKEN_SERVICIO no configurado: la API acepta pedidos sin autenticación (solo para desarrollo).")
        yield

    app = FastAPI(title="SIGAgro IA - Diagnóstico fitosanitario", version="1.0.0", lifespan=ciclo_de_vida)
    app.state.config = config
    app.state.gestor = gestor

    def autenticar(authorization: str | None = Header(default=None)) -> None:
        if not config.token_servicio:
            return
        esperado = f"Bearer {config.token_servicio}"
        if not authorization or not hmac.compare_digest(authorization.encode(), esperado.encode()):
            raise HTTPException(status_code=401, detail="Token de servicio inválido o ausente.")

    def modelo_listo():
        if gestor.clasificador is None:
            raise HTTPException(status_code=503, detail=gestor.error or "El modelo todavía se está cargando.")
        return gestor.clasificador

    @app.get("/salud")
    def salud():
        return {
            "estado": "OK" if gestor.clasificador else "MODELO_NO_DISPONIBLE",
            "modelo": config.modelo_id,
            "error": gestor.error,
            "versionContrato": VERSION_CONTRATO,
        }

    @app.get("/modelo", dependencies=[Depends(autenticar)])
    def modelo():
        c = modelo_listo()
        return {
            "id": c.id,
            "version": c.version,
            "etiquetas": c.etiquetas,
            "cultivosCubiertos": cultivos_cubiertos(c),
            "umbralConfianza": config.umbral_confianza,
            "margenMinimo": config.margen_minimo,
            "estimaGravedad": False,
            "versionContrato": VERSION_CONTRATO,
        }

    @app.post("/inferencia", dependencies=[Depends(autenticar)])
    async def inferencia(solicitud: SolicitudInferencia):
        c = modelo_listo()
        inicio = time.perf_counter()
        try:
            resultado = await run_in_threadpool(
                analizar, solicitud.imagen, c, config, cultivo=solicitud.cultivo, tipo_muestra=solicitud.tipoMuestra
            )
        except SolicitudDemasiadoGrande as e:
            raise HTTPException(status_code=413, detail=str(e)) from e
        resultado["tiempoProcesamientoMs"] = round((time.perf_counter() - inicio) * 1000)
        log.info(
            "Inferencia %s: estado=%s etiqueta=%s confianza=%s (%d ms)",
            solicitud.origen,
            resultado["estado"],
            resultado.get("etiquetaModelo"),
            resultado.get("confianza"),
            resultado["tiempoProcesamientoMs"],
        )
        return resultado

    return app


app = crear_app()
