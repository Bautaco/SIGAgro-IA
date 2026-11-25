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
from .conocimiento import info_de
from .config import Config, cargar_config
from .en_linea import CULTIVOS_ARGENTINA, AnalizadorEnLinea, crear_analizador
from .modelo import GestorModelo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sigagro-ia")


class SolicitudInferencia(BaseModel):
    imagen: str = Field(..., min_length=1, description="Data URL base64 (data:image/jpeg;base64,...) o base64 plano")
    formato: str | None = Field(None, max_length=20)
    cultivo: str | None = Field(None, max_length=100, description="Cultivo del lote indicado por el usuario")
    tipoMuestra: Literal["PLANTA", "CULTIVO", "SUELO"] | None = None
    origen: Literal["MUESTRA_MONITOREO", "EVIDENCIA_LABOR"] = "MUESTRA_MONITOREO"
    observaciones: str | None = Field(None, max_length=2000, description="Observaciones de campo (contexto para el análisis en línea)")


_SIN_DEFINIR = object()


def crear_app(
    config: Config | None = None,
    gestor: GestorModelo | None = None,
    cargar_modelo: bool = True,
    en_linea: AnalizadorEnLinea | None | object = _SIN_DEFINIR,
) -> FastAPI:
    config = config or cargar_config()
    gestor = gestor or GestorModelo()
    if en_linea is _SIN_DEFINIR:
        en_linea = crear_analizador(
            config.en_linea_habilitado, config.modelo_en_linea, config.timeout_en_linea, config.esfuerzo_en_linea
        )
    if en_linea:
        log.info("Análisis en línea habilitado con %s (cultivos no cubiertos y resultados no concluyentes).", en_linea.id)
    else:
        log.info("Análisis en línea deshabilitado (sin ANTHROPIC_API_KEY o EN_LINEA=no): solo modelo local.")

    @asynccontextmanager
    async def ciclo_de_vida(_: FastAPI):
        if cargar_modelo and gestor.clasificador is None:
            await run_in_threadpool(gestor.cargar, config.modelo_id, config.modelo_revision)
            for ruta in config.modelos_adicionales:
                await run_in_threadpool(gestor.cargar_adicional, ruta)
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
            "enLinea": {"habilitado": bool(en_linea), "modelo": en_linea.id if en_linea else None},
            "modelosAdicionales": [{"id": m.id, "version": m.version, "cultivos": cultivos_cubiertos(m)} for m in gestor.adicionales],
            "erroresModelosAdicionales": gestor.errores_adicionales,
        }

    @app.get("/modelo", dependencies=[Depends(autenticar)])
    def modelo():
        c = modelo_listo()
        locales = sorted({x for m in [c, *gestor.adicionales] for x in cultivos_cubiertos(m)})
        return {
            "id": c.id,
            "version": c.version,
            "etiquetas": c.etiquetas,
            # Con el análisis en línea se cubren también los demás cultivos de Argentina
            "cultivosCubiertos": sorted(set(locales) | (set(CULTIVOS_ARGENTINA) if en_linea else set())),
            "cultivosModeloLocal": locales,
            "modelosAdicionales": [
                {"id": m.id, "version": m.version, "etiquetas": m.etiquetas, "cultivos": cultivos_cubiertos(m)} for m in gestor.adicionales
            ],
            "analisisEnLinea": {"habilitado": bool(en_linea), "modelo": en_linea.id if en_linea else None},
            "umbralConfianza": config.umbral_confianza,
            "margenMinimo": config.margen_minimo,
            "estimaGravedad": False,
            # Clases con ficha (nombre en español): el Ingeniero elige de esta lista al rectificar un diagnóstico,
            # así la foto validada queda etiquetada para reentrenar
            "clases": [
                {"etiqueta": e, "nombre": i.nombre, "cultivo": i.cultivo, "tipo": i.tipo}
                for e in sorted({e for m in [c, *gestor.adicionales] for e in m.etiquetas})
                if (i := info_de(e)) and i.cultivo
            ],
            "versionContrato": VERSION_CONTRATO,
        }

    @app.post("/inferencia", dependencies=[Depends(autenticar)])
    async def inferencia(solicitud: SolicitudInferencia):
        c = modelo_listo()
        inicio = time.perf_counter()
        try:
            resultado = await run_in_threadpool(
                analizar,
                solicitud.imagen,
                c,
                config,
                cultivo=solicitud.cultivo,
                tipo_muestra=solicitud.tipoMuestra,
                en_linea=en_linea,
                observaciones=solicitud.observaciones,
                adicionales=gestor.adicionales,
            )
        except SolicitudDemasiadoGrande as e:
            raise HTTPException(status_code=413, detail=str(e)) from e
        resultado["tiempoProcesamientoMs"] = round((time.perf_counter() - inicio) * 1000)
        log.info(
            "Inferencia %s [%s]: estado=%s etiqueta=%s confianza=%s (%d ms)",
            solicitud.origen,
            resultado.get("origenAnalisis", "LOCAL"),
            resultado["estado"],
            resultado.get("etiquetaModelo"),
            resultado.get("confianza"),
            resultado["tiempoProcesamientoMs"],
        )
        return resultado

    return app


app = crear_app()
