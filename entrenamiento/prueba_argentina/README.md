# Prueba argentina de la IA

Fotos de **lotes argentinos**, con la afección **confirmada por un Ingeniero Agrónomo**, que se usan solo para
medir la IA. Ningún script de entrenamiento lee esta carpeta: el modelo nunca las ve.

## Cómo se llena

1. **Desde SIGAgro (recomendado):** `importar_validadas.py` reserva automáticamente el 25 % de los diagnósticos
   validados (elegidos por id, siempre los mismos) y los guarda acá:

   ```powershell
   $env:SIGAGRO_USUARIO = "usuario-administrador"
   .venv\Scripts\python.exe -m entrenamiento.importar_validadas --fraccion-prueba 0.25
   ```

2. **A mano:** una subcarpeta por afección con el **mismo nombre de etiqueta** que en `entrenamiento/datos/`
   (por ejemplo `Corn___Common_Rust/`, `Wheat___Brown_Rust/`, `Soybean___Healthy/`) y adentro las fotos `.jpg`.
   La foto debe ser de una hoja con síntomas (o sana), tomada a campo con celular, de un lote argentino.

## Cuántas fotos

Para que los porcentajes digan algo: **al menos 20–30 fotos por afección prioritaria**, de varios lotes y días
distintos. Prioridad: maíz, trigo, sorgo, maní y soja.

## Cómo se mide

```powershell
.venv\Scripts\python.exe -m entrenamiento.evaluar_argentina
```

Pasa cada foto por el mismo análisis que el servicio y escribe `modelos/sigagro-extensivos/METRICAS_ARGENTINA.md`
(por cultivo y por afección: cuántas reciben diagnóstico y con qué acierto, y las confusiones más frecuentes).
Repetirlo después de cada reentrenamiento para ver si la IA mejora con fotos reales.

Las fotos son de productores: no se suben al repositorio (ver `.gitignore`).
