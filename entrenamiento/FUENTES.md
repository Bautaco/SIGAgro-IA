# Fuentes de datos y modelo base (atribución)

El modelo `sigagro-extensivos` (soja, girasol, maní) se entrena con conjuntos públicos de imágenes publicados en
Hugging Face por **Project-AgML**, todos bajo licencia **CC BY 4.0** (uso permitido con atribución). Las clases
se unifican y renombran en `preparar_datos.py`; la clase `Leaf_scars` del girasol se descarta por ambigua.

| Cultivo | Conjunto (Hugging Face) | Origen citado | Licencia |
|---|---|---|---|
| Soja | [Project-AgML/MH_SoyaHealthVision_disease_classification_leaf](https://huggingface.co/datasets/Project-AgML/MH_SoyaHealthVision_disease_classification_leaf) | MH-SoyaHealthVision, Mendeley Data, DOI [10.17632/hkbgh5s3b7.1](https://data.mendeley.com/datasets/hkbgh5s3b7/1) — campos de soja de Maharashtra, India | CC BY 4.0 |
| Soja | [Project-AgML/soybean_leaf_disease_classification_brazil](https://huggingface.co/datasets/Project-AgML/soybean_leaf_disease_classification_brazil) | Brasil (ver la ficha del conjunto) | CC BY 4.0 |
| Girasol | [Project-AgML/sunflower_disease_classification](https://huggingface.co/datasets/Project-AgML/sunflower_disease_classification) | Mendeley Data, DOI [10.17632/b83hmrzth8.1](https://doi.org/10.17632/b83hmrzth8.1) | CC BY 4.0 |
| Maní | [Project-AgML/groundnut_leaf_disease_classification_2](https://huggingface.co/datasets/Project-AgML/groundnut_leaf_disease_classification_2) | Aishwarya & Padmanabha Reddy (2023), *Data in Brief*; Mendeley Data, DOI [10.17632/22p2vcbxfk.3](https://doi.org/10.17632/22p2vcbxfk.3) | CC BY 4.0 |

**Modelo base:** [`wambugu71/crop_leaf_diseases_vit`](https://huggingface.co/wambugu71/crop_leaf_diseases_vit)
(ViT tiny, licencia MIT), revisión `7d5b32bcd6f83a2f57e7e0346358fad276296877`.

## Limitaciones

- Las fotos son de India, Brasil y otras regiones, no de lotes argentinos: la precisión real a campo puede ser menor
  que la medida en el conjunto de prueba (ver `modelos/sigagro-extensivos/METRICAS.md`).
- Cada cultivo proviene de una fuente distinta: el modelo podría apoyarse en el fondo o el tipo de foto. Por eso el
  servicio usa este modelo solo para el cultivo sembrado en el lote y marca como INCONCLUSO si reconoce otro cultivo.
- Enfermedades importantes en Argentina que **no** están en los datos (p. ej. en girasol Verticillium, Sclerotinia,
  Phomopsis; en soja tizón de la hoja por Cercospora kikuchii) no pueden ser reconocidas por este modelo: para esos
  casos queda el análisis en línea o la revisión del Ingeniero Agrónomo.

## Agregar fotos propias (recomendado)

1. Crear o completar carpetas `entrenamiento/datos/<Etiqueta>/` (p. ej. `Soybean___Frog_Eye_Leaf_Spot/`) con fotos
   de lotes propios, identificadas por un Ingeniero Agrónomo. Para una clase nueva, agregar además su entrada en
   `app/conocimiento.py` (sin productos ni dosis).
2. Reentrenar: `.venv\Scripts\python.exe -m entrenamiento.entrenar`
3. Revisar `modelos/sigagro-extensivos/METRICAS.md` y reiniciar el servicio.
