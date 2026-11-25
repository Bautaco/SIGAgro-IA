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

## Ampliación: más cultivos y enfermedades presentes en Argentina (`ampliar_datos.py`)

Se suman fotos de maíz, trigo, sorgo, papa, poroto, algodón y caña de azúcar, y más fotos de clases débiles de soja
y maní. Solo se usan clases de problemas presentes en Argentina; se descartan las demás (p. ej. MSV y MLN del maíz,
roseta del maní, virus del enrollamiento del algodón, langosta africana) y las clases ambiguas o demasiado generales.
Tope de 900 fotos por clase y 500 por fuente, para que ninguna fuente domine.

| Cultivo | Conjunto | Clases usadas | Licencia |
|---|---|---|---|
| Maíz, papa, soja | [Project-AgML/plant_village_classification](https://huggingface.co/datasets/Project-AgML/plant_village_classification) — PlantVillage (Hughes y Salathé, 2015, arXiv:1511.08060); copia pública en Mendeley Data por Pandian y Gopal, DOI [10.17632/tywbtsjrjv.1](https://doi.org/10.17632/tywbtsjrjv.1) | Mancha gris, roya común, tizón foliar y sano (maíz); tizón temprano, tizón tardío y sana (papa); soja sana | CC0 1.0 (copia de Mendeley; la ficha de Hugging Face no declara licencia) |
| Maíz, papa, soja | [Project-AgML/plant_doc_classification](https://huggingface.co/datasets/Project-AgML/plant_doc_classification) — PlantDoc (Singh et al., 2020), **fotos a campo** | Mancha gris, tizón foliar, roya (maíz); tizón temprano y tardío (papa); soja sana | CC BY-SA 4.0 |
| Maíz | [Project-AgML/corn_leaf_pest_classification](https://huggingface.co/datasets/Project-AgML/corn_leaf_pest_classification) | Tizón foliar (Helminthosporium), gusano cogollero (*Spodoptera frugiperda*), sano | CC BY 4.0 |
| Trigo | Wheat Leaf Dataset — Getachew (2021), Mendeley Data DOI [10.17632/wgd66f8n6h.1](https://doi.org/10.17632/wgd66f8n6h.1) (Etiopía, a campo) | Roya amarilla, septoriosis, sano | CC BY 4.0 |
| Trigo | Disease Dataset of Wheat (2025), Mendeley Data DOI [10.17632/5gc7hwydwg.1](https://doi.org/10.17632/5gc7hwydwg.1) (Bangladesh; solo las fotos originales, no las aumentadas) | Mancha borrosa (leaf blight), brusone, sano | CC BY 4.0 |
| Sorgo | Sorghum Disease Image Dataset (2024), Mendeley Data DOI [10.17632/fgb3cmfrg2.1](https://doi.org/10.17632/fgb3cmfrg2.1) | Antracnosis, roya, carbón de la panoja, moho del grano (no trae plantas sanas) | CC BY 4.0 |
| Soja | [Project-AgML/soybean_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/soybean_leaf_disease_classification) | Tizón bacteriano, mancha marrón (septoria), sana | CC BY 4.0 |
| Maní | [Project-AgML/groundnut_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/groundnut_leaf_disease_classification) | Mancha por Alternaria, roya, sano | CC BY 4.0 |
| Papa | [Project-AgML/potato_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/potato_leaf_disease_classification) | Tizón tardío (Phytophthora), virosis, sana | CC BY 4.0 |
| Poroto | [Project-AgML/bean_disease_uganda](https://huggingface.co/datasets/Project-AgML/bean_disease_uganda) (iBean, Makerere AI Lab) | Mancha angular, roya, sano | CC BY 4.0 |
| Poroto | [Project-AgML/bean_disease_classification_tanzania](https://huggingface.co/datasets/Project-AgML/bean_disease_classification_tanzania) | Antracnosis, roya, sano | CC BY 4.0 |
| Algodón | [Project-AgML/cotton_leaf_disease_classification_bangladesh_2](https://huggingface.co/datasets/Project-AgML/cotton_leaf_disease_classification_bangladesh_2) (solo `raw/`, no las copias aumentadas) | Bacteriosis, daño por herbicida, sano | CC BY 4.0 |
| Caña de azúcar | [Project-AgML/sugarcane_leaf_disease_classification](https://huggingface.co/datasets/Project-AgML/sugarcane_leaf_disease_classification) | Roya marrón, carbón, hoja amarilla, pokkah boeng, mancha parda, sana | CC BY 4.0 |

### Segunda ampliación (prioridad: maíz, trigo, sorgo, maní y soja)

| Cultivo | Conjunto | Clases usadas | Licencia |
|---|---|---|---|
| Trigo | Wheat nitrogen deficiency and leaf rust image dataset — Arya y Singh (2020), Mendeley Data DOI [10.17632/th422bg4yd.1](https://doi.org/10.17632/th422bg4yd.1) (solo `WheatLeafRust`) | **Roya de la hoja** (*Puccinia triticina*, la principal enfermedad del trigo en Argentina), sano | CC BY 4.0 |
| Maní | Dataset of groundnut plant leaf images — Aishwarya y Padmanabha Reddy (2023), Mendeley Data DOI [10.17632/22p2vcbxfk.3](https://doi.org/10.17632/22p2vcbxfk.3) (conjunto completo; la copia de Hugging Face es una parte) | Viruela temprana y tardía, roya, deficiencia nutricional, sano (se descarta «early rust»: síntomas mínimos) | CC BY 4.0 |
| Maíz | [Project-AgML/crop_pest_disease_classification](https://huggingface.co/datasets/Project-AgML/crop_pest_disease_classification) — CCMT (Ghana, fotos a campo; solo `raw/`) | Gusano cogollero (*Spodoptera frugiperda*); las demás etiquetas no indican el cultivo y se descartan | CC BY 4.0 |

Las fotos repetidas (misma foto en dos fuentes o copias del mismo conjunto) se descartan por huella visual (dHash)
dentro de cada etiqueta. Los conjuntos que traen copias aumentadas se leen solo desde sus fotos originales: si no,
variantes de una misma hoja quedarían a la vez en entrenamiento y en prueba, inflando las métricas.

**Brechas para Argentina (sin datos públicos con licencia abierta, quedan para el análisis en línea o el Ingeniero):**
maíz — chicharrita / achaparramiento (*Dalbulus maidis*, Spiroplasma), mancha blanca (*Phaeosphaeria*), barrenador
(*Diatraea*); trigo — mancha amarilla (*Drechslera tritici-repentis*), fusariosis de la espiga; sorgo — plantas
sanas (no hay clase «sano»), tizón (*Exserohilum*); maní — carbón (*Thecaphora frezii*, afecta la caja); soja —
tizón de la hoja / mancha púrpura (*Cercospora kikuchii*), mancha anillada; girasol — Verticillium, Sclerotinia,
Phomopsis. Las fotos propias validadas por un Ingeniero (`importar_validadas.py`) son la forma de cubrirlas.

Notas: PlantVillage son fotos de laboratorio con fondo uniforme; por eso se combinan con PlantDoc y otras fotos a
campo. El sorgo no tiene clase «sano» en los datos: una foto de sorgo sano puede clasificarse como enfermo con baja
confianza (el umbral del servicio la marca como inconclusa).

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
