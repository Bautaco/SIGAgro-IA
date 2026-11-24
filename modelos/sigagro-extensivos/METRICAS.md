# Modelo `sigagro-extensivos` — métricas

- Versión: `local-20261124-2011` · base: `wambugu71/crop_leaf_diseases_vit` (MIT) · técnica: extractor fijo + capa de clasificación entrenada
- Datos: 10863 fotos de 4 conjuntos públicos CC BY 4.0 (ver `entrenamiento/FUENTES.md`) · semilla 42
- División: 8690 entrenamiento (2 variantes por foto) · 1086 validación · 1087 prueba

**Acierto en el conjunto de prueba (1087 fotos no vistas): 78.7%** · F1 macro: 76.6%

| Etiqueta | Precisión | Sensibilidad | F1 | Fotos de prueba |
|---|---|---|---|---|
| Groundnut___Early_Leaf_Spot | 76.2% | 71.9% | 74.0% | 89 |
| Groundnut___Healthy | 83.7% | 82.8% | 83.2% | 93 |
| Groundnut___Late_Leaf_Spot | 91.3% | 91.3% | 91.3% | 69 |
| Groundnut___Nutrient_Deficiency | 67.5% | 81.8% | 74.0% | 33 |
| Groundnut___Rust | 70.4% | 86.4% | 77.6% | 22 |
| Soybean___Defoliating_Caterpillars | 78.7% | 56.7% | 65.9% | 150 |
| Soybean___Diabrotica_speciosa | 85.6% | 83.3% | 84.5% | 150 |
| Soybean___Frog_Eye_Leaf_Spot | 23.5% | 70.6% | 35.3% | 17 |
| Soybean___Healthy | 73.5% | 88.2% | 80.2% | 110 |
| Soybean___Mosaic | 78.4% | 81.7% | 80.0% | 71 |
| Soybean___Rust | 81.5% | 62.4% | 70.7% | 85 |
| Soybean___Septoria_Brown_Spot | 39.4% | 48.1% | 43.3% | 27 |
| Sunflower___Downy_Mildew | 96.4% | 91.5% | 93.9% | 59 |
| Sunflower___Gray_Mold | 100.0% | 100.0% | 100.0% | 47 |
| Sunflower___Healthy | 96.8% | 93.8% | 95.3% | 65 |

**Precisión:** de las fotos que el modelo asignó a la clase, cuántas eran correctas. **Sensibilidad:** de las fotos de la clase, cuántas reconoció.

Las fotos provienen de conjuntos públicos de otros países: la precisión con fotos de lotes argentinos puede ser
menor. Conviene validarlo con fotos propias y reentrenar agregándolas a `entrenamiento/datos/`.
