# Modelo `sigagro-extensivos` — métricas

- Versión: `local-20261125-1637` · base: `wambugu71/crop_leaf_diseases_vit` (MIT) · técnica: extractor fijo + capa de clasificación entrenada
- Datos: 28966 fotos de conjuntos públicos con licencia abierta (ver `entrenamiento/FUENTES.md`) · semilla 42
- División: 23172 entrenamiento (2 variantes por foto) · 2897 validación · 2897 prueba

**Acierto en el conjunto de prueba (2897 fotos no vistas): 87.7%** · F1 macro: 89.0%

| Etiqueta | Precisión | Sensibilidad | F1 | Fotos de prueba |
|---|---|---|---|---|
| Bean___Angular_Leaf_Spot | 88.6% | 90.7% | 89.7% | 43 |
| Bean___Anthracnose | 96.2% | 100.0% | 98.0% | 50 |
| Bean___Healthy | 95.5% | 94.4% | 95.0% | 90 |
| Bean___Rust | 95.6% | 95.6% | 95.6% | 90 |
| Corn___Common_Rust | 96.7% | 95.1% | 95.9% | 61 |
| Corn___Fall_Armyworm | 97.5% | 98.7% | 98.1% | 78 |
| Corn___Gray_Leaf_Spot | 73.2% | 91.2% | 81.2% | 57 |
| Corn___Healthy | 98.9% | 100.0% | 99.4% | 90 |
| Corn___Northern_Leaf_Blight | 90.4% | 73.3% | 81.0% | 90 |
| Cotton___Bacterial_Blight | 96.2% | 100.0% | 98.0% | 25 |
| Cotton___Healthy | 92.6% | 100.0% | 96.2% | 25 |
| Cotton___Herbicide_Damage | 100.0% | 96.4% | 98.2% | 28 |
| Groundnut___Alternaria_Leaf_Spot | 91.7% | 97.8% | 94.6% | 45 |
| Groundnut___Early_Leaf_Spot | 72.1% | 68.9% | 70.5% | 90 |
| Groundnut___Healthy | 77.6% | 81.7% | 79.6% | 93 |
| Groundnut___Late_Leaf_Spot | 97.7% | 94.4% | 96.0% | 90 |
| Groundnut___Nutrient_Deficiency | 96.4% | 96.4% | 96.4% | 83 |
| Groundnut___Rust | 96.2% | 90.6% | 93.3% | 85 |
| Potato___Early_Blight | 91.8% | 90.3% | 91.1% | 62 |
| Potato___Healthy | 85.3% | 82.9% | 84.1% | 35 |
| Potato___Late_Blight | 92.9% | 86.7% | 89.7% | 90 |
| Potato___Virus | 80.8% | 84.0% | 82.4% | 50 |
| Sorghum___Anthracnose | 98.0% | 100.0% | 99.0% | 50 |
| Sorghum___Grain_Mold | 100.0% | 86.0% | 92.5% | 50 |
| Sorghum___Head_Smut | 86.8% | 97.9% | 92.0% | 47 |
| Sorghum___Rust | 100.0% | 98.0% | 99.0% | 49 |
| Soybean___Bacterial_Blight | 95.8% | 100.0% | 97.9% | 23 |
| Soybean___Defoliating_Caterpillars | 70.6% | 51.3% | 59.5% | 150 |
| Soybean___Diabrotica_speciosa | 88.5% | 87.3% | 87.9% | 150 |
| Soybean___Frog_Eye_Leaf_Spot | 27.3% | 70.6% | 39.3% | 17 |
| Soybean___Healthy | 77.0% | 88.2% | 82.2% | 110 |
| Soybean___Mosaic | 73.1% | 80.3% | 76.5% | 71 |
| Soybean___Rust | 72.6% | 70.9% | 71.8% | 86 |
| Soybean___Septoria_Brown_Spot | 71.4% | 63.6% | 67.3% | 55 |
| Sugarcane___Brown_Rust | 86.1% | 100.0% | 92.5% | 31 |
| Sugarcane___Brown_Spot | 94.9% | 74.0% | 83.1% | 50 |
| Sugarcane___Healthy | 93.3% | 97.7% | 95.5% | 43 |
| Sugarcane___Pokkah_Boeng | 73.3% | 73.3% | 73.3% | 30 |
| Sugarcane___Smut | 73.7% | 90.3% | 81.2% | 31 |
| Sugarcane___Yellow_Leaf | 87.0% | 94.0% | 90.4% | 50 |
| Sunflower___Downy_Mildew | 95.2% | 100.0% | 97.5% | 59 |
| Sunflower___Gray_Mold | 100.0% | 97.9% | 98.9% | 47 |
| Sunflower___Healthy | 100.0% | 96.9% | 98.4% | 65 |
| Wheat___Blast | 100.0% | 100.0% | 100.0% | 29 |
| Wheat___Brown_Rust | 91.7% | 95.7% | 93.6% | 23 |
| Wheat___Healthy | 95.4% | 95.4% | 95.4% | 65 |
| Wheat___Leaf_Blight | 100.0% | 100.0% | 100.0% | 35 |
| Wheat___Septoria | 100.0% | 100.0% | 100.0% | 10 |
| Wheat___Yellow_Rust | 90.9% | 95.2% | 93.0% | 21 |

**Precisión:** de las fotos que el modelo asignó a la clase, cuántas eran correctas. **Sensibilidad:** de las fotos de la clase, cuántas reconoció.

Las fotos provienen de conjuntos públicos de otros países: la precisión con fotos de lotes argentinos puede ser
menor. Conviene validarlo con fotos propias y reentrenar agregándolas a `entrenamiento/datos/`.
