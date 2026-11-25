"""
Información de referencia de cada clase del modelo, en español.

Reglas de seguridad agronómica:
- Solo describe el problema (agente causal, síntomas, condiciones, consecuencias) y alternativas de control
  GENERALES. No nombra productos comerciales, mezclas ni dosis: los productos y dosis salen del catálogo de
  SIGAgro cargado desde etiquetas oficiales y los decide el Ingeniero Agrónomo.
- Es información técnica general de referencia: no reemplaza el diagnóstico profesional.

Para cambiar de modelo (MODELO_ID) hay que agregar aquí sus etiquetas; las desconocidas se informan con el
nombre crudo y sin información de referencia (nunca se inventa).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FUENTE = "Referencia técnica general incluida en el servicio de IA de SIGAgro; debe validarla el Ingeniero Agrónomo."

CONTROL_FUNGICO_GENERAL = [
    "Monitorear el lote y estimar incidencia y severidad antes de decidir un tratamiento.",
    "Usar cultivares o híbridos con mejor comportamiento sanitario en próximas siembras.",
    "Rotación de cultivos y manejo del rastrojo para reducir el inóculo.",
    "Si se supera el umbral de daño económico, evaluar un fungicida registrado para el cultivo y la enfermedad, según la etiqueta y la receta del Ingeniero Agrónomo.",
]

CONTROL_PLAGAS_GENERAL = [
    "Monitorear el lote (paño vertical, conteo por metro o porcentaje de defoliación) para estimar la densidad de la plaga.",
    "Comparar con el umbral de daño económico para el estado fenológico del cultivo antes de decidir un control.",
    "Preservar los enemigos naturales y evaluar alternativas de control biológico.",
    "Si se supera el umbral, evaluar un insecticida registrado para el cultivo y la plaga, según la etiqueta y la receta del Ingeniero Agrónomo.",
]

CONTROL_VIRUS_GENERAL = [
    "Usar semilla sana de origen conocido y cultivares con mejor comportamiento.",
    "Monitorear los insectos vectores (pulgones) y controlar malezas hospederas.",
    "No existe control curativo para las virosis: el manejo es preventivo.",
]


@dataclass(frozen=True)
class InfoClase:
    cultivo: str | None  # None = la clase no corresponde a un cultivo (p. ej. imagen inválida)
    tipo: str  # ENFERMEDAD | PLAGA | MALEZA | DEFICIENCIA | SANO | INVALIDA
    nombre: str
    agente: str | None = None
    descripcion: str | None = None
    sintomas: list[str] = field(default_factory=list)
    condiciones: list[str] = field(default_factory=list)
    consecuencias: list[str] = field(default_factory=list)
    control: list[str] = field(default_factory=list)


# Cultivos (texto libre del lote, en minúsculas y sin tildes) -> cultivo del modelo
SINONIMOS_CULTIVO = {
    "maiz": "Maíz",
    "maíz": "Maíz",
    "corn": "Maíz",
    "trigo": "Trigo",
    "wheat": "Trigo",
    "papa": "Papa",
    "potato": "Papa",
    "arroz": "Arroz",
    "rice": "Arroz",
    "soja": "Soja",
    "soya": "Soja",
    "soybean": "Soja",
    "girasol": "Girasol",
    "sunflower": "Girasol",
    "mani": "Maní",
    "peanut": "Maní",
    "groundnut": "Maní",
    "sorgo": "Sorgo",
    "sorghum": "Sorgo",
    "poroto": "Poroto",
    "porotos": "Poroto",
    "frijol": "Poroto",
    "bean": "Poroto",
    "algodon": "Algodón",
    "cotton": "Algodón",
    "caña de azucar": "Caña de azúcar",
    "cana de azucar": "Caña de azúcar",
    "caña": "Caña de azúcar",
    "cana": "Caña de azúcar",
    "sugarcane": "Caña de azúcar",
}

CLASES: dict[str, InfoClase] = {
    "Corn___Common_Rust": InfoClase(
        cultivo="Maíz",
        tipo="ENFERMEDAD",
        nombre="Roya común del maíz",
        agente="Puccinia sorghi (hongo)",
        descripcion="Enfermedad foliar causada por un hongo que forma pústulas sobre las hojas y reduce el área fotosintética.",
        sintomas=[
            "Pústulas pequeñas, ovaladas o alargadas, de color pardo-rojizo (canela).",
            "Pústulas en ambas caras de la hoja que liberan un polvo color óxido al frotarlas.",
            "En ataques severos, clorosis y secado de las hojas.",
        ],
        condiciones=[
            "Temperaturas moderadas y alta humedad relativa.",
            "Períodos prolongados de rocío o mojado foliar.",
            "Híbridos susceptibles.",
        ],
        consecuencias=[
            "Pérdida de área foliar verde y menor llenado de granos.",
            "Pérdidas de rendimiento mayores cuando el ataque es temprano y severo.",
        ],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Corn___Gray_Leaf_Spot": InfoClase(
        cultivo="Maíz",
        tipo="ENFERMEDAD",
        nombre="Mancha gris del maíz",
        agente="Cercospora zeae-maydis (hongo)",
        descripcion="Enfermedad foliar favorecida por el rastrojo de maíz en superficie y la alta humedad.",
        sintomas=[
            "Lesiones rectangulares y alargadas, de color gris a castaño.",
            "Lesiones delimitadas por las nervaduras de la hoja.",
            "Las lesiones pueden unirse y secar grandes áreas de la hoja.",
        ],
        condiciones=[
            "Alta humedad relativa y rocío prolongado.",
            "Temperaturas cálidas.",
            "Rastrojo de maíz en superficie (siembra directa, maíz sobre maíz).",
        ],
        consecuencias=[
            "Necrosis foliar y menor llenado de granos.",
            "Mayor riesgo de quebrado de tallos.",
        ],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Corn___Healthy": InfoClase(cultivo="Maíz", tipo="SANO", nombre="Maíz sin síntomas visibles"),
    "Wheat___Brown_Rust": InfoClase(
        cultivo="Trigo",
        tipo="ENFERMEDAD",
        nombre="Roya anaranjada (roya de la hoja) del trigo",
        agente="Puccinia triticina (hongo)",
        descripcion="Enfermedad foliar del trigo que forma pústulas anaranjadas y reduce el área foliar sana.",
        sintomas=[
            "Pústulas pequeñas y redondeadas, color anaranjado a marrón.",
            "Pústulas distribuidas al azar, principalmente en la cara superior de la hoja.",
        ],
        condiciones=[
            "Temperaturas templadas y humedad o rocío en las hojas.",
            "Cultivares susceptibles.",
        ],
        consecuencias=[
            "Pérdida de área foliar y menor peso de los granos.",
        ],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Wheat___Yellow_Rust": InfoClase(
        cultivo="Trigo",
        tipo="ENFERMEDAD",
        nombre="Roya amarilla (estriada) del trigo",
        agente="Puccinia striiformis (hongo)",
        descripcion="Enfermedad foliar del trigo de desarrollo rápido en ambientes frescos y húmedos.",
        sintomas=[
            "Pústulas amarillas dispuestas en estrías a lo largo de las nervaduras.",
            "Clorosis y secado de las hojas en ataques severos.",
        ],
        condiciones=[
            "Temperaturas frescas y alta humedad.",
            "Cultivares susceptibles.",
        ],
        consecuencias=[
            "Puede causar pérdidas de rendimiento importantes en cultivares susceptibles.",
        ],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Wheat___Healthy": InfoClase(cultivo="Trigo", tipo="SANO", nombre="Trigo sin síntomas visibles"),
    "Potato___Early_Blight": InfoClase(
        cultivo="Papa",
        tipo="ENFERMEDAD",
        nombre="Tizón temprano de la papa",
        agente="Alternaria solani (hongo)",
        descripcion="Enfermedad foliar que comienza en las hojas inferiores y más viejas.",
        sintomas=[
            "Manchas pardas con anillos concéntricos (aspecto de diana).",
            "Halo amarillento alrededor de las manchas.",
            "Comienza en las hojas inferiores.",
        ],
        condiciones=[
            "Alternancia de períodos húmedos y secos, temperaturas cálidas.",
            "Plantas estresadas o con deficiencias nutricionales.",
        ],
        consecuencias=["Defoliación prematura y menor rendimiento de tubérculos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Potato___Late_Blight": InfoClase(
        cultivo="Papa",
        tipo="ENFERMEDAD",
        nombre="Tizón tardío de la papa",
        agente="Phytophthora infestans (oomiceto)",
        descripcion="Enfermedad de avance muy rápido en ambientes frescos y húmedos.",
        sintomas=[
            "Lesiones de aspecto acuoso, verde oscuro a marrón, que avanzan rápidamente.",
            "Moho blanquecino en el envés de las hojas con alta humedad.",
        ],
        condiciones=["Temperaturas frescas con alta humedad o lluvias frecuentes."],
        consecuencias=[
            "Destrucción rápida del follaje.",
            "Pudrición de tubérculos.",
        ],
        control=[
            "Atender con urgencia: la enfermedad avanza rápido.",
            "Eliminar plantas y restos infectados; usar semilla sana.",
            "Evaluar un tratamiento registrado para el cultivo y la enfermedad según la etiqueta y la receta del Ingeniero Agrónomo.",
        ],
    ),
    "Potato___Healthy": InfoClase(cultivo="Papa", tipo="SANO", nombre="Papa sin síntomas visibles"),
    "Rice___Brown_Spot": InfoClase(
        cultivo="Arroz",
        tipo="ENFERMEDAD",
        nombre="Mancha parda del arroz",
        agente="Bipolaris oryzae (hongo)",
        descripcion="Enfermedad foliar asociada a suelos pobres y plantas con deficiencias nutricionales.",
        sintomas=["Manchas ovaladas de color marrón, a veces con centro grisáceo."],
        condiciones=[
            "Deficiencias nutricionales o estrés del cultivo.",
            "Alta humedad.",
        ],
        consecuencias=["Menor área foliar y menor calidad y peso de granos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Rice___Leaf_Blast": InfoClase(
        cultivo="Arroz",
        tipo="ENFERMEDAD",
        nombre="Piricularia (quemado) del arroz",
        agente="Pyricularia oryzae / Magnaporthe oryzae (hongo)",
        descripcion="Una de las enfermedades más importantes del arroz.",
        sintomas=["Lesiones en forma de huso (romboidales) con centro gris y borde marrón."],
        condiciones=[
            "Alta humedad y rocío prolongado.",
            "Exceso de fertilización nitrogenada.",
        ],
        consecuencias=["Pérdida de área foliar; si afecta el cuello de la panoja, grandes pérdidas de rendimiento."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Rice___Healthy": InfoClase(cultivo="Arroz", tipo="SANO", nombre="Arroz sin síntomas visibles"),
    # --- Modelo sigagro-extensivos (soja, girasol, maní): ver entrenamiento/ ---
    "Soybean___Frog_Eye_Leaf_Spot": InfoClase(
        cultivo="Soja",
        tipo="ENFERMEDAD",
        nombre="Mancha ojo de rana de la soja",
        agente="Cercospora sojina (hongo)",
        descripcion="Enfermedad foliar que se transmite por semilla y sobrevive en el rastrojo.",
        sintomas=[
            "Manchas circulares a angulares de 1 a 5 mm, con centro gris a castaño claro.",
            "Borde bien definido castaño rojizo a púrpura (aspecto de ojo).",
            "En ataques intensos las manchas se unen y la hoja se seca.",
        ],
        condiciones=[
            "Clima cálido y húmedo, lluvias frecuentes y rocío prolongado.",
            "Rastrojo de soja infectado y semilla de origen dudoso.",
            "Cultivares susceptibles.",
        ],
        consecuencias=["Defoliación anticipada y pérdidas de rendimiento en cultivares susceptibles."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Soybean___Rust": InfoClase(
        cultivo="Soja",
        tipo="ENFERMEDAD",
        nombre="Roya de la soja",
        agente="Phakopsora pachyrhizi (hongo, roya asiática)",
        descripcion="Enfermedad foliar de rápido avance que puede defoliar el cultivo en pocas semanas.",
        sintomas=[
            "Lesiones pequeñas castañas a castaño rojizas, primero en las hojas inferiores.",
            "Pústulas (uredinios) en el envés de la hoja, visibles con lupa.",
            "Amarillamiento y caída prematura de hojas.",
        ],
        condiciones=[
            "Mojado foliar prolongado y alta humedad.",
            "Temperaturas moderadas.",
            "Llegada de esporas desde zonas con la enfermedad (seguir las alertas regionales).",
        ],
        consecuencias=["Defoliación prematura, menor llenado y peso de granos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Soybean___Septoria_Brown_Spot": InfoClase(
        cultivo="Soja",
        tipo="ENFERMEDAD",
        nombre="Mancha marrón de la soja",
        agente="Septoria glycines (hongo)",
        descripcion="Enfermedad de fin de ciclo que avanza desde las hojas inferiores hacia arriba.",
        sintomas=[
            "Manchas pequeñas, irregulares, castaño oscuras.",
            "Halo amarillento alrededor de las manchas y amarillamiento de la hoja.",
            "Comienza en el estrato inferior del canopeo.",
        ],
        condiciones=["Clima cálido y húmedo.", "Rastrojo de soja en superficie (soja sobre soja)."],
        consecuencias=["Defoliación anticipada y pérdidas de rendimiento cuando sube por el canopeo."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Soybean___Mosaic": InfoClase(
        cultivo="Soja",
        tipo="ENFERMEDAD",
        nombre="Mosaico de la soja (virosis)",
        agente="Virus del mosaico de la soja (SMV), transmitido por semilla y pulgones",
        descripcion="Virosis que deforma las hojas y reduce el crecimiento.",
        sintomas=[
            "Mosaico de zonas verde claro y verde oscuro en las hojas.",
            "Hojas rugosas, abolladas o deformadas.",
            "Plantas de menor porte.",
        ],
        condiciones=["Semilla infectada y presencia de pulgones vectores."],
        consecuencias=["Menor crecimiento y rendimiento; posible manchado de la semilla."],
        control=CONTROL_VIRUS_GENERAL,
    ),
    "Soybean___Defoliating_Caterpillars": InfoClase(
        cultivo="Soja",
        tipo="PLAGA",
        nombre="Orugas defoliadoras de la soja",
        agente="Larvas de lepidópteros (p. ej. oruga medidora Rachiplusia nu, oruga de las leguminosas Anticarsia gemmatalis)",
        descripcion="Larvas que consumen el área foliar; el daño se evalúa como porcentaje de defoliación.",
        sintomas=[
            "Hojas con perforaciones o comidas desde el borde.",
            "Presencia de orugas o excrementos sobre las hojas.",
        ],
        condiciones=["Veranos cálidos; mayor riesgo durante la floración y el llenado de granos."],
        consecuencias=["Pérdida de área foliar que, en estados reproductivos, reduce el rendimiento."],
        control=CONTROL_PLAGAS_GENERAL,
    ),
    "Soybean___Diabrotica_speciosa": InfoClase(
        cultivo="Soja",
        tipo="PLAGA",
        nombre="Vaquita de San Antonio (Diabrotica speciosa)",
        agente="Diabrotica speciosa (coleóptero; el adulto come hojas y la larva ataca raíces)",
        descripcion="Coleóptero verde con manchas amarillas; el adulto produce perforaciones en las hojas.",
        sintomas=[
            "Perforaciones pequeñas y redondeadas en las hojas.",
            "Presencia de adultos verdes con manchas amarillas sobre el cultivo.",
        ],
        condiciones=["Mayor actividad con temperaturas cálidas, en especial en cultivos jóvenes."],
        consecuencias=["Defoliación; en cultivos recién emergidos puede afectar el stand de plantas."],
        control=CONTROL_PLAGAS_GENERAL,
    ),
    "Soybean___Healthy": InfoClase(cultivo="Soja", tipo="SANO", nombre="Soja sin síntomas visibles"),
    "Sunflower___Downy_Mildew": InfoClase(
        cultivo="Girasol",
        tipo="ENFERMEDAD",
        nombre="Mildiu del girasol",
        agente="Plasmopara halstedii (oomicete)",
        descripcion="Enfermedad sistémica que infecta la plántula desde el suelo y produce plantas enanas.",
        sintomas=[
            "Clorosis que acompaña las nervaduras de las hojas.",
            "Eflorescencia blanca en el envés de las hojas en condiciones húmedas.",
            "Plantas enanas con capítulos pequeños y erectos.",
        ],
        condiciones=[
            "Suelo húmedo o encharcado durante la siembra y la emergencia.",
            "Lotes con antecedentes de la enfermedad (inóculo en el suelo).",
        ],
        consecuencias=["Pérdida de plantas y de rendimiento; las plantas infectadas no producen o producen poco."],
        control=[
            "Usar híbridos con resistencia o tolerancia al mildiu.",
            "Rotación de cultivos y evitar lotes con antecedentes y mal drenaje.",
            "Consultar con el Ingeniero Agrónomo el tratamiento de semilla registrado para el cultivo.",
        ],
    ),
    "Sunflower___Gray_Mold": InfoClase(
        cultivo="Girasol",
        tipo="ENFERMEDAD",
        nombre="Podredumbre gris del girasol",
        agente="Botrytis cinerea (hongo)",
        descripcion="Enfermedad favorecida por la humedad que afecta hojas y capítulos.",
        sintomas=[
            "Manchas necróticas castañas en hojas o en el capítulo.",
            "Moho gris pulverulento sobre los tejidos afectados.",
        ],
        condiciones=["Alta humedad, lluvias y temperaturas templadas, en especial desde la floración."],
        consecuencias=["Podredumbre de tejidos y capítulos con pérdida de rendimiento y calidad."],
        control=[
            "Monitorear el lote desde la floración, sobre todo con pronóstico de lluvias.",
            "Densidad y fecha de siembra que favorezcan la aireación del cultivo.",
            "Consultar al Ingeniero Agrónomo las alternativas registradas para el cultivo.",
        ],
    ),
    "Sunflower___Healthy": InfoClase(cultivo="Girasol", tipo="SANO", nombre="Girasol sin síntomas visibles"),
    "Groundnut___Early_Leaf_Spot": InfoClase(
        cultivo="Maní",
        tipo="ENFERMEDAD",
        nombre="Viruela temprana del maní",
        agente="Passalora arachidicola (= Cercospora arachidicola, hongo)",
        descripcion="Enfermedad foliar frecuente del maní, favorecida por el rastrojo infectado.",
        sintomas=[
            "Manchas circulares castaño claras a oscuras, más visibles en el haz de la hoja.",
            "Halo amarillento alrededor de las manchas.",
        ],
        condiciones=["Alta humedad, lluvias y temperaturas cálidas; maní sobre maní."],
        consecuencias=["Defoliación, menor llenado de cajas y pérdidas de rendimiento."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Groundnut___Late_Leaf_Spot": InfoClase(
        cultivo="Maní",
        tipo="ENFERMEDAD",
        nombre="Viruela tardía del maní",
        agente="Nothopassalora personata (= Cercosporidium personatum, hongo)",
        descripcion="Viruela muy común en zonas maniseras; avanza rápido con clima húmedo.",
        sintomas=[
            "Manchas circulares castaño oscuras a casi negras.",
            "Esporulación oscura en el envés de la hoja; halo amarillo poco marcado.",
        ],
        condiciones=["Alta humedad, lluvias frecuentes y temperaturas cálidas; rastrojo de maní."],
        consecuencias=["Defoliación severa y pérdidas importantes de rendimiento."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Groundnut___Rust": InfoClase(
        cultivo="Maní",
        tipo="ENFERMEDAD",
        nombre="Roya del maní",
        agente="Puccinia arachidis (hongo)",
        descripcion="Enfermedad foliar que forma pústulas anaranjadas y acelera el secado de las hojas.",
        sintomas=[
            "Pústulas anaranjadas a castaño rojizas en el envés de la hoja.",
            "Las hojas se secan pero suelen quedar adheridas a la planta.",
        ],
        condiciones=["Alta humedad y temperaturas cálidas."],
        consecuencias=["Pérdida de área foliar y menor rendimiento."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Groundnut___Nutrient_Deficiency": InfoClase(
        cultivo="Maní",
        tipo="DEFICIENCIA",
        nombre="Síntomas compatibles con deficiencia nutricional (maní)",
        descripcion="Amarillamiento o decoloración de hojas no causado por patógenos; la causa debe confirmarse a campo.",
        sintomas=["Clorosis general o internerval de las hojas.", "Menor crecimiento de las plantas."],
        condiciones=["Suelos con baja disponibilidad de nutrientes, pH extremo o problemas de nodulación."],
        consecuencias=["Menor crecimiento y rendimiento si no se corrige la causa."],
        control=[
            "Confirmar la causa con análisis de suelo y de tejido vegetal.",
            "Verificar la nodulación y la inoculación de la semilla.",
            "Definir la corrección con el Ingeniero Agrónomo según el diagnóstico.",
        ],
    ),
    "Groundnut___Healthy": InfoClase(cultivo="Maní", tipo="SANO", nombre="Maní sin síntomas visibles"),
    # --- Ampliación de sigagro-extensivos: más cultivos y enfermedades presentes en Argentina ---
    "Corn___Northern_Leaf_Blight": InfoClase(
        cultivo="Maíz",
        tipo="ENFERMEDAD",
        nombre="Tizón foliar del maíz",
        agente="Exserohilum turcicum (hongo; antes Helminthosporium turcicum)",
        descripcion="Enfermedad foliar que forma lesiones grandes y alargadas; sobrevive en el rastrojo de maíz.",
        sintomas=[
            "Lesiones elípticas grandes (de 2 a 15 cm), de color gris verdoso a castaño, con forma de cigarro.",
            "Las lesiones comienzan en las hojas inferiores y avanzan hacia arriba.",
            "Con humedad, las lesiones se cubren de un moho oscuro (esporulación).",
        ],
        condiciones=[
            "Temperaturas moderadas (18 a 27 °C) con rocío o lluvias frecuentes.",
            "Rastrojo de maíz en superficie y maíz sobre maíz.",
            "Híbridos susceptibles.",
        ],
        consecuencias=[
            "Pérdida de área foliar antes o durante el llenado de granos.",
            "Pérdidas de rendimiento importantes si el ataque es temprano.",
        ],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Corn___Fall_Armyworm": InfoClase(
        cultivo="Maíz",
        tipo="PLAGA",
        nombre="Gusano cogollero del maíz",
        agente="Spodoptera frugiperda (lepidóptero; daña la larva)",
        descripcion="Principal plaga del maíz en Argentina; la larva se alimenta del cogollo y de las hojas en formación.",
        sintomas=[
            "Hojas con raspaduras transparentes («ventanitas») en plantas jóvenes.",
            "Perforaciones irregulares en las hojas que se despliegan del cogollo.",
            "Aserrín húmedo (excrementos) en el cogollo y presencia de larvas con una «Y» invertida en la cabeza.",
        ],
        condiciones=[
            "Temperaturas cálidas; mayor presión en siembras tardías y de segunda.",
            "Ataques más severos en los primeros estados vegetativos.",
        ],
        consecuencias=[
            "Defoliación y daño del cogollo que reduce el crecimiento.",
            "En ataques severos, pérdida de plantas y de rendimiento.",
        ],
        control=CONTROL_PLAGAS_GENERAL,
    ),
    "Wheat___Septoria": InfoClase(
        cultivo="Trigo",
        tipo="ENFERMEDAD",
        nombre="Mancha de la hoja del trigo (septoriosis)",
        agente="Zymoseptoria tritici (hongo; antes Septoria tritici)",
        descripcion="Enfermedad foliar frecuente en la región triguera; el hongo sobrevive en el rastrojo de trigo.",
        sintomas=[
            "Manchas alargadas de color castaño, limitadas por las nervaduras.",
            "Dentro de las manchas, puntos negros pequeños (picnidios), visibles a contraluz.",
            "Avanza de las hojas inferiores a las superiores.",
        ],
        condiciones=[
            "Temperaturas templadas y lluvias frecuentes (la lluvia salpica las esporas).",
            "Trigo sobre trigo y rastrojo en superficie.",
        ],
        consecuencias=["Pérdida de área foliar verde y menor peso de los granos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Wheat___Leaf_Blight": InfoClase(
        cultivo="Trigo",
        tipo="ENFERMEDAD",
        nombre="Mancha borrosa del trigo",
        agente="Bipolaris sorokiniana (hongo)",
        descripcion="Enfermedad foliar favorecida por el calor y la humedad; también puede afectar raíces y granos.",
        sintomas=[
            "Manchas ovaladas de color castaño oscuro, de bordes difusos (borrosos).",
            "Las manchas pueden rodearse de un halo amarillento y unirse entre sí.",
        ],
        condiciones=["Temperaturas cálidas con alta humedad, frecuentes al final del ciclo.", "Rastrojo de trigo y semilla infectada."],
        consecuencias=["Secado anticipado de hojas y menor llenado de granos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Wheat___Blast": InfoClase(
        cultivo="Trigo",
        tipo="ENFERMEDAD",
        nombre="Brusone del trigo (piricularia)",
        agente="Magnaporthe oryzae, patotipo Triticum (hongo; Pyricularia)",
        descripcion="Enfermedad de la espiga presente en el NEA y el NOA; puede causar pérdidas graves en pocos días.",
        sintomas=[
            "Espigas blanqueadas total o parcialmente por encima del punto de ataque.",
            "Lesión oscura en el raquis de la espiga.",
            "Granos chuzos o ausentes en la parte afectada; en hojas, lesiones en forma de ojo.",
        ],
        condiciones=["Temperaturas altas y humedad durante la espigazón (lluvias, rocío prolongado)."],
        consecuencias=["Pérdidas de rendimiento muy altas en años favorables.", "Menor calidad del grano."],
        control=[
            "Ajustar la fecha de siembra para evitar que la espigazón coincida con calor y humedad.",
            "Usar semilla sana y, si hay, cultivares con mejor comportamiento.",
            "Monitorear desde la espigazón; si el riesgo es alto, evaluar un fungicida registrado según la etiqueta y la receta del Ingeniero Agrónomo.",
        ],
    ),
    "Potato___Virus": InfoClase(
        cultivo="Papa",
        tipo="ENFERMEDAD",
        nombre="Síntomas compatibles con virosis de la papa",
        agente="Virus de la papa (p. ej. PVY, PLRV) transmitidos por pulgones y por tubérculo semilla",
        descripcion="Las virosis se transmiten por tubérculos semilla y por insectos vectores; no tienen control curativo.",
        sintomas=[
            "Mosaico (zonas verde claro y verde oscuro) y arrugado de las hojas.",
            "Enrollamiento de las hojas hacia arriba o necrosis de nervaduras.",
            "Plantas con menor crecimiento.",
        ],
        condiciones=["Tubérculo semilla infectado.", "Alta población de pulgones."],
        consecuencias=["Menor rendimiento y degeneración de la semilla propia."],
        control=CONTROL_VIRUS_GENERAL,
    ),
    "Soybean___Bacterial_Blight": InfoClase(
        cultivo="Soja",
        tipo="ENFERMEDAD",
        nombre="Tizón bacteriano de la soja",
        agente="Pseudomonas savastanoi pv. glycinea (bacteria)",
        descripcion="Enfermedad bacteriana foliar favorecida por clima fresco y lluvias con viento.",
        sintomas=[
            "Manchas pequeñas y angulosas, al principio acuosas, que luego se vuelven castañas a negras.",
            "Halo amarillo alrededor de las manchas.",
            "Las manchas se unen y el tejido muerto se rompe, dejando la hoja rasgada.",
        ],
        condiciones=["Temperaturas frescas, lluvias con viento y heridas en las hojas.", "Semilla y rastrojo infectados."],
        consecuencias=["Pérdida de área foliar; en general, daño moderado."],
        control=[
            "Usar semilla sana y cultivares con mejor comportamiento.",
            "Rotación de cultivos y manejo del rastrojo.",
            "Los fungicidas no controlan bacterias: definir el manejo con el Ingeniero Agrónomo.",
        ],
    ),
    "Groundnut___Alternaria_Leaf_Spot": InfoClase(
        cultivo="Maní",
        tipo="ENFERMEDAD",
        nombre="Mancha foliar por Alternaria (maní)",
        agente="Alternaria spp. (hongo)",
        descripcion="Enfermedad foliar que aparece sobre todo en hojas maduras y con estrés.",
        sintomas=["Manchas castañas irregulares, a veces con anillos concéntricos y halo amarillo.", "Secado del borde y de la punta de los folíolos."],
        condiciones=["Alta humedad y temperaturas templadas a cálidas.", "Plantas estresadas o al final del ciclo."],
        consecuencias=["Defoliación y menor rendimiento si el ataque es intenso."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Bean___Angular_Leaf_Spot": InfoClase(
        cultivo="Poroto",
        tipo="ENFERMEDAD",
        nombre="Mancha angular del poroto",
        agente="Pseudocercospora griseola (hongo)",
        descripcion="Una de las enfermedades foliares más importantes del poroto en el NOA.",
        sintomas=[
            "Manchas angulosas, limitadas por las nervaduras, de color gris a castaño.",
            "En el envés, con humedad, se ve un moho oscuro.",
            "En vainas, manchas redondeadas de borde rojizo.",
        ],
        condiciones=["Temperaturas moderadas y alternancia de períodos húmedos y secos.", "Semilla y rastrojo infectados."],
        consecuencias=["Defoliación prematura y menor rendimiento."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Bean___Rust": InfoClase(
        cultivo="Poroto",
        tipo="ENFERMEDAD",
        nombre="Roya del poroto",
        agente="Uromyces appendiculatus (hongo)",
        descripcion="Enfermedad foliar que forma pústulas y se dispersa por el viento.",
        sintomas=[
            "Pústulas pequeñas y redondeadas de color castaño rojizo, sobre todo en el envés.",
            "Halo clorótico alrededor de las pústulas.",
            "Las pústulas liberan un polvo color óxido al frotarlas.",
        ],
        condiciones=["Temperaturas moderadas con rocío prolongado o alta humedad."],
        consecuencias=["Defoliación y menor llenado de granos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Bean___Anthracnose": InfoClase(
        cultivo="Poroto",
        tipo="ENFERMEDAD",
        nombre="Antracnosis del poroto",
        agente="Colletotrichum lindemuthianum (hongo)",
        descripcion="Enfermedad que se transmite por semilla y afecta hojas, tallos y vainas.",
        sintomas=[
            "Nervaduras ennegrecidas en el envés de las hojas.",
            "En vainas, lesiones hundidas de color castaño oscuro con borde rojizo.",
            "Lesiones alargadas y oscuras en tallos y pecíolos.",
        ],
        condiciones=["Temperaturas frescas a moderadas con alta humedad y lluvias.", "Semilla infectada."],
        consecuencias=["Pérdida de rendimiento y de calidad comercial del grano."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Bean___Healthy": InfoClase(cultivo="Poroto", tipo="SANO", nombre="Poroto sin síntomas visibles"),
    "Cotton___Bacterial_Blight": InfoClase(
        cultivo="Algodón",
        tipo="ENFERMEDAD",
        nombre="Bacteriosis del algodón (mancha angular)",
        agente="Xanthomonas citri pv. malvacearum (bacteria)",
        descripcion="Enfermedad bacteriana que afecta hojas, tallos y cápsulas; se transmite por semilla.",
        sintomas=[
            "Manchas angulosas, acuosas al principio y luego castañas, limitadas por las nervaduras.",
            "Lesiones que siguen las nervaduras (nervadura negra).",
            "En cápsulas, manchas acuosas que pueden pudrir la fibra.",
        ],
        condiciones=["Temperaturas altas, lluvias con viento y alta humedad.", "Semilla y rastrojo infectados."],
        consecuencias=["Defoliación, caída de cápsulas y menor calidad de la fibra."],
        control=[
            "Usar semilla sana y cultivares con mejor comportamiento.",
            "Rotación de cultivos y destrucción del rastrojo.",
            "Los fungicidas no controlan bacterias: definir el manejo con el Ingeniero Agrónomo.",
        ],
    ),
    "Cotton___Herbicide_Damage": InfoClase(
        cultivo="Algodón",
        tipo="DEFICIENCIA",
        nombre="Síntomas compatibles con daño por herbicida (algodón)",
        descripcion="Daño no causado por patógenos: suele deberse a deriva o residuos de herbicidas hormonales u otros.",
        sintomas=[
            "Hojas deformadas, angostas, en forma de copa o con bordes dentados.",
            "Nervaduras paralelas o engrosadas y crecimiento anormal de los brotes.",
        ],
        condiciones=["Aplicaciones cercanas con viento (deriva), equipos mal lavados o residuos en el suelo."],
        consecuencias=["Retraso del crecimiento y pérdida de estructuras reproductivas según la intensidad."],
        control=[
            "Registrar fecha, lote y aplicaciones cercanas para identificar el origen.",
            "Revisar la limpieza de los equipos y las condiciones de aplicación (viento, inversión térmica).",
            "Definir con el Ingeniero Agrónomo cómo seguir según la intensidad del daño.",
        ],
    ),
    "Cotton___Healthy": InfoClase(cultivo="Algodón", tipo="SANO", nombre="Algodón sin síntomas visibles"),
    "Sugarcane___Brown_Rust": InfoClase(
        cultivo="Caña de azúcar",
        tipo="ENFERMEDAD",
        nombre="Roya marrón de la caña de azúcar",
        agente="Puccinia melanocephala (hongo)",
        descripcion="Enfermedad foliar presente en Tucumán; su importancia depende de la variedad.",
        sintomas=[
            "Manchas alargadas, amarillentas al principio y luego castaño rojizas.",
            "Pústulas en el envés que liberan esporas de color castaño.",
            "Secado de hojas en variedades susceptibles.",
        ],
        condiciones=["Temperaturas moderadas con alta humedad y rocío.", "Variedades susceptibles y cañas jóvenes."],
        consecuencias=["Pérdida de área foliar y menor producción de caña y azúcar."],
        control=[
            "Usar variedades con mejor comportamiento (principal medida de manejo).",
            "Monitorear los cañaverales jóvenes en épocas favorables.",
            "Definir el manejo con el Ingeniero Agrónomo.",
        ],
    ),
    "Sugarcane___Smut": InfoClase(
        cultivo="Caña de azúcar",
        tipo="ENFERMEDAD",
        nombre="Carbón de la caña de azúcar",
        agente="Sporisorium scitamineum (hongo)",
        descripcion="Enfermedad sistémica que se transmite por la caña semilla y por esporas en el viento.",
        sintomas=[
            "Estructura negra en forma de látigo en el ápice del tallo, cubierta de esporas.",
            "Macollos finos, numerosos y con hojas angostas («aspecto de pasto»).",
        ],
        condiciones=["Caña semilla infectada.", "Clima cálido y seco que favorece la dispersión de esporas."],
        consecuencias=["Pérdida de producción y de cepas en variedades susceptibles."],
        control=[
            "Usar caña semilla sana (semilleros saneados) y variedades con mejor comportamiento.",
            "Eliminar las plantas con látigos antes de que liberen esporas.",
        ],
    ),
    "Sugarcane___Yellow_Leaf": InfoClase(
        cultivo="Caña de azúcar",
        tipo="ENFERMEDAD",
        nombre="Síndrome de la hoja amarilla de la caña de azúcar",
        agente="Virus de la hoja amarilla de la caña (SCYLV), transmitido por pulgones y por caña semilla",
        descripcion="Virosis que provoca amarillamiento de la nervadura central de las hojas.",
        sintomas=[
            "Amarillamiento intenso de la nervadura central en el envés de las hojas.",
            "El amarillamiento avanza hacia la lámina y puede terminar en secado.",
        ],
        condiciones=["Caña semilla infectada.", "Presencia de pulgones vectores."],
        consecuencias=["Menor crecimiento y pérdida de rendimiento."],
        control=CONTROL_VIRUS_GENERAL,
    ),
    "Sugarcane___Pokkah_Boeng": InfoClase(
        cultivo="Caña de azúcar",
        tipo="ENFERMEDAD",
        nombre="Pokkah boeng de la caña de azúcar",
        agente="Fusarium spp. (complejo Fusarium fujikuroi; hongo)",
        descripcion="Enfermedad que afecta las hojas jóvenes del cogollo en períodos de crecimiento rápido.",
        sintomas=[
            "Hojas jóvenes cloróticas, arrugadas y deformadas en la base.",
            "Hojas del cogollo enroscadas; en casos graves, pudrición del cogollo.",
        ],
        condiciones=["Clima cálido y húmedo después de un período seco (crecimiento rápido).", "Variedades susceptibles."],
        consecuencias=["Generalmente daño leve; en casos graves, muerte del cogollo."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Sugarcane___Brown_Spot": InfoClase(
        cultivo="Caña de azúcar",
        tipo="ENFERMEDAD",
        nombre="Mancha parda de la caña de azúcar",
        agente="Cercospora longipes (hongo)",
        descripcion="Enfermedad foliar que aparece sobre todo en hojas maduras.",
        sintomas=["Manchas ovaladas a alargadas de color castaño rojizo, a veces con halo amarillo.", "Las manchas se unen y secan partes de la hoja."],
        condiciones=["Alta humedad y temperaturas cálidas.", "Variedades susceptibles."],
        consecuencias=["Pérdida de área foliar en ataques intensos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Sugarcane___Healthy": InfoClase(cultivo="Caña de azúcar", tipo="SANO", nombre="Caña de azúcar sin síntomas visibles"),
    "Sorghum___Anthracnose": InfoClase(
        cultivo="Sorgo",
        tipo="ENFERMEDAD",
        nombre="Antracnosis del sorgo",
        agente="Colletotrichum sublineola (hongo)",
        descripcion="Afecta hojas, nervadura central, tallo (podredumbre roja) y panoja.",
        sintomas=[
            "Manchas pequeñas, redondeadas o elípticas, con centro claro y borde rojizo a púrpura.",
            "Lesiones rojizas en la nervadura central.",
            "Al cortar el tallo, médula con zonas rojizas (podredumbre roja).",
        ],
        condiciones=["Temperaturas cálidas con lluvias frecuentes y alta humedad.", "Rastrojo infectado e híbridos susceptibles."],
        consecuencias=["Pérdida de área foliar, quebrado de tallos y menor rendimiento."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Sorghum___Rust": InfoClase(
        cultivo="Sorgo",
        tipo="ENFERMEDAD",
        nombre="Roya del sorgo",
        agente="Puccinia purpurea (hongo)",
        descripcion="Enfermedad foliar que forma pústulas; suele aparecer en la segunda mitad del ciclo.",
        sintomas=["Pústulas alargadas de color castaño rojizo en ambas caras de la hoja.", "Manchas púrpuras o rojizas alrededor de las pústulas."],
        condiciones=["Temperaturas moderadas y alta humedad."],
        consecuencias=["Pérdida de área foliar; daño mayor en híbridos susceptibles y ataques tempranos."],
        control=CONTROL_FUNGICO_GENERAL,
    ),
    "Sorghum___Head_Smut": InfoClase(
        cultivo="Sorgo",
        tipo="ENFERMEDAD",
        nombre="Carbón de la panoja del sorgo",
        agente="Sporisorium reilianum (hongo)",
        descripcion="Enfermedad sistémica que reemplaza la panoja por una masa de esporas.",
        sintomas=[
            "Panoja reemplazada por una agalla cubierta por una membrana blanquecina.",
            "Al romperse la membrana queda expuesta una masa de esporas negras con filamentos.",
        ],
        condiciones=["Esporas en el suelo que infectan la plántula; suelos secos y temperaturas moderadas a la emergencia."],
        consecuencias=["Las plantas afectadas no producen grano."],
        control=[
            "Usar híbridos con mejor comportamiento.",
            "Rotación de cultivos para reducir las esporas en el suelo.",
            "Definir el tratamiento de semilla con el Ingeniero Agrónomo.",
        ],
    ),
    "Sorghum___Grain_Mold": InfoClase(
        cultivo="Sorgo",
        tipo="ENFERMEDAD",
        nombre="Moho del grano del sorgo",
        agente="Complejo de hongos (p. ej. Fusarium, Curvularia, Alternaria)",
        descripcion="Hongos que colonizan el grano en formación cuando hay lluvias durante la maduración.",
        sintomas=["Granos con moho blanco, rosado o negro.", "Granos descoloridos, livianos o deformados."],
        condiciones=["Lluvias y alta humedad entre la floración y la madurez fisiológica."],
        consecuencias=["Menor peso y calidad del grano; posible presencia de micotoxinas."],
        control=[
            "Usar híbridos de grano con mejor comportamiento y ajustar la fecha de siembra.",
            "Cosechar en cuanto el grano llega a la humedad adecuada.",
            "Analizar el grano si se sospechan micotoxinas.",
        ],
    ),
    "Invalid": InfoClase(cultivo=None, tipo="INVALIDA", nombre="La imagen no corresponde a una hoja de cultivo analizable"),
}


ARCHIVO_CLASES_APRENDIDAS = "clases_aprendidas.json"


def cargar_clases_aprendidas(carpeta_modelos: Path | None = None) -> int:
    """Agrega las afecciones que el modelo aprendió de diagnósticos validados por Ingenieros Agrónomos.

    `entrenamiento/importar_validadas.py` las crea cuando junta suficientes fotos de una afección que no estaba
    en el modelo, y el entrenamiento copia `clases_aprendidas.json` junto al modelo. Su ficha tiene solo lo que
    escribió el Ingeniero (nombre y cultivo): nunca se inventan síntomas, agente ni condiciones.
    No reemplaza las clases con ficha técnica. Devuelve cuántas clases se agregaron.
    """
    carpeta = carpeta_modelos or Path(__file__).resolve().parent.parent / "modelos"
    agregadas = 0
    for archivo in sorted(carpeta.glob(f"*/{ARCHIVO_CLASES_APRENDIDAS}")):
        try:
            datos = json.loads(archivo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for etiqueta, c in datos.items():
            if etiqueta in CLASES or not isinstance(c, dict) or not c.get("nombre"):
                continue
            n = c.get("fotos")
            CLASES[etiqueta] = InfoClase(
                cultivo=c.get("cultivo"),
                tipo="APRENDIDA",
                nombre=str(c["nombre"]),
                descripcion=(
                    f"Afección incorporada al modelo a partir de {n} diagnósticos validados por Ingenieros Agrónomos en SIGAgro. "
                    if n else "Afección incorporada al modelo a partir de diagnósticos validados por Ingenieros Agrónomos en SIGAgro. "
                ) + "Todavía no tiene ficha técnica: la información la aporta el Ingeniero Agrónomo.",
                control=["Definir el manejo con el Ingeniero Agrónomo."],
            )
            agregadas += 1
    return agregadas


def info_de(etiqueta: str) -> InfoClase | None:
    return CLASES.get(etiqueta)


def normalizar_cultivo(texto: str | None) -> str | None:
    """Texto libre ('Maíz 1ra', 'trigo') -> cultivo del modelo, o None si no se reconoce."""
    if not texto:
        return None
    t = texto.strip().lower()
    t = t.translate(str.maketrans("áéíóú", "aeiou"))
    for clave, cultivo in SINONIMOS_CULTIVO.items():
        if t == clave or t.startswith(clave + " ") or t.startswith(clave + "-"):
            return cultivo
    return None


cargar_clases_aprendidas()
