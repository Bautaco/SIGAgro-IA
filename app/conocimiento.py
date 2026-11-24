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

from dataclasses import dataclass, field

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
    "Invalid": InfoClase(cultivo=None, tipo="INVALIDA", nombre="La imagen no corresponde a una hoja de cultivo analizable"),
}


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
