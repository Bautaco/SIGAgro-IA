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
