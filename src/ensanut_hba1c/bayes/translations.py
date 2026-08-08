"""
English/Spanish translations used by the ENSANUT weighted Bayes heatmaps.

Place this file in:
    02_naive_bayes_followup/py/nb_heatmap_translations.py

The plotting module imports it automatically. Unknown labels are cleaned and
converted to readable title case instead of being left with underscores.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping


PLOT_TEXT = {
    "en": {
        "primary_title": "Primary questionnaire sections · weighted Bayes evidence",
        "secondary_title": "Secondary questionnaire sections · weighted Bayes evidence",
        "positive_panel": "Weighted positive Bayes evidence (NB+)",
        "negative_panel": "Weighted negative Bayes evidence (NB−)",
        "primary_net_title": "Primary questionnaire sections · net weighted Bayes evidence",
        "secondary_net_title": "Secondary questionnaire sections · net weighted Bayes evidence",
        "net_panel": "Net weighted Bayes evidence",
        "positive_colorbar": "Mean weighted NB+",
        "negative_colorbar": "Mean weighted NB−",
        "net_colorbar": "Mean net weighted evidence",
        "participants": "Participants",
        "cluster": "Cluster",
        "primary_section": "Primary section",
        "secondary_section": "Secondary section",
        "variables": "Variables",
        "coverage": "Coverage",


    },
    "es": {
        "primary_title": "Secciones primarias del cuestionario · evidencia Bayes ponderada",
        "secondary_title": "Secciones secundarias del cuestionario · evidencia Bayes ponderada",
        "positive_panel": "Evidencia Bayes positiva ponderada (NB+)",
        "negative_panel": "Evidencia Bayes negativa ponderada (NB−)",
        "primary_net_title": "Secciones primarias · evidencia Bayes neta ponderada",
        "secondary_net_title": "Secciones secundarias · evidencia Bayes neta ponderada",
        "net_panel": "Evidencia Bayes neta ponderada",
        "positive_colorbar": "Media ponderada NB+",
        "negative_colorbar": "Media ponderada NB−",
        "net_colorbar": "Media de evidencia neta ponderada",
        "participants": "Participantes",
        "cluster": "Clúster",
        "primary_section": "Sección primaria",
        "secondary_section": "Sección secundaria",
        "variables": "Variables",
        "coverage": "Cobertura",
    },
}


# Keys are normalized internally: accents, punctuation and repeated spaces
# are ignored when looking up a translation.
PRIMARY_SECTION_TRANSLATIONS = {
    "MICRONUTRIENTES": "Micronutrients",
    "ACTIVIDAD FISICA": "Physical activity",
    "AGREGACION POR GRUPOS NUTRICIONALES": "Food-group aggregation",
    "ANTROPOMETRIA Y PRESION ARTERIAL": "Anthropometry and blood pressure",
    "ANTROPOMETRIA": "Anthropometry",
    "PRESION ARTERIAL": "Blood pressure",
    "ENF": "Disease module",
    "ENFERMEDADES": "Disease module",
    "ETIQUETADO FRONTAL": "Front-of-package labeling",
    "FRECUENCIA DE ALIMENTOS": "Food-frequency module",
    "LACTANCIA": "Breastfeeding",
    "HOGAR": "Household",
    "HEMOGLOBINA": "Hemoglobin",
    "SALUD ADOLESCENTES": "Adolescent health",
    "SALUD DE ADOLESCENTES": "Adolescent health",
    "RESIDENTES": "Household residents",
    "SALUD ADULTOS": "Adult health",
    "SALUD DE ADULTOS": "Adult health",
    "CARACTERISTICAS DE LA VIVIENDA": "Housing characteristics",
    "VIVIENDA": "Housing",
    "SEGURIDAD ALIMENTARIA": "Food security",
    "PROGRAMAS SOCIALES": "Social programs",
    "APOYO DE PROGRAMAS SOCIALES": "Social-program support",
    "SOCIODEMOGRAFICAS": "Sociodemographic characteristics",
    "CARACTERISTICAS SOCIODEMOGRAFICAS": "Sociodemographic characteristics",
    "SERVICIOS SALUD": "Health services",
    "SERVICIOS DE SALUD": "Health services",
    "MEDICAMENTOS": "Medications",
    "USUARIOS SERVICIOS SALUD": "Healthcare users",
    "USUARIOS DE SERVICIOS DE SALUD": "Healthcare users",
    "UTILIZACION SERVICIOS SALUD": "Healthcare utilization",
    "UTILIZACION DE SERVICIOS DE SALUD": "Healthcare utilization",
    "ESTUDIOS LABORATORIO E IMAGEN": "Laboratory and imaging studies",
    "ESTUDIOS DE LABORATORIO E IMAGEN": "Laboratory and imaging studies",
    "ATENCION MEDICA": "Medical care",
    "MUESTRAS DE SANGRE Y DIAGNOSTICO": "Blood samples and diagnostics",
    "MUESTRAS DE SANGRE": "Blood-sample module",
    "DIABETES MELLITUS": "Diabetes mellitus",
    "HIPERTENSION ARTERIAL": "Arterial hypertension",
    "VACUNACION ADULTOS Y ADULTOS MAYORES": "Adult and older-adult vaccination",
    "VACUNACION": "Vaccination",
    "SARCOPENIA": "Sarcopenia",
    "MEMORIA": "Memory assessment",
    "EVALUACION DE MEMORIA": "Memory assessment",
    "SALUD SEXUAL Y REPRODUCTIVA": "Sexual and reproductive health",
    "CAIDAS Y FRACTURAS": "Falls and fractures",
    "ACTIVIDADES INSTRUMENTALES DE LA VIDA DIARIA": "Instrumental activities of daily living",
    "ACTIVIDADES BASICAS DE LA VIDA DIARIA": "Basic activities of daily living",
    "SOBREPESO Y OBESIDAD": "Overweight and obesity",
    "FUNCIONAMIENTO": "Functioning",
    "ANTECEDENTES FAMILIARES": "Family medical history",
    "PROGRAMAS PREVENTIVOS": "Preventive programs",
    "CONSUMO DE SUSTANCIAS": "Substance use",
    "OXIGENO EN CASA": "Home oxygen use",
    "USO DE OXIGENO EN CASA": "Home oxygen use",
    "SINTOMATOLOGIA DEPRESIVA": "Depressive symptomatology",
    "DEPRESION": "Depressive symptomatology",
}


SECONDARY_SECTION_TRANSLATIONS = {
    # Already-English aliases are included so labels read from previously
    # translated summary tables preserve publication-style sentence case.
    "I UTILIZATION": "I. Utilization",
    "II MEDICAL CARE": "II. Medical care",
    "III MEDICATIONS": "III. Medications",
    "IV LABORATORY OR IMAGING STUDIES": "IV. Laboratory or imaging studies",
    "IV ARTERIAL HYPERTENSION": "IV. Arterial hypertension",
    "XVI BASIC ACTIVITIES OF DAILY LIVING": "XVI. Basic activities of daily living",
    "XVII INSTRUMENTAL ACTIVITIES OF DAILY LIVING": "XVII. Instrumental activities of daily living",
    "ADOLESCENT MODULE": "Adolescent module",
    "I UTILIZACION": "I. Utilization",
    "II ATENCION": "II. Medical care",
    "III CARACTERISTICAS": "III. Characteristics",
    "IV ESTUDIOS DE LABORATORIO O GABINETE": "IV. Laboratory or imaging studies",
    "V ENFERMEDAD CARDIOVASCULAR": "V. Cardiovascular disease",
    "VI DIABETES": "VI. Diabetes",
    "VII HIPERTENSION ARTERIAL": "VII. Arterial hypertension",
    "VIII DISLIPIDEMIAS": "VIII. Dyslipidemia",
    "IX ENFERMEDAD RENAL": "IX. Kidney disease",
    "X SALUD MENTAL": "X. Mental health",
    "XI SALUD SEXUAL Y REPRODUCTIVA": "XI. Sexual and reproductive health",
    "XII ACTIVIDAD FISICA": "XII. Physical activity",
    "XIII ALIMENTACION": "XIII. Diet",
    "XIV CONSUMO DE TABACO": "XIV. Tobacco use",
    "XV CONSUMO DE ALCOHOL": "XV. Alcohol use",
    "XVI MEDICAMENTOS": "XVI. Medications",
    "XVII VACUNACION": "XVII. Vaccination",
    "XVIII FUNCIONAMIENTO": "XVIII. Functioning",
    "XIX CAIDAS Y FRACTURAS": "XIX. Falls and fractures",
    "XX MEMORIA": "XX. Memory",
    "XVII ACTIVIDADES INSTRUMENTALES DE LA VIDA DIARIA": "XVII. Instrumental activities of daily living",
    "VI ENFERMEDAD RENAL HIPERCOLESTEROLEMIA": "VI. Kidney disease and hypercholesterolemia",
    "VIII ESCALA DE EXPERIENCIAS DE INSEGURIDAD DEL AGUA EN EL HOGAR": "VIII. Household Water Insecurity Experiences Scale",
    "MODULO ADOLESCENTES": "Adolescent module",
    "MODULO ADULTOS": "Adult module",
    "MODULO ACTIVIDAD": "Physical-activity module",
    "MODULO ETIQUETADO": "Labeling module",
    "MODULO HOGAR": "Household module",
    "MODULO SALUD": "Health module",
    "MODULO FRECUENCIA DE ALIMENTOS": "Food-frequency module",
    "MUESTRAS DE SANGRE": "Blood-sample module",
    "DIAGNOSTICO": "Diagnostics",
    "UTILIZACION": "Utilization",
    "ATENCION": "Medical care",
    "CARACTERISTICAS": "Characteristics",
    "ESTUDIOS DE LABORATORIO O GABINETE": "Laboratory or imaging studies",
    "ENFERMEDAD CARDIOVASCULAR": "Cardiovascular disease",
    "DIABETES": "Diabetes",
    "HIPERTENSION ARTERIAL": "Arterial hypertension",
    "DISLIPIDEMIAS": "Dyslipidemia",
    "ENFERMEDAD RENAL": "Kidney disease",
    "SALUD MENTAL": "Mental health",
    "SALUD SEXUAL Y REPRODUCTIVA": "Sexual and reproductive health",
    "ACTIVIDAD FISICA": "Physical activity",
    "ALIMENTACION": "Diet",
    "CONSUMO DE TABACO": "Tobacco use",
    "CONSUMO DE ALCOHOL": "Alcohol use",
    "MEDICAMENTOS": "Medications",
    "VACUNACION": "Vaccination",
    "FUNCIONAMIENTO": "Functioning",
    "CAIDAS Y FRACTURAS": "Falls and fractures",
    "MEMORIA": "Memory",
    # Corrections for the Adult module
    # Backward-compatible alias for summaries generated by older versions.
    "IV SITUACION DE SALUD Y HEALTHCARE UTILIZATION DE SALUD": "IV. Health situation and healthcare utilization",
    "XVII ACTIVIDADES INSTRUMENTALES DE LA VIDA DIARI": "XVII. Instrumental activities of daily living",
    "XI ACCIDENTES": "XI. Accidents",
    "VII ANTECEDENTES HEREDO FAMILIARES": "VII. Family medical history",
    "XIII USO DE SUSTANCIAS": "XIII. Substance use",
    "XII ATAQUE Y VIOLENCIA PARA ADULTOS DE 20 ANOS O MAS": "XII. Attack and violence in adults (20+ years)",

    # Corrections for the Household and Residents module
    "IV SITUACION DE SALUD Y UTILIZACION DE SALUD": "IV. Health situation and healthcare utilization",
    "V OTRAS CARACTERISTICAS DEL HOGAR": "V. Other household characteristics",
    "V OTRAS CHARACTERISTICS DEL HOGAR": "V. Other household characteristics",
    "II IDENTIFICACION DE HOGARES": "II. Household identification",

    # Other missing submodules
    "MODULO ACTIVIDAD FISICA ADULTOS": "Adult physical activity module",
    "MODULO PHYSICAL ACTIVITY ADULTOS": "Adult physical activity module",
    "CRONICAS": "Chronic diseases",
    "MICRONUTRIENTES": "Micronutrients",
    "MICRONUTRIENTE": "Micronutrients",
    "MACRONUTRIENTES": "Macronutrients",
    "NUTRIENTES": "Nutrients",
    "GRUPOS NUTRICIONALES": "Nutritional groups",
    "ESTADO NUTRICIONAL": "Nutritional status",
    "NUTRICION": "Nutrition",
    }


def _normalize_key(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.upper().strip()
    text = re.sub(r"[_\-]+", " ", text)
    # Periods are structural punctuation, not part of translation keys.
    text = re.sub(r"[^\w\s>·]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text


_ROMAN_PREFIX_PATTERN = re.compile(
    r"^\s*(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|"
    r"IX|VIII|VII|VI|V|IV|III|II|I)\s*[.\-–—:)]*\s+",
    flags=re.IGNORECASE,
)


def _strip_leading_roman_numeral(value: object) -> str:
    """Remove a leading Roman section number from an English label."""
    text = "" if value is None else str(value).strip()
    return _ROMAN_PREFIX_PATTERN.sub("", text, count=1).strip()


_PRIMARY_NORMALIZED = {
    _normalize_key(key): value for key, value in PRIMARY_SECTION_TRANSLATIONS.items()
}
_SECONDARY_NORMALIZED = {
    _normalize_key(key): value for key, value in SECONDARY_SECTION_TRANSLATIONS.items()
}


def _readable_fallback(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    # Preserve common acronyms and Roman numerals.
    words = []
    for word in text.split():
        normalized = _normalize_key(word)
        if normalized in {
            "ENF", "IMSS", "ISSSTE", "ENSANUT",
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX",
            "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII",
            "XVIII", "XIX", "XX",
        }:
            words.append(normalized)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def translate_primary(value: object, language: str = "en") -> str:
    if language.lower().startswith("es"):
        return _readable_fallback(value)
    key = _normalize_key(value)
    translated = _PRIMARY_NORMALIZED.get(key, _readable_fallback(value))
    return _strip_leading_roman_numeral(translated)



SECONDARY_PHRASE_TRANSLATIONS = {
    # Full labels must precede generic word-level translations. This avoids
    # mixed Spanish/English output when the exact section alias is absent.
    "SITUACION DE SALUD Y UTILIZACION DE SERVICIOS DE SALUD": "Health situation and healthcare utilization",
    "OTRAS CARACTERISTICAS DEL HOGAR": "Other household characteristics",
    "MODULO ACTIVIDAD FISICA ADULTOS": "Adult physical activity module",
    "CRONICAS": "Chronic diseases",
    "MICRONUTRIENTES": "Micronutrients",
    "ESTUDIOS DE LABORATORIO O GABINETE": "Laboratory or imaging studies",
    "ESTUDIOS DE LABORATORIO": "Laboratory studies",
    "ESTUDIOS DE GABINETE": "Imaging studies",
    "ENFERMEDAD CARDIOVASCULAR": "Cardiovascular disease",
    "ENFERMEDADES CARDIOVASCULARES": "Cardiovascular diseases",
    "SALUD SEXUAL Y REPRODUCTIVA": "Sexual and reproductive health",
    "HIPERTENSION ARTERIAL": "Arterial hypertension",
    "ACTIVIDAD FISICA": "Physical activity",
    "CONSUMO DE TABACO": "Tobacco use",
    "CONSUMO DE ALCOHOL": "Alcohol use",
    "SEGURIDAD ALIMENTARIA": "Food security",
    "ATENCION MEDICA": "Medical care",
    "SERVICIOS DE SALUD": "Health services",
    "UTILIZACION DE SERVICIOS": "Healthcare utilization",
    "MUESTRAS DE SANGRE": "Blood samples",
    "PRESION ARTERIAL": "Blood pressure",
    "SALUD MENTAL": "Mental health",
    "ENFERMEDAD RENAL": "Kidney disease",
    "CAIDAS Y FRACTURAS": "Falls and fractures",
    "ANTECEDENTES FAMILIARES": "Family medical history",
    "PROGRAMAS PREVENTIVOS": "Preventive programs",
    "SOBREPESO Y OBESIDAD": "Overweight and obesity",
    "ALIMENTACION": "Diet",
    "VACUNACION": "Vaccination",
    "MEDICAMENTOS": "Medications",
    "FUNCIONAMIENTO": "Functioning",
    "MEMORIA": "Memory",
    "DIABETES": "Diabetes",
    "DISLIPIDEMIAS": "Dyslipidemia",
    "CARACTERISTICAS": "Characteristics",
    "UTILIZACION": "Utilization",
    "ATENCION": "Medical care",
    "DIAGNOSTICO": "Diagnostics",
    "ACTIVIDADES INSTRUMENTALES DE LA VIDA DIARIA": "Instrumental activities of daily living",
    "ACTIVIDADES BASICAS DE LA VIDA DIARIA": "Basic activities of daily living",
    "ENFERMEDAD RENAL HIPERCOLESTEROLEMIA": "Kidney disease and hypercholesterolemia",
    "HIPERCOLESTEROLEMIA": "Hypercholesterolemia",
    "ESCALA DE EXPERIENCIAS DE INSEGURIDAD DEL AGUA EN EL HOGAR": "Household Water Insecurity Experiences Scale",
}


def _translate_secondary_compositionally(value: object) -> str:
    key = _normalize_key(value)

    roman_match = re.match(
        r"^((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX))\s+(.*)$",
        key,
    )
    if roman_match:
        body = roman_match.group(2).strip()
    else:
        body = key

    translated = SECONDARY_PHRASE_TRANSLATIONS.get(body)
    if translated is None:
        translated = _PRIMARY_NORMALIZED.get(body)

    if translated is None:
        # Longest-phrase replacement catches labels with qualifiers while
        # preserving readable English for previously unseen submodules.
        translated_text = body.title()
        for source, target in sorted(
            SECONDARY_PHRASE_TRANSLATIONS.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            translated_text = re.sub(
                rf"\b{re.escape(source.title())}\b",
                target,
                translated_text,
                flags=re.IGNORECASE,
            )
        translated = translated_text

    # English labels intentionally omit questionnaire section numbering.
    return _strip_leading_roman_numeral(translated)


def translate_secondary(value: object, language: str = "en") -> str:
    if language.lower().startswith("es"):
        return _readable_fallback(value)
    key = _normalize_key(value)
    if key in _SECONDARY_NORMALIZED:
        translated = _SECONDARY_NORMALIZED[key]
    elif key in _PRIMARY_NORMALIZED:
        translated = _PRIMARY_NORMALIZED[key]
    else:
        translated = _translate_secondary_compositionally(value)
    return _strip_leading_roman_numeral(translated)


def split_hierarchy_label(value: object) -> tuple[str, str | None]:
    """
    Split a hierarchical label into primary and secondary components.

    Supported separators include ``>``, a middle dot, a dash, or a period
    immediately followed by a Roman-numbered subsection. This covers labels
    such as ``SALUD ADULTOS - XVII. ...`` without splitting ordinary hyphens.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return "", None

    if ">" in text:
        parent, child = text.split(">", 1)
        return parent.strip(), child.strip() or None

    if " · " in text:
        parent, child = text.split(" · ", 1)
        return parent.strip(), child.strip() or None

    roman = (
        r"(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|"
        r"IX|VIII|VII|VI|V|IV|III|II|I)"
    )
    match = re.match(
        rf"^(.*?)\s*(?:>|·|[-–—]|\.)\s*({roman}\b.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip() or None

    return text, None

def translate_group(value: object, level: str, language: str = "en") -> str:
    level_key = str(level).lower()
    parent, child = split_hierarchy_label(value)

    if level_key.startswith("prim"):
        return translate_primary(parent, language=language)

    if child:
        parent_text = translate_primary(parent, language=language)
        child_text = translate_secondary(child, language=language)
        return f"{parent_text} · {child_text}"

    return translate_secondary(parent, language=language)


def text(key: str, language: str = "en") -> str:
    language_key = "es" if language.lower().startswith("es") else "en"
    return PLOT_TEXT.get(language_key, PLOT_TEXT["en"]).get(key, key)


def translate_mapping(
    values: Mapping[object, object],
    *,
    level: str,
    language: str = "en",
) -> dict[object, str]:
    return {
        key: translate_group(value, level=level, language=language)
        for key, value in values.items()
    }
