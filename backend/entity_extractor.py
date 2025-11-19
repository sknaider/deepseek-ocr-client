#!/usr/bin/env python3
"""
Medical Entity Extractor for Spanish Clinical Documents
Extracts diagnoses, medications, procedures, vital signs, lab results, and dates
from OCR text of medical documents in Peru/LatAm

Optimized for Spanish medical terminology and Peruvian healthcare documentation
"""

import re
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

# Optional spaCy support
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using rule-based extraction only.")


@dataclass
class Entity:
    """Represents an extracted medical entity"""
    text: str                    # Entity text
    type: str                    # Entity type (diagnostico, medicamento, etc.)
    confidence: float            # Confidence score (0-1)
    start_pos: int               # Start position in original text
    end_pos: int                 # End position in original text
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    context_before: str = ""     # Text before entity
    context_after: str = ""      # Text after entity
    validated: bool = True       # Whether entity passed validation


@dataclass
class MedicationEntity(Entity):
    """Medication entity with dose, frequency, duration"""
    nombre: Optional[str] = None
    dosis: Optional[str] = None
    frecuencia: Optional[str] = None
    duracion: Optional[str] = None
    via: Optional[str] = None


@dataclass
class VitalSignEntity(Entity):
    """Vital sign entity with value and unit"""
    nombre: str = ""
    valor: str = ""
    unidad: str = ""
    valor_numerico: Optional[float] = None


@dataclass
class LabResultEntity(Entity):
    """Lab result entity with value, unit, and reference range"""
    nombre: str = ""
    valor: str = ""
    unidad: str = ""
    rango_min: Optional[float] = None
    rango_max: Optional[float] = None
    valor_numerico: Optional[float] = None


@dataclass
class ExtractionResult:
    """Complete extraction result for a document"""
    diagnosticos: List[Entity] = field(default_factory=list)
    medicamentos: List[MedicationEntity] = field(default_factory=list)
    procedimientos: List[Entity] = field(default_factory=list)
    fechas_importantes: List[Entity] = field(default_factory=list)
    signos_vitales: Dict[str, VitalSignEntity] = field(default_factory=dict)
    resultados_laboratorio: List[LabResultEntity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalEntityExtractor:
    """
    Extract medical entities from Spanish clinical documents

    Uses hybrid approach:
    1. Regex patterns for structured data (vital signs, doses, dates)
    2. Rule-based matching for medical terminology
    3. Optional spaCy NER for general entities
    """

    def __init__(self, config_path: str = "medical_ner_config.yaml"):
        """
        Initialize medical entity extractor

        Args:
            config_path: Path to medical NER configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.nlp = None

        # Load spaCy model if available and enabled
        if SPACY_AVAILABLE and self.config['model'].get('use_rules', True):
            self._load_spacy_model()

        # Compile regex patterns
        self.patterns = self._compile_patterns()

        # Load medical terminology
        self.medical_terms = self.config.get('medical_terms', {})
        self.abbreviations = self.config.get('abbreviations', {})

        self.logger.info("Medical Entity Extractor initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('MedicalEntityExtractor')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)

        return logger

    def _load_spacy_model(self):
        """Load spaCy Spanish model"""
        model_name = self.config['model'].get('spacy_model', 'es_core_news_md')

        try:
            self.logger.info(f"Loading spaCy model: {model_name}")
            self.nlp = spacy.load(model_name)
            self.logger.info(f"spaCy model loaded successfully")
        except OSError:
            self.logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
            self.logger.warning("Falling back to rule-based extraction only")
            self.nlp = None

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns from config"""
        patterns = {}
        pattern_config = self.config.get('patterns', {})

        for entity_type, pattern_list in pattern_config.items():
            patterns[entity_type] = []
            for pattern_dict in pattern_list:
                try:
                    regex = re.compile(pattern_dict['pattern'], re.IGNORECASE)
                    patterns[entity_type].append(regex)
                except Exception as e:
                    self.logger.warning(f"Invalid regex for {entity_type}: {e}")

        return patterns

    def extract_entities(self, text: str) -> ExtractionResult:
        """
        Extract all medical entities from text

        Args:
            text: OCR text from medical document

        Returns:
            ExtractionResult with all extracted entities
        """
        if not text or len(text.strip()) == 0:
            return ExtractionResult()

        # Expand abbreviations
        text = self._expand_abbreviations(text)

        # Extract each entity type
        result = ExtractionResult()

        # Extract vital signs (structured data)
        result.signos_vitales = self._extract_vital_signs(text)

        # Extract lab results
        result.resultados_laboratorio = self._extract_lab_results(text)

        # Extract medications
        result.medicamentos = self._extract_medications(text)

        # Extract dates
        result.fechas_importantes = self._extract_dates(text)

        # Extract diagnoses (using spaCy + rules)
        result.diagnosticos = self._extract_diagnoses(text)

        # Extract procedures
        result.procedimientos = self._extract_procedures(text)

        # Add metadata
        result.metadata = {
            'extraction_timestamp': datetime.utcnow().isoformat(),
            'text_length': len(text),
            'total_entities': (
                len(result.diagnosticos) +
                len(result.medicamentos) +
                len(result.procedimientos) +
                len(result.fechas_importantes) +
                len(result.signos_vitales) +
                len(result.resultados_laboratorio)
            ),
            'extraction_method': 'hybrid' if self.nlp else 'rule_based'
        }

        return result

    def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations to full terms"""
        expanded = text

        for abbr, full_term in self.abbreviations.items():
            # Match abbreviation as whole word
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(pattern, f"{abbr} ({full_term})", expanded, flags=re.IGNORECASE)

        return expanded

    def _extract_vital_signs(self, text: str) -> Dict[str, VitalSignEntity]:
        """Extract vital signs (PA, FC, Temp, etc.)"""
        vital_signs = {}

        # Blood pressure
        for pattern in self.patterns.get('presion_arterial', []):
            for match in pattern.finditer(text):
                try:
                    sistolica = match.group(1)
                    diastolica = match.group(2)
                    valor = f"{sistolica}/{diastolica}"

                    # Validate
                    if self._validate_blood_pressure(float(sistolica), float(diastolica)):
                        vital_signs['presion_arterial'] = VitalSignEntity(
                            text=match.group(0),
                            type='signo_vital',
                            confidence=0.95,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            nombre='Presión Arterial',
                            valor=valor,
                            unidad='mmHg'
                        )
                except (IndexError, ValueError) as e:
                    continue

        # Heart rate
        for pattern in self.patterns.get('frecuencia_cardiaca', []):
            for match in pattern.finditer(text):
                try:
                    valor = match.group(1)
                    if self._validate_heart_rate(float(valor)):
                        vital_signs['frecuencia_cardiaca'] = VitalSignEntity(
                            text=match.group(0),
                            type='signo_vital',
                            confidence=0.9,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            nombre='Frecuencia Cardíaca',
                            valor=valor,
                            unidad='lpm',
                            valor_numerico=float(valor)
                        )
                except (IndexError, ValueError):
                    continue

        # Temperature
        for pattern in self.patterns.get('temperatura', []):
            for match in pattern.finditer(text):
                try:
                    valor = match.group(1)
                    if self._validate_temperature(float(valor)):
                        vital_signs['temperatura'] = VitalSignEntity(
                            text=match.group(0),
                            type='signo_vital',
                            confidence=0.95,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            nombre='Temperatura',
                            valor=valor,
                            unidad='°C',
                            valor_numerico=float(valor)
                        )
                except (IndexError, ValueError):
                    continue

        # Respiratory rate
        for pattern in self.patterns.get('frecuencia_respiratoria', []):
            for match in pattern.finditer(text):
                try:
                    valor = match.group(1)
                    if self._validate_respiratory_rate(float(valor)):
                        vital_signs['frecuencia_respiratoria'] = VitalSignEntity(
                            text=match.group(0),
                            type='signo_vital',
                            confidence=0.9,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            nombre='Frecuencia Respiratoria',
                            valor=valor,
                            unidad='rpm',
                            valor_numerico=float(valor)
                        )
                except (IndexError, ValueError):
                    continue

        # Oxygen saturation
        for pattern in self.patterns.get('saturacion_oxigeno', []):
            for match in pattern.finditer(text):
                try:
                    valor = match.group(1)
                    if self._validate_oxygen_saturation(float(valor)):
                        vital_signs['saturacion_oxigeno'] = VitalSignEntity(
                            text=match.group(0),
                            type='signo_vital',
                            confidence=0.95,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            nombre='Saturación de Oxígeno',
                            valor=valor,
                            unidad='%',
                            valor_numerico=float(valor)
                        )
                except (IndexError, ValueError):
                    continue

        # Glucose
        for pattern in self.patterns.get('glucosa', []):
            for match in pattern.finditer(text):
                try:
                    valor = match.group(1)
                    vital_signs['glucosa'] = VitalSignEntity(
                        text=match.group(0),
                        type='signo_vital',
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        nombre='Glucosa',
                        valor=valor,
                        unidad='mg/dL',
                        valor_numerico=float(valor)
                    )
                except (IndexError, ValueError):
                    continue

        return vital_signs

    def _extract_lab_results(self, text: str) -> List[LabResultEntity]:
        """Extract laboratory results with values and ranges"""
        lab_results = []

        for pattern in self.patterns.get('resultado_lab', []):
            for match in pattern.finditer(text):
                try:
                    nombre = match.group(1).strip()
                    valor = match.group(2)
                    unidad = match.group(3)

                    # Extract reference range if present
                    rango_min = None
                    rango_max = None
                    if match.lastindex >= 5:
                        try:
                            rango_min = float(match.group(4))
                            rango_max = float(match.group(5))
                        except (ValueError, TypeError):
                            pass

                    lab_result = LabResultEntity(
                        text=match.group(0),
                        type='resultado_laboratorio',
                        confidence=0.85,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        nombre=nombre,
                        valor=valor,
                        unidad=unidad,
                        rango_min=rango_min,
                        rango_max=rango_max,
                        valor_numerico=float(valor) if valor else None
                    )

                    lab_results.append(lab_result)

                except (IndexError, ValueError) as e:
                    continue

        return lab_results

    def _extract_medications(self, text: str) -> List[MedicationEntity]:
        """Extract medications with dose, frequency, and duration"""
        medications = []

        # Known medication names
        medication_names = self.medical_terms.get('medicamentos_comunes', [])

        # Find medication mentions
        for med_name in medication_names:
            # Case-insensitive search for medication name
            pattern = re.compile(r'\b' + re.escape(med_name) + r'\b', re.IGNORECASE)

            for match in pattern.finditer(text):
                # Extract context around medication (50 chars before/after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                # Extract dose
                dosis = self._extract_dose_from_context(context)

                # Extract frequency
                frecuencia = self._extract_frequency_from_context(context)

                # Extract duration
                duracion = self._extract_duration_from_context(context)

                medication = MedicationEntity(
                    text=match.group(0),
                    type='medicamento',
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    nombre=match.group(0),
                    dosis=dosis,
                    frecuencia=frecuencia,
                    duracion=duracion,
                    metadata={
                        'context': context
                    }
                )

                medications.append(medication)

        return medications

    def _extract_dose_from_context(self, context: str) -> Optional[str]:
        """Extract dose from medication context"""
        for pattern in self.patterns.get('dosis', []):
            match = pattern.search(context)
            if match:
                return match.group(0)
        return None

    def _extract_frequency_from_context(self, context: str) -> Optional[str]:
        """Extract frequency from medication context"""
        for pattern in self.patterns.get('frecuencia', []):
            match = pattern.search(context)
            if match:
                return match.group(0)
        return None

    def _extract_duration_from_context(self, context: str) -> Optional[str]:
        """Extract duration from medication context"""
        for pattern in self.patterns.get('duracion', []):
            match = pattern.search(context)
            if match:
                return match.group(0)
        return None

    def _extract_dates(self, text: str) -> List[Entity]:
        """Extract important dates (admission, discharge, procedures)"""
        dates = []

        for pattern in self.patterns.get('fecha', []):
            for match in pattern.finditer(text):
                # Get context to determine date type
                context_start = max(0, match.start() - 30)
                context = text[context_start:match.start()].lower()

                # Determine date type from context
                date_type = 'fecha'
                if 'ingreso' in context or 'admisión' in context:
                    date_type = 'fecha_ingreso'
                elif 'alta' in context or 'egreso' in context:
                    date_type = 'fecha_alta'
                elif 'cirugía' in context or 'operación' in context:
                    date_type = 'fecha_cirugia'
                elif 'procedimiento' in context:
                    date_type = 'fecha_procedimiento'

                date_entity = Entity(
                    text=match.group(0),
                    type=date_type,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={'date_type': date_type}
                )

                dates.append(date_entity)

        return dates

    def _extract_diagnoses(self, text: str) -> List[Entity]:
        """Extract medical diagnoses"""
        diagnoses = []

        # Use known diagnoses from config
        known_diagnoses = []
        known_diagnoses.extend(self.medical_terms.get('enfermedades_cronicas', []))
        known_diagnoses.extend(self.medical_terms.get('enfermedades_agudas', []))

        for diagnosis in known_diagnoses:
            pattern = re.compile(r'\b' + re.escape(diagnosis) + r'\b', re.IGNORECASE)

            for match in pattern.finditer(text):
                diag_entity = Entity(
                    text=match.group(0),
                    type='diagnostico',
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={'diagnosis_name': diagnosis}
                )

                diagnoses.append(diag_entity)

        # Use spaCy if available for additional diagnoses
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Look for medical-sounding entities (PROPN, NOUN phrases)
                if ent.label_ in ['MISC', 'PER', 'ORG']:  # Spanish model labels
                    # Check if it looks like a medical diagnosis
                    if any(keyword in ent.text.lower() for keyword in
                           ['síndrome', 'enfermedad', 'trastorno', 'insuficiencia']):
                        diag_entity = Entity(
                            text=ent.text,
                            type='diagnostico',
                            confidence=0.6,  # Lower confidence for NER-extracted
                            start_pos=ent.start_char,
                            end_pos=ent.end_char,
                            metadata={'source': 'spacy_ner', 'label': ent.label_}
                        )
                        diagnoses.append(diag_entity)

        return diagnoses

    def _extract_procedures(self, text: str) -> List[Entity]:
        """Extract medical procedures"""
        procedures = []

        # Keywords that indicate procedures
        procedure_keywords = [
            'cirugía', 'operación', 'intervención', 'procedimiento',
            'biopsia', 'endoscopia', 'colonoscopia', 'laparoscopía',
            'cateterismo', 'angioplastía', 'tomografía', 'resonancia',
            'radiografía', 'ecografía', 'ultrasonido'
        ]

        for keyword in procedure_keywords:
            # Find keyword and extract surrounding context as procedure
            pattern = re.compile(r'([A-Za-zÁ-úÑñ\s]*' + re.escape(keyword) + r'[A-Za-zÁ-úÑñ\s]{0,40})', re.IGNORECASE)

            for match in pattern.finditer(text):
                procedure_text = match.group(1).strip()

                # Skip if too short (likely false positive)
                if len(procedure_text) < 10:
                    continue

                proc_entity = Entity(
                    text=procedure_text,
                    type='procedimiento',
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={'keyword': keyword}
                )

                procedures.append(proc_entity)

        return procedures

    # Validation methods

    def _validate_blood_pressure(self, sistolica: float, diastolica: float) -> bool:
        """Validate blood pressure values"""
        validation = self.config['validation']['signos_vitales']
        return (
            validation['presion_arterial_sistolica']['min'] <= sistolica <= validation['presion_arterial_sistolica']['max'] and
            validation['presion_arterial_diastolica']['min'] <= diastolica <= validation['presion_arterial_diastolica']['max']
        )

    def _validate_heart_rate(self, value: float) -> bool:
        """Validate heart rate value"""
        validation = self.config['validation']['signos_vitales']['frecuencia_cardiaca']
        return validation['min'] <= value <= validation['max']

    def _validate_temperature(self, value: float) -> bool:
        """Validate temperature value"""
        validation = self.config['validation']['signos_vitales']['temperatura']
        return validation['min'] <= value <= validation['max']

    def _validate_respiratory_rate(self, value: float) -> bool:
        """Validate respiratory rate value"""
        validation = self.config['validation']['signos_vitales']['frecuencia_respiratoria']
        return validation['min'] <= value <= validation['max']

    def _validate_oxygen_saturation(self, value: float) -> bool:
        """Validate oxygen saturation value"""
        validation = self.config['validation']['signos_vitales']['saturacion_oxigeno']
        return validation['min'] <= value <= validation['max']

    def to_dict(self, result: ExtractionResult) -> Dict:
        """
        Convert extraction result to dictionary format

        Args:
            result: ExtractionResult object

        Returns:
            Dictionary with structured entity data
        """
        output = {
            'diagnosticos': [e.text for e in result.diagnosticos],
            'medicamentos': [],
            'procedimientos': [e.text for e in result.procedimientos],
            'fechas_importantes': {},
            'signos_vitales': {},
            'resultados_laboratorio': [],
            'metadata': result.metadata
        }

        # Format medications
        for med in result.medicamentos:
            med_dict = {
                'nombre': med.nombre,
                'texto_completo': med.text,
                'confidence': med.confidence
            }
            if med.dosis:
                med_dict['dosis'] = med.dosis
            if med.frecuencia:
                med_dict['frecuencia'] = med.frecuencia
            if med.duracion:
                med_dict['duracion'] = med.duracion

            output['medicamentos'].append(med_dict)

        # Format vital signs
        for name, vital in result.signos_vitales.items():
            output['signos_vitales'][name] = {
                'valor': vital.valor,
                'unidad': vital.unidad,
                'confidence': vital.confidence
            }

        # Format lab results
        for lab in result.resultados_laboratorio:
            lab_dict = {
                'nombre': lab.nombre,
                'valor': lab.valor,
                'unidad': lab.unidad,
                'confidence': lab.confidence
            }
            if lab.rango_min is not None and lab.rango_max is not None:
                lab_dict['rango_normal'] = f"{lab.rango_min}-{lab.rango_max}"

            output['resultados_laboratorio'].append(lab_dict)

        # Group dates by type
        for date in result.fechas_importantes:
            date_type = date.metadata.get('date_type', 'fecha')
            if date_type not in output['fechas_importantes']:
                output['fechas_importantes'][date_type] = []
            output['fechas_importantes'][date_type].append({
                'fecha': date.text,
                'confidence': date.confidence
            })

        return output

    def to_json(self, result: ExtractionResult) -> str:
        """
        Convert extraction result to JSON string

        Args:
            result: ExtractionResult object

        Returns:
            JSON string
        """
        import json
        return json.dumps(self.to_dict(result), ensure_ascii=False, indent=2)


# Convenience function for quick extraction
def extract_medical_entities(text: str, config_path: str = "medical_ner_config.yaml") -> Dict:
    """
    Quick extraction function

    Args:
        text: OCR text from medical document
        config_path: Path to config file

    Returns:
        Dictionary with extracted entities
    """
    extractor = MedicalEntityExtractor(config_path)
    result = extractor.extract_entities(text)
    return extractor.to_dict(result)


if __name__ == '__main__':
    # Example usage
    sample_text = """
    HISTORIA CLÍNICA

    Paciente: Juan Pérez
    Edad: 55 años
    Fecha de ingreso: 15/11/2025

    Diagnósticos:
    - Diabetes Mellitus Tipo 2
    - Hipertensión Arterial

    Medicamentos:
    - Metformina 850mg cada 12 horas
    - Losartán 50mg 1 vez al día
    - Atorvastatina 20mg en la noche

    Signos vitales:
    PA: 140/90 mmHg
    FC: 85 lpm
    Temp: 36.5°C
    FR: 18 rpm
    SpO2: 98%

    Laboratorio:
    Hemoglobina: 13.5 g/dL (12-16)
    Glucosa: 145 mg/dL (70-100)
    Creatinina: 1.1 mg/dL (0.6-1.2)

    Procedimientos:
    - Tomografía computarizada de abdomen
    """

    print("=== Medical Entity Extraction Demo ===\n")
    print("Input text:")
    print(sample_text)
    print("\n" + "="*80 + "\n")

    extractor = MedicalEntityExtractor()
    result = extractor.extract_entities(sample_text)

    print("Extracted entities (JSON):")
    print(extractor.to_json(result))
