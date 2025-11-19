#!/usr/bin/env python3
"""
Medical Entity Extraction Example
Demonstrates how to extract medical entities from Spanish clinical documents

This example shows:
1. Basic entity extraction from OCR text
2. Extracting specific entity types
3. Working with extracted entities
4. Integration with batch processing
"""

import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from entity_extractor import MedicalEntityExtractor, extract_medical_entities


# Example 1: Historia Clínica (Medical History)
HISTORIA_CLINICA = """
HOSPITAL NACIONAL EDGARDO REBAGLIATI MARTINS
HISTORIA CLÍNICA

Paciente: Juan Carlos Pérez González
Edad: 55 años
Sexo: Masculino
Fecha de ingreso: 15/11/2025
Fecha de alta: 20/11/2025

DIAGNÓSTICOS:
1. Diabetes Mellitus Tipo 2 descompensada
2. Hipertensión Arterial estadio 2
3. Insuficiencia Renal Crónica estadio 3

MEDICAMENTOS AL INGRESO:
- Metformina 850mg cada 12 horas vía oral
- Losartán 50mg 1 vez al día vía oral
- Atorvastatina 20mg en la noche vía oral
- Omeprazol 20mg cada 24 horas vía oral

SIGNOS VITALES AL INGRESO:
PA: 160/95 mmHg
FC: 92 lpm
Temp: 36.8°C
FR: 20 rpm
SpO2: 96%
Peso: 85 kg
Talla: 1.70 m

EXÁMENES DE LABORATORIO:
Hemoglobina: 12.5 g/dL (12-16)
Hematocrito: 38% (37-47)
Leucocitos: 8500 /mm3 (4000-11000)
Glucosa: 245 mg/dL (70-100)
Creatinina: 1.8 mg/dL (0.6-1.2)
Urea: 65 mg/dL (15-40)
HbA1c: 9.2% (4-6)
Colesterol total: 220 mg/dL (< 200)
Triglicéridos: 185 mg/dL (< 150)

PROCEDIMIENTOS REALIZADOS:
- Tomografía computarizada de abdomen (16/11/2025)
- Ecografía renal bilateral (17/11/2025)

EVOLUCIÓN:
Paciente ingresa por descompensación metabólica. Se inicia tratamiento con
insulina NPH 12 UI cada 12 horas. Control estricto de glucemia. Mejora
progresiva de función renal con hidratación.

TRATAMIENTO AL ALTA:
- Insulina NPH 12 UI cada 12 horas subcutánea
- Metformina 500mg cada 12 horas vía oral
- Losartán 100mg 1 vez al día vía oral
- Atorvastatina 40mg en la noche vía oral
- Omeprazol 20mg cada 24 horas vía oral

RECOMENDACIONES:
- Control ambulatorio en 7 días
- Dieta diabética 1800 kcal/día
- Control de glucemia capilar 3 veces al día
"""


# Example 2: Resultado de Laboratorio
RESULTADO_LAB = """
LABORATORIO CLÍNICO SAN JOSÉ

Paciente: María Elena Torres Vega
Fecha: 18/11/2025

HEMOGRAMA COMPLETO:
Hemoglobina: 13.5 g/dL (12-16)
Hematocrito: 40% (37-47)
Leucocitos: 7200 /mm3 (4000-11000)
Neutrófilos: 65% (40-75)
Linfocitos: 28% (20-40)
Plaquetas: 250000 /mm3 (150000-400000)

PERFIL BIOQUÍMICO:
Glucosa: 98 mg/dL (70-100)
Creatinina: 0.9 mg/dL (0.6-1.2)
Urea: 28 mg/dL (15-40)
Ácido úrico: 5.2 mg/dL (2.4-6.0)
Colesterol total: 180 mg/dL (< 200)
HDL: 55 mg/dL (> 40)
LDL: 105 mg/dL (< 130)
Triglicéridos: 100 mg/dL (< 150)

FUNCIÓN HEPÁTICA:
TGO: 28 U/L (5-40)
TGP: 32 U/L (5-40)
Bilirrubina total: 0.8 mg/dL (0.3-1.2)
Fosfatasa alcalina: 85 U/L (40-150)
"""


# Example 3: Receta Médica
RECETA = """
Dr. Roberto Sánchez Flores
CMP: 12345
Medicina Interna

Fecha: 19/11/2025

Paciente: Ana Lucía Mendoza Ríos

Rp/:
1. Amoxicilina 500mg cada 8 horas por 7 días vía oral
2. Ibuprofeno 400mg cada 8 horas por 5 días vía oral (si hay dolor)
3. Paracetamol 500mg cada 6 horas por fiebre >38°C vía oral

Indicaciones:
- Tomar con alimentos
- Completar tratamiento antibiótico
- Control en 7 días si no hay mejoría
"""


def print_separator(title: str = ""):
    """Print a formatted separator"""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)
    print()


def example_1_basic_extraction():
    """Example 1: Basic entity extraction"""
    print_separator("EJEMPLO 1: Extracción Básica de Entidades")

    print("Texto de entrada (Historia Clínica):")
    print("-" * 80)
    print(HISTORIA_CLINICA[:300] + "...\n")

    # Extract entities
    print("Extrayendo entidades médicas...")
    extractor = MedicalEntityExtractor()
    result = extractor.extract_entities(HISTORIA_CLINICA)

    # Convert to dictionary
    entities = extractor.to_dict(result)

    print("\n📊 ENTIDADES EXTRAÍDAS:\n")

    # Print diagnoses
    if entities['diagnosticos']:
        print("🔍 DIAGNÓSTICOS:")
        for diag in entities['diagnosticos']:
            print(f"  • {diag}")
        print()

    # Print medications
    if entities['medicamentos']:
        print("💊 MEDICAMENTOS:")
        for med in entities['medicamentos']:
            print(f"  • {med['nombre']}")
            if med.get('dosis'):
                print(f"    Dosis: {med['dosis']}")
            if med.get('frecuencia'):
                print(f"    Frecuencia: {med['frecuencia']}")
        print()

    # Print vital signs
    if entities['signos_vitales']:
        print("🫀 SIGNOS VITALES:")
        for name, vital in entities['signos_vitales'].items():
            print(f"  • {vital['valor']} {vital['unidad']} (Confianza: {vital['confidence']:.0%})")
        print()

    # Print lab results
    if entities['resultados_laboratorio']:
        print("🧪 LABORATORIO:")
        for lab in entities['resultados_laboratorio'][:5]:  # First 5
            rango = lab.get('rango_normal', 'N/A')
            print(f"  • {lab['nombre']}: {lab['valor']} {lab['unidad']} (Rango: {rango})")
        print()

    # Print metadata
    print(f"📈 METADATA:")
    print(f"  Total entidades: {entities['metadata']['total_entities']}")
    print(f"  Método: {entities['metadata']['extraction_method']}")
    print(f"  Longitud texto: {entities['metadata']['text_length']} caracteres")


def example_2_lab_results():
    """Example 2: Extract lab results"""
    print_separator("EJEMPLO 2: Resultados de Laboratorio")

    extractor = MedicalEntityExtractor()
    result = extractor.extract_entities(RESULTADO_LAB)
    entities = extractor.to_dict(result)

    print("🧪 RESULTADOS DE LABORATORIO EXTRAÍDOS:\n")

    for lab in entities['resultados_laboratorio']:
        nombre = lab['nombre']
        valor = lab['valor']
        unidad = lab['unidad']
        rango = lab.get('rango_normal', 'N/A')

        print(f"{nombre:25s} {valor:>8s} {unidad:12s} (Normal: {rango})")

    print(f"\n✅ Total: {len(entities['resultados_laboratorio'])} resultados extraídos")


def example_3_medications():
    """Example 3: Extract medications with details"""
    print_separator("EJEMPLO 3: Extracción de Medicamentos")

    extractor = MedicalEntityExtractor()
    result = extractor.extract_entities(RECETA)
    entities = extractor.to_dict(result)

    print("💊 MEDICAMENTOS RECETADOS:\n")

    for i, med in enumerate(entities['medicamentos'], 1):
        print(f"{i}. {med['nombre']}")
        if med.get('dosis'):
            print(f"   Dosis: {med['dosis']}")
        if med.get('frecuencia'):
            print(f"   Frecuencia: {med['frecuencia']}")
        if med.get('duracion'):
            print(f"   Duración: {med['duracion']}")
        print(f"   Confianza: {med['confidence']:.0%}")
        print()


def example_4_json_output():
    """Example 4: JSON output for integration"""
    print_separator("EJEMPLO 4: Salida JSON para Integración")

    extractor = MedicalEntityExtractor()
    result = extractor.extract_entities(HISTORIA_CLINICA)

    # Get JSON string
    json_output = extractor.to_json(result)

    print("📄 Salida JSON (primeros 1000 caracteres):\n")
    print(json_output[:1000] + "...")

    # Save to file
    output_file = "extracted_entities.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_output)

    print(f"\n✅ JSON completo guardado en: {output_file}")


def example_5_batch_integration():
    """Example 5: Integration with batch processing"""
    print_separator("EJEMPLO 5: Integración con Procesamiento por Lotes")

    print("""
Para usar extracción de entidades con procesamiento por lotes:

1. Habilitar en batch_config.yaml:

   advanced:
     enable_medical_ner: true

2. Ejecutar batch processing:

   python batch_process.py \\
       --input ./medical_docs \\
       --output ./processed \\
       --preset historia_clinica

3. Verificar salida:

   processed/
   ├── documento1/
   │   ├── result.mmd         # OCR text
   │   ├── metadata.json      # Processing metadata
   │   ├── entities.json      # ← Extracted medical entities
   │   └── result_with_boxes.jpg

4. Leer entidades extraídas:

   import json
   with open('processed/documento1/entities.json') as f:
       entities = json.load(f)

   print(f"Diagnósticos: {entities['diagnosticos']}")
   print(f"Medicamentos: {entities['medicamentos']}")
   print(f"Signos vitales: {entities['signos_vitales']}")
    """)


def main():
    """Run all examples"""
    print_separator("DEMOSTRACIÓN: Extracción de Entidades Médicas en Español")

    print("""
Este script demuestra la extracción de entidades médicas de documentos clínicos
en español usando el sistema DeepSeek-OCR + Medical NER.

Entidades extraídas:
  • Diagnósticos (enfermedades, condiciones)
  • Medicamentos (nombre, dosis, frecuencia, duración)
  • Signos vitales (PA, FC, Temp, FR, SpO2, etc.)
  • Resultados de laboratorio (valores, unidades, rangos)
  • Fechas importantes (ingreso, alta, procedimientos)
  • Procedimientos médicos realizados
    """)

    input("\nPresiona Enter para continuar...")

    # Run examples
    example_1_basic_extraction()
    input("\nPresiona Enter para continuar...")

    example_2_lab_results()
    input("\nPresiona Enter para continuar...")

    example_3_medications()
    input("\nPresiona Enter para continuar...")

    example_4_json_output()
    input("\nPresiona Enter para continuar...")

    example_5_batch_integration()

    print_separator("FIN DE LA DEMOSTRACIÓN")

    print("""
Para más información, consulta:
  • README_NER.md - Documentación completa
  • medical_ner_config.yaml - Configuración de entidades
  • backend/entity_extractor.py - Código fuente
    """)


if __name__ == '__main__':
    main()
