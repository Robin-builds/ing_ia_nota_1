# IACC Analytics Assistant - 
**Agente de análisis de gestión comercial con LLM + RAG**  
Evaluación Parcial N°1 — ISY0101 Ingeniería de Soluciones con IA — DuocUC 2026 - Robinson Arriagada Borquez

---

## Descripción

Sistema de consulta en lenguaje natural sobre datos de gestión comercial del Instituto Profesional IACC. Permite a analistas y supervisores obtener reportes de conversión, cumplimiento de metas y análisis de leads sin necesidad de conocimientos técnicos, en menos de 5 minutos.

**Stack tecnológico:**
- LangChain — orquestación del agente
- FAISS — base de datos vectorial local
- GitHub Models API (gpt-4o) — generación de respuestas (gratuito)
- sentence-transformers (all-MiniLM-L6-v2) — embeddings 100% locales
- RAGAS — evaluación del pipeline RAG
- Python 3.11

---

## Estructura del repositorio

```
iacc-analytics-assistant/
├── main.py                        # Punto de entrada (modo consola)
├── requirements.txt               # Dependencias
├── .env                           # Plantilla de variables de entorno
├── src/
│   ├── agent.py                   # Agente principal (pipeline completo)
│   ├── ingest.py                  # Ingesta y normalización de datos CSV
│   ├── indexer.py                 # Construcción y carga del índice FAISS
│   ├── prompts.py                 # System prompt y templates RAG
│   └── build_index.py             # Script de indexación offline
├── evaluation/
│   ├── eval_dataset.py            # Dataset sintético (5 casos IACC)
│   └── run_evaluation.py          # Evaluación RAGAS
└── data/
    ├── raw/                       # CSVs originales exportados de HubSpot
    ├── processed/                 # CSVs normalizados (post Excel)
    ├── knowledge_base/            # Glosario y docs institucionales (.md)
    └── faiss_index/               # Índice FAISS persistido (generado)
```

---

## Requisitos previos

- Python 3.11+
- Cuenta GitHub (para GitHub Models API — gratuita)
- Token GitHub: https://github.com/settings/tokens

> **Sin costo**: El proyecto usa GitHub Models API (gpt-4o gratuito) y
> embeddings locales con sentence-transformers. No requiere OpenAI ni Anthropic.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Robin-builds/ing_ia_nota_1.git

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env
---

## Preparación de datos

### Datos HubSpot (obligatorio)

1. Exportar datos desde HubSpot CRM en formato CSV
2. Normalizar en Excel aplicando los siguientes ajustes:
   - Renombrar columnas al esquema unificado (ver tabla abajo)
   - Estandarizar nombres con `PROPER()` para consistencia
   - Convertir fechas a formato `YYYY-MM-DD`
   - Guardar como `data/processed/hubspot_normalized.csv`

**Esquema de columnas requerido en el CSV normalizado:**

| Columna | Descripción |
|---------|-------------|
| `periodo` | Semana o período (ej: `2025-W18`) |
| `ejecutivo` | Nombre del propietario del contacto |
| `equipo` | `Grupo Aleatorio` o `Grupo Pro` |
| `leads_asignados` | Cantidad de leads asignados en el período |
| `matriculas` | Conteo usando *Date entered "matriculado"* |

### Base de conocimiento institucional (recomendado)

---

## Ejecución

### Paso 1 — Construir el índice FAISS

```bash
python -m src.build_index
```
### Paso 2 — Iniciar el agente

```bash
python main.py
```

Ejemplo de sesión:

```
Analista > ¿Cuántos leads matricularon esta semana por ejecutivo?

Agente > | Ejecutivo      | Equipo          | Leads | Matrículas | Conversión |
         |----------------|-----------------|-------|------------|------------|
         | Ana Torres     | Grupo Pro       |   48  |     3      |   6.25%    |
         | Pedro Soto     | Grupo Aleatorio |   61  |     2      |   3.28%    |
         | María Vega     | Grupo Pro       |   39  |     4      |  10.26%    |
         
         Fuente: [HubSpot - export 2025-05-01]
```

### Paso 3 — Evaluar el pipeline (opcional)

```bash
python -m evaluation.run_evaluation
```

Genera un reporte RAGAS con las métricas de calidad del sistema.

---

## Evaluación RAGAS

| Métrica | Umbral mínimo | Descripción |
|---------|--------------|-------------|
| **Faithfulness**     | ≥ 0.85 | El LLM no inventa datos ausentes en el contexto |
| Context Precision    |    —   | Los chunks recuperados son relevantes           |
| Context Recall       |    —   | Se recuperaron todos los chunks necesarios      |
| Answer Relevancy     |    —   | La respuesta responde la pregunta formulada     |

---

## Restricciones de privacidad

Este sistema cumple con la **Ley N° 19.628** sobre Protección de la Vida Privada:

- Los embeddings se generan **100% localmente** (sentence-transformers)
- El índice FAISS se almacena **localmente** (no en servicios cloud)
- Las consultas al LLM usan **datos agregados y anonimizados**
- Datos nominales de estudiantes (RUT, nombre, contacto) **nunca se indexan ni transmiten**

---

## Declaración de uso de IA

En el desarrollo de este proyecto se utilizó Claude-code como apoyo para estructuración de código y ayuda a solucionar problemas tecnicos. Todas las decisiones de diseño, reglas de negocio y justificaciones son de autoría propia, como este documento.

Citación: https://bibliotecas.duoc.cl/ia
