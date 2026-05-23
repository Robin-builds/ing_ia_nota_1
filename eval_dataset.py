"""
evaluation/eval_dataset.py
--------------------------
Dataset sintético de evaluación para el pipeline RAG de IACC.
Contiene 5 pares pregunta / ground_truth / contexto esperado,
basados en los casos de consulta documentados en el análisis del caso.
"""

# ── Dataset de evaluación ─────────────────────────────────────────────────
# Cada entrada representa un caso real de consulta de un analista IACC.
# ground_truth: respuesta esperada correcta según reglas institucionales.
# contexts: fragmentos que el retriever DEBE encontrar para responder bien.

EVAL_DATASET = [
    {
        "question": "¿Cuántos leads matricularon esta semana por ejecutivo?",
        "ground_truth": (
            "Tabla con columnas: Ejecutivo, Equipo, Leads Asignados, Matrículas, "
            "Tasa de Conversión (%). "
            "La tasa se calcula como Matrículas / Leads Asignados × 100. "
            "El numerador usa Date entered 'matriculado', no la fecha de creación del contacto. "
            "La fuente debe ser citada como [HubSpot - export YYYY-MM-DD]."
        ),
        "reference_context": (
            "El campo correcto para contar matrículas es Date entered 'matriculado'. "
            "Fórmula: Tasa de Conversión = Matrículas / Leads Asignados × 100."
        ),
    },
    {
        "question": "¿Qué se considera un lead activo en IACC?",
        "ground_truth": (
            "Leads activos en IACC: En gestión, Interesado, Volver a llamar, "
            "Intento de contacto, Postulante, Nuevo, En curso, Conectado, "
            "Mal momento, Sin calificar. "
            "Leads inactivos: No desea contacto, Inubicable, Sin interés, No califica."
        ),
        "reference_context": (
            "Clasificación propia de IACC. "
            "Activos: En gestión, Interesado, Volver a llamar, Intento de contacto, "
            "Postulante, Nuevo, En curso, Conectado, Mal momento, Sin calificar. "
            "Inactivos: No desea contacto, Inubicable, Sin interés, No califica."
        ),
    },
    {
        "question": "¿Cómo va el equipo Grupo Pro en el cumplimiento de su meta mensual?",
        "ground_truth": (
            "Tabla con: Ejecutivo, Matrículas reales, Meta asignada, "
            "Cumplimiento (%). "
            "Fórmula: Cumplimiento = Matrículas reales / Meta asignada × 100. "
            "El equipo debe ser referenciado como 'Grupo Pro', no 'Grupo Regular'."
        ),
        "reference_context": (
            "Cumplimiento de Meta = Matrículas reales / Meta asignada × 100. "
            "Los equipos se denominan Grupo Aleatorio y Grupo Pro. "
            "Nunca usar el término Grupo Regular."
        ),
    },
    {
        "question": "¿Cuántos leads del mes están sin gestionar?",
        "ground_truth": (
            "Conteo de leads en estados activos sin interacción registrada, "
            "agrupados por equipo. "
            "El agente debe advertir que pueden existir gestiones no reflejadas "
            "en HubSpot si se realizaron por canales externos."
        ),
        "reference_context": (
            "Leads sin gestionar: leads en estado activo sin interacción registrada. "
            "Advertencia: pueden existir gestiones no capturadas en HubSpot."
        ),
    },
    {
        "question": "¿Qué carrera está convirtiendo más este mes?",
        "ground_truth": (
            "Ranking de programas de interés por tasa de conversión. "
            "Fórmula: Matrículas / Leads Asignados × 100, agrupado por programa. "
            "Si el período es menor a 5 días hábiles, advertir inestabilidad estadística."
        ),
        "reference_context": (
            "Tasa de conversión por programa = Matrículas / Leads Asignados × 100. "
            "Si el período analizado es menor a 5 días hábiles, el dato es estadísticamente inestable."
        ),
    },
]
