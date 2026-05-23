"""
evaluation/run_evaluation.py
-----------------------------
Evaluación del pipeline RAG de IACC con el framework RAGAS.
Mide 4 métricas de calidad: Faithfulness, Context Precision,
Context Recall y Answer Relevancy.

Uso (desde la raíz del proyecto):
    python -m evaluation.run_evaluation

Prerequisitos:
    - Índice FAISS construido (python -m src.build_index)
    - Variables de entorno configuradas en .env
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)

from src.agent import IACCAnalyticsAgent
from src.indexer import cargar_indice, obtener_retriever
from evaluation.eval_dataset import EVAL_DATASET

# Cargar variables de entorno
load_dotenv()

# ── Umbral mínimo de aprobación ───────────────────────────────────────────
UMBRAL_FAITHFULNESS = 0.85


def ejecutar_evaluacion():
    print("=" * 60)
    print("  Evaluación RAGAS — IACC Analytics Assistant")
    print("=" * 60)

    # ── Inicializar agente ─────────────────────────────────────────────────
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    agente = IACCAnalyticsAgent(index_path=faiss_index_path, k=4)

    # ── Construir dataset de evaluación RAGAS ─────────────────────────────
    preguntas = []
    respuestas_generadas = []
    contextos_recuperados = []
    ground_truths = []

    print(f"\nProcesando {len(EVAL_DATASET)} casos de evaluación...\n")

    for i, caso in enumerate(EVAL_DATASET, 1):
        print(f"  [{i}/{len(EVAL_DATASET)}] {caso['question'][:60]}...")

        # Recuperar contexto desde FAISS
        contexto_texto, docs = agente._recuperar_contexto(caso["question"])
        contextos_como_lista = [doc.page_content for doc in docs]

        # Generar respuesta del agente
        agente.reiniciar_sesion()
        respuesta = agente.consultar(caso["question"])

        # Acumular para dataset RAGAS
        preguntas.append(caso["question"])
        respuestas_generadas.append(respuesta)
        contextos_recuperados.append(contextos_como_lista)
        ground_truths.append(caso["ground_truth"])

    # ── Crear Dataset compatible con RAGAS ────────────────────────────────
    ragas_dataset = Dataset.from_dict({
        "question": preguntas,
        "answer": respuestas_generadas,
        "contexts": contextos_recuperados,
        "ground_truth": ground_truths,
    })

    # ── Ejecutar evaluación RAGAS ──────────────────────────────────────────
    print("\nEjecutando evaluación RAGAS...")
    resultados = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
        ],
    )

    # ── Mostrar resultados ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTADOS DE EVALUACIÓN")
    print("=" * 60)

    metricas = {
        "faithfulness": resultados["faithfulness"],
        "context_precision": resultados["context_precision"],
        "context_recall": resultados["context_recall"],
        "answer_relevancy": resultados["answer_relevancy"],
    }

    for metrica, valor in metricas.items():
        estado = "✅" if valor >= UMBRAL_FAITHFULNESS else "⚠️"
        print(f"  {estado} {metrica:<25} {valor:.4f}")

    # Verificar umbral de Faithfulness
    print("\n" + "-" * 60)
    if metricas["faithfulness"] >= UMBRAL_FAITHFULNESS:
        print(f"  ✅ Faithfulness ≥ {UMBRAL_FAITHFULNESS} — Pipeline APROBADO")
    else:
        print(f"  ❌ Faithfulness < {UMBRAL_FAITHFULNESS} — Pipeline REQUIERE AJUSTE")
        print("     Revisar: chunks recuperados, system prompt, k del retriever.")

    # ── Persistir resultados en JSON ───────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("evaluation") / f"resultados_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "modelo": os.getenv("LLM_MODEL"),
                "n_casos": len(EVAL_DATASET),
                "umbral_faithfulness": UMBRAL_FAITHFULNESS,
                "metricas": metricas,
                "aprobado": metricas["faithfulness"] >= UMBRAL_FAITHFULNESS,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n  Resultados guardados en: {output_path}")
    return metricas


if __name__ == "__main__":
    ejecutar_evaluacion()
