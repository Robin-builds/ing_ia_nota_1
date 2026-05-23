"""
src/build_index.py
------------------
Script de indexación offline.
Carga datos HubSpot normalizados + documentos institucionales,
construye el índice FAISS y lo persiste en disco.

Uso (desde la raíz del proyecto):
    python -m src.build_index

Prerequisitos:
    - Archivo CSV normalizado en data/processed/hubspot_normalized.csv
    - Documentos institucionales en data/knowledge_base/
    - Variables de entorno configuradas en .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.ingest import cargar_hubspot_csv, cargar_documentos_institucionales
from src.indexer import construir_indice

# Cargar variables de entorno desde .env
load_dotenv()


def main():
    # ── Rutas desde variables de entorno ──────────────────────────────────
    processed_path = os.getenv("DATA_PROCESSED_PATH", "data/processed")
    knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ── Paso 1: Cargar datos HubSpot normalizados ──────────────────────────
    hubspot_csv = Path(processed_path) / "hubspot_normalized.csv"

    if not hubspot_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {hubspot_csv}\n"
            "Exporta y normaliza los datos de HubSpot antes de ejecutar este script."
        )

    # Fecha del export (usar nombre del archivo o ingresar manualmente)
    fecha_export = input("Ingresa la fecha del export HubSpot (YYYY-MM-DD): ").strip()

    documentos_hubspot = cargar_hubspot_csv(str(hubspot_csv), fecha_export)

    # ── Paso 2: Cargar documentos institucionales ──────────────────────────
    documentos_institucionales = cargar_documentos_institucionales(knowledge_base_path)

    # ── Paso 3: Unificar todos los documentos ─────────────────────────────
    todos_los_documentos = documentos_hubspot + documentos_institucionales
    print(f"\n[build_index] Total documentos a indexar: {len(todos_los_documentos)}")

    # ── Paso 4: Construir y persistir índice FAISS ─────────────────────────
    construir_indice(
        documentos=todos_los_documentos,
        index_path=faiss_index_path,
        embedding_model=embedding_model,
    )

    print("\n✅ Índice construido exitosamente.")
    print(f"   Documentos indexados: {len(todos_los_documentos)}")
    print(f"   Ubicación del índice: {faiss_index_path}")


if __name__ == "__main__":
    main()
