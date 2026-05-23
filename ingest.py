"""
src/ingest.py
-------------
Módulo de ingesta y normalización de datos HubSpot CSV.
Convierte registros tabulares en chunks de texto para indexación en FAISS.
"""

import pandas as pd
from pathlib import Path
from langchain_core.documents import Document


# ── Plantilla de chunk para registros HubSpot ──────────────────────────────
HUBSPOT_CHUNK_TEMPLATE = """Período: {periodo}
Ejecutivo: {ejecutivo} | Equipo: {equipo}
Leads asignados: {leads} | Matrículas: {matriculas}
Tasa de conversión: {conversion:.2f}%
Fuente: HubSpot export {fecha_export}"""


def normalizar_nombre(nombre: str) -> str:
    """
    Estandariza nombres propios: capitaliza y elimina espacios extra.
    Resuelve inconsistencias de acentos y mayúsculas en exports de HubSpot.
    """
    if pd.isna(nombre):
        return "Sin asignar"
    return " ".join(str(nombre).strip().title().split())


def cargar_hubspot_csv(filepath: str, fecha_export: str) -> list[Document]:
    """
    Carga un archivo CSV normalizado de HubSpot y genera documentos LangChain.

    Args:
        filepath: Ruta al archivo CSV procesado (post normalización Excel).
        fecha_export: Fecha del export en formato YYYY-MM-DD.

    Returns:
        Lista de objetos Document listos para indexar en FAISS.

    Columnas requeridas en el CSV:
        - periodo: Semana o período del dato (ej: "2025-W18")
        - ejecutivo: Nombre del propietario del contacto
        - equipo: "Grupo Aleatorio" o "Grupo Pro"
        - leads_asignados: Cantidad de leads asignados en el período
        - matriculas: Conteo usando Date entered "matriculado"
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")

    # Estandarizar nombres para evitar duplicados por acentos/mayúsculas
    df["ejecutivo"] = df["ejecutivo"].apply(normalizar_nombre)
    df["equipo"] = df["equipo"].apply(normalizar_nombre)

    # Calcular tasa de conversión con fórmula institucional IACC
    # Fórmula: Matrículas / Leads Asignados × 100
    df["conversion"] = (
        df["matriculas"] / df["leads_asignados"].replace(0, float("nan")) * 100
    ).fillna(0.0)

    documentos = []
    for _, row in df.iterrows():
        # Construir texto del chunk con plantilla estructurada
        texto = HUBSPOT_CHUNK_TEMPLATE.format(
            periodo=row.get("periodo", "Sin período"),
            ejecutivo=row["ejecutivo"],
            equipo=row["equipo"],
            leads=int(row["leads_asignados"]),
            matriculas=int(row["matriculas"]),
            conversion=row["conversion"],
            fecha_export=fecha_export,
        )

        # Metadata para trazabilidad (fuente citada en respuestas)
        metadata = {
            "fuente": "HubSpot",
            "fecha_export": fecha_export,
            "ejecutivo": row["ejecutivo"],
            "equipo": row["equipo"],
            "periodo": row.get("periodo", "Sin período"),
        }

        documentos.append(Document(page_content=texto, metadata=metadata))

    print(f"[ingest] {len(documentos)} documentos generados desde {Path(filepath).name}")
    return documentos


def cargar_documentos_institucionales(knowledge_base_path: str) -> list[Document]:
    """
    Carga documentos institucionales (Markdown y PDF) para la base de conocimiento RAG.
    Incluye glosario de KPIs, criterios de clasificación de leads y reglamentos.

    Args:
        knowledge_base_path: Carpeta con archivos .md y .pdf institucionales.

    Returns:
        Lista de objetos Document fragmentados con RecursiveCharacterTextSplitter.
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Cargar archivos Markdown del knowledge base
    loader = DirectoryLoader(
        knowledge_base_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos_raw = loader.load()

    # Fragmentar con overlap para preservar contexto entre chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    documentos = splitter.split_documents(documentos_raw)

    print(
        f"[ingest] {len(documentos)} chunks generados desde documentos institucionales"
    )
    return documentos
