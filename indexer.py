"""
src/indexer.py
--------------
Construcción y gestión del índice FAISS para el pipeline RAG de IACC.
Maneja la vectorización de documentos y persistencia del índice local.

Usa sentence-transformers (all-MiniLM-L6-v2) para embeddings 100% locales.
Sin costo, sin API externa. Cumple restricción de procesamiento local (Ley 19.628).
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def _get_embeddings(embedding_model: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Inicializa el modelo de embeddings local con sentence-transformers.

    Usa all-MiniLM-L6-v2 por defecto:
    - 384 dimensiones, ~80MB de descarga única
    - Rendimiento competitivo con text-embedding-3-small en tareas en español
    - Procesamiento 100% local (sin llamadas a APIs externas)

    Args:
        embedding_model: Nombre del modelo HuggingFace a usar.

    Returns:
        Instancia HuggingFaceEmbeddings lista para vectorizar.
    """
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def construir_indice(
    documentos: list[Document],
    index_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> FAISS:
    """
    Vectoriza los documentos y construye el índice FAISS local.

    Args:
        documentos: Lista de Documents (HubSpot chunks + docs institucionales).
        index_path: Ruta donde persistir el índice.
        embedding_model: Modelo sentence-transformers a usar.

    Returns:
        Instancia FAISS lista para búsqueda semántica.
    """
    print(f"[indexer] Vectorizando {len(documentos)} documentos...")

    # Inicializar modelo de embeddings local
    embeddings = _get_embeddings(embedding_model)

    # Construir índice FAISS desde documentos
    vectorstore = FAISS.from_documents(documentos, embeddings)

    # Persistir índice para reutilización (evita re-vectorizar en cada ejecución)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_path)

    print(f"[indexer] Índice guardado en: {index_path}")
    return vectorstore


def cargar_indice(
    index_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> FAISS:
    """
    Carga un índice FAISS previamente construido desde disco.

    Args:
        index_path: Ruta al índice FAISS persistido.
        embedding_model: Debe coincidir con el modelo usado al construir.

    Returns:
        Instancia FAISS lista para búsqueda semántica.
    """
    embeddings = _get_embeddings(embedding_model)

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"No se encontró el índice FAISS en: {index_path}\n"
            "Ejecuta primero: python -m src.build_index"
        )

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # Requerido por LangChain >= 0.2
    )
    print(f"[indexer] Índice cargado desde: {index_path}")
    return vectorstore


def obtener_retriever(vectorstore: FAISS, k: int = 4):
    """
    Configura el retriever semántico sobre el índice FAISS.

    Args:
        vectorstore: Índice FAISS cargado.
        k: Número de chunks a recuperar por consulta.
            Usar k=4 para consultas simples, k=8 para consultas multi-ejecutivo.

    Returns:
        Retriever configurado para búsqueda por similitud.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
