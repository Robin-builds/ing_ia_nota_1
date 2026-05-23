"""
src/agent.py
------------
Agente principal IACC Analytics Assistant.
Implementa el pipeline RAG completo: retrieval → ensamblado de prompt → generación.

Usa GitHub Models API (gpt-4o) via cliente OpenAI compatible.
Requiere: GITHUB_TOKEN en .env
"""

import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS

from src.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from src.indexer import cargar_indice, obtener_retriever


class IACCAnalyticsAgent:
    """
    Agente de análisis de gestión comercial para IACC.

    Flujo por consulta:
    1. Recibe consulta en lenguaje natural
    2. Recupera chunks semánticamente relevantes desde FAISS
    3. Ensambla prompt con contexto recuperado
    4. Genera respuesta via GitHub Models API (gpt-4o)
    5. Retorna respuesta con fuentes citadas
    """

    def __init__(self, index_path: str, k: int = 4):
        """
        Inicializa el agente cargando el índice FAISS y el cliente OpenAI (GitHub Models).

        Args:
            index_path: Ruta al índice FAISS persistido.
            k: Número de chunks a recuperar por consulta.
        """
        # Cargar índice FAISS y configurar retriever
        self.vectorstore = cargar_indice(index_path)
        self.retriever = obtener_retriever(self.vectorstore, k=k)
        self.k = k

        # Inicializar cliente OpenAI apuntando a GitHub Models API
        # Compatible con la misma interfaz que OpenAI, sin costo adicional
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_TOKEN"),
        )
        self.model = os.getenv("LLM_MODEL", "gpt-4o")

        # Historial de conversación para contexto multi-turno
        self.historial: list[dict] = []

        print(f"[agent] IACC Analytics Assistant listo — modelo: {self.model}")

    def _recuperar_contexto(self, consulta: str) -> tuple[str, list]:
        """
        Ejecuta la búsqueda semántica en FAISS y formatea el contexto recuperado.

        Args:
            consulta: Pregunta del usuario en lenguaje natural.

        Returns:
            Tupla (texto_contexto, lista_documentos_recuperados).
        """
        # Para consultas que mencionan múltiples ejecutivos, ampliar k
        k_dinamico = self.k * 2 if any(
            palabra in consulta.lower()
            for palabra in ["todos", "cada", "equipo completo", "todos los ejecutivos"]
        ) else self.k

        # Recuperar chunks más similares semánticamente
        documentos = self.vectorstore.similarity_search(
            consulta, k=k_dinamico
        )

        # Formatear contexto con metadata de trazabilidad
        fragmentos = []
        for i, doc in enumerate(documentos, 1):
            fuente = doc.metadata.get("fuente", "Desconocida")
            fecha = doc.metadata.get("fecha_export", "Sin fecha")
            fragmentos.append(
                f"[Fragmento {i} — {fuente} {fecha}]\n{doc.page_content}"
            )

        contexto_texto = "\n\n".join(fragmentos)
        return contexto_texto, documentos

    def consultar(self, pregunta: str) -> str:
        """
        Procesa una consulta del usuario y retorna la respuesta del agente.

        Args:
            pregunta: Consulta en lenguaje natural del analista IACC.

        Returns:
            Respuesta generada por el LLM con fuentes citadas.
        """
        # Paso 1: Recuperar contexto relevante desde FAISS
        contexto, docs_recuperados = self._recuperar_contexto(pregunta)

        # Paso 2: Ensamblar prompt de usuario con contexto RAG
        prompt_usuario = RAG_PROMPT_TEMPLATE.format(
            contexto_recuperado=contexto,
            consulta_usuario=pregunta,
        )

        # Paso 3: Agregar al historial de conversación
        self.historial.append({"role": "user", "content": prompt_usuario})

        # Paso 4: Llamar a GitHub Models API con system prompt + historial
        # Interfaz idéntica a OpenAI Chat Completions
        respuesta = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.historial,
            ],
        )

        # Extraer texto de respuesta
        texto_respuesta = respuesta.choices[0].message.content

        # Paso 5: Guardar respuesta en historial para contexto multi-turno
        self.historial.append({"role": "assistant", "content": texto_respuesta})

        return texto_respuesta

    def reiniciar_sesion(self):
        """Limpia el historial de conversación para iniciar nueva sesión."""
        self.historial = []
        print("[agent] Sesión reiniciada.")
