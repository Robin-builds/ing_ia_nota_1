"""
main.py
-------
Punto de entrada principal del IACC Analytics Assistant.
Ejecuta el agente en modo conversacional por consola.

Uso:
    python main.py

Prerequisitos:
    - Índice FAISS construido (ejecutar primero: python src/build_index.py)
    - Variables de entorno configuradas en .env
"""

import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from src.agent import IACCAnalyticsAgent

# Cargar variables de entorno
load_dotenv(override=True)


def main():
    print("=" * 60)
    print("  IACC Analytics Assistant")
    print("  Agente de análisis de gestión comercial")
    print("=" * 60)
    print("Escribe tu consulta en lenguaje natural.")
    print("Comandos: 'salir' para terminar | 'nueva sesion' para reiniciar\n")

    # Inicializar agente con índice FAISS
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    k = int(os.getenv("RETRIEVER_K", "4"))

    agente = IACCAnalyticsAgent(index_path=faiss_index_path, k=k)

    # ── Loop de conversación ───────────────────────────────────────────────
    while True:
        try:
            consulta = input("\nAnalista > ").strip()

            # Comandos de control
            if consulta.lower() in ["salir", "exit", "quit"]:
                print("Sesión finalizada.")
                break

            if consulta.lower() == "nueva sesion":
                agente.reiniciar_sesion()
                continue

            if not consulta:
                continue

            # Procesar consulta y mostrar respuesta
            print("\nAgente > ", end="", flush=True)
            respuesta = agente.consultar(consulta)
            print(respuesta)

        except KeyboardInterrupt:
            print("\n\nSesión interrumpida.")
            break
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()
