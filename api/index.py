import os
import json
import logging
import asyncio
from typing import Dict, Any

# Dependencias de Llama Index
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response.schema import Response

# 1. Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE LLAMA INDEX Y CARGA DEL ÍNDICE ---

# Define la ruta donde GitHub Actions dejó los archivos del índice
STORAGE_DIR = "./storage"
INDEX: Any = None
QUERY_ENGINE: BaseQueryEngine = None

def initialize_index():
    """
    Inicializa y carga el índice de Llama Index desde la carpeta 'storage/'.
    Esta función se ejecuta una vez al iniciar el servidor.
    """
    global INDEX, QUERY_ENGINE
    
    # 1. Configurar LLM (debe usar la clave de entorno configurada en GitHub Secrets/Vercel)
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("La variable de entorno OPENAI_API_KEY no está configurada.")
        raise EnvironmentError("OPENAI_API_KEY es requerida.")

    Settings.llm = OpenAI(model="gpt-4-turbo", temperature=0.1)
    
    logger.info("Intentando cargar el índice desde: %s", STORAGE_DIR)

    try:
        # 2. Verificar si la carpeta de índice existe.
        if not os.path.exists(STORAGE_DIR):
            logger.error(
                "Error: La carpeta de índice '%s' no fue encontrada. Asegúrese de que el GitHub Action se ejecutó y subió la carpeta 'storage/' al repositorio.",
                STORAGE_DIR
            )
            # Esto forzará el fallo del deploy de Vercel si el índice no está listo
            raise FileNotFoundError(
                "Índice de Llama Index no encontrado. Falta la carpeta 'storage/'."
            )
            
        # 3. Cargar el índice desde la ruta existente.
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        INDEX = load_index_from_storage(storage_context)
        
        # 4. Crear el motor de consulta (Query Engine)
        QUERY_ENGINE = INDEX.as_query_engine()
        
        logger.info("✅ Índice de datos cargado y Query Engine inicializado exitosamente.")

    except Exception as e:
        logger.error("❌ Error CRÍTICO al cargar el índice: %s", e)
        # Es vital levantar la excepción para que el servidor no inicie con un índice roto
        raise e 

# Ejecuta la inicialización al inicio del script
try:
    initialize_index()
except Exception:
    # Si la inicialización falla, el script terminará y el deploy de Vercel fallará
    logger.critical("Fallo al inicializar el servidor debido a error en el índice.")
    pass

# --- FUNCIÓN HANDLER PRINCIPAL (para Vercel/servidor sin FastAPI/Flask) ---

async def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Función principal de entrada para la API de Vercel.
    Procesa solicitudes HTTP POST con el formato { "query": "..." }.
    """
    # 1. Comprobación del motor de consulta
    if QUERY_ENGINE is None:
        return {
            "statusCode": 503,
            "body": json.dumps({"error": "Servicio no disponible. El índice falló al cargar."}),
            "headers": {"Content-Type": "application/json"},
        }

    try:
        # 2. Parsear el cuerpo de la solicitud
        if event.get('body'):
            body_data = json.loads(event['body'])
            query = body_data.get('query', '')
        else:
            query = event.get('query', '') # Soporte básico para query params si es necesario

        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Falta el parámetro 'query'."}),
                "headers": {"Content-Type": "application/json"},
            }
        
        logger.info("Consulta recibida: %s", query)
        
        # 3. Ejecutar la consulta de manera asíncrona
        # Nota: Vercel a menudo requiere que las tareas de red/IO se manejen con async
        response: Response = await asyncio.to_thread(QUERY_ENGINE.query, query)
        
        # 4. Construir la respuesta
        result = {
            "response": str(response),
            "source_nodes": [
                {
                    "text": node.text.split("...")[0] + "...", # Mostrar solo el inicio
                    "score": float(node.score),
                } 
                for node in response.source_nodes
            ]
        }

        return {
            "statusCode": 200,
            "body": json.dumps(result),
            "headers": {"Content-Type": "application/json"},
        }

    except Exception as e:
        logger.exception("Error durante la ejecución de la consulta.")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Error interno del servidor: {str(e)}"}),
            "headers": {"Content-Type": "application/json"},
        }

# Código de prueba para ejecución local (opcional)
if __name__ == '__main__':
    logger.info("Ejecutando la API localmente...")
    
    # Simulación de una consulta (debes manejar la clave de entorno localmente)
    test_event = {
        'body': json.dumps({"query": "¿Cuáles son los requisitos para la solicitud de asilo?"}),
        'context': {}
    }
    
    # Ejecutar el handler (requiere que el índice exista localmente)
    try:
        response = asyncio.run(handler(test_event, None))
        print("\n--- Respuesta de Prueba ---\n")
        print(response['body'])
    except Exception as e:
        print(f"\n--- Error de Prueba ---\nFallo al correr la prueba. Asegúrate de tener la carpeta 'storage/' y la clave OpenAI configuradas localmente. Error: {e}")
