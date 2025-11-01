from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# Directorio donde se guarda el índice después de la pre-indexación local
INDEX_DIR = "./storage"

def load_knowledge_base():
    """
    Carga el índice persistente de la base de conocimiento (RAG).
    """
    try:
        # Intenta cargar un índice previamente guardado
        index = VectorStoreIndex.load_from_disk(INDEX_DIR)
        print("✅ Índice de conocimiento cargado desde el disco.")
        return index
    except Exception as e:
        # Si no existe, indica un error
        print(f"❌ No se pudo cargar el índice de conocimiento persistente: {e}")
        print("ERROR: Debe ejecutar el script de indexación localmente.")
        return None

def retrieve_context(index, query: str):
    """
    Realiza una búsqueda semántica para obtener el contexto relevante.
    """
    if index:
        # Obtiene los 3 fragmentos más relevantes para inyectar en el prompt
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        
        # Devolvemos solo el texto relevante
        return str(response)
    return ""