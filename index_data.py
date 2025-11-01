from dotenv import load_dotenv 
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.settings import Settings 
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI 

# 1. CARGA DE VARIABLES DE ENTORNO
load_dotenv() 

DATA_DIR = "./data"
INDEX_DIR = "./storage"

def configure_settings():
    """Configura el nuevo sistema Settings de llama-index."""
    # Configurar el modelo de embeddings (¡Más barato y mejor!)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Configurar el LLM por defecto 
    Settings.llm = OpenAI(model="gpt-4o")
    
    # Opcional: Configurar el tamaño de chunking
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20

def build_knowledge_base():
    """
    Lee los documentos de la carpeta 'data' y construye el índice vectorial.
    Este método fuerza la creación desde cero.
    """
    print("Iniciando la lectura e indexación de documentos. Esto puede tardar...")

    # 1. Cargar documentos
    reader = SimpleDirectoryReader(input_dir=DATA_DIR, exclude_hidden=False)
    documents = reader.load_data()

    # 2. Configurar el contexto 
    configure_settings()

    # 3. Construir el índice (¡El proceso costoso!)
    print("Creando nuevo índice desde cero (¡ÚLTIMO INTENTO DE INDEXACIÓN!)")
    
    # Creamos el contexto de almacenamiento sin intentar cargar nada
    storage_context = StorageContext.from_defaults() 

    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, # Pasamos el contexto de almacenamiento
        show_progress=True 
    )

    # 4. Guardar el índice (MÉTODO FINAL Y ROBUSTO)
    # Creamos el directorio si no existe para evitar errores de permisos
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        
    index.storage_context.persist(persist_dir=INDEX_DIR)
    
    print("========================================================")
    print("✅ ÉXITO: Índice creado y guardado en la carpeta './storage'.")
    print("========================================================")

if __name__ == "__main__":
    build_knowledge_base()