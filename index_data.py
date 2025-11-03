import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.settings import Settings 
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv # Importado para uso local, aunque GitHub Actions usa variables de entorno

# --- RUTAS ---
# Directorio donde están tus documentos (ej: archivos PDF, DOCX)
DATA_DIR = "./data" 
# Directorio donde se guardará el índice creado. ¡DEBE coincidir con el workflow de GitHub!
INDEX_DIR = "./storage"

def configure_settings():
    """Configura los modelos y parámetros de Llama Index."""
    
    # 1. Carga de variables de entorno para pruebas locales
    # En GitHub Actions, esto no es necesario, pero ayuda a la ejecución manual.
    load_dotenv()
    
    # 2. Verificar la clave de OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: La variable de entorno OPENAI_API_KEY no está configurada. Saliendo.")
        raise EnvironmentError("OPENAI_API_KEY es requerida para la indexación.")

    # 3. Configurar el modelo de embeddings (¡Más barato y mejor!)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # 4. Configurar el LLM por defecto (usado para algunas operaciones de Llama Index)
    Settings.llm = OpenAI(model="gpt-4o") 
    
    # 5. Configurar el tamaño de chunking
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20

def build_knowledge_base():
    """
    Lee los documentos, construye el índice vectorial y lo guarda en disco.
    """
    
    print("========================================================")
    print("Iniciando el proceso de indexación de documentos.")
    print("========================================================")

    # 1. Configurar Modelos y Clave
    configure_settings()

    # 2. Cargar documentos
    print(f"Cargando documentos desde: {DATA_DIR}...")
    try:
        reader = SimpleDirectoryReader(input_dir=DATA_DIR, exclude_hidden=False)
        documents = reader.load_data()
    except Exception as e:
        print(f"ERROR al cargar documentos: {e}")
        # Si no hay documentos, el script debe fallar para que la Action lo reporte.
        if not os.listdir(DATA_DIR):
            print("El directorio 'data/' está vacío. Asegúrate de tener documentos allí.")
        raise
    
    if not documents:
        print("ADVERTENCIA: No se encontraron documentos válidos para indexar.")
        return

    # 3. Construir el índice
    print(f"Creando nuevo índice con {len(documents)} documento(s).")
    
    # Creamos el contexto de almacenamiento sin intentar cargarlo (porque lo vamos a crear)
    storage_context = StorageContext.from_defaults()

    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        show_progress=True 
    )

    # 4. Guardar el índice
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        
    index.storage_context.persist(persist_dir=INDEX_DIR)
    
    print("========================================================")
    print(f"✅ ÉXITO: Índice creado y guardado en la carpeta '{INDEX_DIR}'.")
    print("========================================================")

if __name__ == "__main__":
    try:
        build_knowledge_base()
    except Exception as e:
        print(f"\n❌ FALLO CRÍTICO EN LA INDEXACIÓN: {e}")
        # Forzar la salida con un código de error para que GitHub Action lo marque como fallo.
        exit(1)
