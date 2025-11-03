import os
import logging
import zipfile
import shutil
from pathlib import Path
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.text_splitter import TokenTextSplitter
from pinecone import Pinecone
import requests

# --- 1. CONFIGURACIÓN INICIAL ---
# Configurar Logger para ver el progreso en los logs de GitHub Actions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar el Embedder de OpenAI
# El modelo 'text-embedding-3-small' es el más económico y eficiente.
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Configurar el Chunking (Fragmentación)
# Usamos un tamaño grande (2048) para reducir el número total de vectores a subir.
Settings.text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=50)

# --- 2. CONSTANTES DE PINE CONE (LEÍDAS DE GITHUB SECRETS) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
DATA_URL = "http://mondoviaggi.ec/data.zip"
TEMP_DIR = Path("./temp_data")

# --- 3. FUNCIÓN PRINCIPAL DE INDEXACIÓN ---
def run_indexing():
    # Verificación de secretos cruciales
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME]):
        logger.error("ERROR: Faltan variables de entorno de Pinecone. Revisa los GitHub Secrets.")
        return

    # --- A. DESCARGAR Y PREPARAR DATOS ---
    logger.info(f"Descargando datos desde: {DATA_URL}")
    try:
        response = requests.get(DATA_URL, stream=True)
        if response.status_code == 200:
            zip_path = Path("data.zip")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Descarga completada. Extrayendo archivos...")
            
            # Crear directorio temporal
            TEMP_DIR.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_DIR)
            os.remove(zip_path) # Limpiar el ZIP descargado
        else:
            logger.error(f"ERROR: No se pudo descargar el ZIP. Código de estado: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"Error durante la descarga o extracción: {e}")
        return

    # --- B. CONEXIÓN E INDEXACIÓN A PINECONE ---
    
    # 1. Conexión a Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Nota: Llama Index maneja el 'environment' internamente con el host del índice
    except Exception as e:
        logger.error(f"Error al inicializar Pinecone: {e}")
        return

    # 2. Conexión al índice
    logger.info(f"Conectando al índice de Pinecone: {INDEX_NAME}...")
    
    # Intentar obtener el índice y verificar si está en estado READY
    if INDEX_NAME not in pc.list_indexes().names:
        logger.error(f"ERROR: El índice '{INDEX_NAME}' no existe o no está listo. Revisa tu panel de Pinecone.")
        return
    
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # 3. Cargar Documentos de la carpeta temporal
    logger.info(f"Cargando documentos de la carpeta: {TEMP_DIR}")
    documents = SimpleDirectoryReader(TEMP_DIR).load_data()
    
    # 4. Crear el índice y subir los vectores
    logger.info(f"Comenzando la indexación y subida de {len(documents)} documentos a Pinecone...")

    # Esto inicia el proceso que solía tomar más de 6 horas
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store, # Esto asegura que se sube a Pinecone, no a local
        show_progress=True
    )

    logger.info(f"Indexación completada. Los vectores han sido subidos a Pinecone.")

    # --- C. LIMPIEZA FINAL ---
    try:
        # Eliminar la carpeta temporal una vez que el proceso termina
        shutil.rmtree(TEMP_DIR)
        logger.info("Limpieza completada. Directorio temporal eliminado.")
    except Exception as e:
        logger.error(f"Error durante la limpieza: {e}")

# Ejecutar la función principal
if __name__ == "__main__":
    run_indexing()
