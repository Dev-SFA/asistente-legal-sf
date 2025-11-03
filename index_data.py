import os
import requests
import zipfile
import io
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.settings import Settings 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE ARCHIVOS ---
# ⚠️ IMPORTANTE: REEMPLAZA ESTA URL CON LA DIRECCIÓN DE TU ARCHIVO ZIP PÚBLICO
DOCUMENTS_ZIP_URL = "http://mondoviaggi.ec/data.zip" # <-- ¡CÁMBIAME!
# Directorio donde se descomprimirán los archivos temporalmente
TEMP_DATA_DIR = "./temp_data" 
# Directorio donde se guardará el índice (subido a GitHub)
INDEX_DIR = "./storage"

def configure_settings():
    """Configura los modelos y parámetros de Llama Index."""
    load_dotenv()
    
    # 1. Verificar la clave de OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: La variable de entorno OPENAI_API_KEY no está configurada.")
        raise EnvironmentError("OPENAI_API_KEY es requerida para la indexación.")

    # 2. Configurar el modelo de embeddings de OpenAI
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # 3. Configurar el LLM por defecto 
    Settings.llm = OpenAI(model="gpt-4o") 
    
    # 4. Configurar el tamaño de chunking
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20

def download_and_extract_data(url, target_dir):
    """Descarga el ZIP de la URL y lo descomprime."""
    
    print(f"Descargando archivo desde: {url}")
    
    # 1. Crear directorio temporal
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 2. Descargar el archivo ZIP
    try:
        response = requests.get(url, stream=True, timeout=300) # 5 minutos de tiempo de espera
    except requests.exceptions.Timeout:
        raise ConnectionError("El servidor tardó demasiado en responder al intento de descarga (Timeout).")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Error de conexión al descargar el ZIP: {e}")

    if response.status_code != 200:
        raise ConnectionError(f"Fallo al descargar el archivo. Código de estado HTTP: {response.status_code}. Verifica la URL.")
    
    # 3. Descomprimir en memoria y extraer
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"Extrayendo {len(z.namelist())} archivos a {target_dir}...")
            z.extractall(target_dir)
    except zipfile.BadZipFile:
        raise ValueError("El archivo descargado no es un archivo ZIP válido o está corrupto.")


def build_knowledge_base():
    """
    Descarga los documentos, construye el índice vectorial y lo guarda en disco.
    """
    
    print("========================================================")
    print("Iniciando el proceso de indexación de documentos remotos.")
    print("========================================================")

    try:
        configure_settings()
        
        # 1. Descargar y descomprimir
        download_and_extract_data(DOCUMENTS_ZIP_URL, TEMP_DATA_DIR)

        # 2. Cargar documentos desde la carpeta temporal
        print(f"Cargando documentos desde la carpeta temporal: {TEMP_DATA_DIR}...")
        reader = SimpleDirectoryReader(input_dir=TEMP_DATA_DIR, exclude_hidden=False)
        documents = reader.load_data()
        
        if not documents:
            raise FileNotFoundError("No se encontraron documentos válidos para indexar después de la descompresión.")

        # 3. Construir y Guardar el índice
        print(f"Creando nuevo índice con {len(documents)} documento(s).")
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR)
            
        index.storage_context.persist(persist_dir=INDEX_DIR)
        
        print("========================================================")
        print(f"✅ ÉXITO: Índice creado y guardado en la carpeta '{INDEX_DIR}'.")
        print("========================================================")
        
    except Exception as e:
        print(f"\n❌ FALLO CRÍTICO EN LA INDEXACIÓN: {e}")
        # Asegurarse de que el Action falle si hay un error
        exit(1)
        
    finally:
        # 4. Limpieza: Eliminar la carpeta temporal
        if os.path.exists(TEMP_DATA_DIR):
            print(f"Limpiando carpeta temporal: {TEMP_DATA_DIR}")
            shutil.rmtree(TEMP_DATA_DIR)


if __name__ == "__main__":
    build_knowledge_base()

