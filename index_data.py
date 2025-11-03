import os
import requests
import zipfile
import shutil
from pathlib import Path

# **Correcciones de Importación FINALIZADAS:**
# Las clases principales se importan ahora desde el módulo 'llama_index.core'
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.settings import Settings

# Librería de Pinecone para la inicialización
from pinecone import Pinecone

# --- CONFIGURACIÓN ---
# La URL de tu archivo ZIP en el hosting
DATA_URL = "http://mondoviaggi.ec/data.zip"
# El nombre del archivo ZIP
ZIP_FILENAME = "data.zip"
# Directorio temporal para la descarga y extracción
TEMP_DIR = "temp_data"
# Nombre de tu índice en Pinecone
INDEX_NAME = "sf-abogados-01"

def download_and_extract_data(url: str, zip_path: Path, extract_dir: Path):
    """Descarga el archivo ZIP y lo extrae en un directorio temporal."""
    print(f"Descargando datos desde: {url}")
    
    # Asegurar que la ruta temporal exista y esté vacía
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 1. Descargar el archivo
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Descarga completada.")
    except requests.exceptions.RequestException as e:
        print(f"Error durante la descarga: {e}")
        # Intentar limpiar el archivo si se creó parcialmente
        if zip_path.exists():
            zip_path.unlink()
        raise

    # 2. Extraer el contenido
    try:
        print(f"Extrayendo archivos a: {extract_dir.resolve()}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraer solo si no es un archivo de un solo nivel
            members = zip_ref.namelist()
            if len(members) == 1 and members[0].endswith('/'):
                # Si solo contiene una carpeta raíz, extraer al directorio superior
                zip_ref.extractall(extract_dir)
                # Mover el contenido de la subcarpeta un nivel arriba
                subfolder = extract_dir / members[0]
                if subfolder.exists():
                    for item in os.listdir(subfolder):
                        shutil.move(subfolder / item, extract_dir)
                    shutil.rmtree(subfolder)
            else:
                zip_ref.extractall(extract_dir)
        print("Extracción completada.")
    except zipfile.BadZipFile:
        print("Error: El archivo descargado no es un ZIP válido.")
        raise
    finally:
        # 3. Limpiar el archivo ZIP original
        if zip_path.exists():
            zip_path.unlink()

def main():
    """Ejecuta el proceso completo de descarga, indexación y subida a Pinecone."""
    print("--- INICIANDO PROCESO DE INDEXACIÓN ---")
    
    # 1. Rutas de Archivo
    zip_path = Path(ZIP_FILENAME)
    extract_dir = Path(TEMP_DIR)

    # 2. Descargar y Extraer
    try:
        download_and_extract_data(DATA_URL, zip_path, extract_dir)
    except Exception as e:
        print(f"Fallo en la fase de descarga/extracción: {e}")
        # No limpiar para depuración si la descarga falló
        return

    # 3. Configurar Entorno (usa los Secrets de GitHub Actions)
    print("Configurando API Keys y modelos...")
    try:
        # OpenAI Key se lee de OPENAI_API_KEY en el entorno
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Pinecone Key y Environment se leen de PINECONE_API_KEY y PINECONE_ENVIRONMENT
        # El entorno ahora no se necesita con Serverless, pero se mantiene la conexión
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        # El 'environment' (Region) ya no es obligatorio, pero lo pasamos si existe
        pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1") 

        # Inicializar cliente de Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
    except Exception as e:
        print(f"Error al configurar Pinecone/OpenAI: {e}")
        shutil.rmtree(extract_dir)
        return

    # 4. Leer Documentos
    print(f"Cargando documentos desde el directorio: {extract_dir.name}")
    try:
        # SimpleDirectoryReader ahora se importa desde llama_index.core
        reader = SimpleDirectoryReader(input_dir=extract_dir.as_posix())
        documents = reader.load_data()
        print(f"Cargados {len(documents)} documentos.")
    except Exception as e:
        print(f"Error al leer documentos: {e}")
        shutil.rmtree(extract_dir)
        return

    # 5. Crear el VectorStore y el Índice (Conexión a Pinecone)
    print(f"Conectando al índice de Pinecone: {INDEX_NAME}")
    try:
        # El índice DEBE haber sido creado manualmente en Pinecone ANTES de correr este script.
        # Esto ya lo hiciste: sf-abogados-01 con 1536 dimensiones.
        pinecone_index = pc.Index(INDEX_NAME)

        # Crear VectorStore que apunte al índice
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            # Se usa el mismo modelo de embedding
            embed_dim=1536  
        )

        print(f"Comenzando la indexación y subida de {len(documents)} documentos a Pinecone...")
        # El VectorStoreIndex ahora se importa desde llama_index.core
        index = VectorStoreIndex.from_documents(
            documents=documents,
            vector_store=vector_store,
            show_progress=True # Muestra la barra de progreso
        )
        
        print("Indexación completada. Los vectores han sido subidos a Pinecone.")

    except Exception as e:
        print(f"Error crítico durante la indexación/subida: {e}")
        raise # Permitir que el proceso falle para ver el log completo

    finally:
        # 6. Limpieza
        print("Iniciando limpieza de archivos temporales...")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        print("Limpieza completada. Directorio temporal eliminado.")
        print("--- PROCESO FINALIZADO ---")

if __name__ == "__main__":
    main()
