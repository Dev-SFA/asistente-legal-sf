import os
import requests
import zipfile
import io
import shutil
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone # <-- ¡CORRECCIÓN CLAVE! Usa 'pinecone', no 'pinecone-client'
# Estas librerías son necesarias para el parseo avanzado de documentos
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# --- CONFIGURACIÓN ---
# Los valores se leen de las variables de entorno configuradas en GitHub Actions
INDEX_NAME = "sf-abogados-01"
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 50
DATA_URL = os.environ.get("DATA_URL") # URL del hosting de datos (leído de GitHub Secrets)
TEMP_DATA_DIR = "temp_data"

def download_and_extract_data(url: str, output_dir: str):
    """Descarga un archivo ZIP desde la URL externa y extrae su contenido."""
    if not url:
        raise ValueError("La variable de entorno DATA_URL no está configurada. El script no puede descargar los datos.")
    
    print(f"Descargando datos desde: {url}")
    
    # Crea el directorio temporal si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx

        # Maneja el archivo en memoria antes de extraer
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(output_dir)
        
        print(f"Descarga y extracción completadas en: {output_dir}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar o conectar: {e}")
        return False
    except zipfile.BadZipFile:
        print("Error: El archivo descargado no es un archivo ZIP válido.")
        return False
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la descarga: {e}")
        return False


def get_documents_from_dir(directory: str):
    """Carga documentos, los divide en nodos (chunks) y extrae texto."""
    all_documents = []
    
    # Recorrer todos los archivos en el directorio
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  -> Procesando archivo: {file_path}")
            
            try:
                # 1. Particionamiento: dividir el documento en elementos estructurales
                elements = partition(filename=file_path)
                
                # 2. Chunking: agrupar los elementos en fragmentos lógicos (chunks)
                # Esto es crucial para la calidad de la recuperación
                chunks = chunk_by_title(elements)
                
                # 3. Almacenar los chunks como documentos
                for i, chunk in enumerate(chunks):
                    if chunk.text.strip():
                        all_documents.append({
                            "text": chunk.text,
                            "metadata": {
                                "file_name": file,
                                "chunk_id": f"{file}_{i}", # ID único
                            }
                        })
                        
            except Exception as e:
                print(f"ERROR al procesar el archivo {file}: {e}")
                continue

    return all_documents


def index_data(documents):
    """Genera embeddings en lotes y sube los vectores a Pinecone."""
    print(f"Comenzando la indexación y subida de {len(documents)} documentos a Pinecone...")

    # 1. Inicializar Clientes de Pinecone y OpenAI
    try:
        # Se inicializan leyendo las variables de entorno
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY")) 
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Conecta a tu índice
        if INDEX_NAME not in pc.list_indexes().names:
             raise ValueError(f"El índice {INDEX_NAME} no existe o no está disponible.")
             
        pinecone_index = pc.Index(INDEX_NAME)

    except Exception as e:
        print(f"Error de inicialización de clientes: {e}")
        return

    # 2. Procesar y subir en lotes
    vectors_to_upsert = []
    total_vectors = 0

    # Usamos tqdm para mostrar el progreso en la consola de GitHub Actions
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Generando embeddings y subiendo a Pinecone"):
        batch = documents[i:i + BATCH_SIZE]
        texts = [doc["text"] for doc in batch]

        # Generar embeddings
        try:
            response = openai_client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL
            )
            embeddings = [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error al generar embeddings en el lote {i}: {e}. Saltando lote.")
            continue

        # Preparar vectores para upsert
        for j, doc in enumerate(batch):
            vector = {
                'id': doc['metadata']['chunk_id'],
                'values': embeddings[j],
                'metadata': doc['metadata'] | {"text": doc["text"]} # Guardar el texto original en el metadata
            }
            vectors_to_upsert.append(vector)
            total_vectors += 1

        # Subir el lote a Pinecone
        if len(vectors_to_upsert) >= BATCH_SIZE:
            try:
                pinecone_index.upsert(
                    vectors=vectors_to_upsert,
                    namespace="" # Usamos el namespace por defecto (cadena vacía)
                )
                vectors_to_upsert = []
            except Exception as e:
                print(f"Error al subir lote a Pinecone: {e}. Descartando lote fallido.")
                vectors_to_upsert = []

    # Subir vectores restantes
    if vectors_to_upsert:
        try:
            pinecone_index.upsert(
                vectors=vectors_to_upsert,
                namespace=""
            )
        except Exception as e:
            print(f"Error al subir lote final a Pinecone: {e}")

    print("\n--------------------------------------------------")
    print(f"Indesación completada. Total de vectores procesados: {total_vectors}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    print("\nINICIANDO PROCESO DE INDEXACIÓN")

    # 1. Descargar y extraer los datos (Desde el hosting externo)
    if not download_and_extract_data(DATA_URL, TEMP_DATA_DIR):
        print("PROCESO FALLIDO: No se pudieron descargar y extraer los datos.")
    else:
        # 2. Cargar, parsear y chunkear documentos
        print(f"Cargando documentos desde el directorio {TEMP_DATA_DIR}...")
        documents_to_index = get_documents_from_dir(TEMP_DATA_DIR)
        print(f"Cargados {len(documents_to_index)} documentos listos para indexar.")

        if documents_to_index:
            # 3. Indexar y subir a Pinecone
            index_data(documents_to_index)
        else:
            print("No se encontraron documentos para indexar.")

    # 4. Limpieza de archivos temporales (¡Importante para GitHub Actions!)
    print("Iniciando limpieza de archivos temporales...")
    if os.path.exists(TEMP_DATA_DIR):
        try:
            shutil.rmtree(TEMP_DATA_DIR)
            print("Limpieza completada. Directorio temporal eliminado.")
        except Exception as e:
            print(f"Advertencia: No se pudo eliminar el directorio temporal: {e}")

    print("\nPROCESO FINALIZADO")
