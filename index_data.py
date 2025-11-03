import os
import requests
import zipfile
import io
import shutil
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import uuid

# --- CONFIGURACIÓN ---
INDEX_NAME = "sf-abogados-01"
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 50
DATA_URL = os.environ.get("DATA_URL")
TEMP_DATA_DIR = "temp_data"

def download_and_extract_data(url: str, output_dir: str):
    """Descarga un archivo ZIP desde la URL externa y extrae su contenido."""
    if not url:
        raise ValueError("La variable de entorno DATA_URL no está configurada. El script no puede descargar los datos.")
    
    print(f"Descargando datos desde: {url}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 

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


def index_data_optimized(directory: str):
    """Procesa documentos de forma optimizada, generando embeddings y subiendo a Pinecone
    sin cargar todos los documentos a la memoria.
    """
    print("Comenzando la indexación optimizada y subida a Pinecone...")

    try:
        # **CORRECCIÓN CRÍTICA DE SINTAXIS:** Inicialización correcta de Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY")) 
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        print("[DEBUG] Clientes de Pinecone y OpenAI inicializados.")

        if INDEX_NAME not in pc.list_indexes().names:
             raise ValueError(f"El índice {INDEX_NAME} no existe o no está disponible.")
             
        pinecone_index = pc.Index(INDEX_NAME)

    except Exception as e:
        print(f"Error de inicialización de clientes: {e}")
        return

    vectors_to_upsert = []
    total_vectors = 0
    document_count = 0

    # Iterar sobre los archivos en el directorio
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  -> Procesando archivo: {file_path}")
            
            try:
                # 1. Particionamiento: dividir el documento
                elements = partition(filename=file_path)
                
                # 2. Chunking: agrupar en fragmentos lógicos
                chunks = chunk_by_title(elements)
                
                # 3. Procesar los chunks del archivo
                for i, chunk in enumerate(chunks):
                    if chunk.text.strip():
                        text = chunk.text
                        chunk_id = f"{file}_{i}_{uuid.uuid4()}" # ID único para evitar colisiones
                        
                        # Generar embeddings para el chunk
                        try:
                            # [DEBUG] Mensaje antes de la llamada a OpenAI
                            print(f"[DEBUG] Generando embedding para chunk {total_vectors+1}")
                            
                            response = openai_client.embeddings.create(
                                input=[text],
                                model=EMBEDDING_MODEL
                            )
                            embedding = response.data[0].embedding
                            
                            # Preparar vector para upsert
                            vector = {
                                'id': chunk_id,
                                'values': embedding,
                                'metadata': {
                                    "file_name": file,
                                    "chunk_id": chunk_id,
                                    "text": text
                                }
                            }
                            vectors_to_upsert.append(vector)
                            total_vectors += 1

                        except Exception as e:
                            print(f"Error al generar embedding para chunk en {file}: {e}. Saltando chunk.")
                            continue

                        # Subir el lote a Pinecone cuando alcance BATCH_SIZE
                        if len(vectors_to_upsert) >= BATCH_SIZE:
                            # [DEBUG] Mensaje antes de la llamada a Pinecone
                            print(f"[DEBUG] Subiendo lote de {len(vectors_to_upsert)} vectores a Pinecone. Total: {total_vectors}")
                            try:
                                pinecone_index.upsert(
                                    vectors=vectors_to_upsert,
                                    namespace=""
                                )
                                vectors_to_upsert = [] # Limpiar el lote para el siguiente
                            except Exception as e:
                                print(f"Error al subir lote a Pinecone: {e}. Descartando lote fallido.")
                                vectors_to_upsert = []

            except Exception as e:
                print(f"ERROR FATAL al procesar el archivo {file}: {e}")
                continue
            
            document_count += 1
            print(f"  -> Archivo {document_count} procesado. Total de vectores subidos: {total_vectors}")

    # Subir vectores restantes (lote final)
    if vectors_to_upsert:
        print(f"[DEBUG] Subiendo lote final de {len(vectors_to_upsert)} vectores restantes.")
        try:
            pinecone_index.upsert(
                vectors=vectors_to_upsert,
                namespace=""
            )
            total_vectors += len(vectors_to_upsert)
        except Exception as e:
            print(f"Error al subir lote final a Pinecone: {e}")

    print("\n--------------------------------------------------")
    print(f"Indesación completada. Total de vectores procesados y subidos: {total_vectors}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    print("\nINICIANDO PROCESO DE INDEXACIÓN")

    if not download_and_extract_data(DATA_URL, TEMP_DATA_DIR):
        print("PROCESO FALLIDO: No se pudieron descargar y extraer los datos.")
    else:
        index_data_optimized(TEMP_DATA_DIR)

    print("Iniciando limpieza de archivos temporales...")
    if os.path.exists(TEMP_DATA_DIR):
        try:
            shutil.rmtree(TEMP_DATA_DIR)
            print("Limpieza completada. Directorio temporal eliminado.")
        except Exception as e:
            print(f"Advertencia: No se pudo eliminar el directorio temporal: {e}")

    print("\nPROCESO FINALIZADO")
