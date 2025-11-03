import os
from pinecone import Pinecone

# --- CONFIGURACIÓN ---
INDEX_NAME = "sf-abogados-01"

def check_count():
    """Conecta a Pinecone y obtiene el conteo de vectores."""
    print("--- VERIFICANDO CONTEO DE VECTORES ---")
    
    # 1. Configurar Entorno (Lee las variables de entorno)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not pinecone_api_key:
        # En GitHub Actions esto no fallará, pero es bueno para pruebas locales
        print("Error: La variable de entorno PINECONE_API_KEY no está configurada.")
        return

    # 2. Inicializar cliente de Pinecone
    try:
        # Nota: El 'environment' ya no es necesario para Serverless,
        # pero el cliente acepta el parámetro por compatibilidad si está en el entorno
        pc = Pinecone(api_key=pinecone_api_key) 
        
        # 3. Conectar al índice
        if INDEX_NAME not in pc.list_indexes().names:
             print(f"Error: El índice '{INDEX_NAME}' no existe.")
             return
             
        pinecone_index = pc.Index(INDEX_NAME)

        # 4. Obtener el estado del índice (incluye conteo por namespace)
        index_stats = pinecone_index.describe_index_stats()
        
        # 5. Mostrar los resultados
        print(f"✅ Conexión exitosa al índice: {INDEX_NAME}")
        print(f"🌐 Namespaces encontrados: {list(index_stats.namespaces.keys())}")
        
        total_vectors = index_stats.total_vector_count
        print(f"✨ CONTEO TOTAL DE VECTORES EN EL ÍNDICE: {total_vectors}")

        if total_vectors > 0:
            print("¡Éxito! Los vectores están cargados. El dashboard tenía un retraso.")
        else:
             print("Atención: El conteo directo sigue en 0. Posible fallo en la subida.")
        
    except Exception as e:
        print(f"Error durante la verificación: {e}")

if __name__ == "__main__":
    check_count()
