import os
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Dict

# --- CONFIGURACIÓN ---
INDEX_NAME = "sf-abogados-01"
# Modelo de embedding que debe coincidir con el usado en index_data.py
EMBEDDING_MODEL = "text-embedding-ada-002" 
TOP_K = 5  # Recuperar los 5 fragmentos más relevantes de Pinecone

def get_relevant_context(query_text: str) -> List[Dict]:
    """
    Busca en Pinecone los fragmentos de texto más relevantes
    para la pregunta del usuario (Retrieval).
    """
    print(f"\n--- BUSCANDO CONTEXTO PARA: {query_text} ---")
    
    try:
        # 1. Inicializar Clientes de Pinecone y OpenAI
        # Las credenciales se leen automáticamente de las variables de entorno
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Conecta a tu índice donde confirmamos que están los 1566 vectores
        pinecone_index = pc.Index(INDEX_NAME)

        # 2. Convertir la pregunta del usuario a un vector (Embedding)
        print("Generando embedding de la pregunta...")
        response = openai_client.embeddings.create(
            input=[query_text],
            model=EMBEDDING_MODEL
        )
        query_vector = response.data[0].embedding

        # 3. Consultar el índice de Pinecone
        print(f"Consultando Pinecone (Top {TOP_K})...")
        results = pinecone_index.query(
            vector=query_vector,
            top_k=TOP_K,
            # include_metadata=True es CRUCIAL para obtener el texto original
            include_metadata=True
            # No especificamos namespace, usa el namespace vacío por defecto ('')
        )

        # 4. Procesar y devolver el contexto
        relevant_context = []
        for match in results.matches:
            # metadata['text'] contiene el contenido del fragmento
            text_fragment = match.metadata.get('text', 'No text found')
            
            # Devolvemos el texto y el score de similitud
            relevant_context.append({
                "text": text_fragment,
                "score": match.score
            })
            
        print(f"✅ Se encontraron {len(relevant_context)} fragmentos relevantes.")
        return relevant_context

    except Exception as e:
        print(f"Error en la consulta a la base de conocimiento: {e}")
        return []

# --- FUNCIÓN DE PRUEBA ---
if __name__ == "__main__":
    # Esta parte solo se ejecuta si corres el script directamente (python knowledge.py)
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
        print("ATENCIÓN: Las variables de entorno PINECONE_API_KEY y OPENAI_API_KEY deben estar configuradas en tu entorno local para probar esta función.")
    
    # Prueba con una pregunta de ejemplo
    test_query = "¿Cuál es el proceso para disolver una compañía en Ecuador según la ley?"
    context = get_relevant_context(test_query)
    
    # Muestra los fragmentos recuperados
    for item in context:
        print("-" * 50)
        print(f"Score de Similitud: {item['score']:.4f}")
        print(f"Contexto: {item['text'][:200]}...")
