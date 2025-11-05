import os
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI

# --- CONFIGURACIÓN DE MODELOS Y LÍMITES ---
# INDEX_NAME se define aquí y también debe inyectarse en Cloud Build para coincidir
INDEX_NAME = "sf-abogados-01" 
EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATION_MODEL = "gpt-5-nano"  # Modelo de OpenAI
TOP_K = 5  

# --- MODELO DE DATOS DE ENTRADA (Incluye reCAPTCHA) ---
class QueryModel(BaseModel):
    """Define la estructura de la solicitud JSON que recibirá el API."""
    question: str
    recaptcha_token: str # Token de seguridad enviado por el frontend

# --- INICIALIZACIÓN DE CLIENTES ---
try:
    # Puerto necesario para Cloud Run (usa la variable de entorno, si no existe usa 8080)
    PORT = int(os.environ.get("PORT", 8080))
    
    # Obtener claves de Variables de Entorno (se inyectarán de forma segura en Cloud Run)
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT") # <--- AHORA SE LEE

    if not PINECONE_API_KEY or not OPENAI_API_KEY or not RECAPTCHA_SECRET_KEY or not PINECONE_ENVIRONMENT: # <--- ¡ACTUALIZADO!
        # Este error ahora solo ocurrirá si olvidas una clave en Cloud Build
        raise ValueError("Faltan variables de entorno esenciales (API Keys, Ambiente Pinecone o Secreto reCAPTCHA).")

    # Inicialización correcta de Pinecone (usa api_key y environment)
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Intenta conectar al índice. Si el índice no existe, fallará aquí.
    pinecone_index = pc.Index(INDEX_NAME)
    
except Exception as e:
    # Muestra el error de Python en los logs de Cloud Run
    print(f"ERROR FATAL DE INICIALIZACIÓN: {e}")
    # Nota: El error de aquí hará que el contenedor se caiga y Cloud Run reporte el fallo de despliegue.
    # Si falla, revisa el log de Cloud Run para ver este mensaje.

app = FastAPI(title="Asistente Legal SF API (RAG con GPT-4o Mini)")

# --- LÓGICA DE SEGURIDAD ---

async def validate_recaptcha(token: str, min_score: float = 0.5):
    """Valida el token de reCAPTCHA con Google antes de llamar a las APIs costosas."""
    # RECAPTCHA_SECRET_KEY se lee al inicio de la aplicación
    response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={
            'secret': RECAPTCHA_SECRET_KEY,
            'response': token
        }
    )
    
    result = response.json()
    
    # Verifica si Google lo marcó como exitoso y si el puntaje supera el mínimo
    if result.get('success') and result.get('score', 0) >= min_score:
        return True
    else:
        return False

# --- LÓGICA RAG ---

def generate_embedding(text):
    """Genera el embedding para un texto dado."""
    response = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def retrieve_context(embedding):
    """Consulta Pinecone para recuperar los fragmentos más relevantes."""
    query_results = pinecone_index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    return query_results

def generate_final_response(query, context):
    """Genera la respuesta final utilizando el contexto recuperado y GPT-4o Mini."""
    
    system_prompt = (
        "Eres un Asistente Legal experto en derecho Ecuatoriano. Tu única fuente de información son los "
        "fragmentos de contexto proporcionados. Si la respuesta no está en el contexto, debes indicar: "
        "'Lo siento, la información específica no se encuentra en mis documentos legales.'"
        "Debes responder con precisión, mencionando artículos de ley o códigos si están disponibles en el contexto. "
        "Tu respuesta debe estar en ESPAÑOL y ser formal. Tu análisis debe ser riguroso."
    )
    
    context_text = "\n\n".join([item['metadata']['text'] for item in context.matches])
    
    user_prompt = (
        f"Pregunta del Usuario: {query}\n\n"
        f"CONTEXTO PROPORCIONADO:\n{context_text}"
    )

    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# --- ENDPOINT PRINCIPAL ---

@app.post("/query")
async def process_query(data: QueryModel):
    """Endpoint principal para recibir la pregunta y devolver la respuesta."""
    try:
        # 1. SEGURIDAD: Validar reCAPTCHA antes de cualquier API costosa
        # Verifica si RECAPTCHA_SECRET_KEY fue cargado
        if not 'RECAPTCHA_SECRET_KEY' in globals():
             raise HTTPException(status_code=500, detail="Configuración de seguridad incompleta.")
             
        if not await validate_recaptcha(data.recaptcha_token):
             raise HTTPException(status_code=403, detail="Validación reCAPTCHA fallida. Acceso denegado.")

        # 2. Generar embedding de la pregunta
        query_embedding = generate_embedding(data.question)

        # 3. Recuperar contexto de Pinecone
        query_results = retrieve_context(query_embedding)

        if not query_results.matches:
            return {"answer": "Lo siento, no encontré información relevante en los documentos legales para responder a tu pregunta."}

        # 4. Generar Respuesta Final con GPT-4o Mini
        final_answer = generate_final_response(data.question, query_results)
        
        # 5. Devolver la respuesta al frontend
        return {"answer": final_answer}

    except Exception as e:
        print(f"Error procesando la consulta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# --- INICIO LOCAL (Para pruebas) ---
if __name__ == "__main__":
    port_local = int(os.environ.get("PORT", 8000))
    # Se espera que el servidor de uvicorn sea llamado por Cloud Run,
    # pero esta sección está aquí para pruebas locales.
    uvicorn.run(app, host="0.0.0.0", port=port_local)
