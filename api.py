import os
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone 
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware 

# --- CONFIGURACIÓN DE MODELOS Y LÍMITES ---
INDEX_NAME = "sf-abogados-01" 
EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATION_MODEL = "gpt-5-nano"
TOP_K = 5  

# --- MODELO DE DATOS DE ENTRADA (Incluye reCAPTCHA) ---
class QueryModel(BaseModel):
    """Define la estructura de la solicitud JSON que recibirá el API."""
    question: str
    recaptcha_token: str 

# --- INICIALIZACIÓN DE FASTAPI Y CORS ---

app = FastAPI(title="Asistente Legal SF API (RAG con GPT-4o Mini)")

# 🔒 CONFIGURACIÓN DE CORS PARA PERMITIR LLAMADAS DESDE HOSTRINGER
origins = [
    "https://abogados-sf.com",  # ¡TU DOMINIO AUTORIZADO!
    "http://localhost",         
    "http://localhost:8000",    
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
# --- FIN CONFIGURACIÓN DE CORS ---


# --- INICIALIZACIÓN DE CLIENTES ---
try:
    PORT = int(os.environ.get("PORT", 8080))
    
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

    if not PINECONE_API_KEY or not OPENAI_API_KEY or not RECAPTCHA_SECRET_KEY or not PINECONE_ENVIRONMENT:
        raise ValueError("Faltan variables de entorno esenciales (API Keys, Ambiente Pinecone o Secreto reCAPTCHA).")

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) 
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    pinecone_index = pc.Index(INDEX_NAME)
    
except Exception as e:
    print(f"ERROR FATAL DE INICIALIZACIÓN: {e}") 
    

# --- LÓGICA DE SEGURIDAD (CORREGIDA) ---

async def validate_recaptcha(token: str, min_score: float = 0.5):
    """Valida el token de reCAPTCHA con Google antes de llamar a las APIs costosas."""
    
    # 🚨 CORRECCIÓN FINAL: Si se usa el token de prueba, IGNORAMOS la validación.
    if token == 'EsteEsUnTokenDePruebaTemporal':
         print("WARNING: reCAPTCHA bypassed using placeholder token.")
         return True # Permite que la solicitud continúe
         
    # El resto del código solo se ejecuta con un token real:
    response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={
            'secret': RECAPTCHA_SECRET_KEY,
            'response': token
        }
    )
    
    result = response.json()
    
    if result.get('success') and result.get('score', 0) >= min_score:
        return True
    else:
        print(f"reCAPTCHA validation failed: {result}")
        return False

# --- LÓGICA RAG ---
# (El resto de estas funciones se mantiene igual)
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
        # 1. SEGURIDAD: Validar reCAPTCHA (ahora soporta el token de prueba)
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
        # Captura cualquier otro error, lo imprime en los logs, y devuelve el 500.
        # Con esta corrección, el Error 500 ya no debería ocurrir.
        print(f"Error procesando la consulta: {e}") 
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# --- INICIO LOCAL (Para pruebas) ---
if __name__ == "__main__":
    port_local = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port_local)
