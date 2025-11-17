import os
import uvicorn
import requests
import json 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
# Librerías necesarias para SendGrid API
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# --- CONFIGURACIÓN DE MODELOS Y LÍMITES ---
INDEX_NAME = "sf-abogados-01"
EMBEDDING_MODEL = "text-embedding-ada-002"
GENERATION_MODEL = "gpt-4o-mini" # Modelo Correcto
TOP_K = 5

# --- CONTACTOS Y DETALLES DE VENTA ---
PHONE_NUMBER = "+593 98 375 6678"
SALES_EMAIL = "leads@abogados-sf.com" 
CONSULTATION_COST = "40 USD"
CONSULTATION_CREDIT_MESSAGE = f"Recuerda que este monto, en caso de que llevemos contigo el caso, **se acredita al costo total del servicio como descuento**."

# --- MODELO DE DATOS DE ENTRADA (INCLUYE MEMORIA DE CHAT) ---
class QueryModel(BaseModel):
    """Define la estructura de la solicitud JSON que recibirá el API."""
    question: str
    recaptcha_token: str
    history: list[dict] = [] # ACEPTA EL HISTORIAL

# --- INICIALIZACIÓN DE FASTAPI Y CORS ---

app = FastAPI(title="Asistente Legal SF API (RAG con GPT-4o Mini)")

# 🔒 CONFIGURACIÓN DE CORS
origins = ["https://abogados-sf.com", "http://localhost", "http://localhost:8000", "http://localhost:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- INICIALIZACIÓN DE CLIENTES ---
pc = None
openai_client = None
pinecone_index = None
SENDGRID_API_KEY = None 

try:
    PORT = int(os.environ.get("PORT", 8080))
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY") 

    # CHEQUEO DE VARIABLES
    missing_vars = []
    if not PINECONE_API_KEY: missing_vars.append("PINECONE_API_KEY")
    if not OPENAI_API_KEY: missing_vars.append("OPENAI_API_KEY")
    if not RECAPTCHA_SECRET_KEY: missing_vars.append("RECAPTCHA_SECRET_KEY")
    if not PINECONE_ENVIRONMENT: missing_vars.append("PINECONE_ENVIRONMENT")
    if not SENDGRID_API_KEY: missing_vars.append("SENDGRID_API_KEY") 

    if missing_vars:
        raise ValueError(f"Faltan variables de entorno esenciales: {', '.join(missing_vars)}")

    # Inicialización de clientes
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)

except Exception as e:
    print(f"ERROR FATAL DE INICIALIZACIÓN: {e}")
    # En caso de error de inicialización, lanzamos la excepción para Cloud Run 
    # y evitamos que el servicio se quede colgado.
    raise e


# --- LÓGICA DE ENVÍO DE EMAIL (VÍA SENDGRID API) ---

def send_summary_email(summary_content: str, recipient: str = SALES_EMAIL):
    """
    Función para enviar el resumen interno por correo electrónico usando la API de SendGrid.
    summary_content es el texto extraído del campo 'summary_email' del JSON.
    """
    
    if not SENDGRID_API_KEY:
        print("ERROR DE CONFIGURACIÓN: SENDGRID_API_KEY no definida. Email no enviado.")
        return False
        
    # Valores por defecto para el email
    subject_line = "Alerta de Lead: Revisión Manual de Contenido (Error de Formato)"
    body_content = summary_content # El contenido completo del LLM como fallback

    try:
        # 1. Buscar los marcadores de Subject y Body
        subject_tag = "Subject:"
        body_tag = "Body:"
        
        # El LLM DEBE generar "Subject:" y "Body:"
        if subject_tag in summary_content and body_tag in summary_content:
            
            # Buscar la posición inicial de cada tag
            subject_start = summary_content.find(subject_tag)
            body_start = summary_content.find(body_tag)
            
            # 2. Lógica para extraer el asunto (entre Subject: y Body:)
            if subject_start != -1 and body_start > subject_start:
                # Extraer lo que está DESPUÉS del tag 'Subject:' y ANTES del tag 'Body:'
                subject_line = summary_content[subject_start + len(subject_tag):body_start].strip()
                
                # 3. Lógica para extraer el cuerpo (todo después de Body:)
                body_content = summary_content[body_start + len(body_tag):].strip()

        elif subject_tag in summary_content:
            # En caso de que falte Body:, usamos el subject y enviamos todo el contenido como cuerpo.
            subject_line = summary_content.split(subject_tag)[1].strip()
            body_content = summary_content
            print("ADVERTENCIA: Solo se encontró 'Subject:'. Usando el contenido completo como cuerpo.")
            
    except Exception as e:
        # Captura cualquier error de parsing y envía el contenido crudo
        print(f"ERROR DE PARSING FATAL en send_summary_email: {e}")
        subject_line = "Error Inesperado en el Lead (Fallo de Parsing)"
        body_content = "Contenido original fallido:\n" + summary_content


    try:
        # Crear el objeto Mail y enviar
        message = Mail(
            from_email=SALES_EMAIL,             
            to_emails=recipient,              
            subject=subject_line,
            plain_text_content=body_content      
        )
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        if response.status_code in [200, 202]:
            print(f"ÉXITO: Email de resumen enviado a {recipient}. Código: {response.status_code}")
            return True
        else:
            print(f"ERROR SG: Fallo al enviar email. Código: {response.status_code}. Cuerpo: {response.body.decode() if response.body else 'No body'}")
            return False

    except Exception as e:
        print(f"ERROR FATAL al enviar email por SendGrid: {e}")
        return False
        
# --- LÓGICA DE SEGURIDAD (reCAPTCHA) ---
async def validate_recaptcha(token: str, min_score: float = 0.5):
    if token == 'EsteEsUnTokenDePruebaTemporal':
        return True

    response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={'secret': RECAPTCHA_SECRET_KEY, 'response': token}
    )
    result = response.json()
    if result.get('success') and result.get('score', 0) >= min_score:
        return True
    else:
        return False

# --- LÓGICA RAG Y EMBEDDINGS (SIN CAMBIOS) ---
def generate_embedding(text):
    response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def retrieve_context(embedding):
    query_results = pinecone_index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    return query_results

def generate_final_response(query, context, history):
    """
    Genera la respuesta final utilizando el contexto, la memoria (history)
    y el Super Prompt final, forzando la salida a JSON.
    """
    
    # --- SUPER PROMPT COMPACTO Y OPTIMIZADO (VERSIÓN JSON FINAL) ---
    system_prompt = (
        "Eres Agorito, un Asistente Legal Virtual, experto en Derecho Constitucional, Civil y de Familia de la ley Ecuatoriana. "
        "Tu personalidad es **vendedora, carismática y siempre profesional**. "
        "Tu objetivo es **generar una respuesta JSON** con dos claves: `summary_email` (para el equipo de ventas) y `user_response` (para el cliente). \n\n"
        
        "**Reglas Clave:**\n"
        "1. **Alcance:** Limítate a Constitucional, Civil y Familia. Si no aplica, el `user_response` debe usar la 'Regla de Cierre de Contraste' (ver abajo) y `summary_email` debe ser `''`."
        f"2. **Venta (CTA):** Después del análisis (Nivel 6-7, sin citar artículos o dar pasos procesales), guía siempre a la Consulta de Pago de {CONSULTATION_COST} (acreditable)."
        "3. **Datos (4 Claves):** Si el cliente acepta el CTA, solicita los 4 datos (Nombre, WhatsApp, Correo, Preferencia de Consulta). Sé acumulativo. **NO** repitas la lista completa, solo pide lo que falta."
        
        "**Formato de Salida ESTRICTO (JSON):**\n"
        "**Condición A: VENTA FINALIZADA (4 Datos Recolectados):**\n"
        "   - **`summary_email`:** Contiene el resumen profesional del caso, comenzando con **'Subject:'** y seguido de **'Body:'**. \n"
        "   - **`user_response`:** Contiene **ÚNICAMENTE** el mensaje de confirmación de agendamiento: '¡Perfecto! Ya tengo toda la información. Pronto alguien de nuestro equipo se pondrá en contacto contigo a través de tu [WhatsApp o correo] para coordinar la fecha y hora de tu consulta de 40 USD, que se acreditará al costo total del servicio.'\n\n"
        "**Condición B: CONVERSACIÓN, ANÁLISIS, O CESE DE INTERACCIÓN:**\n"
        "   - **`summary_email`:** Debe ser una **cadena vacía** (`''`).\n"
        "   - **`user_response`:** Debe ser el mensaje de Agorito para el cliente (e.g., análisis legal, pregunta de seguimiento de datos, o la despedida concisa si el cliente dijo 'gracias').\n\n"
        
        "**Contenido del `summary_email` (Solo Condición A):**\n"
        "Subject: [New Prospect - Legal Advice]. Body: **Client Details:** Name: [Name], WhatsApp Number: [Number], Email: [Email, if available], **Consultation Type:** [Presencial/Virtual], City/Location: [Client's City/Location]. **Case Analysis (For Internal Use):** [ANÁLISIS LEGAL COMPLETO, citando Artículos y Leyes Relevantes]. **Recommendation to the Firm (ESTRATEGIA):** [Proponer una estrategia legal sólida de 3 a 5 pasos]. **Client's Objective:** [Describir lo que el cliente desea lograr]."
        
        "**Reglas de Emergencia (user_response):**"
        f" - **Regla de Cierre de Contraste:** 'Lamentablemente, ese asunto está fuera de nuestra especialidad. Si lo desea, puede contactarnos directamente al {PHONE_NUMBER} o envíe un correo a {SALES_EMAIL} para ver si podemos recomendarle un colega.'"
        " - **Cese de Interacción (Final):** Si el cliente responde con un simple 'gracias' después de la confirmación, el `user_response` debe ser: 'A ti. Feliz día.' o '¡Gracias a ti!'"
    )

    # 3. Formatear el Contexto RAG y la Pregunta
    context_text = "\n\n".join([item['metadata']['text'] for item in context.matches])

    rag_prompt = (
        f"CONTEXTO PROPORCIONADO PARA EL ANÁLISIS (RAG):\n{context_text}\n\n"
        f"Pregunta más reciente del Usuario: {query}"
    )

    # 4. Construir la Matriz de Mensajes
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Añadir historial de conversación
    messages.extend(history)

    # Añadir el prompt RAG (Contexto + Pregunta actual)
    messages.append({"role": "user", "content": rag_prompt})

    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.0,
        # 🔑 FORZAR LA SALIDA A JSON
        response_format={"type": "json_object"},
    )

    final_response_text = response.choices[0].message.content
    return final_response_text

# --- ENDPOINT PRINCIPAL (CON PARSING JSON) ---

@app.post("/query")
async def process_query(data: QueryModel):
    """Endpoint principal para recibir la pregunta y devolver la respuesta."""
    try:
        # 1. Validación de Seguridad
        if not await validate_recaptcha(data.recaptcha_token):
             raise HTTPException(status_code=403, detail="Validación reCAPTCHA fallida. Acceso denegado.")

        # 2. Generación de Respuesta (RAG y LLM)
        query_embedding = generate_embedding(data.question)
        query_results = retrieve_context(query_embedding)
        raw_llm_response = generate_final_response(data.question, query_results, data.history)
        
        # DEBUGGING
        print(f"DEBUG: RAW LLM RESPONSE (JSON):\n{raw_llm_response}")
        # ------------------------------------------------------------------------------------------

        # 3. Lógica para PARSEAR el JSON y ENVIAR el resumen interno
        try:
            llm_output = json.loads(raw_llm_response)
            
            summary_content = llm_output.get("summary_email", "").strip()
            user_response = llm_output.get("user_response", "").strip()
            
            # 🔑 Llamar a SendGrid SOLO si el campo summary_email NO está vacío
            if summary_content:
                # La función send_summary_email ahora usa el contenido de summary_email
                send_summary_email(summary_content)
                
        except json.JSONDecodeError as e:
            # En caso de que el modelo falle al generar JSON
            print(f"ERROR DECODE: Fallo al decodificar JSON de la respuesta del LLM. {e}")
            # Devolvemos un error genérico pero claro al usuario
            user_response = "Disculpa, ha ocurrido un error de procesamiento. Por favor, reformula tu pregunta o contáctanos directamente al +593 98 375 6678."
            summary_content = "" # Aseguramos que no se intente enviar email con formato malo
            
        except Exception as e:
            # Otros errores de procesamiento
            print(f"ERROR INESPERADO en el parsing de la respuesta: {e}")
            user_response = "Disculpa, ha ocurrido un error de procesamiento. Por favor, reformula tu pregunta o contáctanos directamente al +593 98 375 6678."
            summary_content = ""

        # 4. Devolver la respuesta al usuario
        return {"answer": user_response}

    except Exception as e:
        # Esto captura errores de Pinecone, OpenAI, o en el embedding/context retrieval.
        # Es la causa más probable del 504 si no es un JSONDecodeError.
        print(f"Error CRÍTICO procesando la consulta: {e}")
        # Se lanza la excepción con código 500
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# --- INICIO LOCAL (Para pruebas) ---
if __file__ == "__main__":
    port_local = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port_local)
