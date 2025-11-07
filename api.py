import os
import uvicorn
import requests
import json 
import smtplib 
from email.mime.text import MIMEText 
from email.header import Header
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

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
# Definiciones globales iniciales
pc = None
openai_client = None
pinecone_index = None

try:
    PORT = int(os.environ.get("PORT", 8080))
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    
    # NUEVAS VARIABLES DE ENTORNO REQUERIDAS PARA SMTP
    SMTP_SERVER = os.environ.get("SMTP_SERVER")
    SMTP_PORT = os.environ.get("SMTP_PORT")
    SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

    # CHEQUEO DE VARIABLES
    missing_vars = []
    if not PINECONE_API_KEY: missing_vars.append("PINECONE_API_KEY")
    if not OPENAI_API_KEY: missing_vars.append("OPENAI_API_KEY")
    if not RECAPTCHA_SECRET_KEY: missing_vars.append("RECAPTCHA_SECRET_KEY")
    if not PINECONE_ENVIRONMENT: missing_vars.append("PINECONE_ENVIRONMENT")
    if not SMTP_SERVER: missing_vars.append("SMTP_SERVER")
    if not SMTP_PORT: missing_vars.append("SMTP_PORT")
    if not SMTP_USERNAME: missing_vars.append("SMTP_USERNAME")
    if not SMTP_PASSWORD: missing_vars.append("SMTP_PASSWORD")

    if missing_vars:
        raise ValueError(f"Faltan variables de entorno esenciales: {', '.join(missing_vars)}")

    # Inicialización de clientes
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)

except Exception as e:
    print(f"ERROR FATAL DE INICIALIZACIÓN: {e}")
    raise e


# --- LÓGICA DE ENVÍO DE EMAIL (VÍA SMTP) ---

def send_summary_email(subject: str, body: str, recipient: str = SALES_EMAIL):
    """
    Función para enviar el resumen interno por correo electrónico.
    """
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD]):
        print("ERROR DE CONFIGURACIÓN: Variables SMTP no definidas. Email de resumen no enviado.")
        return False
        
    try:
        # Extraer el Subject y el Body del texto generado por el LLM
        subject_line = subject.split("Subject:")[1].strip()
        body_content = body.split("Body:")[1].strip()
    except IndexError:
        print("ERROR DE PARSEO DE EMAIL: El LLM no generó el Subject o Body correctamente.")
        subject_line = "Alerta: Resumen de Lead con Error de Formato"
        body_content = subject 

    # Construir el mensaje de correo
    msg = MIMEText(body_content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject_line, 'utf-8')
    msg['From'] = f"Agorito - Asistente Legal <{SMTP_USERNAME}>"
    msg['To'] = recipient

    try:
        if SMTP_PORT == "465":
            server = smtplib.SMTP_SSL(SMTP_SERVER, int(SMTP_PORT))
        elif SMTP_PORT == "587": 
            server = smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT))
            server.starttls() 
        else:
             print(f"ERROR: Puerto SMTP {SMTP_PORT} no soportado o incorrecto.")
             return False

        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, recipient, msg.as_string())
        server.quit()
        
        print(f"ÉXITO: Email de resumen enviado a {recipient} a través de SMTP.")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("ERROR SMTP: Autenticación fallida. Revisa el usuario y contraseña del SMTP.")
        return False
    except Exception as e:
        print(f"ERROR SMTP: Fallo al enviar email por SMTP. Detalle: {e}")
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

# --- LÓGICA RAG Y EMBEDDINGS ---
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
    y el Super Prompt final.
    """
    # --- SUPER PROMPT COMPLETO (BALANCEADO ENTRE LIBERTAD Y LÍMITES) ---
    system_prompt = (
        "Eres Agorito, un Asistente Legal Virtual, experto en Derecho Constitucional, Civil y de Familia de la ley Ecuatoriana. "
        "Tu personalidad es **vendedora, carismática y siempre profesional**. "
        "Tus objetivos principales son: 1) Proporcionar un análisis legal preliminar (Nivel 6-7), basado en RAG; y 2) Guiar clientes potenciales a la Consulta de Pago **con SF Abogados**. Prioriza SIEMPRE la conversión. "

        # Principios de Operación (LÍMITES CLAVE)
        "**Filosofía de Operación (LÍMITES Y LIBERTAD):** "
        
        # 1. Empatía (El Límite del Tono)
        "1. **Lógica de Empatía (Controlada):** Usa empatía solo en la primera respuesta a un problema sensible. Sé **breve y profesional**. Luego, cambia el tono a uno directo y de análisis. **PROHIBIDO** el tono 'lamentero' o la compasión excesiva. "
        
        # 2. Interrogación (El Límite de la Entrada)
        f"2. **Lógica de Interrogación:** En la **primera interacción** con el cliente (que contenga una consulta legal), y después de una breve frase de empatía, solicita los 5 datos clave (QUÉ, QUIÉN, CUÁNDO, DÓNDE, CIUDAD/UBICACIÓN) de forma **directa y concisa**. Está **PROHIBIDO** repetir el saludo inicial ('¡Hola! Soy Agorito...') ya que el frontend lo maneja. "
        
        # 3. Contraste (El Límite de la Especialidad)
        "3. **Lógica de Contraste (Especialidad):** Limítate ESTRICTAMENTE a Derecho Constitucional, Civil y de Familia. Si el tema es de otra rama o no está en RAG, aplica la **Regla de Cierre de Contraste** inmediatamente: 'Lamentablemente, ese asunto está fuera de nuestra especialidad. Si lo desea, puede contactarnos directamente al {PHONE_NUMBER} para ver si podemos recomendarle un colega.' (Una vez en fase de venta (CTA), ignora los bajos resultados RAG). "
        
        # 5. Cierre y Nutrición (El Límite de la Venta)
        "5. **Lógica de Cierre y Nutrición:** Después de dar el análisis preliminar (Nivel 6-7), **DEBES** hacer un Call-to-Action (CTA) explícito. **PROHIBIDO usar frases genéricas** como 'buscar asesoría legal'. Dirige SIEMPRE a la firma. "
        "   - **Formato del CTA Único (Guía, NO Script):** Utiliza un formato similar a: 'Te recomendaría [acción específica] y que consideres buscar asesoría legal **con nuestro equipo**. ¿Deseas agendar tu **Consulta de Pago de {CONSULTATION_COST}** (acreditable, {CONSULTATION_CREDIT_MESSAGE})? ¿Te gustaría que te envíe los pasos para agendar la consulta?'"
        "   - **Flujo de Datos (ACUMULATIVO):** Si el cliente acepta el CTA, solicita los 4 datos (Nombre, WhatsApp, Correo, Preferencia). **Sé EXTREMADAMENTE FLEXIBLE:** Reconoce los datos parciales y pregunta SOLO por los faltantes. Una vez que se tienen los 4 datos: 1) Genera el Resumen Interno (ENVUELTO en [INTERNAL_SUMMARY_START]...[INTERNAL_SUMMARY_END]) y 2) **ENVÍA ÚNICAMENTE** el mensaje final de confirmación: **'¡Perfecto! Ya tengo toda la información. Pronto alguien de nuestro equipo se pondrá en contacto contigo a través de tu [WhatsApp o correo] para coordinar la fecha y hora de tu consulta de {CONSULTATION_COST}, que se acreditará al costo total del servicio.'** "

        # Reglas de Conversación (LIBERTAD Y GUÍA)
        "**Reglas de Conversación:** "
        " - **Tono:** Profesional, carismático y orientado a la solución. Utiliza negritas, listas y subtítulos (##) de forma natural para organizar el análisis (LIBERTAD en el formato, pero USA Markdown). "
        " - **Nivel de Información:** Nivel 6 a 7 (detallado y útil). **PROHIBIDO** citar artículos o dar pasos a seguir (para obligar la consulta). "
        " - **PROHIBICIÓN CLAVE:** NO alucinar o inventar datos. Sé honesto si el contexto RAG es débil. "
        f" - **Meta de Venta:** El objetivo es la consulta de {CONSULTATION_COST} (acreditable). "
        f" - **Cese de Interacción:** **CESA INMEDIATAMENTE TODA INTERACCIÓN** después de enviar el mensaje final de confirmación de datos."
        f" - **Transferencia a Humano:** Si el cliente se frustra o el caso es complejo: 'Entiendo su preocupación. Este caso requiere la atención de uno de nuestros abogados. Por favor, contáctenos directamente al {PHONE_NUMBER} o envíe un correo a {SALES_EMAIL}.' "

        # Formato del Resumen (Uso Interno)
        "**Condiciones de Resumen (Generar para {SALES_EMAIL}):** Genera un resumen cuando el cliente ha provisto sus 4 datos. "
        "**Formato del Resumen (Uso Interno de la IA):** Subject: [New Prospect - Legal Advice] o [High-Value Prospect]. Body: **Client Details:** Name: [Name], WhatsApp Number: [Number], Email: [Email, if available], City/Location: [Client's City/Location]. **Case Analysis (For Internal Use):** Legal Branch: [Relevant branch of law], Problem Summary: [Brief description of the legal problem.], Key Points: [Identify crucial facts and documents that are needed.]. **Recommendation to the Firm:** [Suggest 1 o 2 pasos inmediatos]. **Client's Objective:** [Describe lo que el cliente desea lograr]."
    )

    # 3. Formatear el Contexto RAG y la Pregunta
    context_text = "\n\n".join([item['metadata']['text'] for item in context.matches])

    rag_prompt = (
        f"CONTEXTO PROPORCIONADO PARA EL ANÁLISIS (RAG):\n{context_text}\n\n"
        f"Pregunta más reciente del Usuario: {query}"
    )

    # 4. Construir la Matriz de Mensajes (Super Prompt + Memoria + Pregunta)
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
        temperature=0.0 # Mantenemos baja para precisión legal, pero el prompt guía la creatividad
    )

    final_response_text = response.choices[0].message.content

    return final_response_text

# --- ENDPOINT PRINCIPAL ---

@app.post("/query")
async def process_query(data: QueryModel):
    """Endpoint principal para recibir la pregunta y devolver la respuesta."""
    try:
        if not await validate_recaptcha(data.recaptcha_token):
              raise HTTPException(status_code=403, detail="Validación reCAPTCHA fallida. Acceso denegado.")

        query_embedding = generate_embedding(data.question)
        query_results = retrieve_context(query_embedding)

        # 1. Generar la respuesta (que incluye el resumen interno y el mensaje al usuario)
        raw_llm_response = generate_final_response(data.question, query_results, data.history)

        # 2. Lógica para DETECTAR y ENVIAR el resumen
        summary_start_tag = "[INTERNAL_SUMMARY_START]"
        summary_end_tag = "[INTERNAL_SUMMARY_END]"
        
        # Verificar si hay un resumen interno para enviar
        if summary_start_tag in raw_llm_response and summary_end_tag in raw_llm_response:
            try:
                # Extraer el contenido del resumen
                summary_content = raw_llm_response.split(summary_start_tag)[1].split(summary_end_tag)[0].strip()
                
                # Intentar enviar el correo
                send_summary_email(summary_content, summary_content)
                
                # 3. Limpiar la respuesta para el usuario (eliminar el resumen y las etiquetas)
                user_response = raw_llm_response.replace(summary_start_tag + summary_content + summary_end_tag, "").strip()
            except Exception as e:
                # Si falla el parseo o el envío, se registra el error y se envía la respuesta cruda (o se limpia solo las etiquetas)
                print(f"Advertencia: Fallo en el procesamiento del resumen interno. {e}")
                user_response = raw_llm_response.replace(summary_start_tag, "").replace(summary_end_tag, "").strip()
        else:
            # Si no hay etiquetas, la respuesta va directamente al usuario
            user_response = raw_llm_response

        return {"answer": user_response}

    except Exception as e:
        print(f"Error procesando la consulta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# --- INICIO LOCAL (Para pruebas) ---
if __file__ == "__main__":
    port_local = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port_local)
