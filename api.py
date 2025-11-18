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
GENERATION_MODEL = "gpt-4o-mini"
TOP_K = 5

# --- CONTACTOS Y DETALLES DE VENTA ---
PHONE_NUMBER = "+593 98 375 6678"
SALES_EMAIL = "leads@abogados-sf.com" 
CONSULTATION_COST = "40 USD"
CONSULTATION_CREDIT_MESSAGE = f"Recuerda que este monto, en caso de que llevemos contigo el caso, **se acredita al costo total del servicio como descuento**."

# --- MODELO DE DATOS DE ENTRADA ---
class QueryModel(BaseModel):
    """Define la estructura de la solicitud JSON que recibirá el API."""
    question: str
    recaptcha_token: str
    history: list[dict] = []

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
    raise e


# --- LÓGICA DE ENVÍO DE EMAIL (VÍA SENDGRID API) ---

def send_summary_email(summary_content: str, recipient: str = SALES_EMAIL):
    """Función para enviar el resumen interno por correo electrónico usando la API de SendGrid."""
    
    if not SENDGRID_API_KEY:
        print("ERROR DE CONFIGURACIÓN: SENDGRID_API_KEY no definida. Email no enviado.")
        return False
        
    subject_line = "Alerta de Lead: Revisión Manual de Contenido (Error de Formato)"
    body_content = summary_content

    try:
        subject_tag = "Subject:"
        body_tag = "Body:"
        
        if subject_tag in summary_content and body_tag in summary_content:
            
            subject_start = summary_content.find(subject_tag)
            body_start = summary_content.find(body_tag)
            
            if subject_start != -1 and body_start > subject_start:
                subject_line = summary_content[subject_start + len(subject_tag):body_start].strip()
                body_content = summary_content[body_start + len(body_tag):].strip()

        elif subject_tag in summary_content:
            subject_line = summary_content.split(subject_tag)[1].strip()
            body_content = summary_content
            print("ADVERTENCIA: Solo se encontró 'Subject:'. Usando el contenido completo como cuerpo.")
            
    except Exception as e:
        print(f"ERROR DE PARSING FATAL en send_summary_email: {e}")
        subject_line = "Error Inesperado en el Lead (Fallo de Parsing)"
        body_content = "Contenido original fallido:\n" + summary_content


    try:
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

# --- FUNCIÓN CLAVE CON OPTIMIZACIÓN DE PROMPT Y MEJORA DEL RESUMEN ---

def generate_final_response(query, context, history):
    """
    Genera la respuesta final. El CONTEXTO RAG se inyecta en el System Prompt.
    Versión con SUMMARY_EMAIL simplificado y flujo de contacto forzado.
    """
    
    # 1. Preparar RAG Context
    context_text = "\n\n".join([item['metadata']['text'] for item in context.matches])
    
    # 2. SUPER PROMPT COMPACTO Y OPTIMIZADO (VERSIÓN JSON FINAL)
    system_prompt = (
        "Eres Agorito, un Asistente Legal Virtual, experto en Derecho Constitucional, Civil y de Familia de la ley Ecuatoriana. "
        "Tu personalidad es **vendedora, carismática y siempre profesional**. "
        "Tu objetivo es **generar una respuesta JSON** con dos claves: `summary_email` (para el equipo de ventas) y `user_response` (para el cliente). \n\n"
        
        "**Reglas Clave de Venta (Flujo Secuencial y NO negociable):**\n"
        "1. **Alcance:** Limítate a Constitucional, Civil y Familia. Si no aplica, el `user_response` debe usar la 'Regla de Cierre de Contraste' (ver abajo) y `summary_email` debe ser `''`."
        "2. **Fase 1 (Recolección de Hechos - FLUIDA Y CONSOLIDADA):** Tu única tarea es recabar los **5 hechos clave del caso (QUÉ, QUIÉN, CUÁNDO, DÓNDE OCURRIÓ, CIUDAD DE RESIDENCIA ACTUAL/JURISDICCIÓN)** y datos de apoyo (ej. Edad). **PROHIBIDO** preguntar por Nombre, WhatsApp, Correo o Preferencia de Consulta en esta fase. **NO AVANCES a la Fase 2 hasta que se hayan recopilado los 5 hechos + Edad. Para la ubicación, CONSOLIDA las preguntas de 'dónde' y 'ciudad' en una sola pregunta de ubicación y jurisdicción.** Usa un lenguaje natural y conversacional."
        f"3. **Fase 2 (Análisis VENDEDOR y CTA MEJORADO):** SOLO después de recopilar los 5 hechos + Edad, ofrece un análisis preliminar de **Nivel 6-8 (semi-profundo y satisfactorio para el cliente)**. Este análisis DEBE: \n"
        "   - **EVITAR la repetición o listado de los datos** que acabas de recolectar. Inicia directamente con una frase de transición, como 'Según los datos proporcionados, su caso se enmarca en...' o 'Con la información clave, puedo ofrecerle el siguiente análisis preliminar...'.\n"
        "   - **Generar un texto ÚNICO y COHESIVO** (no uses listas ni viñetas como 'Rama Legal:', 'Problema Central:', etc.). El análisis debe fluir de forma profesional y persuasiva, integrando la Identificación de la Rama Legal, el Problema Central y el Derecho Vulnerado.\n"
        "   - **Evitar citar artículos o dar pasos procesales completos.**\n"
        "   - **DEBE TERMINAR con el CTA mejorado:** 'Podemos ofrecerte una estrategia ideal para llevar tu caso. ¿Quisieras agendar una consulta de {CONSULTATION_COST}, entendiendo que este monto es un **adelanto** que se acredita al costo total del servicio si decides trabajar con nosotros?'"
        "4. **Fase 3 (Recolección de Contacto - ESTRICTA Y POR PASOS):** **SOLO SI el cliente acepta el CTA**, tu objetivo es obtener los **4 datos de contacto** (Nombre, WhatsApp, Correo, Preferencia) en la siguiente secuencia de pasos: \n"
        "   - **Paso 3.1 (Nombre y WhatsApp):** Si el cliente acepta, tu **PRIMERA** respuesta debe ser pedir **ÚNICAMENTE** el **Nombre** y el **Número de WhatsApp**."
        "   - **Paso 3.2 (Correo y Preferencia):** Después de recibir el Nombre y WhatsApp, tu **SIGUIENTE** respuesta debe ser pedir **ÚNICAMENTE** el **Correo Electrónico** y la **Preferencia de Consulta** (Virtual/Presencial)."
        "   - **ACTIVACIÓN CONDICIÓN A:** Solo se activa la `Condición A` si los **CUATRO DATOS (Nombre, WhatsApp, Correo, Preferencia)** están presentes en el historial, y solo entonces el `summary_email` debe generarse."
        
        "**Formato de Salida ESTRICTO (JSON):**\n"
        "**Condición A: VENTA FINALIZADA (4 Datos de Contacto Recolectados):**\n"
        "   - **`summary_email`:** Contiene el resumen profesional, comenzando con **'Subject:'** y seguido de **'Body:'**. \n"
        "   - **`user_response`:** Contiene **ÚNICAMENTE** el mensaje de confirmación de agendamiento: '¡Perfecto! Ya tengo toda la información. Pronto alguien de nuestro equipo se pondrá en contacto contigo a través de tu [WhatsApp o correo] para coordinar la fecha y hora de tu consulta de 40 USD, que se acreditará al costo total del servicio.' **NO INCLUYAS RESÚMENES DE DATOS EN ESTE CAMPO.**\n\n"
        "**Condición B: CONVERSACIÓN, ANÁLISIS, O CESE DE INTERACCIÓN:**\n"
        "   - **`summary_email`:** Debe ser una **cadena vacía** (`''`).\n"
        "   - **`user_response`:** Debe ser el mensaje de Agorito para el cliente (e.g., recolección de hechos, análisis legal, o pregunta de seguimiento de datos de contacto). **El `user_response` NUNCA puede ser el mensaje de Condición A a menos que se hayan obtenido los 4 datos de contacto.**\n\n"
        
        # 🔑 TEMPLATE SIMPLIFICADO Y CORREGIDO
        "**Contenido del `summary_email` (Solo Condición A - FORMATO SIMPLIFICADO Y COMPLETO):**\n"
        "Subject: [New Lead] - [Resumen breve del caso, ej. Demanda de alimentos en Quito]\n"
        "Body: \n"
        "**Datos del Cliente (Lead):**\n"
        "Name: [Nombre]\n" 
        "WhatsApp Number: [Número de WhatsApp]\n"
        "Email: [Correo Electrónico]\n"
        "Consultation Type: [Virtual o Presencial]\n"
        "City/Location: [Ciudad/Ubicación - Recopilado en Fase 1]\n\n"
        "**Resumen del Caso (4 W's):**\n"
        "QUÉ (What happened): [Resumen conciso del evento basado en la conversación].\n"
        "QUIÉN (Who is involved): [Lista de partes clave involucradas].\n"
        "CUÁNDO (When did it happen): [Cronología o fecha del evento].\n"
        "DÓNDE (Where did it happen): [Lugar donde ocurrió el evento].\n\n"
        
        # *** ANÁLISIS OFRECIDO AL CLIENTE Y ESTRATEGIA INTERNA ***
        "**Análisis Preliminar Ofrecido al Cliente:**\n"
        "[Incluir aquí EXACTAMENTE el texto del análisis preliminar de Nivel 6-8 que se le dio al cliente en la Fase 2, justo antes de preguntar por la consulta de 40 USD.]\n\n"
        
        "**Estrategia Legal Sugerida (Para el equipo de ventas):**\n"
        "[Escribir un análisis de 2-3 frases y la estrategia legal concisa (3 a 5 pasos) aquí. Este es el análisis de alto nivel para el equipo. No se requiere citar leyes, solo la estrategia sugerida.]\n"
        
        # INYECCIÓN DE CONTEXTO PARA OPTIMIZACIÓN DE VELOCIDAD
        f"**CONTEXTO RAG PARA EL ANÁLISIS:** Utiliza el siguiente contexto legal extraído para responder a la pregunta del usuario. Si el contexto es débil o irrelevante, sigue las reglas de Contraste/Venta.\n"
        f"--- CONTEXTO ---\n{context_text}\n--- FIN CONTEXTO ---\n"
        
        "**Reglas de Emergencia (user_response):**"
        f" - **Regla de Cierre de Contraste:** 'Lamentablemente, ese asunto está fuera de nuestra especialidad. Si lo desea, puede contactarnos directamente al {PHONE_NUMBER} o envíe un correo a {SALES_EMAIL} para ver si podemos recomendarle un colega.'"
        " - **Cese de Interacción (Final):** Si el cliente responde con un simple 'gracias' después de la confirmación, el `user_response` debe ser: 'A ti. Feliz día.' o '¡Gracias a ti!'"
    )

    # 3. Construir la Matriz de Mensajes
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Añadir historial de conversación
    messages.extend(history)

    # Añadir la pregunta actual del usuario
    messages.append({"role": "user", "content": query})

    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    final_response_text = response.choices[0].message.content
    return final_response_text

# --- ENDPOINT PRINCIPAL ---

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
        
        print(f"DEBUG: RAW LLM RESPONSE (JSON):\n{raw_llm_response}")

        # 3. Lógica para PARSEAR el JSON y ENVIAR el resumen interno
        try:
            llm_output = json.loads(raw_llm_response)
            
            summary_content = llm_output.get("summary_email", "").strip()
            user_response = llm_output.get("user_response", "").strip()
            
            # Llamar a SendGrid SOLO si el campo summary_email NO está vacío
            if summary_content:
                send_summary_email(summary_content)
                
        except json.JSONDecodeError as e:
            # En caso de que el modelo falle al generar JSON
            print(f"ERROR DECODE: Fallo al decodificar JSON de la respuesta del LLM. {e}")
            user_response = "Disculpa, ha ocurrido un error de procesamiento. Por favor, reformula tu última respuesta o contáctanos directamente al +593 98 375 6678."
            summary_content = ""
            
        except Exception as e:
            # Otros errores de procesamiento
            print(f"ERROR INESPERADO en el parsing de la respuesta: {e}")
            user_response = "Disculpa, ha ocurrido un error de procesamiento. Por favor, reformula tu pregunta o contáctanos directamente al +593 98 375 6678."
            summary_content = ""

        # 4. Devolver la respuesta al usuario
        return {"answer": user_response}

    except Exception as e:
        # Esto captura errores de Pinecone, OpenAI, o en el embedding/context retrieval.
        print(f"Error CRÍTICO procesando la consulta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# --- INICIO LOCAL (Para pruebas) ---
if __name__ == "__main__":
    port_local = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port_local)
