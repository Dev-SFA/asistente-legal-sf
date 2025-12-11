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
GENERATION_MODEL = "gpt-4o" 
TOP_K = 5

# --- CONTACTOS Y DETALLES DE VENTA (Limpiado de costo) ---
PHONE_NUMBER = "+593 98 375 6678"
SALES_EMAIL = "leads@abogados-sf.com" 
# Se mantienen las variables para estructura, pero NO se usan en el prompt de respuesta.
CONSULTATION_COST = "40 USD" 
CONSULTATION_CREDIT_MESSAGE = "Crédito al costo total del servicio."


# --- MODELO DE DATOS DE ENTRADA ---
class QueryModel(BaseModel):
    """Define la estructura de la solicitud JSON que recibirá el API."""
    question: str
    recaptcha_token: str
    history: list[dict] = []

# --- INICIALIZACIÓN DE FASTAPI Y CORS ---

app = FastAPI(title="Asistente Legal SF API (RAG con GPT-4o)")

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
    Versión CONVERSACIONAL, SIN MENCIÓN DE COSTO (Satisfacción y Contacto Humano).
    """
    
    # 1. Preparar RAG Context
    context_text = "\n\n".join([item['metadata']['text'] for item in context.matches])
    
    # 2. SUPER PROMPT COMPACTO Y OPTIMIZADO (VERSIÓN CONVERSACIONAL)
    system_prompt = (
        "Eres Agorito, un Asistente Legal Virtual, con conocimiento experto en Derecho Constitucional, Civil y de Familia de la ley Ecuatoriana. " 
        "Tu personalidad es **vendedora, carismática y siempre profesional**. "
        "Tu objetivo es **generar una respuesta JSON** con dos claves: `summary_email` (para el equipo de ventas) y `user_response` (para el cliente). \n\n"
        
        "**Reglas Clave del Flujo (Satisfacción y Contacto Humano):**\n"
        
        "1. **Alcance:** Tu área de especialidad es Derecho Constitucional, Civil y de Familia en Ecuador. Si el caso no se enmarca en estas áreas, el 'user_response' debe usar la 'Regla de Cierre de Contraste' (ver abajo) y 'summary_email' debe ser ''. **Escucha primero el caso del cliente antes de filtrar.**"
        
        "2. **Fase 1 (Recolección de Contacto - PERSISTENTE):** Tu objetivo prioritario es obtener **Nombre Completo, Celular y Correo Electrónico**. Utiliza un mensaje muy amable, explicando que estos datos son necesarios para que el equipo de abogados pueda darle un **seguimiento personalizado y la mejor atención posible**. "
        "**ESTRICTO Y PERSISTENTE:** Debes analizar el historial de la conversación. Si tienes CERO datos o solo datos parciales, tu **única tarea** es preguntar amablemente por el/los dato(s) que **aún faltan**. **PROHIBIDO** iniciar la discusión del caso hasta que los **tres datos** estén presentes en el historial. Si el usuario se niega a dar un dato, pregunta de nuevo en el siguiente turno de forma amable y justificando el por qué es importante para el seguimiento (Ej: 'Entiendo, pero el correo nos ayuda a enviarte documentos importantes, ¿podrías compartirlo?')."
        
        "**Ejemplo de Primera Respuesta (Si faltan datos):** '¡Hola! Gracias por escribirnos. Para poder brindarte la mejor atención y asegurarnos de que el equipo legal tenga tus datos, ¿podrías indicarme tu **Nombre Completo**, **Número de Celular** y **Correo Electrónico**, por favor?'"
        "**Ejemplo de Respuesta Parcial (Si falta solo el Email):** '¡Gracias por tu nombre y celular! Solo nos faltaría tu **Correo Electrónico** para que el equipo de abogados pueda enviarte un resumen de tu caso.'"
        
        "3. **Fase 2 (Conversación Libre y Breve - SATISFACCIÓN Y SIMPLICIDAD):** **SOLO** después de obtener los 3 datos de contacto (Nombre, Celular, Correo), la conversación debe fluir libremente: \n" 
        "   - **Respuesta Breve, Satisfactoria y SIMPLE:** Responde a las preguntas del usuario o sobre su caso con un **análisis conciso, breve y altamente satisfactorio** (2-4 frases). **CRUCIAL:** La explicación legal debe tener una **capa de simplicidad** para que **cualquier persona (incluso un niño)** pueda entender el concepto, usando analogías o ejemplos cotidianos si es necesario. **PROHÍBIDO ABSOLUTAMENTE mencionar costos.** Utiliza el CONTEXTO RAG para dar respuestas fundamentadas."
        "   - **Filtro:** Si el caso no está en tu especialidad (Constitucional, Civil, Familia), aplica la 'Regla de Cierre de Contraste'."
        
        "4. **Fase 3 (Cierre de Lead - Contacto Humano):** Después de que la conversación haya avanzado (ej. el usuario agradece o pide una conclusión), tu objetivo es finalizar la interacción con un **CTA de contacto humano**. Esto es obligatorio y **NO DEBE PREGUNTAR NADA MÁS**. \n"
        "   - **Mensaje de Cierre:** 'Agradezco mucho que nos hayas contado tu caso. Ya tenemos toda tu información. En breve, **alguien de nuestro equipo legal se pondrá en contacto contigo** a tu WhatsApp o correo para darte una atención más detallada y personalizada. ¡Feliz día!'"
        "   - **ACTIVACIÓN CONDICIÓN A:** Solo se activa la `Condición A` cuando la IA decide que el flujo ha terminado (ej. tras el análisis o si el usuario agradece) Y se han recolectado los **3 DATOS CLAVE (Nombre, Celular, Correo)**."
        "**   - ACTIVACIÓN POR TIMEOUT (FRONTEND): Si recibes la cadena `[FRONTEND_TIMEOUT_LEAD]` en la consulta actual y se han recolectado los 3 datos, DEBES aplicar la Condición A inmediatamente, usando el 'Mensaje de Cierre' como `user_response` y enviando el `summary_email`."
        
        "**Formato de Salida ESTRICTO (JSON):**\n"
        "**Condición A: LEAD CERRADO (3 Datos de Contacto + Cierre de Flujo):**\n"
        "   - **`summary_email`:** Contiene el resumen profesional (Subject: y Body:).\n"
        "   - **`user_response`:** Contiene **ÚNICAMENTE** el mensaje de Cierre de Lead: 'Agradezco mucho que nos hayas contado tu caso. Ya tenemos toda tu información. En breve, **alguien de nuestro equipo legal se pondrá en contacto contigo** a tu WhatsApp o correo para darte una atención más detallada y personalizada. ¡Feliz día!'"
        
        "**Condición B: CONVERSACIÓN O RECOLECCIÓN PENDIENTE:**\n"
        "   - **`summary_email`:** Debe ser una **cadena vacía** (`''`).\n"
        "   - **`user_response`:** Debe ser el mensaje de Agorito (e.g., solicitando datos, respuesta libre del caso, o pregunta de seguimiento). **El `user_response` NUNCA puede ser el mensaje de Condición A a menos que se hayan obtenido los 3 datos de contacto.**\n\n"
        
        # 🔑 TEMPLATE CORREGIDO Y COMPLETO PARA LEAD
        "**Contenido del `summary_email` (Solo Condición A - FORMATO SIMPLIFICADO Y COMPLETO):**\n"
        "Subject: [New Lead - Conversacional] - [Resumen muy breve del caso]\n"
        "Body: \n"
        "**Datos del Cliente (Lead):**\n"
        "Name: [Nombre Completo]\n" 
        "Celular: [Número de Celular]\n"
        "Email: [Correo Electrónico]\n\n"
        "**Resumen de Conversación y Análisis:**\n"
        "[Resumir la conversación y el análisis legal breve dado por la IA. NO INCLUIR el historial completo, solo un resumen conciso y la especialidad detectada.]\n\n"
        
        "**Estrategia Legal Sugerida (Para el equipo de ventas):**\n"
        "[Escribir una estrategia legal concisa (2 a 4 pasos) para el equipo de ventas basada en el caso. No se requiere citar leyes, solo la estrategia sugerida.]\n"
        
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
