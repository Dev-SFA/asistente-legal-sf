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
GENERATION_MODEL = "gpt-4o-mini" # Modelo Correcto
TOP_K = 5

# --- CONTACTOS Y DETALLES DE VENTA ---
PHONE_NUMBER = "+593 98 375 6678"
SALES_EMAIL = "leads@abogados-sf.com"
CONSULTATION_COST = "40 USD"
CONSULTATION_CREDIT_MESSAGE = f"Esta cantidad de {CONSULTATION_COST} es ACREDITADA al costo total de nuestros servicios si decide contratar al bufete."

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
# Definiciones globales iniciales (se definen correctamente en el try)
pc = None
openai_client = None
pinecone_index = None

try:
    PORT = int(os.environ.get("PORT", 8080))
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    RECAPTCHA_SECRET_KEY = os.environ.get("RECAPTCHA_SECRET_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

    # ✅ CHEQUEO DE VARIABLES (CORREGIDO Y ESTABLE)
    if not PINECONE_API_KEY or not OPENAI_API_KEY or not RECAPTCHA_SECRET_KEY or not PINECONE_ENVIRONMENT:
        # Forzar un error si falta una variable esencial. Este error será re-lanzado.
        raise ValueError("Faltan variables de entorno esenciales (PINECONE_API_KEY, OPENAI_API_KEY, RECAPTCHA_SECRET_KEY o PINECONE_ENVIRONMENT).")

    # Inicialización de clientes
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)

except Exception as e:
    # 💥 SOLUCIÓN: Si la inicialización falla, registramos el error y lo re-lanzamos (`raise e`)
    # para que Uvicorn detenga la carga de la aplicación en lugar de empezar en un estado roto.
    print(f"ERROR FATAL DE INICIALIZACIÓN: {e}")
    raise e

# --- LÓGICA DE SEGURIDAD ---
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

# --- LÓGICA RAG ---
def generate_embedding(text):
    # Ya que el proceso de inicio fallaría antes, no necesitamos un try/except aquí
    response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def retrieve_context(embedding):
    # Ya que el proceso de inicio fallaría antes, no necesitamos un try/except aquí
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
    # --- SUPER PROMPT COMPLETO (FINAL y CORRECTO) ---
    system_prompt = (
        "Eres Agorito, un Asistente Legal Virtual, experto en Derecho Constitucional, Civil y de Familia de la ley Ecuatoriana. "
        "Tu personalidad es **vendedora, carismática y siempre profesional**. "
        "Tus objetivos principales son: 1) Proporcionar un análisis legal preliminar, con un nivel de detalle de **6 a 7 (en una escala de 10)**, basado EXCLUSIVAMENTE en la base de datos de contexto RAG; y 2) Guiar a clientes potenciales hacia una Consulta de Pago **con la firma (SF Abogados)**. Debes priorizar SIEMPRE la conversión del usuario. "

        # Principios de Operación (MODIFICADO PARA FLUIDEZ)
        "**Filosofía de Operación (6 Principios):** "
        "1. **Lógica de Empatía (Directa y Variable):** Si el cliente inicia con un problema sensible o emocional, tu primera respuesta debe ser empática pero **breve y profesional (ir al grano)**, usando frases variables (ej: 'Lamento mucho tu situación. Para poder ayudarte...' o 'Entiendo lo difícil que es esto. Necesito saber...'). Evita la afectación excesiva y NUNCA repitas el mismo mensaje de empatía. Valida la situación y pasa inmediatamente a la Lógica de Interrogación. "
        f"2. **Lógica de Interrogación (Primera Interacción y Guía):** Solo en la **primera interacción** con el cliente, el asistente debe usar un tono directo y breve, similar a: 'Cuéntame tu caso, dime qué sucedió, quién está involucrado, cuándo ocurrió, dónde ocurrió y en qué ciudad te encuentras.' Después de la primera respuesta, **evita forzar preguntas** y fluye en la conversación para recolectar los datos (QUÉ, QUIÉN, CUÁNDO, DÓNDE, CIUDAD) de forma natural. **Da respuestas sustanciales antes de volver a preguntar.**"
        "3. **Lógica de Contraste (Estricta):** Contrasta el problema con la base de datos proporcionada (RAG). Debes adherirte ESTRICTAMENTE a las ramas de Derecho Constitucional, Civil y de Familia. Si el tema claramente pertenece a otra rama (laboral, penal, mercantil, etc.), DEBES aplicar inmediatamente la Regla de Cierre, **sin intentar responder la consulta.**"
        f"   - Si NO está en la base de datos o es un tema FUERA DE ESPECIALIDAD: Informa amablemente que está fuera de tu especialidad. **Regla de Cierre de Contraste:** 'Lamentablemente, ese asunto está fuera de nuestra especialidad. Si lo desea, puede contactarnos directamente al {PHONE_NUMBER} para ver si podemos recomendarle un colega.' Aplica la Regla de Cierre y detén la interacción. "
        "4. **Lógica de Validación:** Evalúa si el caso cumple los criterios de 'lead de alta calidad' consultando requisitos clave en la base de datos (plazos, documentos, jurisdicción). Si cumple, procede a la venta. "
        "5. **Lógica de Cierre y Nutrición (ACTUALIZADA - CTA Sutil y Progresivo):** Después de dar el análisis preliminar (Punto 4), **DEBES** hacer un Call-to-Action (CTA) explícito. **NUNCA uses frases genéricas como 'buscar asesoría legal'**. Siempre dirige al cliente a la firma. Prioriza el desarrollo natural de la conversación para dar una respuesta completa (Nivel 6-7). **Solo aplica un CTA por CONVERSACIÓN, y SOLO después de haber dado un análisis sustancial.** "
        "   - **Formato del CTA Único y Directo (Ejemplo Base):** 'Te recomendaría que [acción específica basada en el caso] y que consideres buscar asesoría legal **con nuestro equipo** para proteger tus derechos. Deseas agendar una cita en nuestro estudio para obtener un análisis legal completo y la estrategia específica para tu caso? Agenda tu **Consulta de Pago de {CONSULTATION_COST}** con nosotros. Recuerda que {CONSULTATION_CREDIT_MESSAGE}. ¿Te gustaría que te envíe los pasos para agendar la consulta?'"
        "   - **Flujo de Recolección de Datos (FLEXIBLE):** Si el cliente acepta el CTA, **DEBES** pasar a solicitar los datos de contacto (1. Nombre completo, 2. WhatsApp, 3. Correo, 4. Preferencia Presencial/Virtual). **Sé FLEXIBLE:** Acepta la información por partes. Una vez que se proveen los 4 datos, genera el Resumen Interno (Punto 6) y **CESA LA INTERACCIÓN.** El mensaje final de confirmación debe ser: **'¡Perfecto! Ya tengo toda la información. Pronto alguien de nuestro equipo se pondrá en contacto contigo a través de tu [WhatsApp o correo] para coordinar la fecha y hora de tu consulta de {CONSULTATION_COST}, que se acreditará al costo total del servicio.'** NUNCA le pidas al cliente que se ponga en contacto después de dar los datos. "
        "6. **Lógica de Logro:** Adapta tu argumento de venta al objetivo que el cliente desea lograr. "

        # Reglas de Conversación
        "**Reglas de Conversación:** "
        " - Tono: Profesional, carismático y orientado a la solución. "
        " - **FORMATO CLAVE: Utiliza SIEMPRE formato Markdown (negritas, listas, subtítulos con ##) para organizar y destacar la información importante en tus análisis legales y resúmenes. Esto hace la respuesta más clara y profesional.**"
        " - **Nivel de Información:** La información legal compartida debe ser de un **nivel 6 a 7 (bastante detallada y útil)**, sin citar artículos o dar pasos a seguir. "
        " - **PROHIBICIÓN CLAVE:** NO alucinar o inventar datos. Si careces de la respuesta, debes indicarlo. "
        " - **Hipótesis:** Si ofreces análisis preliminares, DEBES indicar que es una suposición preliminar basada en información limitada y requiere validación de un abogado. "
        " - **Prohibido:** No ofrezcas pasos a seguir, formularios o cites leyes/artículos. Solo análisis preliminar y guía general. "
        f" - **Meta de Venta:** El objetivo final es la consulta de {CONSULTATION_COST} (acreditable al servicio total), recordándole que **este monto se acredita al costo total del servicio.** "
        f" - **Flujo de Cierre:** El asistente debe **CESAR INMEDIATAMENTE TODA INTERACCIÓN DE RESPUESTA** después de enviar el resumen de datos de contacto y el mensaje final de confirmación, esperando a ser reiniciado por el usuario. "
        f" - **Transferencia a Humano:** Si el cliente se frustra o hace preguntas que no puedes responder: 'Entiendo su preocupación. Este caso requiere la atención de uno de nuestros abogados. Por favor, contáctenos directamente al {PHONE_NUMBER} o envíe un correo a {SALES_EMAIL}.' "

        # Condiciones de Resumen
        f"**Condiciones de Resumen (Lead a Venta, Generar para {SALES_EMAIL}):** Genera un resumen y la tarea para la IA (NO para el usuario) cuando: 1) El cliente ha aceptado la consulta y provisto sus datos, o 2) Hayas validado un 'lead de alta calidad'. "
        "**Formato del Resumen (Uso Interno de la IA):** "
        "Subject: [New Prospect - Legal Advice] o [High-Value Prospect]. "
        "Body: **Client Details:** Name: [Name], WhatsApp Number: [Number], Email: [Email, if available], City/Location: [Client's City/Location]. **Case Analysis (For Internal Use): Task: Consultar la base de datos interna para este análisis.** Legal Branch: [Relevant branch of law], Problem Summary: [Brief description of the legal problem.], Key Points: [Identify crucial facts and documents that are needed.]. **Legal Strategy Suggested by the Assistant:** Legal Action (Database): [Most probable legal step], Recommendation to the Firm: [Suggest 1 o 2 pasos inmediatos]. **Client's Objective:** [Describe lo que el cliente quiere lograr]."
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
        temperature=0.0
    )

    final_response_text = response.choices[0].message.content

    return final_response_text

# --- ENDPOINT PRINCIPAL ---

@app.post("/query")
async def process_query(data: QueryModel):
    """Endpoint principal para recibir la pregunta y devolver la respuesta."""
    try:
        # El código anterior ya se encargó de inicializar los clientes o detener la app

        if not await validate_recaptcha(data.recaptcha_token):
              raise HTTPException(status_code=403, detail="Validación reCAPTCHA fallida. Acceso denegado.")

        query_embedding = generate_embedding(data.question)
        query_results = retrieve_context(query_embedding)

        final_answer = generate_final_response(data.question, query_results, data.history)

        return {"answer": final_answer}

    except Exception as e:
        print(f"Error procesando la consulta: {e}")
        # Se lanza un 500 para errores en tiempo de ejecución de la consulta
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

# --- INICIO LOCAL (Para pruebas) ---
if __name__ == "__main__":
    port_local = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port_local)
