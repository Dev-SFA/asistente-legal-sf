import os
import json
from flask import Flask, request, jsonify
# Librerías necesarias para SendGrid API
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
# Asegúrate de tener las librerías de Google Cloud que uses, por ejemplo:
# from google.cloud import secretmanager
# from google.cloud import aiplatform

# Inicialización de la aplicación Flask
app = Flask(__name__)

# --- CONFIGURACIÓN DE VARIABLES DE ENTORNO ---
# Estas variables deben estar configuradas en Google Cloud Run
SALES_EMAIL = os.environ.get("SALES_EMAIL", "leads@abogados-sf.com")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY") 
# Asegúrate de tener tus credenciales LLM aquí
# LLM_PROJECT_ID = os.environ.get("LLM_PROJECT_ID")
# LLM_REGION = os.environ.get("LLM_REGION") 

# --- FUNCIÓN CENTRAL DE ENVÍO DE EMAIL (VÍA SENDGRID API) ---

def send_summary_email(subject: str, body: str, recipient: str = SALES_EMAIL):
    """
    Función para enviar el resumen interno por correo electrónico usando la API de SendGrid.
    Utiliza la clave API configurada como variable de entorno SENDGRID_API_KEY.
    """
    
    if not SENDGRID_API_KEY:
        print("ERROR DE CONFIGURACIÓN: SENDGRID_API_KEY no definida. Email no enviado.")
        return False
        
    try:
        # 1. Parsear el Subject y el Body del texto generado por el LLM
        # Asume el formato: "Subject: [Asunto] Body: [Cuerpo]"
        if "Subject:" in subject and "Body:" in body:
            subject_line = subject.split("Subject:")[1].strip()
            body_content = body.split("Body:")[1].strip()
        else:
            # Fallback en caso de que el formato del LLM sea incorrecto
            print("ADVERTENCIA: Formato de LLM inesperado. Usando texto crudo.")
            subject_line = "Alerta de Lead: Revisión Manual de Contenido"
            body_content = subject
    except Exception:
        # Fallback de seguridad
        subject_line = "Error Inesperado en el Lead"
        body_content = "Contenido fallido: " + subject

    try:
        # 2. Crear el objeto Mail
        message = Mail(
            from_email=SALES_EMAIL,              # Ej: "leads@abogados-sf.com"
            to_emails=recipient,                 # El destinatario final (tú mismo)
            subject=subject_line,
            plain_text_content=body_content      # Envío como texto plano
        )
        
        # 3. Conectar y enviar (la API usa HTTPS/443, no el puerto SMTP)
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        # 4. Revisar la respuesta de la API (200 o 202 es éxito)
        if response.status_code in [200, 202]:
            print(f"ÉXITO: Email de resumen enviado a {recipient}. Código: {response.status_code}")
            return True
        else:
            print(f"ERROR SG: Fallo al enviar email. Código: {response.status_code}. Cuerpo: {response.body}")
            return False

    except Exception as e:
        print(f"ERROR FATAL al enviar email por SendGrid: {e}")
        return False

# --- FUNCIÓN SIMULADA DEL LLM (REEMPLAZAR CON TU LÓGICA REAL) ---

def call_llm_and_check_if_lead_is_ready(chat_history: str) -> tuple[str, bool]:
    """
    [DEBES REEMPLAZAR ESTO CON TU CÓDIGO LLM REAL]
    Simula la llamada al LLM para obtener el siguiente mensaje y verificar si el lead está listo.
    
    El LLM debe retornar el resumen con el formato: "Subject: [Asunto del email] Body: [Cuerpo del email]"
    """
    # Ejemplo de un LLM que ha terminado la conversación y ha capturado los datos
    if "Confirmo cita" in chat_history:
        # 1. Generar el resumen con el formato requerido para el email
        email_content = (
            "Subject: NUEVO LEAD: Consulta por Invasión de Terreno"
            " Body: Se ha capturado un lead del chat. "
            " Cliente: Adrián Rosales. "
            " Contacto: 48787979879 / adrian@gmail.com. "
            " Caso: Problema con invasión de 10m2 de terreno en Quito. "
            " Preferencia: Tarde. "
            " ACCIÓN: Contactar por WhatsApp para agendar cita de 40 USD."
        )
        return email_content, True # Retorna el contenido del email y True para indicar que se debe enviar
    
    # Ejemplo de un LLM que continúa la conversación
    return "Gracias por la información. Por favor, deme su nombre completo.", False

# --- ENDPOINT PRINCIPAL DE LA APLICACIÓN ---

@app.route('/', methods=['POST'])
def handle_chat_request():
    """
    Maneja la solicitud POST con el historial del chat.
    """
    try:
        data = request.get_json()
        chat_history = data.get('history', '')
        
        if not chat_history:
            return jsonify({"response": "Por favor, proporcione el historial de chat."}), 400

        # Llama a tu lógica de LLM (¡REEMPLAZA ESTA FUNCIÓN!)
        llm_output, should_send_email = call_llm_and_check_if_lead_is_ready(chat_history)
        
        # Si la lógica del LLM indica que el lead está listo, envía el resumen
        if should_send_email:
            # Asume que el LLM genera el output con Subject: y Body:
            subject_part = llm_output.split("Body:")[0] 
            body_part = llm_output
            
            # Intenta enviar el email (la función maneja los errores)
            success = send_summary_email(subject=subject_part, body=body_part, recipient=SALES_EMAIL)
            
            if success:
                # Retorna la respuesta final del LLM al usuario del chat
                return jsonify({"response": "¡Perfecto! Hemos registrado sus datos. Pronto nos pondremos en contacto."})
            else:
                # Si falla el envío de email, puedes notificar a una cola de errores
                print("FALLO EN EL ENVÍO DEL EMAIL. EL LEAD NO HA SIDO NOTIFICADO.")
                return jsonify({"response": "Hay un error en nuestro sistema de notificaciones. Por favor, contacte directamente al +593... Gracias."})


        # Si el LLM aún está en conversación, devuelve su respuesta al usuario
        return jsonify({"response": llm_output})

    except Exception as e:
        app.logger.error(f"Error en la solicitud: {e}")
        return jsonify({"error": "Ocurrió un error interno."}), 500

if __name__ == "__main__":
    # La aplicación debe escuchar en el puerto que le asigna el entorno (Cloud Run)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
