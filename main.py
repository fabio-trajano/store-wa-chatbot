import os
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from twilio.rest import Client

# Import our LangChain agent and database helper functions.
from langchain_agent import process_customer_query
from database import init_db, add_pending_approval, get_pending_approval, mark_approved

# Load environment variables from .env file.
load_dotenv()

# Twilio and store configuration.
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
STORE_NAME = os.getenv("STORE_NAME", "My Store")

# Parse the approval group numbers from a comma-separated list.
APPROVAL_GROUP_NUMBERS = [num.strip() for num in os.getenv("APPROVAL_GROUP_NUMBERS").split(",")]

# Initialize Twilio Client.
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Create FastAPI app.
app = FastAPI()

# Initialize the SQLite database (creates table if not exists).
init_db()

def send_whatsapp_message(to_number: str, message: str):
    """
    Helper function to send WhatsApp messages via Twilio.
    """
    try:
        msg = client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            to=f"whatsapp:{to_number}"
        )
        return msg.sid
    except Exception as e:
        print(f"Error sending message to {to_number}: {e}")
        return None

@app.post("/webhook/customer")
async def customer_webhook(request: Request):
    """
    Webhook for customer messages coming from WhatsApp.
    Expects Twilio to send form data containing 'From' (customer phone) and 'Body' (message).
    Processes the query using a LangChain agent and then saves the generated response
    in pending approvals while notifying the approval group.
    """
    form_data = await request.form()
    customer_phone = form_data.get("From", None)
    customer_message = form_data.get("Body", None)

    if not customer_phone or not customer_message:
        raise HTTPException(status_code=400, detail="Missing 'From' or 'Body' in the request")

    # Process the customer query through the LangChain agent.
    try:
        generated_response = process_customer_query(customer_message)
    except Exception as e:
        generated_response = "Desculpe, ocorreu um erro ao processar o seu pedido."
        print("Error processing customer query:", e)

    # Generate a unique ID for this pending approval.
    pending_id = str(uuid.uuid4())[:8]  # shorten uuid to 8 characters

    # Save the pending approval record to the SQLite database.
    add_pending_approval(pending_id, customer_phone, customer_message, generated_response)

    # Send the generated response to all numbers in the approval group for owner review.
    approval_message = (f"[ID: {pending_id}] Resposta sugerida: {generated_response}\n"
                        f"Responda 'APROVAR {pending_id}' para aprovar esta mensagem.")
    for number in APPROVAL_GROUP_NUMBERS:
        send_whatsapp_message(number, approval_message)

    return JSONResponse(content={"status": "pending approval", "id": pending_id})

@app.post("/webhook/approval")
async def approval_webhook(request: Request):
    """
    Webhook for handling messages from the approval group.
    Expects a message in the format 'APROVAR <pending_id>'.
    Once approved by any number in the approval group, sends the final response to the customer,
    and a confirmation message to all the other numbers in the approval group.
    """
    form_data = await request.form()
    group_phone = form_data.get("From", None)
    group_message = form_data.get("Body", "").strip()

    if not group_phone or not group_message:
        raise HTTPException(status_code=400, detail="Missing 'From' or 'Body' in the request")

    # Process only messages starting with "APROVAR"
    if not group_message.upper().startswith("APROVAR"):
        return JSONResponse(content={"status": "ignored"})

    parts = group_message.split()
    if len(parts) < 2:
        return JSONResponse(content={"status": "invalid format; use 'APROVAR <ID>'"})
    pending_id = parts[1].strip()

    # Retrieve the pending approval record from the database.
    record = get_pending_approval(pending_id)
    if not record:
        return JSONResponse(content={"status": "pending id not found"})

    # Mark the record as approved.
    mark_approved(pending_id)

    # Compose the final message to send to the customer in European Portuguese.
    customer_phone = record["customer_phone"]
    final_message = f"Olá, obrigado por entrar em contacto com a {STORE_NAME}. {record['generated_response']}"
    send_whatsapp_message(customer_phone, final_message)

    # Send a confirmation message to the other approval group numbers.
    confirmation_message = f"O pedido [ID: {pending_id}] foi aprovado pelo número {group_phone}."
    for number in APPROVAL_GROUP_NUMBERS:
        # Avoid sending the confirmation to the approving number.
        if number != group_phone:
            send_whatsapp_message(number, confirmation_message)

    return JSONResponse(content={"status": "message sent to customer", "id": pending_id})

# To run the server directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
