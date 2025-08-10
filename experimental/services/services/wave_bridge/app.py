"""WhatsApp Wave Bridge for AI Village Tutoring
Sprint R-3+AF4 Implementation with W&B Prompt Tuning.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
import wandb

from .language_support import SUPPORTED_LANGUAGES, auto_translate_flow, detect_language
from .metrics import ResponseMetrics
from .prompt_tuning import ABTestManager, PromptTuner
from .tutor_engine import AITutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp Wave Bridge", version="1.0.0")

# Initialize components
prompt_tuner = PromptTuner()
ab_test_manager = ABTestManager()
ai_tutor = AITutor()
metrics = ResponseMetrics()

# Initialize W&B for tracking conversations
wandb.init(
    project="aivillage-tutoring",
    job_type="whatsapp-bridge",
    config={
        "response_time_target": 5.0,  # seconds
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "ab_test_enabled": True,
    },
)


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle WhatsApp messages with W&B tracking and prompt optimization
    Target: <5 second response time.
    """
    start_time = time.time()

    try:
        # Parse Twilio webhook data
        form_data = await request.form()
        incoming_msg = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")
        message_sid = form_data.get("MessageSid", "")

        logger.info(
            f"Received WhatsApp message from {from_number}: {incoming_msg[:50]}..."
        )

        # Validate required fields
        if not incoming_msg or not from_number:
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Generate session ID for tracking
        session_id = generate_session_id(from_number, message_sid)

        # Detect language for personalization
        detected_lang = await detect_language(incoming_msg)

        # Log interaction to W&B immediately
        wandb.log(
            {
                "interaction_type": "whatsapp_incoming",
                "session_id": session_id,
                "language_detected": detected_lang,
                "message_length": len(incoming_msg),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_number_hash": hashlib.sha256(from_number.encode()).hexdigest()[
                    :8
                ],
            }
        )

        # Route to prompt-engineered tutor with timeout
        try:
            response = await asyncio.wait_for(
                get_tutor_response(
                    incoming_msg, from_number, session_id, detected_lang
                ),
                timeout=4.5,  # Leave 0.5s buffer for response formatting
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for session {session_id}")
            response = get_fallback_response(detected_lang)

        # Calculate response time
        response_time = time.time() - start_time

        # Track response metrics in background
        background_tasks.add_task(
            log_response_metrics, response, response_time, session_id, detected_lang
        )

        # Format and return Twilio response
        twiml_response = format_whatsapp_response(response)

        logger.info(f"Response sent in {response_time:.2f}s for session {session_id}")

        return PlainTextResponse(
            content=str(twiml_response), media_type="application/xml"
        )

    except Exception as e:
        logger.exception(f"Error processing WhatsApp webhook: {e!s}")

        # Log error to W&B
        wandb.log(
            {
                "error_type": "webhook_processing",
                "error_message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Return fallback response
        error_response = get_error_response()
        return PlainTextResponse(
            content=str(error_response), media_type="application/xml"
        )


async def get_tutor_response(
    message: str, from_number: str, session_id: str, detected_lang: str
) -> dict[str, Any]:
    """Generate tutoring response with W&B prompt optimization."""
    # Check if this is a greeting (new conversation)
    is_greeting = is_greeting_message(message, detected_lang)

    if is_greeting:
        # A/B test greeting responses
        greeting_variant = ab_test_manager.get_greeting_variant(from_number)

        wandb.log(
            {
                "interaction_type": "greeting",
                "ab_variant": greeting_variant,
                "session_id": session_id,
            }
        )

        response = await ai_tutor.generate_greeting(
            variant=greeting_variant, language=detected_lang, user_message=message
        )
    else:
        # Get optimized tutoring prompt from W&B
        optimized_prompt = await prompt_tuner.get_optimized_prompt(
            message_type="tutoring",
            language=detected_lang,
            context={"user_message": message, "session_id": session_id},
        )

        # Generate tutoring response
        response = await ai_tutor.generate_response(
            user_message=message,
            prompt_template=optimized_prompt,
            language=detected_lang,
            session_id=session_id,
        )

    # Handle translation if needed
    if detected_lang != "en" and detected_lang in SUPPORTED_LANGUAGES:
        response = await auto_translate_flow(response, detected_lang)

    return {
        "text": response,
        "language": detected_lang,
        "session_id": session_id,
        "is_greeting": is_greeting,
    }


def generate_session_id(from_number: str, message_sid: str) -> str:
    """Generate unique session ID for tracking."""
    combined = f"{from_number}_{message_sid}_{int(time.time())}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def is_greeting_message(message: str, language: str) -> bool:
    """Detect if message is a greeting."""
    greeting_patterns = {
        "en": ["hello", "hi", "hey", "start", "help"],
        "es": ["hola", "buenos", "ayuda"],
        "hi": ["नमस्ते", "हैलो", "मदद"],
        "sw": ["hujambo", "habari", "msaada"],
        "ar": ["مرحبا", "السلام", "مساعدة"],
        "pt": ["olá", "oi", "ajuda"],
        "fr": ["bonjour", "salut", "aide"],
    }

    patterns = greeting_patterns.get(language, greeting_patterns["en"])
    message_lower = message.lower()

    return any(pattern in message_lower for pattern in patterns)


def format_whatsapp_response(response_data: dict[str, Any]) -> str:
    """Format response for Twilio WhatsApp."""
    response = MessagingResponse()
    message = response.message()
    message.body(response_data["text"])
    return str(response)


def get_fallback_response(language: str) -> dict[str, Any]:
    """Fallback response for timeouts."""
    fallback_messages = {
        "en": "I'm processing your message. Please wait a moment...",
        "es": "Estoy procesando tu mensaje. Por favor espera un momento...",
        "hi": "मैं आपका संदेश प्रोसेस कर रहा हूं। कृपया एक क्षण प्रतीक्षा करें...",
        "sw": "Ninachakata ujumbe wako. Tafadhali subiri kidogo...",
        "ar": "أقوم بمعالجة رسالتك. يرجى الانتظار لحظة...",
        "pt": "Estou processando sua mensagem. Por favor, aguarde um momento...",
        "fr": "Je traite votre message. Veuillez patienter un moment...",
    }

    return {
        "text": fallback_messages.get(language, fallback_messages["en"]),
        "language": language,
        "is_fallback": True,
    }


def get_error_response() -> str:
    """Error response for system failures."""
    response = MessagingResponse()
    message = response.message()
    message.body(
        "Sorry, I'm having technical difficulties. Please try again in a moment."
    )
    return str(response)


async def log_response_metrics(
    response: dict[str, Any], response_time: float, session_id: str, language: str
) -> None:
    """Log detailed response metrics to W&B."""
    # Core metrics
    metrics_data = {
        "response_time": response_time,
        "response_length": len(response.get("text", "")),
        "language": language,
        "session_id": session_id,
        "is_greeting": response.get("is_greeting", False),
        "is_fallback": response.get("is_fallback", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Performance flags
    metrics_data["performance_target_met"] = response_time < 5.0
    metrics_data["response_quality"] = (
        "good"
        if response_time < 3.0
        else "acceptable"
        if response_time < 5.0
        else "slow"
    )

    # Log to W&B
    wandb.log(metrics_data)

    # Update local metrics tracker
    await metrics.update_metrics(metrics_data)

    logger.info(f"Logged metrics for session {session_id}: {response_time:.2f}s")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "whatsapp-wave-bridge",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }


@app.get("/metrics")
async def get_metrics():
    """Get current performance metrics."""
    return await metrics.get_summary()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
