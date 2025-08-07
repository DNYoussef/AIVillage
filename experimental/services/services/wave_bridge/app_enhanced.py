"""Enhanced WhatsApp Wave Bridge with Advanced Prompt Engineering
Part B: Agent Forge Phase 4 - Full Integration
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import os
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
import wandb

from .agent_forge.prompt_engineering.ab_testing import prompt_ab_test
from .agent_forge.prompt_engineering.prompt_baker import prompt_baker

# Import new prompt engineering components
from .agent_forge.prompt_engineering.tutor_prompts import tutor_prompt_engineer

# Import original components
from .language_support import SUPPORTED_LANGUAGES, auto_translate_flow, detect_language
from .metrics import ResponseMetrics
from .prompt_tuning import ABTestManager, PromptTuner
from .tutor_engine import AITutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhatsApp Wave Bridge - Enhanced with Prompt Engineering",
    version="2.0.0",
    description="Advanced AI tutoring with W&B-optimized prompt engineering",
)

# Initialize original components
original_prompt_tuner = PromptTuner()
original_ab_test_manager = ABTestManager()
ai_tutor = AITutor()
metrics = ResponseMetrics()

# Initialize enhanced prompt engineering components
enhanced_prompt_engineer = tutor_prompt_engineer
enhanced_ab_tester = prompt_ab_test
prompt_optimizer = prompt_baker

# Configuration
ENABLE_ENHANCED_PROMPTS = os.getenv("ENABLE_ENHANCED_PROMPTS", "true").lower() == "true"
ENABLE_REAL_TIME_OPTIMIZATION = (
    os.getenv("ENABLE_REAL_TIME_OPTIMIZATION", "true").lower() == "true"
)
PROMPT_OPTIMIZATION_THRESHOLD = float(
    os.getenv("PROMPT_OPTIMIZATION_THRESHOLD", "0.75")
)

# Initialize W&B for enhanced tracking
wandb.init(
    project="aivillage-tutoring",
    job_type="enhanced-whatsapp-bridge",
    config={
        "response_time_target": 5.0,
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "ab_test_enabled": True,
        "enhanced_prompts_enabled": ENABLE_ENHANCED_PROMPTS,
        "real_time_optimization": ENABLE_REAL_TIME_OPTIMIZATION,
        "prompt_engineering_version": "2.0.0",
    },
)


@app.post("/whatsapp/webhook")
async def enhanced_whatsapp_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    """Enhanced WhatsApp webhook with advanced prompt engineering
    Features: W&B tracking, A/B testing, real-time optimization, multi-language support
    """
    start_time = time.time()
    session_context = {}

    try:
        # Parse Twilio webhook data
        form_data = await request.form()
        incoming_msg = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")
        message_sid = form_data.get("MessageSid", "")

        logger.info(
            f"Enhanced webhook received message from {from_number}: {incoming_msg[:50]}..."
        )

        # Validate required fields
        if not incoming_msg or not from_number:
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Generate enhanced session ID
        session_id = generate_enhanced_session_id(from_number, message_sid)
        session_context = {
            "session_id": session_id,
            "from_number": from_number,
            "message_sid": message_sid,
            "start_time": start_time,
        }

        # Detect language with enhanced accuracy
        detected_lang = await detect_language(incoming_msg)
        session_context["detected_language"] = detected_lang

        # Enhanced W&B logging with session context
        wandb.log(
            {
                "enhanced_interaction": True,
                "session_id": session_id,
                "language_detected": detected_lang,
                "message_length": len(incoming_msg),
                "enhanced_features_enabled": ENABLE_ENHANCED_PROMPTS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_number_hash": hashlib.sha256(from_number.encode()).hexdigest()[
                    :8
                ],
            }
        )

        # Route to enhanced tutor with advanced prompt engineering
        try:
            if ENABLE_ENHANCED_PROMPTS:
                response = await get_enhanced_tutor_response(
                    incoming_msg,
                    from_number,
                    session_id,
                    detected_lang,
                    session_context,
                )
            else:
                # Fallback to original implementation
                response = await get_original_tutor_response(
                    incoming_msg, from_number, session_id, detected_lang
                )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout for enhanced session {session_id}")
            response = get_enhanced_fallback_response(detected_lang)

        # Calculate response time
        response_time = time.time() - start_time

        # Enhanced metrics logging in background
        background_tasks.add_task(
            log_enhanced_response_metrics,
            response,
            response_time,
            session_id,
            detected_lang,
            session_context,
        )

        # Real-time prompt optimization (if enabled)
        if ENABLE_REAL_TIME_OPTIMIZATION:
            background_tasks.add_task(
                trigger_real_time_optimization, response, response_time, session_context
            )

        # Format and return enhanced Twilio response
        twiml_response = format_enhanced_whatsapp_response(response)

        logger.info(
            f"Enhanced response sent in {response_time:.2f}s for session {session_id}"
        )

        return PlainTextResponse(
            content=str(twiml_response), media_type="application/xml"
        )

    except Exception as e:
        logger.error(f"Error in enhanced WhatsApp webhook: {e!s}")

        # Enhanced error logging to W&B
        wandb.log(
            {
                "enhanced_error": True,
                "error_type": "webhook_processing",
                "error_message": str(e),
                "session_context": session_context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Return enhanced error response
        error_response = get_enhanced_error_response()
        return PlainTextResponse(
            content=str(error_response), media_type="application/xml"
        )


async def get_enhanced_tutor_response(
    message: str,
    from_number: str,
    session_id: str,
    detected_lang: str,
    session_context: dict[str, Any],
) -> dict[str, Any]:
    """Generate enhanced tutoring response with advanced prompt engineering"""
    # Analyze message context for better prompt selection
    message_context = await analyze_message_context(
        message, detected_lang, session_context
    )

    # Check if this is a greeting (new conversation)
    is_greeting = is_enhanced_greeting_message(message, detected_lang, message_context)

    if is_greeting:
        # Enhanced A/B test greeting responses with context awareness
        greeting_variant = enhanced_ab_tester.get_user_variant(
            from_number, "greeting_style"
        )

        # Generate enhanced greeting with personalized prompt engineering
        if greeting_variant:
            enhanced_prompt = await enhanced_prompt_engineer.generate_prompt_template(
                greeting_style=greeting_variant.configuration.get("style", "friendly"),
                hint_complexity="direct",
                example_type="real-world",
                encouragement_frequency=greeting_variant.configuration.get(
                    "enthusiasm_level", 0.3
                ),
                subject=message_context.get("subject_hint", "general"),
                context=message_context,
            )

            response = await ai_tutor.generate_greeting(
                variant=greeting_variant.variant_name,
                language=detected_lang,
                user_message=message,
                enhanced_prompt=(
                    enhanced_prompt.template_text if enhanced_prompt else None
                ),
            )
        else:
            # Fallback to original greeting
            response = await ai_tutor.generate_greeting(
                variant="friendly", language=detected_lang, user_message=message
            )

        # Log enhanced greeting interaction
        wandb.log(
            {
                "enhanced_greeting": True,
                "greeting_variant": (
                    greeting_variant.variant_id if greeting_variant else "fallback"
                ),
                "context_analysis": message_context,
                "session_id": session_id,
            }
        )

    else:
        # Enhanced tutoring response with optimized prompts

        # Get optimized prompt based on current performance data
        tutoring_variant = enhanced_ab_tester.get_user_variant(
            from_number, "tutoring_approach"
        )

        if tutoring_variant:
            # Generate context-aware prompt template
            optimized_prompt = await enhanced_prompt_engineer.generate_prompt_template(
                greeting_style="friendly",  # Keep consistent for tutoring
                hint_complexity=tutoring_variant.configuration.get(
                    "complexity", "guided"
                ),
                example_type=tutoring_variant.configuration.get(
                    "approach", "real-world"
                ),
                encouragement_frequency=0.3,
                subject=message_context.get("detected_subject", "general"),
                context={
                    **message_context,
                    "user_message": message,
                    "session_id": session_id,
                    "tutoring_style": tutoring_variant.configuration.get(
                        "approach", "guided"
                    ),
                },
            )

            # Generate response with enhanced prompt
            response = await ai_tutor.generate_response(
                user_message=message,
                prompt_template=(
                    optimized_prompt.template_text if optimized_prompt else None
                ),
                language=detected_lang,
                session_id=session_id,
                enhanced_context=message_context,
            )

            # Run A/B test interaction for continuous optimization
            await enhanced_ab_tester.run_test_interaction(
                student_msg=message,
                user_id=from_number,
                test_type="tutoring_approach",
                language=detected_lang,
                context={
                    "session_id": session_id,
                    "variant_used": tutoring_variant.variant_id,
                    **message_context,
                },
            )
        else:
            # Fallback to original prompt optimization
            optimized_prompt = await original_prompt_tuner.get_optimized_prompt(
                message_type="tutoring",
                language=detected_lang,
                context={"user_message": message, "session_id": session_id},
            )

            response = await ai_tutor.generate_response(
                user_message=message,
                prompt_template=optimized_prompt,
                language=detected_lang,
                session_id=session_id,
            )

    # Enhanced translation with context preservation
    if detected_lang != "en" and detected_lang in SUPPORTED_LANGUAGES:
        response = await auto_translate_flow(
            response, detected_lang, preserve_context=message_context
        )

    return {
        "text": response,
        "language": detected_lang,
        "session_id": session_id,
        "is_greeting": is_greeting,
        "context_analysis": message_context,
        "enhanced_features_used": True,
    }


async def analyze_message_context(
    message: str, language: str, session_context: dict[str, Any]
) -> dict[str, Any]:
    """Analyze message context for enhanced prompt selection"""
    context = {
        "message_length": len(message),
        "word_count": len(message.split()),
        "language": language,
        "session_id": session_context.get("session_id", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Detect subject area with enhanced accuracy
    subject_keywords = {
        "mathematics": [
            "math",
            "algebra",
            "calculus",
            "geometry",
            "números",
            "suma",
            "resta",
        ],
        "science": [
            "science",
            "physics",
            "chemistry",
            "biology",
            "experimento",
            "átomo",
        ],
        "programming": [
            "code",
            "python",
            "javascript",
            "programming",
            "función",
            "variable",
        ],
        "language_arts": ["grammar", "writing", "essay", "literatura", "gramática"],
        "history": [
            "history",
            "war",
            "civilization",
            "historia",
            "guerra",
            "civilización",
        ],
    }

    detected_subject = "general"
    max_matches = 0

    message_lower = message.lower()
    for subject, keywords in subject_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in message_lower)
        if matches > max_matches:
            max_matches = matches
            detected_subject = subject

    context["detected_subject"] = detected_subject
    context["subject_confidence"] = max_matches / len(
        subject_keywords.get(detected_subject, [])
    )

    # Detect urgency/difficulty level
    urgency_indicators = [
        "help",
        "confused",
        "don't understand",
        "ayuda",
        "no entiendo",
    ]
    context["urgency_level"] = (
        "high"
        if any(indicator in message_lower for indicator in urgency_indicators)
        else "normal"
    )

    # Question type analysis
    if "?" in message:
        context["question_type"] = "direct_question"
    elif any(
        word in message_lower
        for word in ["explain", "how", "why", "what", "explica", "cómo", "por qué"]
    ):
        context["question_type"] = "explanation_request"
    else:
        context["question_type"] = "statement"

    return context


def is_enhanced_greeting_message(
    message: str, language: str, context: dict[str, Any]
) -> bool:
    """Enhanced greeting detection with context awareness"""
    # Basic greeting patterns (existing logic)
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

    basic_greeting = any(pattern in message_lower for pattern in patterns)

    # Enhanced detection: consider message length and context
    if basic_greeting:
        return True

    # Short messages might be greetings
    if len(message.split()) <= 3 and context.get("question_type") != "direct_question":
        return True

    # First interaction indicators
    first_interaction_phrases = [
        "need help",
        "necesito ayuda",
        "मदद चाहिए",
        "j'ai besoin d'aide",
    ]
    return any(phrase in message_lower for phrase in first_interaction_phrases)


async def get_original_tutor_response(
    message: str, from_number: str, session_id: str, detected_lang: str
) -> dict[str, Any]:
    """Fallback to original tutor response implementation"""
    # This maintains compatibility with the original implementation
    is_greeting = is_enhanced_greeting_message(message, detected_lang, {})

    if is_greeting:
        greeting_variant = original_ab_test_manager.get_greeting_variant(from_number)
        response = await ai_tutor.generate_greeting(
            variant=greeting_variant, language=detected_lang, user_message=message
        )
    else:
        optimized_prompt = await original_prompt_tuner.get_optimized_prompt(
            message_type="tutoring",
            language=detected_lang,
            context={"user_message": message, "session_id": session_id},
        )

        response = await ai_tutor.generate_response(
            user_message=message,
            prompt_template=optimized_prompt,
            language=detected_lang,
            session_id=session_id,
        )

    if detected_lang != "en" and detected_lang in SUPPORTED_LANGUAGES:
        response = await auto_translate_flow(response, detected_lang)

    return {
        "text": response,
        "language": detected_lang,
        "session_id": session_id,
        "is_greeting": is_greeting,
        "enhanced_features_used": False,
    }


def generate_enhanced_session_id(from_number: str, message_sid: str) -> str:
    """Generate enhanced session ID with additional entropy"""
    timestamp = int(time.time() * 1000)  # Millisecond precision
    combined = f"{from_number}_{message_sid}_{timestamp}_{ENABLE_ENHANCED_PROMPTS}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def format_enhanced_whatsapp_response(response_data: dict[str, Any]) -> str:
    """Format enhanced response for Twilio WhatsApp"""
    response = MessagingResponse()
    message = response.message()

    # Add enhanced metadata as invisible markers (for debugging)
    response_text = response_data["text"]
    if response_data.get("enhanced_features_used", False):
        # Add invisible enhancement marker
        response_text += "\u200b"  # Zero-width space as marker

    message.body(response_text)
    return str(response)


def get_enhanced_fallback_response(language: str) -> dict[str, Any]:
    """Enhanced fallback response with better language support"""
    fallback_messages = {
        "en": "I'm processing your message with advanced AI. Please wait a moment... 🤔",
        "es": "Estoy procesando tu mensaje con IA avanzada. Por favor espera un momento... 🤔",
        "hi": "मैं उन्नत AI के साथ आपका संदेश प्रोसेस कर रहा हूं। कृपया एक क्षण प्रतीक्षा करें... 🤔",
        "sw": "Ninachakata ujumbe wako kwa AI ya hali ya juu. Tafadhali subiri kidogo... 🤔",
        "ar": "أقوم بمعالجة رسالتك بالذكاء الاصطناعي المتقدم. يرجى الانتظار لحظة... 🤔",
        "pt": "Estou processando sua mensagem com IA avançada. Por favor, aguarde um momento... 🤔",
        "fr": "Je traite votre message avec une IA avancée. Veuillez patienter un moment... 🤔",
    }

    return {
        "text": fallback_messages.get(language, fallback_messages["en"]),
        "language": language,
        "is_fallback": True,
        "enhanced_features_used": True,
    }


def get_enhanced_error_response() -> str:
    """Enhanced error response with better user experience"""
    response = MessagingResponse()
    message = response.message()
    message.body(
        "I'm experiencing some technical difficulties with my advanced systems. Please try again in a moment, and I'll be right back to help you learn! 🛠️✨"
    )
    return str(response)


async def log_enhanced_response_metrics(
    response: dict[str, Any],
    response_time: float,
    session_id: str,
    language: str,
    session_context: dict[str, Any],
):
    """Enhanced metrics logging with additional context"""
    # Core metrics (original)
    base_metrics = {
        "response_time": response_time,
        "response_length": len(response.get("text", "")),
        "language": language,
        "session_id": session_id,
        "is_greeting": response.get("is_greeting", False),
        "is_fallback": response.get("is_fallback", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Enhanced metrics
    enhanced_metrics = {
        "enhanced_features_used": response.get("enhanced_features_used", False),
        "context_analysis": response.get("context_analysis", {}),
        "prompt_engineering_version": "2.0.0",
        "session_context": session_context,
    }

    # Combined metrics
    all_metrics = {**base_metrics, **enhanced_metrics}

    # Performance analysis
    all_metrics["performance_target_met"] = response_time < 5.0
    all_metrics["response_quality"] = (
        "excellent"
        if response_time < 2.0
        else (
            "good"
            if response_time < 4.0
            else "acceptable" if response_time < 5.0 else "slow"
        )
    )
    all_metrics["enhancement_impact"] = (
        "positive" if enhanced_metrics["enhanced_features_used"] else "baseline"
    )

    # Log to W&B
    wandb.log(all_metrics)

    # Update original metrics tracker
    await metrics.update_metrics(base_metrics)

    logger.info(
        f"Enhanced metrics logged for session {session_id}: {response_time:.2f}s"
    )


async def trigger_real_time_optimization(
    response: dict[str, Any], response_time: float, session_context: dict[str, Any]
):
    """Trigger real-time prompt optimization based on performance"""
    try:
        # Only optimize if performance is below threshold
        if (
            response_time > PROMPT_OPTIMIZATION_THRESHOLD * 5.0
        ):  # 5.0 is the target response time
            # Identify potential optimization opportunities
            optimization_opportunities = {
                "slow_response_time": response_time > 4.0,
                "language_complexity": session_context.get("detected_language", "en")
                != "en",
                "subject_complexity": session_context.get("context_analysis", {}).get(
                    "detected_subject"
                )
                in ["mathematics", "science", "programming"],
                "enhancement_needed": True,
            }

            # Log optimization trigger
            wandb.log(
                {
                    "real_time_optimization_triggered": True,
                    "optimization_opportunities": optimization_opportunities,
                    "session_id": session_context.get("session_id", ""),
                    "response_time": response_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Trigger background optimization (could be expanded to update prompt weights)
            logger.info(
                f"Real-time optimization triggered for session {session_context.get('session_id', '')}"
            )

    except Exception as e:
        logger.error(f"Error in real-time optimization: {e}")


# Enhanced API endpoints


@app.get("/health/enhanced")
async def enhanced_health_check():
    """Enhanced health check with prompt engineering status"""
    return {
        "status": "healthy",
        "service": "whatsapp-wave-bridge-enhanced",
        "version": "2.0.0",
        "features": {
            "enhanced_prompts": ENABLE_ENHANCED_PROMPTS,
            "real_time_optimization": ENABLE_REAL_TIME_OPTIMIZATION,
            "prompt_engineering": True,
            "advanced_ab_testing": True,
        },
        "prompt_engineering": {
            "active_templates": len(enhanced_prompt_engineer.active_templates),
            "ab_tests_running": len(enhanced_ab_tester.active_tests),
            "optimization_threshold": PROMPT_OPTIMIZATION_THRESHOLD,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }


@app.get("/metrics/enhanced")
async def get_enhanced_metrics():
    """Get enhanced performance metrics including prompt engineering data"""
    # Get base metrics
    base_metrics = await metrics.get_summary()

    # Add prompt engineering metrics
    enhanced_metrics = {
        "prompt_engineering": {
            "active_templates": len(enhanced_prompt_engineer.active_templates),
            "template_performance": {
                template_id: {
                    "performance_score": template.performance_score,
                    "interaction_count": template.interaction_count,
                    "confidence_score": getattr(template, "confidence_score", 0.0),
                }
                for template_id, template in enhanced_prompt_engineer.active_templates.items()
            },
            "ab_test_results": {
                test_type: len(variants)
                for test_type, variants in enhanced_ab_tester.active_tests.items()
            },
            "total_interactions": enhanced_ab_tester.total_interactions,
            "optimization_enabled": ENABLE_REAL_TIME_OPTIMIZATION,
        }
    }

    # Combine metrics
    return {**base_metrics, **enhanced_metrics}


@app.post("/admin/optimize-prompts")
async def trigger_prompt_optimization():
    """Admin endpoint to trigger prompt optimization"""
    try:
        # Generate optimization report
        report = await prompt_optimizer.generate_baking_report()

        return {
            "optimization_triggered": True,
            "report": report,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error triggering prompt optimization: {e}")
        return {
            "optimization_triggered": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.get("/admin/ab-test-results")
async def get_ab_test_results():
    """Get detailed A/B test results"""
    try:
        results = {}

        for test_type in enhanced_ab_tester.active_tests.keys():
            test_results = await enhanced_ab_tester.analyze_test_results(test_type)
            results[test_type] = [
                {
                    "variant_id": result.variant_id,
                    "total_interactions": result.total_interactions,
                    "avg_engagement": result.avg_engagement,
                    "conversion_rate": result.conversion_rate,
                    "recommendation": result.recommendation,
                }
                for result in test_results
            ]

        return {
            "ab_test_results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting A/B test results: {e}")
        return {"error": str(e)}


# Original endpoints for backward compatibility
@app.get("/health")
async def health_check():
    """Original health check endpoint"""
    return {
        "status": "healthy",
        "service": "whatsapp-wave-bridge",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }


@app.get("/metrics")
async def get_metrics():
    """Original metrics endpoint"""
    return await metrics.get_summary()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
