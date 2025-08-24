"""Multi-language Support for WhatsApp Wave Bridge
Optimized for edge translation with fallback options.
"""

from datetime import datetime
import logging

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None  # type: ignore[assignment]
from googletrans import Translator
import langdetect
import openai
import wandb

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "priority": 1},
    "es": {"name": "Spanish", "native": "Español", "priority": 2},
    "hi": {"name": "Hindi", "native": "हिन्दी", "priority": 3},
    "sw": {"name": "Swahili", "native": "Kiswahili", "priority": 4},
    "ar": {"name": "Arabic", "native": "العربية", "priority": 5},
    "pt": {"name": "Portuguese", "native": "Português", "priority": 6},
    "fr": {"name": "French", "native": "Français", "priority": 7},
    "de": {"name": "German", "native": "Deutsch", "priority": 8},
    "it": {"name": "Italian", "native": "Italiano", "priority": 9},
    "zh": {"name": "Chinese", "native": "中文", "priority": 10},
}

# Edge model availability (fastest response)
EDGE_SUPPORTED = {"en", "es", "fr", "de", "it", "pt"}

# High-quality model preferences
TRANSLATION_MODELS = {
    "edge": "google-translate-edge",
    "cloud": "google-translate-cloud",
    "anthropic": "claude-3-haiku",
    "openai": "gpt-3.5-turbo",
}


class LanguageDetector:
    """Fast language detection with caching."""

    def __init__(self) -> None:
        self.cache = {}
        self.confidence_threshold = 0.8

    async def detect_language(self, text: str) -> str:
        """Detect language with confidence scoring."""
        # Check cache first
        text_hash = hash(text.lower().strip())
        if text_hash in self.cache:
            return self.cache[text_hash]

        try:
            # Use langdetect for fast detection
            detected = langdetect.detect(text)
            confidence = langdetect.detect_langs(text)[0].prob

            # Fallback to English if confidence is low
            if confidence < self.confidence_threshold:
                detected = "en"

            # Validate against supported languages
            if detected not in SUPPORTED_LANGUAGES:
                detected = self.find_closest_supported(detected)

            # Cache result
            self.cache[text_hash] = detected

            # Log detection to W&B
            wandb.log(
                {
                    "language_detection": {
                        "detected": detected,
                        "confidence": confidence,
                        "text_length": len(text),
                    }
                }
            )

            return detected

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English

    def find_closest_supported(self, lang_code: str) -> str:
        """Map unsupported languages to closest supported ones."""
        language_families = {
            "ro": "fr",  # Romanian -> French
            "ca": "es",  # Catalan -> Spanish
            "eu": "es",  # Basque -> Spanish
            "gl": "pt",  # Galician -> Portuguese
            "nl": "de",  # Dutch -> German
            "da": "de",  # Danish -> German
            "sv": "de",  # Swedish -> German
            "no": "de",  # Norwegian -> German
            "pl": "de",  # Polish -> German
            "cs": "de",  # Czech -> German
            "sk": "de",  # Slovak -> German
            "hu": "de",  # Hungarian -> German
            "fi": "de",  # Finnish -> German
            "ja": "zh",  # Japanese -> Chinese
            "ko": "zh",  # Korean -> Chinese
            "th": "zh",  # Thai -> Chinese
            "vi": "zh",  # Vietnamese -> Chinese
            "ru": "en",  # Russian -> English
            "uk": "en",  # Ukrainian -> English
            "bg": "en",  # Bulgarian -> English
            "hr": "en",  # Croatian -> English
            "sr": "en",  # Serbian -> English
            "sl": "en",  # Slovenian -> English
            "tr": "en",  # Turkish -> English
            "he": "ar",  # Hebrew -> Arabic
            "fa": "ar",  # Persian -> Arabic
            "ur": "hi",  # Urdu -> Hindi
            "bn": "hi",  # Bengali -> Hindi
            "ta": "hi",  # Tamil -> Hindi
            "te": "hi",  # Telugu -> Hindi
            "ml": "hi",  # Malayalam -> Hindi
            "kn": "hi",  # Kannada -> Hindi
            "gu": "hi",  # Gujarati -> Hindi
            "pa": "hi",  # Punjabi -> Hindi
            "mr": "hi",  # Marathi -> Hindi
        }

        return language_families.get(lang_code, "en")


# Global language detector instance
language_detector = LanguageDetector()


async def detect_language(text: str) -> str:
    """Detect language of input text."""
    return await language_detector.detect_language(text)


def edge_model_available(language: str) -> bool:
    """Check if edge translation model is available for language."""
    return language in EDGE_SUPPORTED


class TranslationEngine:
    """Multi-tier translation engine with fallback cascade."""

    def __init__(self) -> None:
        self.google_translator = Translator()
        self.anthropic_client = anthropic.Anthropic() if hasattr(anthropic, "Anthropic") else None
        self.openai_client = openai.OpenAI() if hasattr(openai, "OpenAI") else None

    async def edge_translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Fast edge translation using Google Translate."""
        try:
            # Use Google Translate for edge cases
            result = self.google_translator.translate(text, src=source_lang, dest=target_lang)

            wandb.log(
                {
                    "translation_method": "edge",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "success": True,
                }
            )

            return result.text

        except Exception as e:
            logger.warning(f"Edge translation failed: {e}")
            raise

    async def cloud_translate(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """High-quality cloud translation using AI models."""
        # Try Anthropic first for quality
        if self.anthropic_client:
            try:
                prompt = f"""Translate the following text from {SUPPORTED_LANGUAGES.get(source_lang, {}).get("name", source_lang)} to {SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", target_lang)}.

Maintain the original tone, context, and cultural nuances. For educational content, preserve technical accuracy.

Text to translate: {text}

Translation:"""

                response = await self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )

                translation = response.content[0].text.strip()

                wandb.log(
                    {
                        "translation_method": "anthropic",
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "success": True,
                    }
                )

                return translation

            except Exception as e:
                logger.warning(f"Anthropic translation failed: {e}")

        # Fallback to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a professional translator. Translate text from {source_lang} to {target_lang} while preserving meaning, tone, and cultural context.",
                        },
                        {"role": "user", "content": text},
                    ],
                    temperature=0.1,
                    max_tokens=800,
                )

                translation = response.choices[0].message.content.strip()

                wandb.log(
                    {
                        "translation_method": "openai",
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "success": True,
                    }
                )

                return translation

            except Exception as e:
                logger.warning(f"OpenAI translation failed: {e}")

        # Final fallback to Google Translate
        return await self.edge_translate(text, target_lang, source_lang)


# Global translation engine
translation_engine = TranslationEngine()


async def auto_translate_flow(message: str, target_lang: str, source_lang: str = "en") -> str:
    """Translate with fallback options prioritizing speed and quality."""
    # Skip translation if same language
    if source_lang == target_lang:
        return message

    start_time = datetime.now()

    try:
        # Try edge model first (fastest)
        if edge_model_available(target_lang):
            try:
                result = await translation_engine.edge_translate(message, target_lang, source_lang)
                translation_time = (datetime.now() - start_time).total_seconds()

                wandb.log(
                    {
                        "translation_time": translation_time,
                        "translation_tier": "edge",
                        "success": True,
                    }
                )

                return result

            except Exception as e:
                logger.info(f"Edge translation failed, trying cloud: {e}")

        # Fallback to cloud translation (higher quality)
        result = await translation_engine.cloud_translate(message, target_lang, source_lang)
        translation_time = (datetime.now() - start_time).total_seconds()

        wandb.log(
            {
                "translation_time": translation_time,
                "translation_tier": "cloud",
                "success": True,
            }
        )

        return result

    except Exception as e:
        logger.exception(f"All translation methods failed: {e}")

        wandb.log(
            {
                "translation_time": (datetime.now() - start_time).total_seconds(),
                "translation_tier": "failed",
                "success": False,
                "error": str(e),
            }
        )

        # Return original message with language note
        lang_name = SUPPORTED_LANGUAGES.get(target_lang, {}).get("native", target_lang)
        return f"[Translation to {lang_name} unavailable] {message}"


def get_language_greeting(language: str) -> str:
    """Get culturally appropriate greeting for language."""
    greetings = {
        "en": "Hello! I'm your AI tutor. How can I help you learn today?",
        "es": "¡Hola! Soy tu tutor de IA. ¿Cómo puedo ayudarte a aprender hoy?",
        "hi": "नमस्ते! मैं आपका AI शिक्षक हूं। आज मैं आपकी सीखने में कैसे मदद कर सकता हूं?",
        "sw": "Hujambo! Mimi ni mwalimu wako wa AI. Ninawezaje kukusaidia kujifunza leo?",
        "ar": "مرحباً! أنا مدرسك الذكي. كيف يمكنني مساعدتك في التعلم اليوم؟",
        "pt": "Olá! Eu sou seu tutor de IA. Como posso ajudá-lo a aprender hoje?",
        "fr": "Bonjour! Je suis votre tuteur IA. Comment puis-je vous aider à apprendre aujourd'hui?",
        "de": "Hallo! Ich bin Ihr KI-Tutor. Wie kann ich Ihnen heute beim Lernen helfen?",
        "it": "Ciao! Sono il tuo tutor AI. Come posso aiutarti a imparare oggi?",
        "zh": "你好！我是你的AI导师。今天我如何帮助你学习？",
    }

    return greetings.get(language, greetings["en"])


def get_supported_languages_list(user_language: str = "en") -> str:
    """Get formatted list of supported languages."""
    if user_language == "en":
        return "I support: " + ", ".join(
            [f"{info['native']} ({info['name']})" for info in SUPPORTED_LANGUAGES.values()]
        )
    # Return in user's language
    intro_text = {
        "es": "Apoyo: ",
        "hi": "मैं समर्थन करता हूं: ",
        "sw": "Ninasaidia: ",
        "ar": "أدعم: ",
        "pt": "Eu apoio: ",
        "fr": "Je supporte: ",
        "de": "Ich unterstütze: ",
        "it": "Supporto: ",
        "zh": "我支持: ",
    }

    intro = intro_text.get(user_language, "I support: ")
    return intro + ", ".join([info["native"] for info in SUPPORTED_LANGUAGES.values()])


async def validate_translation_quality(original: str, translated: str, target_lang: str) -> float:
    """Validate translation quality using back-translation
    Returns confidence score 0-1.
    """
    try:
        # Back-translate to English
        back_translation = await auto_translate_flow(translated, "en", target_lang)

        # Simple similarity check (can be enhanced with semantic similarity)
        original_words = set(original.lower().split())
        back_words = set(back_translation.lower().split())

        if len(original_words) == 0:
            return 0.0

        intersection = len(original_words.intersection(back_words))
        union = len(original_words.union(back_words))

        similarity = intersection / union if union > 0 else 0.0

        wandb.log(
            {
                "translation_validation": {
                    "similarity_score": similarity,
                    "target_language": target_lang,
                    "original_length": len(original),
                    "translated_length": len(translated),
                }
            }
        )

        return similarity

    except Exception as e:
        logger.warning(f"Translation validation failed: {e}")
        return 0.5  # Neutral confidence
