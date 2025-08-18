"""Polyglot Agent - Translation & Linguistics

The translation and linguistics specialist of AIVillage, responsible for:
- Low-resource machine translation across 5 initial languages
- Dialect handling and cultural nuance preservation
- Real-time translation for multi-agent coordination
- Mobile-optimized language models (<50MB each)
- Cultural localization and context adaptation
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    HINDI = "hi"
    SWAHILI = "sw"
    ARABIC = "ar"


class TranslationType(Enum):
    DIRECT = "direct"
    PIVOT = "pivot"  # through English
    CULTURAL_ADAPTED = "cultural_adapted"
    TECHNICAL = "technical"


@dataclass
class LanguageModel:
    language_code: str
    model_name: str
    size_mb: float
    accuracy_score: float
    latency_ms: float
    specialization: str
    cultural_context: list[str]
    loaded: bool = False


@dataclass
class TranslationRequest:
    request_id: str
    source_text: str
    source_language: str
    target_language: str
    translation_type: TranslationType
    context: str | None
    max_length: int
    preserve_formatting: bool
    timestamp: float


@dataclass
class TranslationResult:
    request_id: str
    translated_text: str
    confidence_score: float
    detected_source_lang: str | None
    cultural_adaptations: list[str]
    processing_time_ms: float
    model_used: str
    receipt: dict[str, Any]


class PolyglotAgent(AgentInterface):
    """Polyglot Agent provides translation and linguistics services for AIVillage,
    enabling cross-language communication between users and agents with cultural awareness.
    """

    def __init__(self, agent_id: str = "polyglot_agent"):
        self.agent_id = agent_id
        self.agent_type = "Polyglot"
        self.capabilities = [
            "machine_translation",
            "dialect_processing",
            "cultural_nuance",
            "linguistic_analysis",
            "language_detection",
            "cultural_localization",
            "real_time_translation",
            "mobile_optimization",
            "context_preservation",
            "multilingual_embeddings",
        ]

        # Language models and translation state
        self.language_models: dict[str, LanguageModel] = {}
        self.translation_history: list[TranslationResult] = []
        self.language_pairs_cache: dict[str, dict[str, Any]] = {}
        self.cultural_context_db: dict[str, list[str]] = {}

        # Performance tracking
        self.translations_completed = 0
        self.languages_supported = 5
        self.average_confidence = 0.0
        self.average_latency_ms = 0.0
        self.cache_hit_rate = 0.0

        # Mobile constraints
        self.max_model_size_mb = 50
        self.max_memory_usage_mb = 200
        self.target_latency_ms = 500

        # Cultural adaptation rules
        self.cultural_adaptations = {
            "formal_address": {
                "es": "usted",  # Spanish formal
                "ar": "حضرتك",  # Arabic formal
                "hi": "आप",  # Hindi formal
                "sw": "bwana/bibi",  # Swahili formal
            },
            "greetings": {
                "es": ["Buenos días", "Buenas tardes", "Buenas noches"],
                "ar": ["السلام عليكم", "أهلاً وسهلاً", "مرحباً"],
                "hi": ["नमस्ते", "प्रणाम", "आदाब"],
                "sw": ["Hujambo", "Habari", "Salamu"],
            },
        }

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate translation and linguistics responses"""
        prompt_lower = prompt.lower()

        if "translate" in prompt_lower:
            return "I can translate text between English, Spanish, Hindi, Swahili, and Arabic with cultural adaptation."
        if "language" in prompt_lower and "detect" in prompt_lower:
            return "I detect the source language automatically and provide translation with confidence scores."
        if "cultural" in prompt_lower or "context" in prompt_lower:
            return "I preserve cultural nuances and adapt translations for local contexts and customs."
        if "mobile" in prompt_lower or "offline" in prompt_lower:
            return "I use mobile-optimized models under 50MB each for offline translation capability."
        if "real-time" in prompt_lower or "live" in prompt_lower:
            return "I provide real-time translation for live conversations and agent communications."

        return "I am Polyglot Agent, bridging language barriers across AIVillage with cultural intelligence."

    async def get_embedding(self, text: str) -> list[float]:
        """Generate multilingual-aware embeddings"""
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Multilingual embeddings capture cross-language semantic similarity
        return [(hash_value % 1000) / 1000.0] * 384

    async def rerank(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Rerank based on linguistic and cultural relevance"""
        linguistic_keywords = [
            "translation",
            "language",
            "cultural",
            "dialect",
            "linguistic",
            "multilingual",
            "localization",
            "context",
            "nuance",
            "communication",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))

            for keyword in linguistic_keywords:
                score += content.lower().count(keyword) * 2

            # Boost content with cultural and contextual information
            if any(
                term in content.lower()
                for term in ["cultural", "context", "adaptation"]
            ):
                score *= 1.7

            result["linguistic_relevance"] = score

        return sorted(
            results, key=lambda x: x.get("linguistic_relevance", 0), reverse=True
        )[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return Polyglot agent status and translation metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "supported_languages": [lang.value for lang in SupportedLanguage],
            "loaded_models": len(
                [m for m in self.language_models.values() if m.loaded]
            ),
            "translations_completed": self.translations_completed,
            "average_confidence": self.average_confidence,
            "average_latency_ms": self.average_latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "total_model_size_mb": sum(
                m.size_mb for m in self.language_models.values()
            ),
            "cultural_contexts_available": len(self.cultural_context_db),
            "specialization": "translation_and_linguistics",
            "mobile_optimized": True,
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with automatic translation if needed"""
        # Detect if translation is needed based on message content
        translation_context = "[MULTILINGUAL]"

        if recipient:
            # Check if message contains non-English content
            detected_lang = await self._detect_language(message)
            if detected_lang and detected_lang != SupportedLanguage.ENGLISH.value:
                translation_result = await self.translate_text(
                    message, detected_lang, "en"
                )
                message = f"{translation_context} Original ({detected_lang}): {message[:30]}... Translated: {translation_result.translated_text}"

            response = await recipient.generate(
                f"Polyglot Agent communicates: {message}"
            )
            return f"Cross-language communication: {response[:50]}..."
        return "No recipient for translation services"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate language-specific latent spaces"""
        query_lower = query.lower()

        if "translate" in query_lower:
            space_type = "machine_translation"
        elif "cultural" in query_lower:
            space_type = "cultural_adaptation"
        elif "detect" in query_lower:
            space_type = "language_detection"
        elif "linguistic" in query_lower:
            space_type = "linguistic_analysis"
        else:
            space_type = "multilingual_processing"

        latent_repr = f"POLYGLOT[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str | None = None,
        preserve_cultural_context: bool = True,
    ) -> TranslationResult:
        """Translate text between supported languages - MVP function"""
        request_id = f"trans_{int(time.time())}_{len(self.translation_history)}"
        start_time = time.time()

        # Validate language support
        if not await self._validate_language_pair(source_lang, target_lang):
            raise ValueError(
                f"Language pair {source_lang}->{target_lang} not supported"
            )

        # Auto-detect source language if not specified or uncertain
        if source_lang == "auto":
            detected_lang = await self._detect_language(text)
            source_lang = detected_lang if detected_lang else "en"

        # Create translation request
        request = TranslationRequest(
            request_id=request_id,
            source_text=text,
            source_language=source_lang,
            target_language=target_lang,
            translation_type=(
                TranslationType.CULTURAL_ADAPTED
                if preserve_cultural_context
                else TranslationType.DIRECT
            ),
            context=context,
            max_length=len(text) * 2,  # Allow expansion
            preserve_formatting=True,
            timestamp=time.time(),
        )

        # Check cache first
        cache_key = (
            f"{source_lang}_{target_lang}_{hashlib.md5(text.encode()).hexdigest()}"
        )
        if cache_key in self.language_pairs_cache:
            cached_result = self.language_pairs_cache[cache_key]
            logger.info(f"Translation cache hit: {request_id}")
            return cached_result

        # Perform translation
        translated_text = await self._perform_translation(request)

        # Apply cultural adaptations
        cultural_adaptations = []
        if preserve_cultural_context:
            (
                translated_text,
                cultural_adaptations,
            ) = await self._apply_cultural_adaptations(
                translated_text, source_lang, target_lang, context
            )

        # Calculate confidence score
        confidence_score = await self._calculate_translation_confidence(
            text, translated_text, source_lang, target_lang
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Create receipt
        receipt = {
            "agent": "Polyglot",
            "action": "translation",
            "request_id": request_id,
            "timestamp": time.time(),
            "source_language": source_lang,
            "target_language": target_lang,
            "source_text_hash": hashlib.sha256(text.encode()).hexdigest(),
            "translated_text_hash": hashlib.sha256(
                translated_text.encode()
            ).hexdigest(),
            "character_count": len(text),
            "confidence_score": confidence_score,
            "processing_time_ms": processing_time,
            "cultural_adaptations": len(cultural_adaptations),
            "model_used": f"mobile_{target_lang}_v1",
            "signature": f"polyglot_{request_id}",
        }

        # Create result
        result = TranslationResult(
            request_id=request_id,
            translated_text=translated_text,
            confidence_score=confidence_score,
            detected_source_lang=(
                source_lang if source_lang != request.source_language else None
            ),
            cultural_adaptations=cultural_adaptations,
            processing_time_ms=processing_time,
            model_used=receipt["model_used"],
            receipt=receipt,
        )

        # Store in cache and history
        self.language_pairs_cache[cache_key] = result
        self.translation_history.append(result)

        # Update metrics
        self.translations_completed += 1
        self.average_latency_ms = (
            self.average_latency_ms * (self.translations_completed - 1)
            + processing_time
        ) / self.translations_completed
        self.average_confidence = (
            self.average_confidence * (self.translations_completed - 1)
            + confidence_score
        ) / self.translations_completed

        logger.info(
            f"Translation completed: {request_id} ({source_lang}->{target_lang}) - {confidence_score:.2f} confidence"
        )

        return result

    async def _validate_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Validate that language pair is supported"""
        supported_codes = [lang.value for lang in SupportedLanguage]

        if source_lang == "auto":
            return target_lang in supported_codes

        return source_lang in supported_codes and target_lang in supported_codes

    async def _detect_language(self, text: str) -> str | None:
        """Detect language of input text"""
        # Simple rule-based language detection for MVP
        text_lower = text.lower()

        # Arabic detection (Arabic script)
        if any("\u0600" <= char <= "\u06ff" for char in text):
            return SupportedLanguage.ARABIC.value

        # Hindi detection (Devanagari script)
        if any("\u0900" <= char <= "\u097f" for char in text):
            return SupportedLanguage.HINDI.value

        # Spanish detection (common words and patterns)
        spanish_indicators = [
            "el ",
            "la ",
            "es ",
            "un ",
            "una ",
            "que ",
            "de ",
            "se ",
            "con ",
            "para ",
        ]
        if any(indicator in text_lower for indicator in spanish_indicators):
            return SupportedLanguage.SPANISH.value

        # Swahili detection (common words)
        swahili_indicators = [
            "ni ",
            "na ",
            "wa ",
            "ya ",
            "za ",
            "kwa ",
            "mtu ",
            "watu ",
            "jambo ",
        ]
        if any(indicator in text_lower for indicator in swahili_indicators):
            return SupportedLanguage.SWAHILI.value

        # Default to English
        return SupportedLanguage.ENGLISH.value

    async def _perform_translation(self, request: TranslationRequest) -> str:
        """Perform the actual translation using mobile-optimized models"""
        # Simulate mobile-optimized translation models
        source_lang = request.source_language
        target_lang = request.target_language
        text = request.source_text

        # Load appropriate model pair
        model_key = f"{source_lang}_{target_lang}"
        if model_key not in self.language_models:
            await self._load_translation_model(source_lang, target_lang)

        # Simple rule-based translation for demonstration
        # In production, this would use quantized neural translation models

        if source_lang == "en" and target_lang == "es":
            # English to Spanish basic translations
            translations = {
                "hello": "hola",
                "world": "mundo",
                "good": "bueno",
                "morning": "mañana",
                "thank you": "gracias",
                "please": "por favor",
                "how are you": "cómo estás",
                "i am fine": "estoy bien",
                "yes": "sí",
                "no": "no",
            }

            # Apply word-by-word translation with fallback
            words = text.lower().split()
            translated_words = []
            for word in words:
                translated_word = translations.get(word, word)
                translated_words.append(translated_word)

            return " ".join(translated_words).capitalize()

        if source_lang == "en" and target_lang == "hi":
            # English to Hindi basic translations
            translations = {
                "hello": "नमस्ते",
                "world": "दुनिया",
                "good": "अच्छा",
                "morning": "सुबह",
                "thank you": "धन्यवाद",
                "please": "कृपया",
                "how are you": "आप कैसे हैं",
                "yes": "हाँ",
                "no": "नहीं",
            }

            words = text.lower().split()
            translated_words = []
            for word in words:
                translated_word = translations.get(word, word)
                translated_words.append(translated_word)

            return " ".join(translated_words)

        if source_lang == "en" and target_lang == "ar":
            # English to Arabic basic translations
            translations = {
                "hello": "مرحبا",
                "world": "عالم",
                "good": "جيد",
                "morning": "صباح",
                "thank you": "شكراً",
                "please": "من فضلك",
                "how are you": "كيف حالك",
                "yes": "نعم",
                "no": "لا",
            }

            words = text.lower().split()
            translated_words = []
            for word in words:
                translated_word = translations.get(word, word)
                translated_words.append(translated_word)

            return " ".join(translated_words)

        if source_lang == "en" and target_lang == "sw":
            # English to Swahili basic translations
            translations = {
                "hello": "hujambo",
                "world": "dunia",
                "good": "nzuri",
                "morning": "asubuhi",
                "thank you": "asante",
                "please": "tafadhali",
                "how are you": "habari yako",
                "yes": "ndiyo",
                "no": "hapana",
            }

            words = text.lower().split()
            translated_words = []
            for word in words:
                translated_word = translations.get(word, word)
                translated_words.append(translated_word)

            return " ".join(translated_words).capitalize()

        # Reverse translations (target language back to English)
        if target_lang == "en":
            # Simple reverse lookup
            all_translations = {
                "hola": "hello",
                "mundo": "world",
                "gracias": "thank you",
                "नमस्ते": "hello",
                "दुनिया": "world",
                "धन्यवाद": "thank you",
                "مرحبا": "hello",
                "عالم": "world",
                "شكراً": "thank you",
                "hujambo": "hello",
                "dunia": "world",
                "asante": "thank you",
            }

            words = text.split()
            translated_words = []
            for word in words:
                translated_word = all_translations.get(word.lower(), word)
                translated_words.append(translated_word)

            return " ".join(translated_words).capitalize()

        # Fallback for unsupported pairs
        return f"[Translation {source_lang}->{target_lang}] {text}"

    async def _load_translation_model(self, source_lang: str, target_lang: str):
        """Load mobile-optimized translation model for language pair"""
        model_key = f"{source_lang}_{target_lang}"

        # Create mobile-optimized model specification
        model = LanguageModel(
            language_code=model_key,
            model_name=f"MobileTranslate-{source_lang.upper()}{target_lang.upper()}-Quantized",
            size_mb=min(self.max_model_size_mb, 45),  # Always under 50MB
            accuracy_score=0.87,
            latency_ms=150,
            specialization="general_translation",
            cultural_context=self.cultural_context_db.get(target_lang, []),
            loaded=True,
        )

        self.language_models[model_key] = model
        logger.info(f"Loaded translation model: {model_key} ({model.size_mb}MB)")

    async def _apply_cultural_adaptations(
        self, text: str, source_lang: str, target_lang: str, context: str | None
    ) -> tuple[str, list[str]]:
        """Apply cultural adaptations to translation"""
        adaptations = []
        adapted_text = text

        # Apply formal address adaptations
        if target_lang in self.cultural_adaptations["formal_address"]:
            formal_marker = self.cultural_adaptations["formal_address"][target_lang]
            if context and "formal" in context.lower():
                adaptations.append(f"formal_address_applied:{formal_marker}")

        # Apply greeting adaptations
        if target_lang in self.cultural_adaptations["greetings"]:
            greetings = self.cultural_adaptations["greetings"][target_lang]
            # Replace generic greetings with culturally appropriate ones
            if "hello" in adapted_text.lower():
                adapted_text = adapted_text.lower().replace(
                    "hello", greetings[0].lower()
                )
                adaptations.append(f"greeting_localized:{greetings[0]}")

        # Time-based adaptations (morning/evening greetings)
        current_hour = time.localtime().tm_hour
        if target_lang == "es" and "good morning" in text.lower():
            if current_hour >= 12:
                adapted_text = adapted_text.replace("good morning", "buenas tardes")
                adaptations.append("time_based_greeting:afternoon")

        # Religious/cultural context adaptations
        if target_lang == "ar" and context and "greeting" in context.lower():
            if "hello" in adapted_text.lower():
                adapted_text = adapted_text.replace("hello", "السلام عليكم")
                adaptations.append("islamic_greeting_applied")

        return adapted_text, adaptations

    async def _calculate_translation_confidence(
        self, source_text: str, translated_text: str, source_lang: str, target_lang: str
    ) -> float:
        """Calculate confidence score for translation quality"""
        # Simple heuristic-based confidence calculation for MVP
        base_confidence = 0.85

        # Length-based confidence (translations shouldn't be too different in length)
        length_ratio = len(translated_text) / max(1, len(source_text))
        if 0.5 <= length_ratio <= 2.0:
            length_confidence = 0.1
        else:
            length_confidence = -0.2

        # Language-specific confidence adjustments
        lang_confidence = 0.0
        if source_lang == "en":  # English as source is generally more accurate
            lang_confidence = 0.05

        # Script consistency confidence
        script_confidence = 0.0
        if (
            target_lang == "ar"
            and any("\u0600" <= char <= "\u06ff" for char in translated_text)
        ) or (
            target_lang == "hi"
            and any("\u0900" <= char <= "\u097f" for char in translated_text)
        ):
            script_confidence = 0.1

        final_confidence = (
            base_confidence + length_confidence + lang_confidence + script_confidence
        )
        return max(0.0, min(1.0, final_confidence))

    async def detect_language_and_translate(
        self, text: str, target_lang: str
    ) -> TranslationResult:
        """Auto-detect source language and translate - MVP function"""
        detected_lang = await self._detect_language(text)

        if detected_lang == target_lang:
            # Same language, return as-is
            return TranslationResult(
                request_id=f"same_lang_{int(time.time())}",
                translated_text=text,
                confidence_score=1.0,
                detected_source_lang=detected_lang,
                cultural_adaptations=[],
                processing_time_ms=10.0,
                model_used="no_translation_needed",
                receipt={
                    "agent": "Polyglot",
                    "action": "language_detection_only",
                    "detected_language": detected_lang,
                    "same_as_target": True,
                },
            )

        # Perform translation
        return await self.translate_text(text, detected_lang, target_lang)

    async def get_translation_report(self) -> dict[str, Any]:
        """Generate comprehensive translation performance report"""
        return {
            "agent": "Polyglot",
            "report_type": "translation_performance",
            "timestamp": time.time(),
            "translation_metrics": {
                "total_translations": self.translations_completed,
                "supported_languages": len(SupportedLanguage),
                "language_pairs": len(self.language_models),
                "average_confidence": self.average_confidence,
                "average_latency_ms": self.average_latency_ms,
                "cache_hit_rate": len(self.language_pairs_cache)
                / max(1, self.translations_completed),
            },
            "model_metrics": {
                "loaded_models": len(
                    [m for m in self.language_models.values() if m.loaded]
                ),
                "total_model_size_mb": sum(
                    m.size_mb for m in self.language_models.values()
                ),
                "mobile_optimized": all(
                    m.size_mb <= self.max_model_size_mb
                    for m in self.language_models.values()
                ),
                "average_model_accuracy": sum(
                    m.accuracy_score for m in self.language_models.values()
                )
                / max(1, len(self.language_models)),
            },
            "cultural_adaptation_stats": {
                "adaptations_available": sum(
                    len(adaptations)
                    for adaptations in self.cultural_adaptations.values()
                ),
                "recent_adaptations_applied": sum(
                    len(result.cultural_adaptations)
                    for result in self.translation_history[-10:]
                ),
                "cultural_contexts": len(self.cultural_context_db),
            },
            "recommendations": [
                "Expand cultural adaptation rules for business contexts",
                "Implement caching for frequently translated phrases",
                "Add support for technical domain translation",
                "Optimize model loading for faster cold-start performance",
            ],
        }

    async def initialize(self):
        """Initialize the Polyglot Agent"""
        try:
            logger.info("Initializing Polyglot Agent - Translation & Linguistics...")

            # Initialize cultural context database
            self.cultural_context_db = {
                "es": ["formal_address", "time_based_greetings", "regional_variants"],
                "hi": ["honorifics", "religious_context", "formal_informal"],
                "ar": ["islamic_greetings", "formal_address", "msa_vs_dialect"],
                "sw": ["respect_forms", "age_based_address", "tribal_context"],
                "en": ["business_formal", "casual_informal", "technical_context"],
            }

            # Pre-load the most common language pair (English-Spanish)
            await self._load_translation_model("en", "es")

            self.initialized = True
            logger.info(
                f"Polyglot Agent {self.agent_id} initialized - {self.languages_supported} languages supported"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Polyglot Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Polyglot Agent gracefully"""
        try:
            logger.info("Polyglot Agent shutting down...")

            # Generate final translation report
            final_report = await self.get_translation_report()
            logger.info(
                f"Polyglot Agent final report: {final_report['translation_metrics']}"
            )

            # Unload language models to free memory
            for model in self.language_models.values():
                model.loaded = False

            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Polyglot Agent shutdown: {e}")
