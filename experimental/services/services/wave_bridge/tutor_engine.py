"""
AI Tutor Engine for WhatsApp Wave Bridge
Optimized for educational interactions with response time <5s
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import wandb
import json
import hashlib

# AI/ML imports
import anthropic
import openai
from transformers import pipeline

from .language_support import get_language_greeting, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

class AITutor:
    """
    Main AI Tutor engine with multi-model support and caching
    """

    def __init__(self):
        # Initialize AI clients
        self.anthropic_client = anthropic.Anthropic() if hasattr(anthropic, 'Anthropic') else None
        self.openai_client = openai.OpenAI() if hasattr(openai, 'OpenAI') else None

        # Agent Forge model integration
        self.agent_forge_model = None
        self.agent_forge_tokenizer = None
        self.agent_forge_available = False

        # Response caching for common questions
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour cache

        # Performance tracking
        self.response_times = []
        self.model_performance = {
            'agent_forge': {'avg_time': 0, 'success_rate': 0, 'calls': 0},
            'anthropic': {'avg_time': 0, 'success_rate': 0, 'calls': 0},
            'openai': {'avg_time': 0, 'success_rate': 0, 'calls': 0},
            'fallback': {'avg_time': 0, 'success_rate': 0, 'calls': 0}
        }

        # Initialize Agent Forge model
        self.initialize_agent_forge_model()

        # Initialize subject matter expertise
        self.subject_experts = self.initialize_subject_experts()

    def initialize_agent_forge_model(self):
        """Initialize Agent Forge model for tutoring."""
        try:
            # Import Agent Forge components
            import sys
            from pathlib import Path

            # Add Agent Forge path
            agent_forge_path = Path(__file__).parent.parent.parent / "agent_forge"
            sys.path.append(str(agent_forge_path))

            from agent_forge.rag_integration import AgentForgeRAGSelector
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Auto-select best model
            selector = AgentForgeRAGSelector("./benchmark_results")
            import asyncio
            selection = asyncio.run(selector.select_best_model())

            model_path = selection['model_path']

            logger.info(f"Loading Agent Forge model: {selection['selected_phase']}")

            # Load model
            self.agent_forge_tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.agent_forge_tokenizer.pad_token is None:
                self.agent_forge_tokenizer.pad_token = self.agent_forge_tokenizer.eos_token

            self.agent_forge_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

            if not torch.cuda.is_available():
                self.agent_forge_model = self.agent_forge_model.to("cpu")

            self.agent_forge_model.eval()
            self.agent_forge_available = True

            logger.info(f"✅ Agent Forge model loaded successfully: {selection['selected_phase']}")

        except Exception as e:
            logger.warning(f"Failed to load Agent Forge model: {e}")
            self.agent_forge_available = False

    def initialize_subject_experts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize subject-specific tutoring approaches"""

        experts = {
            'mathematics': {
                'keywords': ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'equation', 'solve', 'calculate'],
                'approach': 'step_by_step',
                'prompt_suffix': 'Show your work step by step and explain each step clearly.',
                'examples_needed': True
            },
            'science': {
                'keywords': ['physics', 'chemistry', 'biology', 'experiment', 'molecule', 'force', 'energy', 'cell'],
                'approach': 'conceptual',
                'prompt_suffix': 'Explain the underlying concepts and real-world applications.',
                'examples_needed': True
            },
            'programming': {
                'keywords': ['code', 'programming', 'python', 'javascript', 'function', 'variable', 'loop', 'debug'],
                'approach': 'practical',
                'prompt_suffix': 'Provide code examples and explain the logic behind them.',
                'examples_needed': True
            },
            'language': {
                'keywords': ['grammar', 'vocabulary', 'writing', 'essay', 'sentence', 'paragraph', 'literature'],
                'approach': 'interactive',
                'prompt_suffix': 'Use examples and ask follow-up questions to reinforce learning.',
                'examples_needed': False
            },
            'history': {
                'keywords': ['history', 'historical', 'war', 'ancient', 'civilization', 'empire', 'revolution'],
                'approach': 'narrative',
                'prompt_suffix': 'Tell the story in an engaging way and connect to broader themes.',
                'examples_needed': False
            },
            'general': {
                'keywords': [],
                'approach': 'adaptive',
                'prompt_suffix': 'Adapt your teaching style to the student\'s needs.',
                'examples_needed': False
            }
        }

        # Log expert configuration
        wandb.config.update({"subject_experts": experts})

        return experts

    async def generate_greeting(
        self,
        variant: str,
        language: str,
        user_message: str
    ) -> str:
        """
        Generate personalized greeting based on A/B test variant
        """

        start_time = time.time()

        try:
            # Get base greeting for language
            base_greeting = get_language_greeting(language)

            # Customize based on variant
            if variant == 'enthusiastic':
                greeting = self.make_enthusiastic(base_greeting, language)
            elif variant == 'professional':
                greeting = self.make_professional(base_greeting, language)
            elif variant == 'friendly':
                greeting = self.make_friendly(base_greeting, language)
            else:
                greeting = base_greeting

            # Add contextual elements based on user message
            if user_message:
                subject = self.detect_subject_interest(user_message)
                if subject != 'general':
                    greeting += f"\n\nI see you're interested in {subject}! That's one of my favorite subjects to teach. ✨"

            response_time = time.time() - start_time

            # Log greeting generation
            wandb.log({
                "greeting_generation": {
                    "variant": variant,
                    "language": language,
                    "response_time": response_time,
                    "subject_detected": subject if user_message else None
                }
            })

            return greeting

        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            return get_language_greeting(language)

    async def generate_response(
        self,
        user_message: str,
        prompt_template: str,
        language: str,
        session_id: str
    ) -> str:
        """
        Generate tutoring response with model fallback chain
        """

        start_time = time.time()

        try:
            # Check cache first
            cache_key = self.get_cache_key(user_message, prompt_template, language)
            if cache_key in self.response_cache:
                cache_entry = self.response_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:

                    wandb.log({
                        "response_generation": {
                            "method": "cache_hit",
                            "response_time": time.time() - start_time,
                            "session_id": session_id
                        }
                    })

                    return cache_entry['response']

            # Detect subject and customize approach
            subject = self.detect_subject_interest(user_message)
            expert_config = self.subject_experts.get(subject, self.subject_experts['general'])

            # Enhance prompt with subject-specific guidance
            enhanced_prompt = self.enhance_prompt_for_subject(
                prompt_template,
                expert_config,
                user_message
            )

            # Try models in order of preference (speed + quality)
            response = None

            # Try Agent Forge model first (our trained model)
            if self.agent_forge_available and self.agent_forge_model:
                try:
                    response = await self.generate_agent_forge_response(
                        enhanced_prompt, user_message, language, session_id
                    )
                    if response:
                        model_used = 'agent_forge'
                        logger.info("Used Agent Forge model for response")
                except Exception as e:
                    logger.error(f"Agent Forge model failed: {e}")

            # Try Anthropic as fallback (good balance of speed and quality)
            if not response and self.anthropic_client:
                try:
                    response = await self.generate_with_anthropic(
                        enhanced_prompt,
                        user_message,
                        language
                    )

                    if response:
                        self.update_model_performance('anthropic', start_time, True)

                except Exception as e:
                    logger.warning(f"Anthropic generation failed: {e}")
                    self.update_model_performance('anthropic', start_time, False)

            # Fallback to OpenAI
            if not response and self.openai_client:
                try:
                    response = await self.generate_with_openai(
                        enhanced_prompt,
                        user_message,
                        language
                    )

                    if response:
                        self.update_model_performance('openai', start_time, True)

                except Exception as e:
                    logger.warning(f"OpenAI generation failed: {e}")
                    self.update_model_performance('openai', start_time, False)

            # Final fallback to rule-based response
            if not response:
                response = self.generate_fallback_response(user_message, subject, language)
                self.update_model_performance('fallback', start_time, True)

            # Cache successful response
            if response:
                self.response_cache[cache_key] = {
                    'response': response,
                    'timestamp': time.time()
                }

            response_time = time.time() - start_time

            # Log response generation
            wandb.log({
                "response_generation": {
                    "subject": subject,
                    "language": language,
                    "response_time": response_time,
                    "session_id": session_id,
                    "response_length": len(response) if response else 0,
                    "cache_miss": True
                }
            })

            return response or "I'm having trouble generating a response right now. Please try rephrasing your question."

        except Exception as e:
            logger.error(f"Error in response generation: {e}")

            # Emergency fallback
            return self.generate_emergency_fallback(language)

    def detect_subject_interest(self, message: str) -> str:
        """Detect subject area from user message"""

        message_lower = message.lower()

        # Score each subject based on keyword matches
        subject_scores = {}
        for subject, config in self.subject_experts.items():
            if subject == 'general':
                continue

            score = sum(1 for keyword in config['keywords'] if keyword in message_lower)
            if score > 0:
                subject_scores[subject] = score

        # Return highest scoring subject or general
        if subject_scores:
            best_subject = max(subject_scores.items(), key=lambda x: x[1])[0]
            return best_subject

        return 'general'

    def enhance_prompt_for_subject(
        self,
        base_prompt: str,
        expert_config: Dict[str, Any],
        user_message: str
    ) -> str:
        """Enhance prompt template with subject-specific guidance"""

        # Add subject-specific instruction
        enhanced_prompt = base_prompt

        if expert_config['prompt_suffix']:
            enhanced_prompt += f"\n\nSpecial instruction: {expert_config['prompt_suffix']}"

        # Add approach-specific guidance
        approach_guides = {
            'step_by_step': "\nBreak down complex problems into clear, numbered steps.",
            'conceptual': "\nFocus on explaining the 'why' behind concepts, not just the 'how'.",
            'practical': "\nProvide concrete examples and hands-on exercises.",
            'interactive': "\nAsk questions to engage the student and check understanding.",
            'narrative': "\nUse storytelling to make information memorable and engaging."
        }

        approach = expert_config.get('approach', 'adaptive')
        if approach in approach_guides:
            enhanced_prompt += approach_guides[approach]

        return enhanced_prompt

    async def generate_with_anthropic(
        self,
        prompt: str,
        user_message: str,
        language: str
    ) -> Optional[str]:
        """Generate response using Anthropic Claude"""

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt.format(user_message=user_message)
                }
            ]

            response = await self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Fast model for sub-5s response
                max_tokens=800,
                temperature=0.3,
                messages=messages
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return None

    async def generate_agent_forge_response(
        self,
        prompt: str,
        user_message: str,
        language: str,
        session_id: str
    ) -> Optional[str]:
        """Generate response using Agent Forge model."""

        try:
            import torch

            # Create educational prompt template
            educational_prompt = f"""You are an expert AI tutor specializing in personalized education. Provide clear, helpful explanations that encourage learning.

Student Question: {user_message}

Guidelines:
- Explain concepts clearly and step-by-step
- Use examples to illustrate points
- Encourage further learning
- Be supportive and patient
- Keep responses concise but thorough

Response:"""

            # Tokenize input
            inputs = self.agent_forge_tokenizer.encode(educational_prompt, return_tensors="pt")

            # Move to appropriate device
            device = next(self.agent_forge_model.parameters()).device
            inputs = inputs.to(device)

            # Generate response
            with torch.no_grad():
                outputs = self.agent_forge_model.generate(
                    inputs,
                    max_length=inputs.size(1) + 200,  # Allow reasonable response length
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.agent_forge_tokenizer.eos_token_id,
                    eos_token_id=self.agent_forge_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode response
            full_response = self.agent_forge_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(educational_prompt):].strip()

            # Clean up response
            if response:
                # Remove any incomplete sentences at the end
                sentences = response.split('.')
                if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                    response = '.'.join(sentences[:-1]) + '.'

                # Ensure reasonable length
                if len(response) > 1000:
                    response = response[:1000] + "..."

                # Update performance tracking
                self.model_performance['agent_forge']['calls'] += 1

                return response.strip()

            return None

        except Exception as e:
            logger.error(f"Agent Forge generation error: {e}")
            self.model_performance['agent_forge']['calls'] += 1  # Track failed attempts
            return None

    async def generate_with_openai(
        self,
        prompt: str,
        user_message: str,
        language: str
    ) -> Optional[str]:
        """Generate response using OpenAI GPT"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Fast and cost-effective
                messages=[
                    {"role": "system", "content": "You are a helpful AI tutor."},
                    {"role": "user", "content": prompt.format(user_message=user_message)}
                ],
                max_tokens=800,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return None

    def generate_fallback_response(
        self,
        user_message: str,
        subject: str,
        language: str
    ) -> str:
        """Generate rule-based fallback response"""

        # Subject-specific fallback templates
        fallback_templates = {
            'mathematics': "I'd be happy to help with your math question about '{message}'. Let me break this down step by step for you...",
            'science': "Great science question about '{message}'! Let me explain the key concepts behind this...",
            'programming': "I can help you with that programming question about '{message}'. Here's how I'd approach this...",
            'language': "That's an interesting language question about '{message}'. Let me help you understand this better...",
            'history': "Your history question about '{message}' is fascinating. Let me share what I know about this topic...",
            'general': "Thank you for your question about '{message}'. I'm here to help you learn and understand this topic better..."
        }

        template = fallback_templates.get(subject, fallback_templates['general'])
        response = template.format(message=user_message[:50])

        # Add language-specific encouragement
        encouragements = {
            'en': "Feel free to ask follow-up questions!",
            'es': "¡No dudes en hacer preguntas de seguimiento!",
            'hi': "अतिरिक्त प्रश्न पूछने में संकोच न करें!",
            'sw': "Hisi kuuliza maswali ya ziada!",
            'ar': "لا تتردد في طرح أسئلة إضافية!",
            'pt': "Sinta-se à vontade para fazer perguntas de acompanhamento!",
            'fr': "N'hésitez pas à poser des questions de suivi!"
        }

        encouragement = encouragements.get(language, encouragements['en'])
        return f"{response}\n\n{encouragement}"

    def generate_emergency_fallback(self, language: str) -> str:
        """Emergency fallback when all systems fail"""

        emergency_responses = {
            'en': "I'm experiencing technical difficulties. Please try asking your question again in a moment.",
            'es': "Estoy experimentando dificultades técnicas. Por favor intenta hacer tu pregunta de nuevo en un momento.",
            'hi': "मुझे तकनीकी कठिनाइयों का सामना कर रहा हूं। कृपया एक क्षण में अपना प्रश्न फिर से पूछने का प्रयास करें।",
            'sw': "Nina matatizo ya kiufundi. Tafadhali jaribu kuuliza swali lako tena baada ya muda.",
            'ar': "أواجه صعوبات تقنية. يرجى المحاولة مرة أخرى في وقت لاحق.",
            'pt': "Estou enfrentando dificuldades técnicas. Tente fazer sua pergunta novamente em um momento.",
            'fr': "Je rencontre des difficultés techniques. Veuillez réessayer votre question dans un moment."
        }

        return emergency_responses.get(language, emergency_responses['en'])

    def make_enthusiastic(self, base_greeting: str, language: str) -> str:
        """Add enthusiastic elements to greeting"""

        enthusiastic_additions = {
            'en': " 🌟 I'm super excited to learn with you today!",
            'es': " 🌟 ¡Estoy súper emocionado de aprender contigo hoy!",
            'hi': " 🌟 आज आपके साथ सीखने के लिए मैं बहुत उत्साहित हूं!",
            'sw': " 🌟 Nina furaha sana kujifunza nawe leo!",
            'ar': " 🌟 أنا متحمس جداً للتعلم معك اليوم!",
            'pt': " 🌟 Estou super animado para aprender com você hoje!",
            'fr': " 🌟 Je suis super excité d'apprendre avec vous aujourd'hui!"
        }

        addition = enthusiastic_additions.get(language, enthusiastic_additions['en'])
        return base_greeting + addition

    def make_professional(self, base_greeting: str, language: str) -> str:
        """Add professional elements to greeting"""

        professional_additions = {
            'en': " I'm committed to providing you with accurate, comprehensive educational support.",
            'es': " Me comprometo a brindarte apoyo educativo preciso y completo.",
            'hi': " मैं आपको सटीक और व्यापक शैक्षिक सहायता प्रदान करने के लिए प्रतिबद्ध हूं।",
            'sw': " Nimejitoa kutoa msaada wa kielimu sahihi na kamili.",
            'ar': " أنا ملتزم بتقديم الدعم التعليمي الدقيق والشامل لك.",
            'pt': " Estou comprometido em fornecer suporte educacional preciso e abrangente.",
            'fr': " Je m'engage à vous fournir un soutien éducatif précis et complet."
        }

        addition = professional_additions.get(language, professional_additions['en'])
        return base_greeting + addition

    def make_friendly(self, base_greeting: str, language: str) -> str:
        """Add friendly elements to greeting"""

        friendly_additions = {
            'en': " 😊 Think of me as your learning buddy - we're in this together!",
            'es': " 😊 ¡Piensa en mí como tu compañero de aprendizaje - estamos juntos en esto!",
            'hi': " 😊 मुझे अपना सीखने का साथी समझें - हम इसमें एक साथ हैं!",
            'sw': " 😊 Nilete kama rafiki yako wa kujifunza - tuko pamoja katika hili!",
            'ar': " 😊 فكر فيّ كصديق التعلم - نحن في هذا معاً!",
            'pt': " 😊 Pense em mim como seu companheiro de aprendizado - estamos juntos nisso!",
            'fr': " 😊 Pensez à moi comme votre ami d'apprentissage - nous sommes ensemble!"
        }

        addition = friendly_additions.get(language, friendly_additions['en'])
        return base_greeting + addition

    def get_cache_key(self, user_message: str, prompt: str, language: str) -> str:
        """Generate cache key for response caching"""

        combined = f"{user_message}_{prompt}_{language}"
        return hashlib.md5(combined.encode()).hexdigest()

    def update_model_performance(self, model: str, start_time: float, success: bool):
        """Update performance tracking for models"""

        response_time = time.time() - start_time

        if model in self.model_performance:
            perf = self.model_performance[model]
            perf['calls'] += 1

            if success:
                # Update running average of response time
                if perf['calls'] == 1:
                    perf['avg_time'] = response_time
                else:
                    perf['avg_time'] = (perf['avg_time'] * (perf['calls'] - 1) + response_time) / perf['calls']

                # Update success rate
                perf['success_rate'] = (perf['success_rate'] * (perf['calls'] - 1) + 1) / perf['calls']
            else:
                perf['success_rate'] = (perf['success_rate'] * (perf['calls'] - 1)) / perf['calls']

        # Log performance update
        wandb.log({
            "model_performance": {
                "model": model,
                "response_time": response_time,
                "success": success,
                "avg_time": self.model_performance[model]['avg_time'],
                "success_rate": self.model_performance[model]['success_rate']
            }
        })

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""

        return {
            "model_performance": self.model_performance,
            "cache_size": len(self.response_cache),
            "avg_response_time": sum(self.response_times[-100:]) / len(self.response_times[-100:]) if self.response_times else 0,
            "subjects_supported": list(self.subject_experts.keys())
        }
