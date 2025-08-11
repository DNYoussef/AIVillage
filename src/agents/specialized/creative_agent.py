"""
Creative Agent - Content Generation and Artistic Task Specialist
"""
import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import hashlib

from src.production.rag.rag_system.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


@dataclass
class CreativeRequest:
    """Request for creative content generation"""
    content_type: str  # 'story', 'poem', 'design', 'music', 'art'
    theme: str
    style: Optional[str] = None
    length: str = 'medium'
    constraints: Dict[str, Any] = None


class CreativeAgent(AgentInterface):
    """
    Specialized agent for creative tasks including:
    - Story and narrative generation
    - Poetry and creative writing
    - Visual design concepts
    - Music composition ideas
    - Art direction and aesthetics
    - Brand and marketing creative
    """
    
    def __init__(self, agent_id: str = "creative_agent"):
        self.agent_id = agent_id
        self.agent_type = "Creative"
        self.capabilities = [
            "story_generation",
            "poetry_writing",
            "visual_design",
            "music_composition",
            "art_direction",
            "brand_creative",
            "character_development",
            "world_building"
        ]
        self.creative_projects = {}
        self.style_library = {}
        self.inspiration_sources = []
        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate creative content based on prompt"""
        if "story" in prompt.lower() or "narrative" in prompt.lower():
            return "I can create compelling stories with rich characters and engaging plots. Specify genre and themes."
        elif "poem" in prompt.lower() or "poetry" in prompt.lower():
            return "I craft poetry in various styles - haiku, sonnets, free verse, and more. What's your theme?"
        elif "design" in prompt.lower() or "visual" in prompt.lower():
            return "I provide visual design concepts, color schemes, layouts, and aesthetic direction."
        elif "music" in prompt.lower():
            return "I can suggest musical compositions, chord progressions, and thematic arrangements."
        elif "art" in prompt.lower():
            return "I offer art direction, conceptual ideas, and aesthetic guidance for visual projects."
        return "I'm a Creative Agent specialized in generating original content across multiple artistic mediums."
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for creative text"""
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_value % 1000) / 1000.0] * 384
    
    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank results based on creative relevance"""
        keywords = ['creative', 'art', 'design', 'story', 'music', 'visual', 'aesthetic', 'inspiration']
        
        for result in results:
            score = 0
            text = str(result.get('content', ''))
            for keyword in keywords:
                score += text.lower().count(keyword)
            result['creative_relevance_score'] = score
            
        return sorted(results, key=lambda x: x.get('creative_relevance_score', 0), reverse=True)[:k]
    
    async def introspect(self) -> dict[str, Any]:
        """Return agent capabilities and status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'capabilities': self.capabilities,
            'active_projects': len(self.creative_projects),
            'style_library_size': len(self.style_library),
            'inspiration_sources': len(self.inspiration_sources),
            'initialized': self.initialized
        }
    
    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with other agents"""
        response = await recipient.generate(f"Creative Agent says: {message}")
        return f"Received response: {response}"
    
    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space for creative generation"""
        creative_type = "visual" if any(word in query.lower() for word in ["design", "art", "visual"]) else "narrative"
        latent_representation = f"CREATIVE[{creative_type}:{query[:50]}]"
        return creative_type, latent_representation

    async def generate_story(self, request: CreativeRequest) -> Dict[str, Any]:
        """Generate a story based on theme and style"""
        try:
            theme = request.theme
            style = request.style or "contemporary"
            length = request.length
            
            # Story elements
            genres = ["fantasy", "sci-fi", "mystery", "romance", "thriller", "drama"]
            characters = [
                {"name": "Elena", "role": "protagonist", "trait": "determined"},
                {"name": "Marcus", "role": "mentor", "trait": "wise"},
                {"name": "Zara", "role": "antagonist", "trait": "cunning"}
            ]
            
            settings = [
                "a bustling cyberpunk city",
                "an ancient mystical forest", 
                "a remote space station",
                "a quaint coastal town",
                "a post-apocalyptic wasteland"
            ]
            
            conflicts = [
                "discovery of a hidden truth",
                "race against time to prevent disaster",
                "internal struggle with identity",
                "battle between good and evil",
                "quest for redemption"
            ]
            
            # Generate story structure
            genre = random.choice(genres)
            setting = random.choice(settings)
            main_character = random.choice(characters)
            conflict = random.choice(conflicts)
            
            if length == "short":
                word_count = "500-1000"
                acts = 3
            elif length == "long":
                word_count = "5000-10000"
                acts = 5
            else:
                word_count = "2000-3000"
                acts = 4
            
            story_outline = {
                "title": f"The {theme.title()} Chronicles",
                "genre": genre,
                "style": style,
                "setting": setting,
                "main_character": main_character,
                "central_conflict": conflict,
                "theme": theme,
                "structure": {
                    "acts": acts,
                    "estimated_word_count": word_count
                },
                "plot_points": [
                    f"Opening: Introduce {main_character['name']} in {setting}",
                    f"Inciting incident: {conflict} emerges",
                    f"Rising action: {main_character['name']} faces challenges",
                    f"Climax: Confrontation with core conflict",
                    f"Resolution: Theme of {theme} is realized"
                ],
                "opening_paragraph": f"In {setting}, {main_character['name']} had always been known for being {main_character['trait']}. But nothing could have prepared them for what was about to unfold—a journey that would test everything they believed about {theme}."
            }
            
            return story_outline
            
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            return {'error': str(e)}
    
    async def create_poetry(self, request: CreativeRequest) -> Dict[str, Any]:
        """Create poetry based on theme and style"""
        try:
            theme = request.theme
            style = request.style or "free_verse"
            
            poetry_forms = {
                "haiku": {"lines": 3, "syllables": [5, 7, 5], "structure": "traditional"},
                "sonnet": {"lines": 14, "rhyme_scheme": "ABAB CDCD EFEF GG", "structure": "shakespearean"},
                "free_verse": {"lines": "variable", "structure": "unmetered", "style": "modern"},
                "limerick": {"lines": 5, "rhyme_scheme": "AABBA", "structure": "humorous"}
            }
            
            if style not in poetry_forms:
                style = "free_verse"
            
            form_info = poetry_forms[style]
            
            # Generate poem based on theme
            if theme.lower() in ["nature", "seasons", "beauty"]:
                sample_lines = [
                    "Whispers of wind through ancient trees",
                    "Sunlight dances on morning dew",
                    "Rivers carve stories in stone",
                    "Petals fall like gentle tears"
                ]
            elif theme.lower() in ["love", "heart", "emotion"]:
                sample_lines = [
                    "Hearts speak in languages beyond words",
                    "Love blooms in unexpected places", 
                    "Memories weave golden threads",
                    "Two souls find their rhythm"
                ]
            elif theme.lower() in ["time", "change", "life"]:
                sample_lines = [
                    "Clock hands sweep away yesterdays",
                    "Seasons turn with patient grace",
                    "Each moment holds infinity",
                    "Change is the only constant friend"
                ]
            else:
                sample_lines = [
                    f"In the realm of {theme}, truth emerges",
                    f"Where {theme} meets the soul",
                    f"Dancing with the spirit of {theme}",
                    f"The essence of {theme} unfolds"
                ]
            
            poem_structure = {
                "title": f"Reflections on {theme.title()}",
                "style": style,
                "form": form_info,
                "theme": theme,
                "verses": [
                    {
                        "stanza": 1,
                        "lines": sample_lines[:2]
                    },
                    {
                        "stanza": 2, 
                        "lines": sample_lines[2:]
                    }
                ],
                "literary_devices": [
                    "metaphor",
                    "imagery",
                    "alliteration" if style != "free_verse" else "enjambment"
                ],
                "mood": "contemplative",
                "tone": "reflective"
            }
            
            return poem_structure
            
        except Exception as e:
            logger.error(f"Poetry creation failed: {e}")
            return {'error': str(e)}
    
    async def design_visual_concept(self, request: CreativeRequest) -> Dict[str, Any]:
        """Create visual design concept"""
        try:
            theme = request.theme
            style = request.style or "modern"
            
            color_palettes = {
                "warm": ["#FF6B35", "#F7931E", "#FFD23F", "#EE4B2B"],
                "cool": ["#4A90E2", "#7ED321", "#50E3C2", "#B8E986"],
                "monochrome": ["#000000", "#333333", "#666666", "#CCCCCC"],
                "earth": ["#8B4513", "#D2691E", "#CD853F", "#F4A460"],
                "vibrant": ["#FF1493", "#00FFFF", "#ADFF2F", "#FF4500"]
            }
            
            design_styles = {
                "modern": {"description": "Clean, minimal, geometric", "fonts": ["Helvetica", "Futura"]},
                "vintage": {"description": "Retro, ornate, textured", "fonts": ["Serif", "Script"]},
                "minimalist": {"description": "Simple, spacious, functional", "fonts": ["Sans-serif"]},
                "artistic": {"description": "Expressive, bold, creative", "fonts": ["Display", "Handwritten"]},
                "corporate": {"description": "Professional, trustworthy, clean", "fonts": ["Arial", "Times"]}
            }
            
            if style not in design_styles:
                style = "modern"
            
            # Select appropriate color palette based on theme
            if theme.lower() in ["nature", "organic", "earth"]:
                palette_name = "earth"
            elif theme.lower() in ["technology", "future", "digital"]:
                palette_name = "cool"
            elif theme.lower() in ["energy", "passion", "excitement"]:
                palette_name = "vibrant"
            else:
                palette_name = "warm"
            
            design_concept = {
                "title": f"{theme.title()} Design Concept",
                "style": style,
                "style_description": design_styles[style]["description"],
                "color_palette": {
                    "name": palette_name,
                    "colors": color_palettes[palette_name],
                    "primary": color_palettes[palette_name][0],
                    "secondary": color_palettes[palette_name][1],
                    "accent": color_palettes[palette_name][2]
                },
                "typography": {
                    "primary_font": design_styles[style]["fonts"][0],
                    "secondary_font": design_styles[style]["fonts"][-1] if len(design_styles[style]["fonts"]) > 1 else design_styles[style]["fonts"][0],
                    "hierarchy": ["Heading 1", "Heading 2", "Body", "Caption"]
                },
                "layout_principles": [
                    "Visual hierarchy",
                    "Balance and proportion",
                    "White space utilization",
                    "Consistent alignment"
                ],
                "mood_board": [
                    f"{theme} imagery",
                    f"{style} textures",
                    "Geometric patterns" if style == "modern" else "Organic shapes",
                    "High contrast elements"
                ],
                "applications": [
                    "Logo design",
                    "Website layout",
                    "Marketing materials",
                    "Brand identity"
                ]
            }
            
            return design_concept
            
        except Exception as e:
            logger.error(f"Visual design concept failed: {e}")
            return {'error': str(e)}
    
    async def compose_music_concept(self, request: CreativeRequest) -> Dict[str, Any]:
        """Create music composition concept"""
        try:
            theme = request.theme
            style = request.style or "contemporary"
            
            musical_keys = ["C major", "G major", "D major", "A minor", "E minor", "F# minor"]
            time_signatures = ["4/4", "3/4", "6/8", "2/4"]
            tempos = {
                "slow": "60-80 BPM (Adagio)",
                "medium": "100-120 BPM (Moderato)", 
                "fast": "140-160 BPM (Allegro)",
                "very_fast": "180+ BPM (Presto)"
            }
            
            instruments_by_style = {
                "classical": ["piano", "violin", "cello", "flute", "oboe"],
                "contemporary": ["guitar", "piano", "drums", "bass", "synthesizer"],
                "jazz": ["saxophone", "trumpet", "piano", "double bass", "drums"],
                "electronic": ["synthesizer", "drum machine", "sampler", "vocoder"],
                "folk": ["acoustic guitar", "violin", "harmonica", "mandolin", "banjo"]
            }
            
            if style not in instruments_by_style:
                style = "contemporary"
            
            # Theme-based musical elements
            if theme.lower() in ["love", "romance", "heart"]:
                suggested_key = "F major"
                tempo_desc = "medium"
                mood = "romantic"
                dynamics = "soft to moderate"
            elif theme.lower() in ["adventure", "journey", "epic"]:
                suggested_key = "D major"
                tempo_desc = "fast"
                mood = "heroic"
                dynamics = "building crescendo"
            elif theme.lower() in ["peace", "calm", "meditation"]:
                suggested_key = "C major"
                tempo_desc = "slow"
                mood = "serene"
                dynamics = "gentle and flowing"
            else:
                suggested_key = random.choice(musical_keys)
                tempo_desc = "medium"
                mood = "expressive"
                dynamics = "varied"
            
            composition_concept = {
                "title": f"Composition: {theme.title()}",
                "style": style,
                "key": suggested_key,
                "time_signature": random.choice(time_signatures),
                "tempo": tempos[tempo_desc],
                "mood": mood,
                "dynamics": dynamics,
                "instrumentation": instruments_by_style[style][:4],
                "structure": {
                    "intro": "8 bars",
                    "verse_a": "16 bars",
                    "chorus": "16 bars", 
                    "verse_b": "16 bars",
                    "chorus": "16 bars",
                    "bridge": "8 bars",
                    "chorus": "16 bars",
                    "outro": "8 bars"
                },
                "chord_progression": [
                    "I - vi - IV - V" if "major" in suggested_key else "i - VI - III - VII",
                    "vi - IV - I - V" if "major" in suggested_key else "VI - iv - i - V"
                ],
                "melodic_elements": [
                    f"Emphasizes {theme} through recurring motifs",
                    "Uses stepwise motion for smoothness",
                    "Incorporates rhythmic variation",
                    "Features call and response patterns"
                ]
            }
            
            return composition_concept
            
        except Exception as e:
            logger.error(f"Music composition failed: {e}")
            return {'error': str(e)}
    
    async def develop_character(self, character_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Develop detailed character profile"""
        try:
            name = character_brief.get('name', 'Unnamed Character')
            role = character_brief.get('role', 'protagonist')
            genre = character_brief.get('genre', 'contemporary')
            
            # Character traits based on role
            protagonist_traits = ["brave", "determined", "curious", "loyal", "resourceful"]
            antagonist_traits = ["cunning", "powerful", "ruthless", "intelligent", "charismatic"]
            mentor_traits = ["wise", "patient", "experienced", "caring", "mysterious"]
            
            if role == "protagonist":
                primary_trait = random.choice(protagonist_traits)
                secondary_traits = random.sample([t for t in protagonist_traits if t != primary_trait], 2)
            elif role == "antagonist":
                primary_trait = random.choice(antagonist_traits)
                secondary_traits = random.sample([t for t in antagonist_traits if t != primary_trait], 2)
            else:  # mentor or supporting
                primary_trait = random.choice(mentor_traits)
                secondary_traits = random.sample([t for t in mentor_traits if t != primary_trait], 2)
            
            # Background elements
            professions = ["teacher", "scientist", "artist", "engineer", "doctor", "detective", "warrior", "merchant"]
            origins = ["small town", "big city", "foreign country", "mystical realm", "space colony", "underground society"]
            
            character_profile = {
                "name": name,
                "role": role,
                "genre": genre,
                "personality": {
                    "primary_trait": primary_trait,
                    "secondary_traits": secondary_traits,
                    "fatal_flaw": "overconfidence" if role == "protagonist" else "underestimating others",
                    "greatest_fear": "failure" if role == "protagonist" else "losing control",
                    "core_desire": "to protect others" if role == "protagonist" else "to achieve power"
                },
                "background": {
                    "profession": random.choice(professions),
                    "origin": random.choice(origins),
                    "formative_event": f"A life-changing encounter that shaped their {primary_trait} nature",
                    "relationships": [
                        {"type": "family", "description": "Complex relationship with sibling"},
                        {"type": "mentor", "description": "Learned crucial skills from wise teacher"},
                        {"type": "rival", "description": "Ongoing competition drives growth"}
                    ]
                },
                "physical_description": {
                    "build": random.choice(["tall and lean", "average height", "short and sturdy", "imposing presence"]),
                    "distinctive_feature": random.choice(["piercing eyes", "confident smile", "graceful movements", "commanding voice"]),
                    "style": f"Dresses in a way that reflects their {primary_trait} personality"
                },
                "character_arc": {
                    "starting_point": f"Begins as someone who is {primary_trait} but limited by {character_profile.get('personality', {}).get('fatal_flaw', 'unknown flaw')}",
                    "challenge": f"Must overcome their {character_profile.get('personality', {}).get('greatest_fear', 'fear')} to grow",
                    "transformation": f"Learns to balance {primary_trait} with {secondary_traits[0]}",
                    "ending_point": "Emerges as a more complete, evolved person"
                },
                "dialogue_style": {
                    "speaking_pattern": f"Speaks in a {primary_trait} manner",
                    "vocabulary": "Complex" if primary_trait in ["intelligent", "wise"] else "Direct",
                    "catchphrase": f"A phrase that embodies {primary_trait}",
                    "internal_thoughts": f"Often contemplates {character_profile.get('personality', {}).get('core_desire', 'their goals')}"
                }
            }
            
            return character_profile
            
        except Exception as e:
            logger.error(f"Character development failed: {e}")
            return {'error': str(e)}
    
    async def initialize(self):
        """Initialize the Creative agent"""
        try:
            logger.info("Initializing Creative Agent...")
            
            # Initialize style library
            self.style_library = {
                "literary_styles": ["gothic", "romantic", "modernist", "minimalist", "surreal"],
                "visual_styles": ["abstract", "realistic", "impressionist", "art_deco", "bauhaus"],
                "musical_styles": ["classical", "jazz", "electronic", "folk", "contemporary"]
            }
            
            # Initialize inspiration sources
            self.inspiration_sources = [
                "nature patterns",
                "human emotions",
                "historical events", 
                "cultural traditions",
                "technological innovations",
                "philosophical concepts"
            ]
            
            self.initialized = True
            logger.info(f"Creative Agent {self.agent_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Creative Agent: {e}")
            self.initialized = False
    
    async def shutdown(self):
        """Cleanup resources"""
        try:
            # Save creative projects if needed
            for project_id in self.creative_projects:
                logger.info(f"Saving creative project: {project_id}")
            
            self.initialized = False
            logger.info(f"Creative Agent {self.agent_id} shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")