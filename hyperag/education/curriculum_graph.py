"""Hypergraph-based Curriculum Graph Implementation
Sprint R-4+AF1: Education Core System - Task A.1
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
from typing import Any

import wandb

# Import from hyperag components
from ..core.hypergraph_kg import Hyperedge, HypergraphKG, Node

logger = logging.getLogger(__name__)

@dataclass
class ConceptNode:
    """Educational concept with metadata"""

    concept_id: str
    name: str
    subject: str
    grade: int
    difficulty_level: float
    content: dict[str, str]  # language -> content
    learning_objectives: list[str]
    assessment_criteria: list[str]
    estimated_time_minutes: int
    prerequisites: list[str]
    follow_up_concepts: list[str]
    cultural_adaptations: dict[str, Any]  # region -> adaptations
    created_at: str = ""
    updated_at: str = ""

@dataclass
class LearningPath:
    """Sequence of concepts forming a learning path"""

    path_id: str
    name: str
    subject: str
    grade_range: tuple[int, int]
    concepts: list[str]  # concept_ids in order
    estimated_duration_hours: float
    difficulty_progression: list[float]
    cultural_region: str
    language: str
    completion_rate: float = 0.0
    effectiveness_score: float = 0.0

class CurriculumGraph:
    """Hypergraph-based curriculum with W&B tracking and cultural adaptation"""

    def __init__(self, project_name: str = "aivillage-education"):
        self.project_name = project_name
        self.graph = HypergraphKG()
        self.concepts = {}  # concept_id -> ConceptNode
        self.learning_paths = {}  # path_id -> LearningPath
        self.subject_taxonomies = {}  # subject -> hierarchy

        # Cultural and regional data
        self.cultural_examples = defaultdict(dict)  # (region, concept) -> examples
        self.regional_contexts = {}  # region -> context data

        # Performance tracking
        self.concept_effectiveness = defaultdict(float)
        self.learning_analytics = defaultdict(list)

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Load base curriculum data
        asyncio.create_task(self.initialize_base_curriculum())

    def initialize_wandb_tracking(self):
        """Initialize W&B tracking for curriculum development"""
        try:
            wandb.init(
                project=self.project_name,
                job_type="curriculum_development",
                config={
                    "curriculum_version": "1.0.0",
                    "grade_range": "K-8",
                    "subjects": ["mathematics", "science", "language_arts", "social_studies"],
                    "languages": ["en", "es", "hi", "fr", "ar", "pt", "sw"],
                    "cultural_regions": ["north_america", "latin_america", "south_asia", "east_africa", "middle_east"]
                }
            )

            logger.info("W&B curriculum tracking initialized")

        except Exception as e:
            logger.error(f"Failed to initialize W&B tracking: {e}")

    async def initialize_base_curriculum(self):
        """Initialize base K-8 curriculum across subjects"""
        # Mathematics curriculum (K-8)
        await self.build_mathematics_curriculum()

        # Science curriculum (K-8)
        await self.build_science_curriculum()

        # Language Arts curriculum (K-8)
        await self.build_language_arts_curriculum()

        # Social Studies curriculum (K-8)
        await self.build_social_studies_curriculum()

        # Create cross-curricular connections
        await self.create_cross_curricular_connections()

        # Generate learning paths
        await self.generate_adaptive_learning_paths()

        logger.info("Base K-8 curriculum initialized")

    async def add_concept(self,
                         subject: str,
                         grade: int,
                         concept: str,
                         prerequisites: list[str],
                         content: dict[str, str],
                         learning_objectives: list[str] = None,
                         cultural_adaptations: dict[str, Any] = None,
                         estimated_time: int = 30) -> str:
        """Add educational concept with multi-language content and cultural adaptations"""
        concept_id = self.generate_concept_id(subject, grade, concept)

        # Create concept node
        concept_node = ConceptNode(
            concept_id=concept_id,
            name=concept,
            subject=subject,
            grade=grade,
            difficulty_level=self.estimate_difficulty(prerequisites, grade),
            content=content or {},
            learning_objectives=learning_objectives or [],
            assessment_criteria=self.generate_assessment_criteria(concept, grade),
            estimated_time_minutes=estimated_time,
            prerequisites=prerequisites,
            follow_up_concepts=[],
            cultural_adaptations=cultural_adaptations or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )

        # Store concept
        self.concepts[concept_id] = concept_node

        # Create knowledge graph node
        kg_node = Node(
            node_id=concept_id,
            node_type="educational_concept",
            properties={
                "name": concept,
                "subject": subject,
                "grade": grade,
                "difficulty": concept_node.difficulty_level,
                "languages": list(content.keys()) if content else []
            }
        )

        await self.graph.add_node(kg_node)

        # Create hyperedge for concept relationships
        if prerequisites:
            edge = Hyperedge(
                edge_id=f"prereq_{concept_id}",
                entities=[concept_id] + prerequisites,
                relation_type="requires_knowledge_of",
                metadata={
                    "subject": subject,
                    "grade": grade,
                    "difficulty": concept_node.difficulty_level,
                    "prerequisite_count": len(prerequisites)
                }
            )

            await self.graph.add_edge(edge)

        # Update follow-up concepts for prerequisites
        for prereq_id in prerequisites:
            if prereq_id in self.concepts:
                self.concepts[prereq_id].follow_up_concepts.append(concept_id)

        # Log to W&B for curriculum coverage tracking
        wandb.log({
            f"curriculum/{subject}/grade_{grade}_concepts": len([c for c in self.concepts.values() if c.subject == subject and c.grade == grade]),
            "curriculum/total_concepts": len(self.concepts),
            "curriculum/languages_covered": len(set().union(*[c.content.keys() for c in self.concepts.values()])),
            "concept_added": True,
            "concept_id": concept_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.info(f"Added concept: {concept} ({subject}, Grade {grade})")

        return concept_id

    def generate_concept_id(self, subject: str, grade: int, concept: str) -> str:
        """Generate unique concept ID"""
        # Create deterministic ID based on content
        content_hash = hashlib.md5(f"{subject}_{grade}_{concept}".encode()).hexdigest()[:8]
        return f"{subject.lower()}_{grade}_{content_hash}"

    def estimate_difficulty(self, prerequisites: list[str], grade: int) -> float:
        """Estimate concept difficulty based on prerequisites and grade level"""
        base_difficulty = grade / 8.0  # Normalize grade to 0-1

        # Add complexity based on prerequisites
        prereq_complexity = len(prerequisites) * 0.1

        # Consider depth of prerequisite chain
        max_depth = 0
        for prereq_id in prerequisites:
            if prereq_id in self.concepts:
                prereq_depth = len(self.concepts[prereq_id].prerequisites)
                max_depth = max(max_depth, prereq_depth)

        depth_complexity = max_depth * 0.05

        total_difficulty = min(1.0, base_difficulty + prereq_complexity + depth_complexity)

        return total_difficulty

    def generate_assessment_criteria(self, concept: str, grade: int) -> list[str]:
        """Generate assessment criteria based on concept and grade level"""
        base_criteria = [
            f"Student can explain {concept} in their own words",
            f"Student can identify examples of {concept}",
            f"Student can apply {concept} to solve problems"
        ]

        # Add grade-appropriate criteria
        if grade <= 2:
            base_criteria.extend([
                f"Student can recognize {concept} visually",
                f"Student can demonstrate {concept} with manipulatives"
            ])
        elif grade <= 5:
            base_criteria.extend([
                f"Student can compare {concept} to related concepts",
                f"Student can create examples of {concept}"
            ])
        else:  # Grades 6-8
            base_criteria.extend([
                f"Student can analyze the relationship between {concept} and other concepts",
                f"Student can evaluate different approaches to {concept}",
                f"Student can synthesize {concept} with prior knowledge"
            ])

        return base_criteria

    async def build_mathematics_curriculum(self):
        """Build comprehensive K-8 mathematics curriculum"""
        # Kindergarten Mathematics
        k_concepts = [
            ("Counting to 10", [], {"en": "Learn to count from 1 to 10", "es": "Aprende a contar del 1 al 10"}),
            ("Number Recognition", ["Counting to 10"], {"en": "Recognize numbers 0-10", "es": "Reconocer números 0-10"}),
            ("Basic Shapes", [], {"en": "Circle, square, triangle, rectangle", "es": "Círculo, cuadrado, triángulo, rectángulo"}),
            ("Patterns", [], {"en": "Simple AB patterns", "es": "Patrones simples AB"}),
            ("Comparing Size", [], {"en": "Big, small, bigger, smaller", "es": "Grande, pequeño, más grande, más pequeño"})
        ]

        for concept, prereqs, content in k_concepts:
            await self.add_concept("mathematics", 0, concept, prereqs, content)

        # Grade 1 Mathematics
        grade1_concepts = [
            ("Counting to 100", ["Counting to 10"], {"en": "Count to 100 by ones and tens", "es": "Contar hasta 100 de uno en uno y de diez en diez"}),
            ("Addition Facts to 10", ["Number Recognition"], {"en": "Basic addition within 10", "es": "Suma básica hasta 10"}),
            ("Subtraction Facts to 10", ["Addition Facts to 10"], {"en": "Basic subtraction within 10", "es": "Resta básica hasta 10"}),
            ("Place Value (Tens and Ones)", ["Counting to 100"], {"en": "Understanding tens and ones", "es": "Entender decenas y unidades"}),
            ("Measurement Length", ["Comparing Size"], {"en": "Compare and order lengths", "es": "Comparar y ordenar longitudes"})
        ]

        for concept, prereqs, content in grade1_concepts:
            await self.add_concept("mathematics", 1, concept, prereqs, content)

        # Grade 2 Mathematics
        grade2_concepts = [
            ("Addition within 100", ["Addition Facts to 10", "Place Value (Tens and Ones)"], {"en": "Add two-digit numbers", "es": "Sumar números de dos dígitos"}),
            ("Subtraction within 100", ["Subtraction Facts to 10", "Place Value (Tens and Ones)"], {"en": "Subtract two-digit numbers", "es": "Restar números de dos dígitos"}),
            ("Skip Counting", ["Counting to 100"], {"en": "Count by 2s, 5s, and 10s", "es": "Contar de 2 en 2, de 5 en 5, y de 10 en 10"}),
            ("Introduction to Multiplication", ["Skip Counting"], {"en": "Equal groups and arrays", "es": "Grupos iguales y matrices"}),
            ("Fractions (Halves, Thirds, Fourths)", ["Basic Shapes"], {"en": "Simple fractions of shapes", "es": "Fracciones simples de formas"})
        ]

        for concept, prereqs, content in grade2_concepts:
            await self.add_concept("mathematics", 2, concept, prereqs, content)

        # Continue building through Grade 8...
        await self.build_advanced_math_concepts()

    async def build_advanced_math_concepts(self):
        """Build Grade 3-8 mathematics concepts"""
        # Grade 3 Mathematics
        grade3_concepts = [
            ("Multiplication Tables", ["Introduction to Multiplication"], {"en": "Memorize multiplication facts 0-12", "es": "Memorizar tablas de multiplicar 0-12"}),
            ("Division Facts", ["Multiplication Tables"], {"en": "Related division facts", "es": "Hechos de división relacionados"}),
            ("Fractions on Number Line", ["Fractions (Halves, Thirds, Fourths)"], {"en": "Represent fractions on number line", "es": "Representar fracciones en la recta numérica"}),
            ("Area and Perimeter", ["Measurement Length"], {"en": "Calculate area and perimeter of rectangles", "es": "Calcular área y perímetro de rectángulos"}),
            ("Data and Graphs", ["Patterns"], {"en": "Create and interpret bar graphs", "es": "Crear e interpretar gráficos de barras"})
        ]

        for concept, prereqs, content in grade3_concepts:
            await self.add_concept("mathematics", 3, concept, prereqs, content)

        # Grade 4 Mathematics
        grade4_concepts = [
            ("Multi-digit Multiplication", ["Multiplication Tables"], {"en": "Multiply multi-digit numbers", "es": "Multiplicar números de múltiples dígitos"}),
            ("Multi-digit Division", ["Division Facts"], {"en": "Divide multi-digit numbers", "es": "Dividir números de múltiples dígitos"}),
            ("Equivalent Fractions", ["Fractions on Number Line"], {"en": "Find equivalent fractions", "es": "Encontrar fracciones equivalentes"}),
            ("Decimal Notation", ["Place Value (Tens and Ones)"], {"en": "Understand decimal place value", "es": "Entender el valor posicional decimal"}),
            ("Angles and Lines", ["Basic Shapes"], {"en": "Types of angles and lines", "es": "Tipos de ángulos y líneas"})
        ]

        for concept, prereqs, content in grade4_concepts:
            await self.add_concept("mathematics", 4, concept, prereqs, content)

        # Grade 5 Mathematics
        grade5_concepts = [
            ("Operations with Decimals", ["Decimal Notation", "Multi-digit Multiplication"], {"en": "Add, subtract, multiply, divide decimals", "es": "Sumar, restar, multiplicar, dividir decimales"}),
            ("Operations with Fractions", ["Equivalent Fractions"], {"en": "Add and subtract fractions", "es": "Sumar y restar fracciones"}),
            ("Volume of Rectangular Prisms", ["Area and Perimeter"], {"en": "Calculate volume using formulas", "es": "Calcular volumen usando fórmulas"}),
            ("Coordinate Plane", ["Data and Graphs"], {"en": "Plot points on coordinate plane", "es": "Graficar puntos en el plano coordenado"}),
            ("Patterns and Relationships", ["Patterns"], {"en": "Analyze numerical patterns", "es": "Analizar patrones numéricos"})
        ]

        for concept, prereqs, content in grade5_concepts:
            await self.add_concept("mathematics", 5, concept, prereqs, content)

        # Grade 6 Mathematics
        grade6_concepts = [
            ("Ratios and Proportions", ["Operations with Fractions"], {"en": "Understand ratios and unit rates", "es": "Entender razones y tasas unitarias"}),
            ("Percent", ["Operations with Decimals"], {"en": "Find percent of a number", "es": "Encontrar el porcentaje de un número"}),
            ("Negative Numbers", ["Coordinate Plane"], {"en": "Operations with integers", "es": "Operaciones con enteros"}),
            ("Algebraic Expressions", ["Patterns and Relationships"], {"en": "Write and evaluate expressions", "es": "Escribir y evaluar expresiones"}),
            ("Statistics and Data", ["Data and Graphs"], {"en": "Measures of center and variability", "es": "Medidas de tendencia central y variabilidad"})
        ]

        for concept, prereqs, content in grade6_concepts:
            await self.add_concept("mathematics", 6, concept, prereqs, content)

        # Grade 7 Mathematics
        grade7_concepts = [
            ("Proportional Relationships", ["Ratios and Proportions"], {"en": "Solve proportional relationship problems", "es": "Resolver problemas de relaciones proporcionales"}),
            ("Operations with Rational Numbers", ["Negative Numbers"], {"en": "Add, subtract, multiply, divide rational numbers", "es": "Sumar, restar, multiplicar, dividir números racionales"}),
            ("Linear Equations", ["Algebraic Expressions"], {"en": "Solve one-step and two-step equations", "es": "Resolver ecuaciones de uno y dos pasos"}),
            ("Geometry and Scale", ["Volume of Rectangular Prisms"], {"en": "Scale drawings and similar figures", "es": "Dibujos a escala y figuras similares"}),
            ("Probability", ["Statistics and Data"], {"en": "Theoretical and experimental probability", "es": "Probabilidad teórica y experimental"})
        ]

        for concept, prereqs, content in grade7_concepts:
            await self.add_concept("mathematics", 7, concept, prereqs, content)

        # Grade 8 Mathematics
        grade8_concepts = [
            ("Linear Functions", ["Linear Equations", "Coordinate Plane"], {"en": "Graph and analyze linear functions", "es": "Graficar y analizar funciones lineales"}),
            ("Systems of Equations", ["Linear Functions"], {"en": "Solve systems of linear equations", "es": "Resolver sistemas de ecuaciones lineales"}),
            ("Exponents and Scientific Notation", ["Operations with Rational Numbers"], {"en": "Work with exponents and scientific notation", "es": "Trabajar con exponentes y notación científica"}),
            ("Pythagorean Theorem", ["Geometry and Scale"], {"en": "Apply Pythagorean theorem", "es": "Aplicar el teorema de Pitágoras"}),
            ("Transformations", ["Coordinate Plane"], {"en": "Translations, reflections, rotations", "es": "Traslaciones, reflexiones, rotaciones"})
        ]

        for concept, prereqs, content in grade8_concepts:
            await self.add_concept("mathematics", 8, concept, prereqs, content)

    async def build_science_curriculum(self):
        """Build K-8 science curriculum with NGSS alignment"""
        # Physical Science concepts
        physical_science_concepts = [
            (0, "Properties of Objects", [], {"en": "Objects have properties", "es": "Los objetos tienen propiedades"}),
            (1, "Light and Sound", ["Properties of Objects"], {"en": "Light and sound travel", "es": "La luz y el sonido viajan"}),
            (2, "Materials and Properties", ["Properties of Objects"], {"en": "Different materials have different properties", "es": "Diferentes materiales tienen diferentes propiedades"}),
            (3, "Forces and Motion", ["Properties of Objects"], {"en": "Forces cause motion", "es": "Las fuerzas causan movimiento"}),
            (4, "Energy Transfer", ["Forces and Motion"], {"en": "Energy can be transferred", "es": "La energía puede transferirse"}),
            (5, "Matter and Molecules", ["Materials and Properties"], {"en": "Matter is made of particles", "es": "La materia está hecha de partículas"}),
            (6, "Chemical Reactions", ["Matter and Molecules"], {"en": "Substances react to form new substances", "es": "Las sustancias reaccionan para formar nuevas sustancias"}),
            (7, "Electromagnetic Waves", ["Light and Sound", "Energy Transfer"], {"en": "Waves carry energy", "es": "Las ondas transportan energía"}),
            (8, "Atomic Structure", ["Chemical Reactions"], {"en": "Atoms make up all matter", "es": "Los átomos forman toda la materia"})
        ]

        for grade, concept, prereqs, content in physical_science_concepts:
            await self.add_concept("science", grade, concept, prereqs, content, estimated_time=45)

        # Life Science concepts
        life_science_concepts = [
            (0, "Basic Needs of Living Things", [], {"en": "All living things have needs", "es": "Todos los seres vivos tienen necesidades"}),
            (1, "Plant and Animal Structures", ["Basic Needs of Living Things"], {"en": "Structures help organisms survive", "es": "Las estructuras ayudan a los organismos a sobrevivir"}),
            (2, "Life Cycles", ["Plant and Animal Structures"], {"en": "All organisms have life cycles", "es": "Todos los organismos tienen ciclos de vida"}),
            (3, "Traits and Environment", ["Life Cycles"], {"en": "Traits help organisms survive", "es": "Los rasgos ayudan a los organismos a sobrevivir"}),
            (4, "Food Webs", ["Traits and Environment"], {"en": "Energy flows through ecosystems", "es": "La energía fluye a través de los ecosistemas"}),
            (5, "Ecosystems", ["Food Webs"], {"en": "Organisms interact in ecosystems", "es": "Los organismos interactúan en los ecosistemas"}),
            (6, "Cell Structure", ["Ecosystems"], {"en": "Cells are the basic unit of life", "es": "Las células son la unidad básica de la vida"}),
            (7, "Genetics and Heredity", ["Cell Structure"], {"en": "Traits are inherited", "es": "Los rasgos se heredan"}),
            (8, "Evolution and Natural Selection", ["Genetics and Heredity"], {"en": "Species change over time", "es": "Las especies cambian con el tiempo"})
        ]

        for grade, concept, prereqs, content in life_science_concepts:
            await self.add_concept("science", grade, concept, prereqs, content, estimated_time=45)

        # Earth and Space Science concepts
        earth_space_concepts = [
            (0, "Weather Patterns", [], {"en": "Weather changes daily", "es": "El clima cambia diariamente"}),
            (1, "Sun and Seasons", ["Weather Patterns"], {"en": "Sun affects Earth's surface", "es": "El sol afecta la superficie de la Tierra"}),
            (2, "Earth Materials", [], {"en": "Earth is made of different materials", "es": "La Tierra está hecha de diferentes materiales"}),
            (3, "Climate and Weather", ["Sun and Seasons"], {"en": "Climate patterns affect regions", "es": "Los patrones climáticos afectan las regiones"}),
            (4, "Rock Cycle", ["Earth Materials"], {"en": "Rocks change over time", "es": "Las rocas cambian con el tiempo"}),
            (5, "Water Cycle", ["Climate and Weather"], {"en": "Water cycles through Earth's systems", "es": "El agua circula a través de los sistemas de la Tierra"}),
            (6, "Plate Tectonics", ["Rock Cycle"], {"en": "Earth's plates move and change", "es": "Las placas de la Tierra se mueven y cambian"}),
            (7, "Solar System", ["Sun and Seasons"], {"en": "Gravity governs solar system", "es": "La gravedad gobierna el sistema solar"}),
            (8, "Universe and Stars", ["Solar System"], {"en": "Stars have life cycles", "es": "Las estrellas tienen ciclos de vida"})
        ]

        for grade, concept, prereqs, content in earth_space_concepts:
            await self.add_concept("science", grade, concept, prereqs, content, estimated_time=45)

    async def build_language_arts_curriculum(self):
        """Build K-8 language arts curriculum"""
        # Reading concepts
        reading_concepts = [
            (0, "Letter Recognition", [], {"en": "Recognize all letters", "es": "Reconocer todas las letras"}),
            (0, "Phonemic Awareness", [], {"en": "Hear sounds in words", "es": "Escuchar sonidos en las palabras"}),
            (1, "Phonics", ["Letter Recognition", "Phonemic Awareness"], {"en": "Connect letters to sounds", "es": "Conectar letras con sonidos"}),
            (1, "Sight Words", ["Letter Recognition"], {"en": "Recognize common words", "es": "Reconocer palabras comunes"}),
            (2, "Reading Fluency", ["Phonics", "Sight Words"], {"en": "Read with accuracy and speed", "es": "Leer con precisión y velocidad"}),
            (2, "Reading Comprehension", ["Reading Fluency"], {"en": "Understand what you read", "es": "Entender lo que lees"}),
            (3, "Main Idea and Details", ["Reading Comprehension"], {"en": "Identify main ideas", "es": "Identificar ideas principales"}),
            (4, "Text Structure", ["Main Idea and Details"], {"en": "Understand how texts are organized", "es": "Entender cómo se organizan los textos"}),
            (5, "Literary Elements", ["Text Structure"], {"en": "Character, setting, plot", "es": "Personaje, escenario, trama"}),
            (6, "Theme and Inference", ["Literary Elements"], {"en": "Infer meaning and identify themes", "es": "Inferir significado e identificar temas"}),
            (7, "Literary Analysis", ["Theme and Inference"], {"en": "Analyze author's purpose and style", "es": "Analizar el propósito y estilo del autor"}),
            (8, "Critical Reading", ["Literary Analysis"], {"en": "Evaluate arguments and evidence", "es": "Evaluar argumentos y evidencia"})
        ]

        for grade, concept, prereqs, content in reading_concepts:
            await self.add_concept("language_arts", grade, concept, prereqs, content, estimated_time=30)

        # Writing concepts
        writing_concepts = [
            (0, "Letter Formation", [], {"en": "Form letters correctly", "es": "Formar letras correctamente"}),
            (1, "Simple Sentences", ["Letter Formation"], {"en": "Write complete sentences", "es": "Escribir oraciones completas"}),
            (2, "Paragraph Writing", ["Simple Sentences"], {"en": "Write organized paragraphs", "es": "Escribir párrafos organizados"}),
            (3, "Narrative Writing", ["Paragraph Writing"], {"en": "Tell stories with beginning, middle, end", "es": "Contar historias con principio, medio, fin"}),
            (4, "Informative Writing", ["Narrative Writing"], {"en": "Explain topics clearly", "es": "Explicar temas claramente"}),
            (5, "Opinion Writing", ["Informative Writing"], {"en": "Support opinions with reasons", "es": "Apoyar opiniones con razones"}),
            (6, "Research Writing", ["Opinion Writing"], {"en": "Use sources to support writing", "es": "Usar fuentes para apoyar la escritura"}),
            (7, "Argumentative Writing", ["Research Writing"], {"en": "Build logical arguments", "es": "Construir argumentos lógicos"}),
            (8, "Academic Writing", ["Argumentative Writing"], {"en": "Write for academic purposes", "es": "Escribir para propósitos académicos"})
        ]

        for grade, concept, prereqs, content in writing_concepts:
            await self.add_concept("language_arts", grade, concept, prereqs, content, estimated_time=40)

    async def build_social_studies_curriculum(self):
        """Build K-8 social studies curriculum"""
        # History and culture concepts
        history_concepts = [
            (0, "Family History", [], {"en": "Learn about family traditions", "es": "Aprender sobre tradiciones familiares"}),
            (1, "Community Helpers", ["Family History"], {"en": "People who help our community", "es": "Personas que ayudan a nuestra comunidad"}),
            (2, "Local History", ["Community Helpers"], {"en": "History of our town/city", "es": "Historia de nuestro pueblo/ciudad"}),
            (3, "Native American History", ["Local History"], {"en": "First peoples of America", "es": "Primeros pueblos de América"}),
            (4, "Colonial America", ["Native American History"], {"en": "European colonization", "es": "Colonización europea"}),
            (5, "American Revolution", ["Colonial America"], {"en": "Birth of the United States", "es": "Nacimiento de los Estados Unidos"}),
            (6, "Ancient Civilizations", [], {"en": "Early civilizations worldwide", "es": "Primeras civilizaciones mundiales"}),
            (7, "Medieval and Renaissance", ["Ancient Civilizations"], {"en": "Middle Ages and Renaissance", "es": "Edad Media y Renacimiento"}),
            (8, "Industrial Revolution", ["American Revolution"], {"en": "Changes in work and society", "es": "Cambios en el trabajo y la sociedad"})
        ]

        for grade, concept, prereqs, content in history_concepts:
            await self.add_concept("social_studies", grade, concept, prereqs, content, estimated_time=40)

        # Geography concepts
        geography_concepts = [
            (0, "Maps and Globes", [], {"en": "Basic map skills", "es": "Habilidades básicas de mapas"}),
            (1, "Neighborhoods", ["Maps and Globes"], {"en": "Features of neighborhoods", "es": "Características de los vecindarios"}),
            (2, "Communities", ["Neighborhoods"], {"en": "Rural, suburban, urban", "es": "Rural, suburbano, urbano"}),
            (3, "States and Regions", ["Communities"], {"en": "US states and regions", "es": "Estados y regiones de EE.UU."}),
            (4, "Physical Geography", ["States and Regions"], {"en": "Landforms and climate", "es": "Accidentes geográficos y clima"}),
            (5, "World Geography", ["Physical Geography"], {"en": "Continents and countries", "es": "Continentes y países"}),
            (6, "Cultural Geography", ["World Geography"], {"en": "How geography affects culture", "es": "Cómo la geografía afecta la cultura"}),
            (7, "Economic Geography", ["Cultural Geography"], {"en": "Resources and trade", "es": "Recursos y comercio"}),
            (8, "Global Issues", ["Economic Geography"], {"en": "Worldwide challenges", "es": "Desafíos mundiales"})
        ]

        for grade, concept, prereqs, content in geography_concepts:
            await self.add_concept("social_studies", grade, concept, prereqs, content, estimated_time=35)

    async def create_cross_curricular_connections(self):
        """Create connections between subjects"""
        # Math-Science connections
        math_science_connections = [
            ("Data and Graphs", "Weather Patterns", "graphing_weather_data"),
            ("Measurement Length", "Properties of Objects", "measuring_objects"),
            ("Fractions", "Food Webs", "energy_transfer_fractions"),
            ("Coordinate Plane", "Solar System", "plotting_planet_positions"),
            ("Statistics and Data", "Ecosystems", "analyzing_population_data")
        ]

        for math_concept, science_concept, connection_type in math_science_connections:
            await self.create_concept_connection(math_concept, science_concept, connection_type)

        # Language Arts-Social Studies connections
        la_ss_connections = [
            ("Reading Comprehension", "Local History", "reading_historical_texts"),
            ("Research Writing", "Ancient Civilizations", "research_projects"),
            ("Literary Elements", "Cultural Geography", "analyzing_cultural_stories"),
            ("Critical Reading", "Global Issues", "evaluating_news_sources")
        ]

        for la_concept, ss_concept, connection_type in la_ss_connections:
            await self.create_concept_connection(la_concept, ss_concept, connection_type)

    async def create_concept_connection(self, concept1_name: str, concept2_name: str, connection_type: str):
        """Create hyperedge connecting concepts across subjects"""
        # Find concept IDs
        concept1_id = None
        concept2_id = None

        for concept_id, concept in self.concepts.items():
            if concept.name == concept1_name:
                concept1_id = concept_id
            elif concept.name == concept2_name:
                concept2_id = concept_id

        if concept1_id and concept2_id:
            edge = Hyperedge(
                edge_id=f"connection_{connection_type}",
                entities=[concept1_id, concept2_id],
                relation_type="cross_curricular_connection",
                metadata={
                    "connection_type": connection_type,
                    "subjects": [self.concepts[concept1_id].subject, self.concepts[concept2_id].subject],
                    "strength": 0.8  # Connection strength
                }
            )

            await self.graph.add_edge(edge)

    async def generate_adaptive_learning_paths(self):
        """Generate adaptive learning paths for different learner profiles"""
        subjects = ["mathematics", "science", "language_arts", "social_studies"]

        for subject in subjects:
            for grade in range(9):  # K-8
                # Standard learning path
                await self.create_learning_path(
                    subject=subject,
                    grade_range=(grade, grade),
                    difficulty_preference="standard",
                    cultural_region="north_america",
                    language="en"
                )

                # Accelerated learning path
                if grade < 8:
                    await self.create_learning_path(
                        subject=subject,
                        grade_range=(grade, grade + 1),
                        difficulty_preference="accelerated",
                        cultural_region="north_america",
                        language="en"
                    )

                # Remedial learning path
                if grade > 0:
                    await self.create_learning_path(
                        subject=subject,
                        grade_range=(grade - 1, grade),
                        difficulty_preference="remedial",
                        cultural_region="north_america",
                        language="en"
                    )

    async def create_learning_path(self,
                                 subject: str,
                                 grade_range: tuple[int, int],
                                 difficulty_preference: str,
                                 cultural_region: str,
                                 language: str) -> str:
        """Create optimized learning path for specific parameters"""
        # Get concepts for grade range and subject
        relevant_concepts = [
            concept for concept in self.concepts.values()
            if concept.subject == subject and
            grade_range[0] <= concept.grade <= grade_range[1]
        ]

        if not relevant_concepts:
            return None

        # Sort concepts by dependencies (topological sort)
        ordered_concepts = await self.topological_sort_concepts(relevant_concepts)

        # Adjust for difficulty preference
        if difficulty_preference == "accelerated":
            # Include more challenging concepts
            ordered_concepts = [c for c in ordered_concepts if c.difficulty_level >= 0.6]
        elif difficulty_preference == "remedial":
            # Focus on foundational concepts
            ordered_concepts = [c for c in ordered_concepts if c.difficulty_level <= 0.7]

        # Calculate path metrics
        total_time = sum(concept.estimated_time_minutes for concept in ordered_concepts) / 60.0
        difficulty_progression = [concept.difficulty_level for concept in ordered_concepts]

        # Create learning path
        path_id = f"{subject}_{grade_range[0]}_{grade_range[1]}_{difficulty_preference}_{cultural_region}_{language}"

        learning_path = LearningPath(
            path_id=path_id,
            name=f"{subject.title()} Path (Grade {grade_range[0]}-{grade_range[1]}, {difficulty_preference.title()})",
            subject=subject,
            grade_range=grade_range,
            concepts=[concept.concept_id for concept in ordered_concepts],
            estimated_duration_hours=total_time,
            difficulty_progression=difficulty_progression,
            cultural_region=cultural_region,
            language=language
        )

        self.learning_paths[path_id] = learning_path

        # Log to W&B
        wandb.log({
            f"learning_paths/{subject}/total": len([p for p in self.learning_paths.values() if p.subject == subject]),
            "learning_path_created": True,
            "path_id": path_id,
            "concepts_count": len(ordered_concepts),
            "estimated_hours": total_time
        })

        return path_id

    async def topological_sort_concepts(self, concepts: list[ConceptNode]) -> list[ConceptNode]:
        """Sort concepts by dependency order"""
        # Create concept lookup
        concept_lookup = {concept.concept_id: concept for concept in concepts}

        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for concept in concepts:
            in_degree[concept.concept_id] = 0

        for concept in concepts:
            for prereq_id in concept.prerequisites:
                if prereq_id in concept_lookup:
                    graph[prereq_id].append(concept.concept_id)
                    in_degree[concept.concept_id] += 1

        # Topological sort using Kahn's algorithm
        queue = deque([concept_id for concept_id in in_degree if in_degree[concept_id] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(concept_lookup[current])

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    async def get_concept_by_id(self, concept_id: str) -> ConceptNode | None:
        """Retrieve concept by ID"""
        return self.concepts.get(concept_id)

    async def get_concepts_by_subject_grade(self, subject: str, grade: int) -> list[ConceptNode]:
        """Get all concepts for a subject and grade"""
        return [
            concept for concept in self.concepts.values()
            if concept.subject == subject and concept.grade == grade
        ]

    async def get_learning_path(self, path_id: str) -> LearningPath | None:
        """Retrieve learning path by ID"""
        return self.learning_paths.get(path_id)

    async def find_optimal_learning_path(self,
                                       subject: str,
                                       current_grade: int,
                                       mastered_concepts: list[str],
                                       learning_style: str,
                                       cultural_region: str,
                                       language: str) -> LearningPath | None:
        """Find optimal learning path based on student profile"""
        # Filter paths by criteria
        candidate_paths = []

        for path in self.learning_paths.values():
            if (path.subject == subject and
                path.grade_range[0] <= current_grade <= path.grade_range[1] + 1 and
                path.cultural_region == cultural_region and
                path.language == language):

                # Calculate path suitability score
                suitability_score = await self.calculate_path_suitability(
                    path, mastered_concepts, learning_style
                )

                candidate_paths.append((path, suitability_score))

        if not candidate_paths:
            return None

        # Return path with highest suitability score
        best_path = max(candidate_paths, key=lambda x: x[1])[0]

        # Log path selection
        wandb.log({
            "optimal_path_selected": True,
            "subject": subject,
            "grade": current_grade,
            "path_id": best_path.path_id,
            "cultural_region": cultural_region,
            "language": language
        })

        return best_path

    async def calculate_path_suitability(self,
                                       path: LearningPath,
                                       mastered_concepts: list[str],
                                       learning_style: str) -> float:
        """Calculate how suitable a learning path is for a student"""
        score = 0.0

        # Check prerequisite mastery
        path_concepts = [await self.get_concept_by_id(cid) for cid in path.concepts]
        path_concepts = [c for c in path_concepts if c is not None]

        prerequisite_mastery = 0
        total_prerequisites = 0

        for concept in path_concepts:
            for prereq_id in concept.prerequisites:
                total_prerequisites += 1
                if prereq_id in mastered_concepts:
                    prerequisite_mastery += 1

        if total_prerequisites > 0:
            score += (prerequisite_mastery / total_prerequisites) * 0.4
        else:
            score += 0.4  # No prerequisites needed

        # Check difficulty progression
        difficulty_variance = 0
        if len(path.difficulty_progression) > 1:
            diffs = path.difficulty_progression
            for i in range(1, len(diffs)):
                if diffs[i] < diffs[i-1]:  # Difficulty decrease is bad
                    difficulty_variance += 0.1

        score += max(0, 0.3 - difficulty_variance)

        # Learning style compatibility
        if learning_style == "visual":
            # Prefer paths with more visual concepts
            visual_concepts = len([c for c in path_concepts if "visual" in c.name.lower() or "shape" in c.name.lower()])
            score += min(0.2, visual_concepts * 0.05)
        elif learning_style == "kinesthetic":
            # Prefer hands-on concepts
            hands_on_concepts = len([c for c in path_concepts if "manipulatives" in str(c.assessment_criteria)])
            score += min(0.2, hands_on_concepts * 0.05)
        else:
            score += 0.1  # Default bonus

        # Path completeness
        score += min(0.1, len(path.concepts) / 20.0)  # Bonus for comprehensive paths

        return score

    async def get_curriculum_statistics(self) -> dict[str, Any]:
        """Get comprehensive curriculum statistics"""
        stats = {
            "total_concepts": len(self.concepts),
            "subjects": {},
            "grades": {},
            "languages": set(),
            "cultural_regions": set(),
            "learning_paths": len(self.learning_paths),
            "cross_curricular_connections": 0,
            "average_difficulty": 0.0,
            "total_estimated_hours": 0.0
        }

        # Analyze concepts
        for concept in self.concepts.values():
            # Subject breakdown
            if concept.subject not in stats["subjects"]:
                stats["subjects"][concept.subject] = 0
            stats["subjects"][concept.subject] += 1

            # Grade breakdown
            if concept.grade not in stats["grades"]:
                stats["grades"][concept.grade] = 0
            stats["grades"][concept.grade] += 1

            # Languages
            stats["languages"].update(concept.content.keys())

            # Cultural regions
            stats["cultural_regions"].update(concept.cultural_adaptations.keys())

            # Accumulate metrics
            stats["total_estimated_hours"] += concept.estimated_time_minutes / 60.0

        # Calculate averages
        if self.concepts:
            stats["average_difficulty"] = sum(c.difficulty_level for c in self.concepts.values()) / len(self.concepts)

        # Count cross-curricular connections
        edges = await self.graph.get_all_edges()
        stats["cross_curricular_connections"] = len([e for e in edges if e.relation_type == "cross_curricular_connection"])

        # Convert sets to lists for JSON serialization
        stats["languages"] = list(stats["languages"])
        stats["cultural_regions"] = list(stats["cultural_regions"])

        return stats

# Global curriculum graph instance
curriculum_graph = CurriculumGraph()
