"""Horticulturist Agent - Smart Agriculture & Permaculture

The agriculture and permaculture specialist of AIVillage, responsible for:
- Crop planning, monitoring, and yield optimization
- Soil health assessment and management recommendations
- Sustainable farming practices and permaculture design
- Smart agriculture with IoT sensor integration
- Environmental monitoring and climate adaptation
- Mobile-optimized farming guidance and alerts
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class CropType(Enum):
    VEGETABLE = "vegetable"
    GRAIN = "grain"
    FRUIT = "fruit"
    HERB = "herb"
    LEGUME = "legume"
    ROOT = "root"


class GrowthStage(Enum):
    SEED = "seed"
    GERMINATION = "germination"
    SEEDLING = "seedling"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    MATURE = "mature"
    HARVEST = "harvest"


class SoilCondition(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class CropProfile:
    crop_id: str
    name: str
    crop_type: CropType
    variety: str
    planting_date: float
    expected_harvest_date: float
    growth_stage: GrowthStage
    days_to_maturity: int
    space_required_sqft: float
    companion_plants: list[str]
    water_requirements: str  # low, medium, high
    sun_requirements: str  # full_sun, partial_sun, shade
    soil_ph_range: tuple[float, float]
    created_timestamp: float


@dataclass
class SoilData:
    location_id: str
    ph_level: float
    nitrogen: float  # ppm
    phosphorus: float  # ppm
    potassium: float  # ppm
    organic_matter: float  # percentage
    moisture: float  # percentage
    temperature: float  # celsius
    condition: SoilCondition
    last_tested: float
    recommendations: list[str]


@dataclass
class EnvironmentalData:
    location_id: str
    temperature: float  # celsius
    humidity: float  # percentage
    light_level: float  # lux
    rainfall: float  # mm
    wind_speed: float  # km/h
    uv_index: float
    timestamp: float
    source: str  # sensor, api, manual


@dataclass
class GardenPlot:
    plot_id: str
    name: str
    size_sqft: float
    location: str
    soil_type: str
    sun_exposure: str
    crops_planted: list[str]  # crop_ids
    planting_zones: dict[str, list[str]]  # zone -> crop_ids
    companion_groupings: list[list[str]]
    created_timestamp: float
    last_updated: float


class HorticulturistAgent(AgentInterface):
    """Horticulturist Agent provides comprehensive smart agriculture and permaculture
    services including crop management, soil monitoring, and sustainable farming guidance.
    """

    def __init__(self, agent_id: str = "horticulturist_agent"):
        self.agent_id = agent_id
        self.agent_type = "Horticulturist"
        self.capabilities = [
            "crop_planning",
            "crop_monitoring",
            "yield_optimization",
            "soil_health_assessment",
            "permaculture_design",
            "companion_planting",
            "pest_management",
            "irrigation_optimization",
            "harvest_scheduling",
            "seasonal_planning",
            "climate_adaptation",
            "mobile_farming_alerts",
            "smart_sensor_integration",
            "sustainable_practices",
        ]

        # Crop and garden management
        self.crop_profiles: dict[str, CropProfile] = {}
        self.garden_plots: dict[str, GardenPlot] = {}
        self.soil_data: dict[str, SoilData] = {}
        self.environmental_data: dict[str, list[EnvironmentalData]] = {}

        # Knowledge database
        self.crop_database: dict[str, dict[str, Any]] = {}
        self.companion_plant_db: dict[str, list[str]] = {}
        self.pest_management_db: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self.crops_managed = 0
        self.successful_harvests = 0
        self.soil_assessments_completed = 0
        self.recommendations_provided = 0
        self.average_yield_improvement = 0.0

        # Smart agriculture settings
        self.mobile_alerts_enabled = True
        self.iot_sensor_integration = True
        self.automated_recommendations = True
        self.organic_practices_preferred = True

        # Regional and seasonal data
        self.growing_seasons = {
            "spring": {"start": 3, "end": 5},
            "summer": {"start": 6, "end": 8},
            "fall": {"start": 9, "end": 11},
            "winter": {"start": 12, "end": 2},
        }

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate agriculture and permaculture responses"""
        prompt_lower = prompt.lower()

        if "crop" in prompt_lower or "plant" in prompt_lower:
            return (
                "I provide crop planning, monitoring, and optimization for sustainable agriculture and higher yields."
            )
        if "soil" in prompt_lower:
            return "I assess soil health and provide recommendations for nutrition, pH balance, and organic matter improvement."
        if "permaculture" in prompt_lower or "sustainable" in prompt_lower:
            return "I design permaculture systems and promote sustainable farming practices using companion planting and natural methods."
        if "pest" in prompt_lower or "disease" in prompt_lower:
            return "I recommend organic pest management strategies and early detection methods for crop protection."
        if "harvest" in prompt_lower or "yield" in prompt_lower:
            return "I optimize harvest timing and provide yield improvement recommendations based on growth monitoring."

        return "I am Horticulturist Agent, your smart agriculture specialist for sustainable farming and permaculture design."

    async def get_embedding(self, text: str) -> list[float]:
        """Generate agriculture-focused embeddings"""
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Agriculture embeddings focus on seasonal patterns and crop relationships
        return [(hash_value % 1000) / 1000.0] * 384

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank based on agricultural relevance"""
        agriculture_keywords = [
            "crop",
            "soil",
            "plant",
            "grow",
            "harvest",
            "farming",
            "agriculture",
            "garden",
            "permaculture",
            "sustainable",
            "organic",
            "pest",
            "yield",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))

            for keyword in agriculture_keywords:
                score += content.lower().count(keyword) * 1.5

            # Boost sustainable and organic content
            if any(term in content.lower() for term in ["sustainable", "organic", "permaculture"]):
                score *= 1.4

            result["agriculture_relevance"] = score

        return sorted(results, key=lambda x: x.get("agriculture_relevance", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return Horticulturist agent status and agricultural metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "crops_managed": self.crops_managed,
            "garden_plots": len(self.garden_plots),
            "successful_harvests": self.successful_harvests,
            "soil_assessments": self.soil_assessments_completed,
            "recommendations_provided": self.recommendations_provided,
            "average_yield_improvement": self.average_yield_improvement,
            "crop_database_entries": len(self.crop_database),
            "companion_plant_combinations": len(self.companion_plant_db),
            "mobile_alerts_enabled": self.mobile_alerts_enabled,
            "organic_practices": self.organic_practices_preferred,
            "specialization": "smart_agriculture_and_permaculture",
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate agricultural insights and recommendations"""
        # Add agricultural context to communications
        if any(keyword in message.lower() for keyword in ["crop", "soil", "plant", "harvest"]):
            agricultural_context = "[AGRICULTURAL INSIGHT]"
            message = f"{agricultural_context} {message}"

        if recipient:
            response = await recipient.generate(f"Horticulturist Agent provides farming guidance: {message}")
            return f"Agricultural guidance delivered: {response[:50]}..."
        return "No recipient for agricultural guidance"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate agriculture-specific latent spaces"""
        query_lower = query.lower()

        if "crop" in query_lower or "plant" in query_lower:
            space_type = "crop_management"
        elif "soil" in query_lower:
            space_type = "soil_analysis"
        elif "pest" in query_lower or "disease" in query_lower:
            space_type = "pest_management"
        elif "permaculture" in query_lower or "sustainable" in query_lower:
            space_type = "permaculture_design"
        else:
            space_type = "general_agriculture"

        latent_repr = f"HORTICULTURIST[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def plan_crop(self, crop_data: dict[str, Any]) -> dict[str, Any]:
        """Plan and schedule crop planting - MVP function"""
        crop_id = crop_data.get("crop_id", f"crop_{int(time.time())}")

        # Get crop information from database or create basic profile
        crop_info = await self._get_crop_info(crop_data["name"])

        # Calculate optimal planting and harvest dates
        planting_date = crop_data.get("planting_date", time.time())
        harvest_date = planting_date + (crop_info["days_to_maturity"] * 24 * 3600)

        # Create crop profile
        crop_profile = CropProfile(
            crop_id=crop_id,
            name=crop_data["name"],
            crop_type=CropType(crop_data.get("type", "vegetable")),
            variety=crop_data.get("variety", "standard"),
            planting_date=planting_date,
            expected_harvest_date=harvest_date,
            growth_stage=GrowthStage.SEED,
            days_to_maturity=crop_info["days_to_maturity"],
            space_required_sqft=crop_info["space_required"],
            companion_plants=crop_info["companions"],
            water_requirements=crop_info["water_needs"],
            sun_requirements=crop_info["sun_needs"],
            soil_ph_range=crop_info["ph_range"],
            created_timestamp=time.time(),
        )

        # Store crop profile
        self.crop_profiles[crop_id] = crop_profile

        # Generate planting recommendations
        recommendations = await self._generate_planting_recommendations(crop_profile)

        # Create planting schedule
        planting_schedule = await self._create_planting_schedule(crop_profile)

        # Create receipt
        receipt = {
            "agent": "Horticulturist",
            "action": "crop_planning",
            "timestamp": time.time(),
            "crop_id": crop_id,
            "crop_name": crop_profile.name,
            "crop_type": crop_profile.crop_type.value,
            "planting_date": planting_date,
            "harvest_date": harvest_date,
            "days_to_maturity": crop_profile.days_to_maturity,
            "space_required": crop_profile.space_required_sqft,
            "companions_recommended": len(crop_profile.companion_plants),
            "signature": f"horticulturist_plan_{crop_id}",
        }

        self.crops_managed += 1
        self.recommendations_provided += len(recommendations)

        logger.info(
            f"Crop planning completed: {crop_profile.name} - harvest expected in {crop_profile.days_to_maturity} days"
        )

        return {
            "status": "success",
            "crop_id": crop_id,
            "crop_profile": crop_profile,
            "recommendations": recommendations,
            "planting_schedule": planting_schedule,
            "receipt": receipt,
        }

    async def _get_crop_info(self, crop_name: str) -> dict[str, Any]:
        """Get crop information from database"""
        # Basic crop database
        crop_db = {
            "tomato": {
                "days_to_maturity": 80,
                "space_required": 4.0,
                "companions": ["basil", "parsley", "marigold"],
                "water_needs": "medium",
                "sun_needs": "full_sun",
                "ph_range": (6.0, 6.8),
            },
            "lettuce": {
                "days_to_maturity": 45,
                "space_required": 1.0,
                "companions": ["carrots", "radish", "onion"],
                "water_needs": "medium",
                "sun_needs": "partial_sun",
                "ph_range": (6.0, 7.0),
            },
            "carrots": {
                "days_to_maturity": 70,
                "space_required": 0.5,
                "companions": ["lettuce", "chives", "parsley"],
                "water_needs": "low",
                "sun_needs": "full_sun",
                "ph_range": (6.0, 6.8),
            },
            "beans": {
                "days_to_maturity": 60,
                "space_required": 2.0,
                "companions": ["corn", "squash", "cucumber"],
                "water_needs": "medium",
                "sun_needs": "full_sun",
                "ph_range": (6.0, 7.0),
            },
            "basil": {
                "days_to_maturity": 40,
                "space_required": 1.0,
                "companions": ["tomato", "pepper", "oregano"],
                "water_needs": "medium",
                "sun_needs": "full_sun",
                "ph_range": (6.0, 7.0),
            },
        }

        crop_name_lower = crop_name.lower()
        if crop_name_lower in crop_db:
            return crop_db[crop_name_lower]
        # Default values for unknown crops
        return {
            "days_to_maturity": 60,
            "space_required": 2.0,
            "companions": [],
            "water_needs": "medium",
            "sun_needs": "full_sun",
            "ph_range": (6.0, 7.0),
        }

    async def _generate_planting_recommendations(self, crop: CropProfile) -> list[str]:
        """Generate planting recommendations for crop"""
        recommendations = []

        # Seasonal recommendations
        current_month = datetime.fromtimestamp(crop.planting_date).month
        if current_month in [12, 1, 2]:  # Winter
            recommendations.append("Consider using cold frames or greenhouses for winter planting")
            recommendations.append("Start seeds indoors if outdoor temperatures are too cold")
        elif current_month in [3, 4, 5]:  # Spring
            recommendations.append("Ideal planting season - prepare soil with compost")
            recommendations.append("Watch for late frost warnings that could damage young plants")
        elif current_month in [6, 7, 8]:  # Summer
            recommendations.append("Ensure adequate water supply during hot weather")
            recommendations.append("Provide shade during extreme heat if needed")
        else:  # Fall
            recommendations.append("Plant for fall harvest - consider succession planting")
            recommendations.append("Prepare for cooler weather and shorter days")

        # Companion planting
        if crop.companion_plants:
            companions_str = ", ".join(crop.companion_plants[:3])
            recommendations.append(f"Plant alongside companions: {companions_str}")
            recommendations.append("Companion plants help with pest control and soil nutrients")

        # Space and layout
        recommendations.append(f"Allocate {crop.space_required_sqft} sq ft per plant")
        recommendations.append(f"Ensure {crop.sun_requirements} exposure")

        # Soil preparation
        ph_min, ph_max = crop.soil_ph_range
        recommendations.append(f"Test soil pH - target range {ph_min}-{ph_max}")
        recommendations.append("Add organic compost 2-3 weeks before planting")

        return recommendations

    async def _create_planting_schedule(self, crop: CropProfile) -> dict[str, Any]:
        """Create detailed planting and care schedule"""
        schedule = {
            "planting_date": datetime.fromtimestamp(crop.planting_date).strftime("%Y-%m-%d"),
            "key_milestones": [],
            "care_tasks": [],
        }

        # Calculate key growth milestones
        germination_date = crop.planting_date + (7 * 24 * 3600)  # 1 week
        seedling_date = crop.planting_date + (21 * 24 * 3600)  # 3 weeks
        mature_date = crop.planting_date + (crop.days_to_maturity * 0.8 * 24 * 3600)

        schedule["key_milestones"] = [
            {
                "stage": "germination",
                "date": datetime.fromtimestamp(germination_date).strftime("%Y-%m-%d"),
                "description": "Seeds should begin sprouting",
            },
            {
                "stage": "seedling",
                "date": datetime.fromtimestamp(seedling_date).strftime("%Y-%m-%d"),
                "description": "Plants establish strong root system",
            },
            {
                "stage": "mature",
                "date": datetime.fromtimestamp(mature_date).strftime("%Y-%m-%d"),
                "description": "Plants reach full maturity",
            },
            {
                "stage": "harvest",
                "date": datetime.fromtimestamp(crop.expected_harvest_date).strftime("%Y-%m-%d"),
                "description": "Ready for harvest",
            },
        ]

        # Regular care tasks
        schedule["care_tasks"] = [
            {"frequency": "daily", "task": "Check soil moisture", "weeks": [1, 2, 3]},
            {
                "frequency": "weekly",
                "task": "Inspect for pests and diseases",
                "weeks": "all",
            },
            {
                "frequency": "bi-weekly",
                "task": "Apply organic fertilizer",
                "weeks": [4, 6, 8, 10],
            },
            {
                "frequency": "monthly",
                "task": "Prune and support plants",
                "weeks": [4, 8, 12],
            },
        ]

        return schedule

    async def assess_soil(self, location_id: str, soil_sample_data: dict[str, Any]) -> dict[str, Any]:
        """Assess soil health and provide recommendations - MVP function"""
        assessment_id = f"soil_{location_id}_{int(time.time())}"

        # Process soil sample data
        ph_level = soil_sample_data.get("ph", 6.5)
        nitrogen = soil_sample_data.get("nitrogen_ppm", 50)
        phosphorus = soil_sample_data.get("phosphorus_ppm", 30)
        potassium = soil_sample_data.get("potassium_ppm", 150)
        organic_matter = soil_sample_data.get("organic_matter_percent", 3.0)
        moisture = soil_sample_data.get("moisture_percent", 25)
        temperature = soil_sample_data.get("temperature_c", 20)

        # Determine soil condition
        soil_condition = await self._evaluate_soil_condition(ph_level, nitrogen, phosphorus, potassium, organic_matter)

        # Generate recommendations
        recommendations = await self._generate_soil_recommendations(
            ph_level, nitrogen, phosphorus, potassium, organic_matter, soil_condition
        )

        # Create soil data record
        soil_data = SoilData(
            location_id=location_id,
            ph_level=ph_level,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            organic_matter=organic_matter,
            moisture=moisture,
            temperature=temperature,
            condition=soil_condition,
            last_tested=time.time(),
            recommendations=recommendations,
        )

        # Store soil data
        self.soil_data[location_id] = soil_data

        # Create receipt
        receipt = {
            "agent": "Horticulturist",
            "action": "soil_assessment",
            "timestamp": time.time(),
            "assessment_id": assessment_id,
            "location_id": location_id,
            "ph_level": ph_level,
            "soil_condition": soil_condition.value,
            "nitrogen_ppm": nitrogen,
            "phosphorus_ppm": phosphorus,
            "potassium_ppm": potassium,
            "organic_matter_percent": organic_matter,
            "recommendations_count": len(recommendations),
            "signature": f"horticulturist_soil_{assessment_id}",
        }

        self.soil_assessments_completed += 1
        self.recommendations_provided += len(recommendations)

        logger.info(f"Soil assessment completed: {location_id} - {soil_condition.value} condition")

        return {
            "status": "success",
            "assessment_id": assessment_id,
            "soil_condition": soil_condition.value,
            "soil_data": soil_data,
            "recommendations": recommendations,
            "receipt": receipt,
        }

    async def _evaluate_soil_condition(self, ph: float, n: float, p: float, k: float, om: float) -> SoilCondition:
        """Evaluate overall soil condition based on key metrics"""
        score = 0

        # pH evaluation (6.0-7.0 is ideal for most crops)
        if 6.0 <= ph <= 7.0:
            score += 20
        elif 5.5 <= ph <= 7.5:
            score += 15
        elif 5.0 <= ph <= 8.0:
            score += 10
        else:
            score += 5

        # Nitrogen evaluation (40-60 ppm is good)
        if 40 <= n <= 60:
            score += 20
        elif 20 <= n <= 80:
            score += 15
        elif 10 <= n <= 100:
            score += 10
        else:
            score += 5

        # Phosphorus evaluation (30-50 ppm is good)
        if 30 <= p <= 50:
            score += 20
        elif 20 <= p <= 70:
            score += 15
        elif 10 <= p <= 90:
            score += 10
        else:
            score += 5

        # Potassium evaluation (120-180 ppm is good)
        if 120 <= k <= 180:
            score += 20
        elif 80 <= k <= 220:
            score += 15
        elif 40 <= k <= 300:
            score += 10
        else:
            score += 5

        # Organic matter evaluation (3-5% is ideal)
        if 3.0 <= om <= 5.0:
            score += 20
        elif 2.0 <= om <= 6.0:
            score += 15
        elif 1.0 <= om <= 7.0:
            score += 10
        else:
            score += 5

        # Convert score to condition
        if score >= 85:
            return SoilCondition.EXCELLENT
        if score >= 70:
            return SoilCondition.GOOD
        if score >= 50:
            return SoilCondition.FAIR
        if score >= 30:
            return SoilCondition.POOR
        return SoilCondition.CRITICAL

    async def _generate_soil_recommendations(
        self,
        ph: float,
        n: float,
        p: float,
        k: float,
        om: float,
        condition: SoilCondition,
    ) -> list[str]:
        """Generate soil improvement recommendations"""
        recommendations = []

        # pH recommendations
        if ph < 6.0:
            recommendations.append("Add agricultural lime to raise soil pH")
            recommendations.append("Apply wood ash in small amounts to increase alkalinity")
        elif ph > 7.5:
            recommendations.append("Add organic sulfur or pine needles to lower pH")
            recommendations.append("Use acidic compost to gradually reduce alkalinity")

        # Nutrient recommendations
        if n < 30:
            recommendations.append("Apply nitrogen-rich fertilizer (blood meal, fish emulsion)")
            recommendations.append("Plant nitrogen-fixing legumes as cover crops")
        elif n > 80:
            recommendations.append("Reduce nitrogen inputs to prevent excess growth")

        if p < 25:
            recommendations.append("Add phosphorus-rich bone meal or rock phosphate")
        elif p > 70:
            recommendations.append("Reduce phosphorus fertilizer to prevent runoff")

        if k < 100:
            recommendations.append("Apply potassium-rich compost or kelp meal")
        elif k > 250:
            recommendations.append("Monitor potassium levels - excess can block other nutrients")

        # Organic matter recommendations
        if om < 2.0:
            recommendations.append("Add 2-3 inches of compost to increase organic matter")
            recommendations.append("Use mulch to protect soil and add organic material")
            recommendations.append("Plant cover crops to build soil biology")
        elif om > 6.0:
            recommendations.append("Good organic matter levels - maintain with regular compost")

        # Condition-specific recommendations
        if condition == SoilCondition.CRITICAL:
            recommendations.append("URGENT: Soil needs immediate attention before planting")
            recommendations.append("Consider raised beds with imported soil mix")
        elif condition == SoilCondition.POOR:
            recommendations.append("Improve soil gradually over one growing season")
            recommendations.append("Focus on adding compost and adjusting pH")

        return recommendations

    async def monitor_crop_growth(self, crop_id: str, growth_data: dict[str, Any]) -> dict[str, Any]:
        """Monitor crop growth and provide care recommendations - MVP function"""
        monitoring_id = f"monitor_{crop_id}_{int(time.time())}"

        if crop_id not in self.crop_profiles:
            return {"status": "error", "message": "Crop not found"}

        crop = self.crop_profiles[crop_id]

        # Process growth observation data
        current_height = growth_data.get("height_cm", 0)
        leaf_condition = growth_data.get("leaf_condition", "good")  # excellent, good, fair, poor
        pest_presence = growth_data.get("pests_observed", [])
        disease_symptoms = growth_data.get("diseases_observed", [])
        flowering_status = growth_data.get("flowering", False)
        fruiting_status = growth_data.get("fruiting", False)

        # Update growth stage based on observations
        updated_stage = await self._determine_growth_stage(crop, current_height, flowering_status, fruiting_status)
        crop.growth_stage = updated_stage

        # Analyze growth progress
        days_since_planting = (time.time() - crop.planting_date) / (24 * 3600)
        expected_progress = days_since_planting / crop.days_to_maturity

        # Generate care recommendations
        care_recommendations = await self._generate_care_recommendations(
            crop, leaf_condition, pest_presence, disease_symptoms, expected_progress
        )

        # Check for harvest readiness
        harvest_assessment = await self._assess_harvest_readiness(crop, growth_data)

        # Create receipt
        receipt = {
            "agent": "Horticulturist",
            "action": "crop_monitoring",
            "timestamp": time.time(),
            "monitoring_id": monitoring_id,
            "crop_id": crop_id,
            "growth_stage": updated_stage.value,
            "days_since_planting": int(days_since_planting),
            "progress_percentage": int(expected_progress * 100),
            "leaf_condition": leaf_condition,
            "pests_detected": len(pest_presence),
            "diseases_detected": len(disease_symptoms),
            "harvest_ready": harvest_assessment["ready"],
            "recommendations_count": len(care_recommendations),
            "signature": f"horticulturist_monitor_{monitoring_id}",
        }

        logger.info(
            f"Crop monitoring completed: {crop.name} - {updated_stage.value} stage, {expected_progress * 100:.1f}% progress"
        )

        return {
            "status": "success",
            "monitoring_id": monitoring_id,
            "growth_stage": updated_stage.value,
            "progress_percentage": expected_progress * 100,
            "care_recommendations": care_recommendations,
            "harvest_assessment": harvest_assessment,
            "receipt": receipt,
        }

    async def _determine_growth_stage(
        self, crop: CropProfile, height: float, flowering: bool, fruiting: bool
    ) -> GrowthStage:
        """Determine current growth stage based on observations"""
        days_since_planting = (time.time() - crop.planting_date) / (24 * 3600)

        if fruiting and crop.crop_type in [CropType.FRUIT, CropType.VEGETABLE]:
            return GrowthStage.FRUITING
        if flowering:
            return GrowthStage.FLOWERING
        if days_since_planting >= crop.days_to_maturity * 0.8:
            return GrowthStage.MATURE
        if days_since_planting >= crop.days_to_maturity * 0.4:
            return GrowthStage.VEGETATIVE
        if days_since_planting >= 21:  # 3 weeks
            return GrowthStage.SEEDLING
        if days_since_planting >= 7:  # 1 week
            return GrowthStage.GERMINATION
        return GrowthStage.SEED

    async def _generate_care_recommendations(
        self,
        crop: CropProfile,
        leaf_condition: str,
        pests: list[str],
        diseases: list[str],
        progress: float,
    ) -> list[str]:
        """Generate care recommendations based on current crop status"""
        recommendations = []

        # Leaf condition recommendations
        if leaf_condition == "poor":
            recommendations.append("Check soil moisture and drainage - yellowing may indicate watering issues")
            recommendations.append("Apply balanced organic fertilizer if nutrient deficiency suspected")
        elif leaf_condition == "fair":
            recommendations.append("Monitor leaf health closely and adjust care as needed")

        # Pest management
        if pests:
            recommendations.append(f"Pest alert: {', '.join(pests[:3])} detected")
            recommendations.append("Apply organic pest control methods (neem oil, beneficial insects)")
            recommendations.append("Remove affected plant parts if infestation is localized")

        # Disease management
        if diseases:
            recommendations.append(f"Disease symptoms detected: {', '.join(diseases[:2])}")
            recommendations.append("Improve air circulation around plants")
            recommendations.append("Apply organic fungicide if fungal disease suspected")

        # Growth stage specific recommendations
        if crop.growth_stage == GrowthStage.FLOWERING:
            recommendations.append("Reduce nitrogen fertilizer during flowering to promote fruiting")
            recommendations.append("Ensure consistent watering to support flower development")
        elif crop.growth_stage == GrowthStage.FRUITING:
            recommendations.append("Increase phosphorus and potassium for fruit development")
            recommendations.append("Support heavy fruiting branches to prevent breaking")

        # Progress-based recommendations
        if progress < 0.5 and crop.growth_stage == GrowthStage.SEEDLING:
            recommendations.append("Consider slow growth - check soil temperature and fertility")
        elif progress > 0.8:
            recommendations.append("Prepare for harvest - monitor ripeness indicators daily")

        return recommendations

    async def _assess_harvest_readiness(self, crop: CropProfile, growth_data: dict[str, Any]) -> dict[str, Any]:
        """Assess if crop is ready for harvest"""
        days_since_planting = (time.time() - crop.planting_date) / (24 * 3600)

        # Basic readiness based on maturity time
        basic_ready = days_since_planting >= crop.days_to_maturity

        # Crop-specific readiness indicators
        ready = False
        indicators = []

        if crop.name.lower() == "tomato":
            color_good = growth_data.get("fruit_color", "") in [
                "red",
                "orange",
                "yellow",
            ]
            firmness_good = growth_data.get("fruit_firmness", "") == "firm"
            ready = basic_ready and color_good and firmness_good
            indicators = [
                "Check fruit color and firmness",
                "Harvest when fully colored but still firm",
            ]
        elif crop.name.lower() == "lettuce":
            head_formed = growth_data.get("head_formed", False)
            leaves_crisp = growth_data.get("leaves_crisp", True)
            ready = basic_ready and head_formed and leaves_crisp
            indicators = [
                "Harvest outer leaves first",
                "Cut head at base when fully formed",
            ]
        elif crop.name.lower() == "carrots":
            root_size = growth_data.get("root_diameter_cm", 0)
            ready = basic_ready and root_size >= 2.5
            indicators = [
                "Check root size by gently brushing soil away",
                "Harvest when roots reach full size",
            ]
        else:
            # Generic harvest assessment
            ready = basic_ready
            indicators = [
                "Check for ripeness indicators specific to crop",
                "Harvest at peak maturity",
            ]

        return {
            "ready": ready,
            "days_to_maturity": crop.days_to_maturity,
            "days_since_planting": int(days_since_planting),
            "harvest_indicators": indicators,
            "estimated_days_to_harvest": max(0, crop.days_to_maturity - int(days_since_planting)),
        }

    async def get_garden_dashboard(self, plot_id: str | None = None) -> dict[str, Any]:
        """Generate comprehensive garden management dashboard"""
        current_time = time.time()

        # Filter crops by plot if specified
        if plot_id and plot_id in self.garden_plots:
            plot = self.garden_plots[plot_id]
            relevant_crops = [self.crop_profiles[cid] for cid in plot.crops_planted if cid in self.crop_profiles]
        else:
            relevant_crops = list(self.crop_profiles.values())

        # Calculate garden metrics
        active_crops = len(relevant_crops)
        crops_ready_for_harvest = sum(
            1 for crop in relevant_crops if (current_time - crop.planting_date) / (24 * 3600) >= crop.days_to_maturity
        )

        # Upcoming tasks
        upcoming_tasks = []
        for crop in relevant_crops:
            days_since_planting = (current_time - crop.planting_date) / (24 * 3600)
            if days_since_planting >= crop.days_to_maturity - 7:  # Harvest within a week
                upcoming_tasks.append(f"Check {crop.name} for harvest readiness")

        return {
            "agent": "Horticulturist",
            "dashboard_type": "garden_management",
            "timestamp": current_time,
            "garden_metrics": {
                "active_crops": active_crops,
                "garden_plots": len(self.garden_plots),
                "crops_ready_harvest": crops_ready_for_harvest,
                "soil_assessments": len(self.soil_data),
                "total_space_managed_sqft": sum(plot.size_sqft for plot in self.garden_plots.values()),
            },
            "performance_metrics": {
                "crops_managed": self.crops_managed,
                "successful_harvests": self.successful_harvests,
                "recommendations_provided": self.recommendations_provided,
                "average_yield_improvement": self.average_yield_improvement,
            },
            "upcoming_tasks": upcoming_tasks,
            "seasonal_guidance": self._get_seasonal_guidance(),
            "recommendations": [
                "Monitor soil moisture levels regularly",
                "Check crops for pest and disease symptoms weekly",
                "Plan succession plantings for continuous harvest",
                "Maintain compost system for soil health",
            ],
        }

    def _get_seasonal_guidance(self) -> list[str]:
        """Get seasonal gardening guidance"""
        current_month = datetime.now().month

        if current_month in [3, 4, 5]:  # Spring
            return [
                "Prepare beds and start cool-season crops",
                "Begin hardening off seedlings started indoors",
                "Apply compost and organic fertilizers",
            ]
        if current_month in [6, 7, 8]:  # Summer
            return [
                "Focus on watering and mulching",
                "Harvest regularly to encourage production",
                "Plant heat-tolerant varieties",
            ]
        if current_month in [9, 10, 11]:  # Fall
            return [
                "Plant cool-season crops for fall harvest",
                "Begin season cleanup and composting",
                "Protect tender plants from frost",
            ]
        # Winter
        return [
            "Plan next year's garden layout",
            "Start seeds indoors for early varieties",
            "Maintain tools and garden infrastructure",
        ]

    async def initialize(self):
        """Initialize the Horticulturist Agent"""
        try:
            logger.info("Initializing Horticulturist Agent - Smart Agriculture & Permaculture System...")

            # Initialize crop database with common crops
            self.crop_database = {
                "tomato": {
                    "type": CropType.VEGETABLE,
                    "season": "warm",
                    "companions": ["basil", "parsley"],
                },
                "lettuce": {
                    "type": CropType.VEGETABLE,
                    "season": "cool",
                    "companions": ["carrots", "radish"],
                },
                "carrots": {
                    "type": CropType.ROOT,
                    "season": "cool",
                    "companions": ["lettuce", "chives"],
                },
                "beans": {
                    "type": CropType.LEGUME,
                    "season": "warm",
                    "companions": ["corn", "squash"],
                },
                "basil": {
                    "type": CropType.HERB,
                    "season": "warm",
                    "companions": ["tomato", "pepper"],
                },
            }

            # Initialize companion planting database
            self.companion_plant_db = {
                "tomato": ["basil", "parsley", "marigold", "nasturtium"],
                "lettuce": ["carrots", "radish", "onion", "garlic"],
                "carrots": ["lettuce", "chives", "parsley", "sage"],
                "beans": ["corn", "squash", "cucumber", "radish"],
                "basil": ["tomato", "pepper", "oregano", "cilantro"],
            }

            self.initialized = True
            logger.info(f"Horticulturist Agent {self.agent_id} initialized - Smart agriculture system ready")

        except Exception as e:
            logger.error(f"Failed to initialize Horticulturist Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Horticulturist Agent gracefully"""
        try:
            logger.info("Horticulturist Agent shutting down...")

            # Generate final agricultural report
            final_report = {
                "crops_managed": self.crops_managed,
                "successful_harvests": self.successful_harvests,
                "soil_assessments": self.soil_assessments_completed,
                "recommendations_provided": self.recommendations_provided,
                "garden_plots_managed": len(self.garden_plots),
            }

            logger.info(f"Horticulturist Agent final report: {final_report}")
            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Horticulturist Agent shutdown: {e}")
